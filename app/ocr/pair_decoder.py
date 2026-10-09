"""Совместное декодирование пары ваучеров одной работы (T20): буксиры k и p.

Если на одну работу вышли оба буксира, у каждого свой ваучер, а времена в двух третях пар
совпадают целиком (план, этап 3). Это два независимых прочтения одних чисел, написанных
разными людьми. Декодер пары пересчитывает top-N записей каждого ваучера (ядро T19,
:func:`app.ocr.decoder.decode`) совместно:

    оценка(a, b) = score_a(a) + score_b(b) + w · связь(a, b),
    связь(a, b) = log(π · [все времена равны] + (1 − π) · близость(Δ)).

- `score_*` — ненормированные оценки ядра (картинка + приоры записи);
- `Δ` — разности фактических времён строк `a − b` в минутах (`24:00` = `00:00` след. суток);
- `близость(Δ) = P(шаблон расходящихся строк | не все равны) · Π_r f_r(|Δ_r|) · m(Δ_r) / 2` —
  по строкам, которые расходятся; `f_r` — плотность |Δ| на минуту (бины), `/ 2` — знак,
  `m` — поправка на кратность 10: обе записи в минутах, кратных 10, поэтому почти вся масса
  бина приходится на `|Δ|`, кратные 10 (доля `mult10_share`), а масса бина сохраняется.
  `π`, шаблоны, `f_r` и доля — из `train` по `pair_id` манифеста (`ocr_lab.fit_number_pairs`).

Кандидаты стороны — её top-N записей ядра плюс top-N записей партнёра (оценённые по своей
картинке): так верная общая запись находится, даже если у одного ваучера она за пределами
top-N. Масса записей, которые не перечислены (`exp(log_z) −` перечисленное), входит как
одна «прочая» запись со связью `rest_link`, поэтому уверенность не завышается.

Если ваучер партнёра уже подтверждён, его значения не перебираются, а работают как сильный
приор: `оценка(a) = score_a(a) + w · связь(a, подтверждённый)`, а сама подтверждённая запись
добавляется в кандидаты.

Контракт для приложения (T26): пара определяется по одной заявке или работе;
:class:`PairContext` принимает контекст партнёра и его распределения подполей (ваучер
ожидает проверки) либо его подтверждённую запись (`b_confirmed`, написание бланка).

Зависимости — только numpy и stdlib.
"""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.ocr.decoder import (
    CHAIN,
    ROWS,
    BinnedLogDensity,
    DecodeContext,
    DecodeResult,
    DecoderInputs,
    DecoderModel,
    FormRow,
    Record,
    _evidence,
    _score_scalar,
    candidate_space,
    decode,
    record_from_forms,
    rerank_result,
)

_NEG_INF = float("-inf")

# Флаги пары (добавляются к флагам DecodeResult каждой стороны).
FLAG_PAIR_SAME = "pair_same"
FLAG_PAIR_DIFFERS = "pair_differs"
FLAG_PAIR_CHANGED = "pair_changed"
FLAG_PAIR_CONFIRMED = "pair_partner_confirmed"

# Бины |Δ| времени строки между ваучерами пары, минуты (Δ = 0 — отдельный случай «равны»).
DELTA_EDGES: tuple[float, ...] = (
    1.0,
    10.0,
    20.0,
    30.0,
    60.0,
    90.0,
    120.0,
    150.0,
    180.0,
    210.0,
    240.0,
    300.0,
    480.0,
    720.0,
    1440.0,
    2880.0,
    4320.0,
)
# Массы бинов по умолчанию (по цифрам train T20: мелкие сдвиги 10–20 мин у выхода и большие
# 100–230 мин у окончания и прихода).
DEFAULT_DELTA_MASSES: tuple[float, ...] = (
    0.01,  # 1–10
    0.20,  # 10–20
    0.10,  # 20–30
    0.04,  # 30–60
    0.03,  # 60–90
    0.12,  # 90–120
    0.14,  # 120–150
    0.06,  # 150–180
    0.10,  # 180–210
    0.10,  # 210–240
    0.05,  # 240–300
    0.03,  # 300–480
    0.01,  # 480–720
    0.005,  # 720–1440
    0.003,  # 1–2 сут
    0.002,  # 2–3 сут
)
# Доля пар, где совпадают все времена: 131 из 195 по плану (67 %).
DEFAULT_SAME_SHARE = 0.67
# Шаблоны расходящихся строк (по CHAIN) среди пар, где не всё совпало.
DEFAULT_DIFF_PATTERNS: dict[tuple[int, ...], float] = {
    (0, 0, 1, 1): 0.35,
    (1, 0, 0, 0): 0.25,
    (1, 1, 0, 0): 0.20,
    (0, 0, 0, 1): 0.08,
    (1, 0, 1, 1): 0.05,
}
DELTA_FLOOR_PER_MINUTE = 1e-7


def diff_patterns() -> list[tuple[int, ...]]:
    """Все 15 шаблонов «какие строки расходятся» (без «все равны»)."""
    out = []
    for mask in range(1, 2 ** len(CHAIN)):
        out.append(tuple((mask >> (len(CHAIN) - 1 - i)) & 1 for i in range(len(CHAIN))))
    return out


def pattern_key(pattern: Sequence[int]) -> str:
    return "".join(str(int(v)) for v in pattern)


def parse_pattern_key(key: str) -> tuple[int, ...]:
    return tuple(int(c) for c in key)


def record_times(record: Record | Mapping[str, FormRow]) -> tuple[datetime, ...]:
    """Фактические времена строк в порядке CHAIN (запись ядра или написание бланка)."""
    if isinstance(record, Record):
        return tuple(record.rows[r].dt for r in CHAIN)
    return tuple(record_from_forms(record, 0.0).rows[r].dt for r in CHAIN)


def time_deltas(a: Sequence[datetime], b: Sequence[datetime]) -> tuple[int, ...]:
    """Разности `a − b` по строкам, минуты."""
    return tuple(int(round((x - y).total_seconds() / 60.0)) for x, y in zip(a, b, strict=True))


@dataclass(frozen=True)
class PairPriors:
    """Приоры связи пары. Логарифмы натуральные.

    - `same_logp = log π` — все четыре времени равны;
    - `pattern_logp` — `log P(шаблон расходящихся строк | не все равны)`, ключ — кортеж по
      CHAIN (1 — строка расходится); неизвестный шаблон — `pattern_floor`;
    - `delta[row]` — лог-плотность |Δ| строки на минуту (`|Δ| ≥ 1`);
    - `mult10_share` — доля ненулевых |Δ|, кратных 10;
    - `rest_link` — связь «прочей» (неперечисленной) записи с любой записью партнёра;
    - `top_n` — записей ядра с каждой стороны, `weight` — множитель связи.
    """

    same_logp: float
    pattern_logp: Mapping[tuple[int, ...], float]
    pattern_floor: float
    delta: Mapping[str, BinnedLogDensity]
    rest_link: float
    mult10_share: float = 0.99
    top_n: int = 10
    weight: float = 1.0
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if not self.same_logp < 0:
            raise ValueError("same_logp: нужна вероятность меньше 1")
        if not 0.0 < self.mult10_share < 1.0:
            raise ValueError("mult10_share должен быть в (0, 1)")
        for row in CHAIN:
            if row not in self.delta:
                raise ValueError(f"нет плотности |Δ| строки {row}")

    @property
    def diff_logp(self) -> float:
        return math.log1p(-math.exp(self.same_logp))

    def link_deltas(self, deltas: Sequence[int]) -> float:
        """Связь по разностям строк (минуты, по CHAIN), без веса."""
        pattern = tuple(int(d != 0) for d in deltas)
        if not any(pattern):
            return self.same_logp
        value = self.diff_logp + self.pattern_logp.get(pattern, self.pattern_floor)
        mult = math.log(10.0 * self.mult10_share)
        other = math.log(10.0 * (1.0 - self.mult10_share) / 9.0)
        for row, d in zip(CHAIN, deltas, strict=True):
            if d != 0:
                value += float(self.delta[row](abs(d))) - math.log(2.0)
                value += mult if d % 10 == 0 else other
        return value

    def link(
        self,
        a: Record | Mapping[str, FormRow] | Sequence[datetime],
        b: Record | Mapping[str, FormRow] | Sequence[datetime],
    ) -> float:
        """Взвешенная связь двух записей."""
        ta = a if isinstance(a, Sequence) else record_times(a)
        tb = b if isinstance(b, Sequence) else record_times(b)
        return self.weight * self.link_deltas(time_deltas(ta, tb))

    @classmethod
    def default(cls) -> PairPriors:
        """Приоры по цифрам плана — чтобы пары работали без файла."""
        density = BinnedLogDensity.from_masses(
            DELTA_EDGES, DEFAULT_DELTA_MASSES, DELTA_FLOOR_PER_MINUTE
        )
        known = sum(DEFAULT_DIFF_PATTERNS.values())
        others = [p for p in diff_patterns() if p not in DEFAULT_DIFF_PATTERNS]
        pattern = {p: math.log(v) for p, v in DEFAULT_DIFF_PATTERNS.items()}
        rest = (1.0 - known) / len(others)
        pattern.update({p: math.log(rest) for p in others})
        priors = cls(
            same_logp=math.log(DEFAULT_SAME_SHARE),
            pattern_logp=pattern,
            pattern_floor=math.log(rest),
            delta={row: density for row in CHAIN},
            rest_link=0.0,
            meta={"source": "defaults_from_plan"},
        )
        # Связь «прочей» записи — как у типичного расхождения: одна строка на 2 часа.
        return replace(priors, rest_link=priors.link_deltas((0, 0, 0, 120)))

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "same_logp": self.same_logp,
            "pattern_logp": {pattern_key(p): v for p, v in self.pattern_logp.items()},
            "pattern_floor": self.pattern_floor,
            "delta": {row: d.to_json() for row, d in self.delta.items()},
            "rest_link": self.rest_link,
            "mult10_share": self.mult10_share,
            "top_n": self.top_n,
            "weight": self.weight,
            "meta": dict(self.meta),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> PairPriors:
        if int(data.get("version", 0)) != 1:
            raise ValueError("неизвестная версия приоров пары")
        return cls(
            same_logp=float(data["same_logp"]),
            pattern_logp={parse_pattern_key(k): float(v) for k, v in data["pattern_logp"].items()},
            pattern_floor=float(data["pattern_floor"]),
            delta={row: BinnedLogDensity.from_json(v) for row, v in data["delta"].items()},
            rest_link=float(data["rest_link"]),
            mult10_share=float(data.get("mult10_share", 0.99)),
            top_n=int(data.get("top_n", 10)),
            weight=float(data.get("weight", 1.0)),
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Контракт входа и выхода
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PairContext:
    """Контекст пары: ваучер `a` и его партнёр `b` по той же заявке или работе.

    - `a`, `b` — контексты ядра (год, заявка, вид работ) каждого ваучера; если `b` не задан,
      берётся `a` (у пары одна заявка);
    - `b_confirmed` — подтверждённая запись партнёра в написании бланка
      (`{строка: (дата на бланке, час, минута)}`). Тогда распределения партнёра не нужны:
      его значения — сильный приор для `a`.
    """

    a: DecodeContext
    b: DecodeContext | None = None
    b_confirmed: Mapping[str, FormRow] | None = None

    @property
    def b_context(self) -> DecodeContext:
        return self.b or self.a


@dataclass(frozen=True)
class PairResult:
    """Итог декодирования пары.

    - `a`, `b` — результаты сторон в формате ядра: `records` переранжированы по
      маргинальной вероятности пары (`p`), `confidence`/`margin` — от неё же, флаги лучшей
      записи пересчитаны, `marginals` строк — от ядра; `b = None`, если партнёр подтверждён;
    - `p_same` — вероятность того, что времена ваучеров совпадают целиком;
    - `joint_confidence` — вероятность лучшей пары `(a, b)` целиком;
    - `flags` — флаги пары (`pair_same` / `pair_differs`, `pair_partner_confirmed`).
    """

    a: DecodeResult
    b: DecodeResult | None
    p_same: float
    joint_confidence: float
    flags: tuple[str, ...]


@dataclass
class _Side:
    """Кандидаты одной стороны: написания, оценки, времена и масса «прочих» записей."""

    forms: list[dict[str, FormRow]]
    scores: list[float]
    times: list[tuple[datetime, ...]]
    rest: float
    core: DecodeResult


def _forms_key(forms: Mapping[str, FormRow]) -> tuple[FormRow, ...]:
    return tuple(forms[r] for r in ROWS)


def _logsumexp(values: Sequence[float]) -> float:
    arr = np.asarray([v for v in values if math.isfinite(v)], dtype=float)
    if arr.size == 0:
        return _NEG_INF
    m = float(arr.max())
    return m + math.log(float(np.exp(arr - m).sum()))


def _core(
    inputs: DecoderInputs, context: DecodeContext, model: DecoderModel, n: int
) -> DecodeResult:
    return decode(inputs, context, replace(model, search=replace(model.search, top_n=n)))


def _build_side(
    core: DecodeResult,
    inputs: DecoderInputs,
    context: DecodeContext,
    model: DecoderModel,
    extra: Sequence[Mapping[str, FormRow]],
) -> _Side:
    """Кандидаты стороны: top-N ядра + `extra` (записи партнёра), оценённые своей картинкой."""
    ev = _evidence(inputs, model.search.prob_floor)
    space = candidate_space(inputs, context, model)
    forms: list[dict[str, FormRow]] = []
    scores: list[float] = []
    seen: set[tuple[FormRow, ...]] = set()
    in_space: list[float] = []
    for rec in core.records:
        f = rec.forms()
        seen.add(_forms_key(f))
        forms.append(f)
        scores.append(rec.score)
        in_space.append(rec.score)
    for other in extra:
        key = _forms_key(other)
        if key in seen:
            continue
        seen.add(key)
        score = _score_scalar(other, ev, context, model)
        if not math.isfinite(score):
            continue
        forms.append(dict(other))
        scores.append(score)
        if space.contains(other):
            in_space.append(score)
    rest = _NEG_INF
    if math.isfinite(core.log_z):
        covered = _logsumexp(in_space)
        ratio = math.exp(covered - core.log_z) if math.isfinite(covered) else 0.0
        if ratio < 1.0 - 1e-12:
            rest = core.log_z + math.log1p(-ratio)
    times = [record_times(f) for f in forms]
    return _Side(forms=forms, scores=scores, times=times, rest=rest, core=core)


def _side_result(
    side: _Side,
    marginal: np.ndarray,
    inputs: DecoderInputs,
    context: DecodeContext,
    keep: int,
    extra_flags: Sequence[str],
) -> DecodeResult:
    order = [i for i in np.argsort(-marginal, kind="stable") if marginal[i] > 0][:keep]
    records = [record_from_forms(side.forms[i], side.scores[i], float(marginal[i])) for i in order]
    flags = list(extra_flags)
    core_top = side.core.records[0].forms() if side.core.records else None
    if records and core_top != records[0].forms():
        flags.append(FLAG_PAIR_CHANGED)
    return rerank_result(side.core, records, inputs, context, extra_flags=flags)


def decode_pair(
    a_inputs: DecoderInputs,
    b_inputs: DecoderInputs | None,
    context: PairContext,
    model: DecoderModel | None = None,
    pair: PairPriors | None = None,
) -> PairResult:
    """Совместно декодировать ваучер `a` и его партнёра `b` (другой буксир, та же работа).

    `b_inputs` — распределения подполей партнёра (формат `runtime.predict_digits`) или
    `None`, если задан `context.b_confirmed`.
    """
    model = model or DecoderModel.default()
    pair = pair or PairPriors.default()
    keep = max(1, model.search.top_n)
    n = max(pair.top_n, keep)
    if context.b_confirmed is not None:
        return _decode_with_confirmed(a_inputs, context, model, pair, n, keep)
    if b_inputs is None:
        raise ValueError("нужны распределения партнёра или его подтверждённая запись")
    ctx_b = context.b_context
    # Сначала ядро каждой стороны, затем добавляем записи партнёра в кандидаты.
    core_a = _core(a_inputs, context.a, model, n)
    core_b = _core(b_inputs, ctx_b, model, n)
    side_a = _build_side(core_a, a_inputs, context.a, model, [r.forms() for r in core_b.records])
    side_b = _build_side(core_b, b_inputs, ctx_b, model, [r.forms() for r in core_a.records])

    sa = np.asarray([*side_a.scores, side_a.rest])
    sb = np.asarray([*side_b.scores, side_b.rest])
    na, nb = len(side_a.scores), len(side_b.scores)
    link = np.full((na + 1, nb + 1), pair.weight * pair.rest_link)
    same = np.zeros((na + 1, nb + 1), dtype=bool)
    for i, ta in enumerate(side_a.times):
        for j, tb in enumerate(side_b.times):
            same[i, j] = ta == tb
            link[i, j] = pair.link(ta, tb)
    total = sa[:, None] + sb[None, :] + link
    finite = np.isfinite(total)
    if not finite.any():
        empty_a = rerank_result(side_a.core, [], a_inputs, context.a)
        empty_b = rerank_result(side_b.core, [], b_inputs, ctx_b)
        return PairResult(empty_a, empty_b, 0.0, 0.0, ())
    m = float(total[finite].max())
    prob = np.where(finite, np.exp(total - m), 0.0)
    prob /= prob.sum()
    p_same = float(prob[same].sum())
    best_i, best_j = np.unravel_index(int(np.argmax(prob)), prob.shape)
    flags = [FLAG_PAIR_SAME if same[best_i, best_j] else FLAG_PAIR_DIFFERS]
    res_a = _side_result(side_a, prob[:na].sum(axis=1), a_inputs, context.a, keep, flags)
    res_b = _side_result(side_b, prob[:, :nb].sum(axis=0), b_inputs, ctx_b, keep, flags)
    return PairResult(res_a, res_b, p_same, float(prob[best_i, best_j]), tuple(flags))


def _decode_with_confirmed(
    a_inputs: DecoderInputs,
    context: PairContext,
    model: DecoderModel,
    pair: PairPriors,
    n: int,
    keep: int,
) -> PairResult:
    confirmed = context.b_confirmed
    assert confirmed is not None
    core = _core(a_inputs, context.a, model, n)
    side = _build_side(core, a_inputs, context.a, model, [confirmed])
    t_conf = record_times(confirmed)
    link = [pair.link(t, t_conf) for t in side.times]
    total = np.asarray([*side.scores, side.rest]) + np.asarray(
        [*link, pair.weight * pair.rest_link]
    )
    finite = np.isfinite(total)
    flags = [FLAG_PAIR_CONFIRMED]
    if not finite.any():
        return PairResult(
            rerank_result(side.core, [], a_inputs, context.a, extra_flags=flags),
            None,
            0.0,
            0.0,
            tuple(flags),
        )
    m = float(total[finite].max())
    prob = np.where(finite, np.exp(total - m), 0.0)
    prob /= prob.sum()
    same = np.asarray([t == t_conf for t in side.times] + [False])
    p_same = float(prob[same].sum())
    best = int(np.argmax(prob))
    flags.append(FLAG_PAIR_SAME if same[best] else FLAG_PAIR_DIFFERS)
    res = _side_result(side, prob[:-1], a_inputs, context.a, keep, flags)
    return PairResult(res, None, p_same, float(prob[best]), tuple(flags))
