"""Декодер номера ваучера (T20): картинка + порядок номеров + имя файла.

Нумерация ваучеров идёт по буксиру и сбрасывается 1 января (`docs/base_logic.md`), поэтому
следующий номер почти всегда равен «максимум подтверждённых номеров (буксир, год) + 1»
(`predict_next` в `app/services/voucher_number.py`). Декодер выбирает номер из объединения
трёх источников кандидатов:

- top-k модели номера (T17, `runtime.predict_number`);
- окно `ожидаемый ± radius` вокруг ожидаемого номера;
- номер из имени файла, если имя разбирается (`parse_voucher_name`, T01).

Оценка кандидата `n` — сумма логарифмов:

- картинка: `w · log P_img(n)` (`runtime.score_number_candidates`, CTC-правдоподобие);
- порядок: `log P(n − ожидаемый)` — дискретное распределение с пиком в 0, в окне
  эмпирическое (из `train`), вне окна — пол на номер. После долгой паузы (дни от последнего
  подтверждённого ваучера) номера могли уйти вперёд — тогда приор смешивается с
  равномерным на `[1, radius + span(пауза)]`, и окно кандидатов расширяется вперёд;
- занятый номер: штраф `used_logp`, если `n` уже подтверждён для этого (буксир, год). Штраф
  большой, но не бесконечный: копии «(2)» и дубли «12a» бывают; для них (по имени файла)
  штраф снимается;
- имя файла: `log(1 − ε)` номеру из имени и `log(ε / 998)` прочим. ε мало (строгий вариант),
  если в имени чистый номер с кодом буксира, совпадающим с буксиром ваучера; иначе — слабый
  вариант с большим ε.

Вероятности — softmax по кандидатам. Ожидаемый номер и занятые номера считает вызывающий:
в приложении — по базе (`predict_next`), в оценке на истории — :func:`history_context` по
ваучерам, которые шли **раньше** текущего (номер текущего ваучера туда не попадает).

Зависимости — только numpy и stdlib (разбор имени файла импортируется лениво).
"""

from __future__ import annotations

import json
import math
from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np

NUMBER_MIN = 1
NUMBER_MAX = 999
N_NUMBERS = NUMBER_MAX - NUMBER_MIN + 1

#: `(кандидаты как строки) -> {кандидат: log P_img}` — `runtime.score_number_candidates`
#: с привязанным кропом или `number_scores.score_candidates` с выходом модели.
NumberScorer = Callable[[Sequence[str]], Mapping[str, float]]

# Коды флагов NumberResult.flags.
FLAG_NUMBER_NO_CANDIDATES = "number_no_candidates"
FLAG_NUMBER_USED = "number_used"
FLAG_NUMBER_FAR = "number_far_from_expected"
FLAG_NUMBER_FILE_MISMATCH = "number_file_mismatch"
FLAG_NUMBER_OVERRIDE = "number_image_override"
FLAG_NUMBER_NO_EXPECTED = "number_no_expected"


# ---------------------------------------------------------------------------
# История номеров: ожидаемый и занятые номера без подглядывания
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberHistoryItem:
    """Подтверждённый ваучер истории: буксир, год, дата (порядок), номер и ключ."""

    tug_code: str
    year: int
    dt: datetime
    number: int
    key: str = ""


class HistoryContext(NamedTuple):
    """Что история говорит о номере: ожидаемый, занятые номера и пауза в днях."""

    expected: int
    used: frozenset[int]
    pause_days: float | None


def history_context(
    history: Iterable[NumberHistoryItem],
    *,
    tug_code: str,
    year: int,
    before: datetime | None,
    exclude: str | None = None,
) -> HistoryContext:
    """Ожидаемый номер, занятые номера и паузу (буксир, год) по ваучерам строго раньше `before`.

    Как `predict_next`: ожидаемый = максимум номеров + 1 (1, если ваучеров в году ещё нет).
    Пауза — дни от последнего такого ваучера до `before` (`None`, если ваучеров нет или
    `before` не задан). `before=None` — все ваучеры истории (прод: всё, что уже подтверждено).
    Ваучер с ключом `exclude` (текущий) не учитывается никогда — так приор не видит его
    истинный номер.
    """
    items = [
        item
        for item in history
        if item.tug_code == tug_code
        and item.year == year
        and (exclude is None or item.key != exclude)
        and (before is None or item.dt < before)
    ]
    numbers = frozenset(item.number for item in items)
    pause: float | None = None
    if items and before is not None:
        pause = (before - max(item.dt for item in items)).total_seconds() / 86400.0
    return HistoryContext(max(numbers, default=0) + 1, numbers, pause)


# ---------------------------------------------------------------------------
# Имя файла
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class FilenameHint:
    """Что даёт имя файла ваучера: номер, код буксира и признак копии/дубля."""

    number: int
    tug_code: str | None = None
    copy: bool = False


def filename_hint(name: str | None) -> FilenameHint | None:
    """Разобрать имя файла (`parse_voucher_name`, T01); `None`, если номера в имени нет."""
    if not name:
        return None
    from app.services.voucher_number import parse_voucher_name

    parsed = parse_voucher_name(name)
    if parsed is None or not NUMBER_MIN <= parsed.number <= NUMBER_MAX:
        return None
    return FilenameHint(
        number=parsed.number,
        tug_code=parsed.tug_code,
        copy=parsed.duplicate or parsed.copy is not None,
    )


# ---------------------------------------------------------------------------
# Приоры
# ---------------------------------------------------------------------------


DEFAULT_RADIUS = 15


def default_offset_probs(radius: int = DEFAULT_RADIUS) -> np.ndarray:
    """Сдвиг «номер − ожидаемый» по умолчанию: 0 — 93 %, дальше геометрический спад."""
    d = np.arange(-radius, radius + 1)
    probs = np.where(d >= 0, 0.5 ** np.abs(d), 0.1 * 0.5 ** np.abs(d)).astype(float)
    probs[radius] = 0.0
    probs = 0.06 * probs / probs.sum()
    probs[radius] = 0.93
    return probs  # в окне 0,99, остаток 0,01 — вне окна


@dataclass(frozen=True)
class NumberPriors:
    """Приоры декодера номера. Логарифмы натуральные.

    - `offset_logp[d + radius]` — `log P(номер − ожидаемый = d)` в окне `|d| ≤ radius`;
    - `outside_logp` — `log P` одного номера вне окна;
    - `used_logp` — добавка за номер, уже подтверждённый для (буксир, год);
    - `file_eps` / `file_eps_weak` — доля ваучеров, где номер в имени файла не равен
      истинному: строгий (чистый номер с кодом своего буксира) и слабый вариант;
    - `image_weight` — множитель `log P_img`; `top_k` — кандидатов из картинки;
    - режим «после паузы»: если от последнего подтверждённого ваучера (буксир, год) прошло не
      меньше `long_pause_days` дней, номера могли уйти вперёд (ваучеры, которых нет в базе).
      Тогда приор — смесь `long_weight · обычный + (1 − long_weight) · равномерный на
      [1, radius + span]`, где `span = ⌈span_factor · rate · пауза⌉`, `rate` — ваучеров в день.
    """

    offset_logp: tuple[float, ...]
    outside_logp: float
    used_logp: float = math.log(1e-3)
    file_eps: float = 0.005
    file_eps_weak: float = 0.05
    image_weight: float = 1.0
    radius: int = DEFAULT_RADIUS
    top_k: int = 5
    long_pause_days: float = 4.0
    long_weight: float = 0.5
    rate: float = 0.93
    span_factor: float = 1.5
    meta: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        if len(self.offset_logp) != 2 * self.radius + 1:
            raise ValueError("offset_logp: нужно 2·radius + 1 значений")
        for name in ("file_eps", "file_eps_weak", "long_weight"):
            if not 0.0 < getattr(self, name) < 1.0:
                raise ValueError(f"{name} должен быть в (0, 1)")

    @classmethod
    def from_offset_probs(
        cls, probs: Sequence[float] | np.ndarray, outside_mass: float, **kwargs: Any
    ) -> NumberPriors:
        """Собрать приоры из вероятностей сдвигов в окне и массы вне окна."""
        p = np.asarray(probs, dtype=float)
        radius = (p.size - 1) // 2
        with np.errstate(divide="ignore"):
            offset = np.log(p)
        n_out = max(1, N_NUMBERS - p.size)
        outside = math.log(max(outside_mass, 1e-12) / n_out)
        # Сдвиг в окне не дешевле, чем номер вне окна: пол против нулевых бинов.
        offset = np.maximum(offset, outside)
        return cls(
            offset_logp=tuple(float(v) for v in offset),
            outside_logp=outside,
            radius=radius,
            **kwargs,
        )

    @classmethod
    def default(cls) -> NumberPriors:
        probs = default_offset_probs()
        return cls.from_offset_probs(probs, 1.0 - float(probs.sum()), meta={"source": "defaults"})

    def short_value(self, d: int) -> float:
        """`log P(d)` обычного режима: окно ±radius, вне окна — пол на номер."""
        if -self.radius <= d <= self.radius:
            return self.offset_logp[d + self.radius]
        return self.outside_logp

    def is_long(self, pause_days: float | None) -> bool:
        return pause_days is not None and pause_days >= self.long_pause_days

    def span(self, pause_days: float | None) -> int:
        """Насколько номера могли уйти вперёд за паузу (0 в обычном режиме)."""
        if not self.is_long(pause_days):
            return 0
        assert pause_days is not None
        return int(math.ceil(self.span_factor * self.rate * pause_days))

    def offset_value(self, d: int, pause_days: float | None = None) -> float:
        """`log P(номер − ожидаемый = d)` с учётом паузы после последнего ваучера."""
        short = self.short_value(d)
        if not self.is_long(pause_days):
            return short
        width = self.radius + self.span(pause_days)
        mixed = self.long_weight * math.exp(short)
        if 1 <= d <= width:
            mixed += (1.0 - self.long_weight) / width
        return math.log(mixed) if mixed > 0 else self.outside_logp

    def file_logp(self, match: bool, strong: bool) -> float:
        eps = self.file_eps if strong else self.file_eps_weak
        return math.log1p(-eps) if match else math.log(eps / (N_NUMBERS - 1))

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["offset_logp"] = list(self.offset_logp)
        data["meta"] = dict(self.meta)
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> NumberPriors:
        known = set(cls.__dataclass_fields__)
        kwargs = {k: v for k, v in data.items() if k in known}
        kwargs["offset_logp"] = tuple(float(v) for v in data["offset_logp"])
        kwargs["meta"] = dict(data.get("meta", {}))
        return cls(**kwargs)

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")


# ---------------------------------------------------------------------------
# Контекст и результат
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberContext:
    """Контекст номера ваучера.

    - `expected` — ожидаемый следующий номер (буксир, год): `predict_next` в приложении или
      :func:`history_context` на истории. `None` — приора порядка нет.
    - `used` — номера, уже подтверждённые для (буксир, год).
    - `pause_days` — дни от последнего подтверждённого ваучера (буксир, год) до текущего;
      после долгой паузы приор порядка шире (номера могли уйти вперёд).
    - `filename` — имя файла ваучера (как загружено), `tug_code` — буксир ваучера (k/p),
      если известен: строгий приор имени — только когда код в имени совпадает с ним.
    """

    expected: int | None = None
    used: frozenset[int] = frozenset()
    pause_days: float | None = None
    filename: str | None = None
    tug_code: str | None = None


@dataclass(frozen=True)
class NumberResult:
    """Итог декодирования номера.

    - `candidates` — `(номер, p)` по убыванию `p` (softmax по всем кандидатам), не больше
      `top` штук; `value = candidates[0][0]`;
    - `confidence = p(top1)`, `margin = p(top1) − p(top2)`;
    - `image_top` — top-1 картинки (для флага `number_image_override`);
    - `scores` — полная оценка каждого кандидата (для отладки и пар).
    """

    candidates: tuple[tuple[int, float], ...]
    confidence: float
    margin: float
    flags: tuple[str, ...]
    expected: int | None
    image_top: int | None
    scores: Mapping[int, float] = field(default_factory=dict, repr=False)

    @property
    def value(self) -> int | None:
        return self.candidates[0][0] if self.candidates else None

    def to_json(self, k: int = 5) -> list[list[float]]:
        """Подполе `voucher_number` формата T13: `[[номер, p], …]`."""
        return [[n, p] for n, p in self.candidates[:k]]


# ---------------------------------------------------------------------------
# Декодирование
# ---------------------------------------------------------------------------


def number_candidates(
    image_top: Sequence[tuple[int, float]],
    context: NumberContext,
    priors: NumberPriors,
    hint: FilenameHint | None = None,
) -> list[int]:
    """Объединение источников: top-k картинки, окно вокруг ожидаемого, номер из имени."""
    cands = {int(n) for n, _ in list(image_top)[: priors.top_k]}
    if context.expected is not None:
        e = int(context.expected)
        hi = e + priors.radius + priors.span(context.pause_days)
        cands.update(range(e - priors.radius, hi + 1))
    if hint is not None:
        cands.add(hint.number)
    return sorted(n for n in cands if NUMBER_MIN <= n <= NUMBER_MAX)


def candidate_prior(
    n: int, context: NumberContext, priors: NumberPriors, hint: FilenameHint | None
) -> float:
    """Приорная часть оценки кандидата (порядок, занятость, имя файла)."""
    score = 0.0
    if context.expected is not None:
        score += priors.offset_value(n - int(context.expected), context.pause_days)
    copy_of_used = hint is not None and hint.copy and hint.number == n
    if n in context.used and not copy_of_used:
        score += priors.used_logp
    if hint is not None:
        strong = hint.tug_code is not None and (
            context.tug_code is None or hint.tug_code == context.tug_code
        )
        score += priors.file_logp(n == hint.number, strong)
    return score


def decode_number(
    image_top: Sequence[tuple[int, float]],
    scorer: NumberScorer,
    context: NumberContext,
    priors: NumberPriors | None = None,
    *,
    top: int = 5,
) -> NumberResult:
    """Выбрать номер ваучера.

    `image_top` — top-k модели (`runtime.predict_number(crop)`), `scorer` — оценка
    произвольных кандидатов той же модели (`lambda c: runtime.score_number_candidates(crop,
    c)`). Без кропа номера можно передать `image_top=()` и `scorer=lambda c: {}`: тогда
    решают порядок и имя файла.
    """
    priors = priors or NumberPriors.default()
    hint = filename_hint(context.filename)
    cands = number_candidates(image_top, context, priors, hint)
    flags: set[str] = set()
    if context.expected is None:
        flags.add(FLAG_NUMBER_NO_EXPECTED)
    image_top1 = int(image_top[0][0]) if image_top else None
    empty_flags = tuple(sorted({*flags, FLAG_NUMBER_NO_CANDIDATES}))
    if not cands:
        return NumberResult((), 0.0, 0.0, empty_flags, context.expected, image_top1)
    raw = scorer([str(n) for n in cands])
    scores: dict[int, float] = {}
    for n in cands:
        img = raw.get(str(n))
        img_term = priors.image_weight * float(img) if img is not None else 0.0
        scores[n] = img_term + candidate_prior(n, context, priors, hint)
    values = np.asarray([scores[n] for n in cands])
    finite = np.isfinite(values)
    if not finite.any():
        return NumberResult((), 0.0, 0.0, empty_flags, context.expected, image_top1, scores)
    m = float(values[finite].max())
    probs = np.where(finite, np.exp(values - m), 0.0)
    probs = probs / probs.sum()
    order = np.argsort(-probs, kind="stable")
    ranked = tuple((cands[i], float(probs[i])) for i in order[:top] if probs[i] > 0)
    best = ranked[0][0]
    p2 = ranked[1][1] if len(ranked) > 1 else 0.0
    if best in context.used and not (hint is not None and hint.copy and hint.number == best):
        flags.add(FLAG_NUMBER_USED)
    if context.expected is not None:
        d = best - int(context.expected)
        if d < -priors.radius or d > priors.radius + priors.span(context.pause_days):
            flags.add(FLAG_NUMBER_FAR)
    if hint is not None and best != hint.number:
        flags.add(FLAG_NUMBER_FILE_MISMATCH)
    if image_top1 is not None and best != image_top1:
        flags.add(FLAG_NUMBER_OVERRIDE)
    return NumberResult(
        candidates=ranked,
        confidence=ranked[0][1],
        margin=ranked[0][1] - p2,
        flags=tuple(sorted(flags)),
        expected=context.expected,
        image_top=image_top1,
        scores=scores,
    )


def decode_number_output(
    output: np.ndarray | None,
    context: NumberContext,
    priors: NumberPriors | None = None,
    *,
    kind: str = "ctc",
    temperature: float | Sequence[float] = 1.0,
    top: int = 5,
) -> NumberResult:
    """`decode_number` по сырому выходу модели номера (логиты CTC или голов, T17/T18)."""
    from app.ocr.number_scores import score_candidates, top_numbers

    priors = priors or NumberPriors.default()
    if output is None:
        return decode_number((), lambda c: {}, context, priors, top=top)
    image_top = top_numbers(output, kind=kind, k=priors.top_k, temperature=temperature)
    return decode_number(
        image_top,
        lambda c: score_candidates(output, c, kind=kind, temperature=temperature),
        context,
        priors,
        top=top,
    )
