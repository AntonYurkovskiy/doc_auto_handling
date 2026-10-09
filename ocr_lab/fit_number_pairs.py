"""Номер ваучера и пары k/p в декодере (T20): приоры по `train` и сравнение вариантов.

Запуск (пути — из `ocr_lab.paths`, то есть от `OCR_WORK_DIR`):

    python -m ocr_lab.fit_number_pairs fit               # приоры номера и пары по train
    python -m ocr_lab.fit_number_pairs eval --split val  # ядро / + номер / + пары → отчёт

- `fit` — приоры номера (`app.ocr.number_decoder.NumberPriors`): распределение сдвига «номер −
  ожидаемый» в окне ±15 и масса вне окна, штраф за занятый номер, доля расхождений имени
  файла с истиной; приоры пары (`app.ocr.pair_decoder.PairPriors`): доля пар с полностью
  совпадающими временами `π`, шаблоны расходящихся строк и плотности |Δ| по строкам — по
  `pair_id` манифеста (T02). Пишет `models/number_priors_v0.json` и `models/pair_priors_v0.json`.
- `eval` — варианты на сплите: 1) ядро T19 (номер — top-1 `number_v0`), 2) ядро + номер
  (без имени файла и с ним), 3) ядро + номер + пары, плюс «партнёр подтверждён». Метрики:
  запись целиком, запись с номером, номер, покрытие при точности 99,5 % — по всем ваучерам и
  на подмножестве с парой. Пишет `reports/decoder_t20_compare_<split>.{md,json}` и JSONL T13
  итогового варианта `models/decoder_v1/<split>_predictions.jsonl` (для `ocr_lab.evaluate`).

Честность оценки номера. Ожидаемый номер и занятые номера считаются только по ваучерам того
же (буксир, год), которые шли **раньше** текущего — как в проде, где подтверждены только
прошлые ваучеры (`app.ocr.number_decoder.history_context`). Номера истории — из имён файлов
(так их заполняет приложение, `apply_filename_fields`), буксир — по коду в имени файла.
Текущий ваучер исключается по `scan_id`, а «раньше» отсчитывается от даты выхода, которую
прочитал декодер (ядро), а не от истины.

В консоль и отчёты идут только агрегаты и `scan_id`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.ocr.decoder import (
    CHAIN,
    BinnedLogDensity,
    DecodeContext,
    DecodeResult,
    DecoderInputs,
    DecoderModel,
    FormRow,
    decode,
)
from app.ocr.number_decoder import (
    NumberContext,
    NumberHistoryItem,
    NumberPriors,
    NumberResult,
    decode_number_output,
    default_offset_probs,
    filename_hint,
    history_context,
)
from app.ocr.pair_decoder import (
    DEFAULT_DELTA_MASSES,
    DEFAULT_DIFF_PATTERNS,
    DEFAULT_SAME_SHARE,
    DELTA_EDGES,
    DELTA_FLOOR_PER_MINUTE,
    PairContext,
    PairPriors,
    decode_pair,
    diff_patterns,
    record_times,
    time_deltas,
)
from ocr_lab import paths
from ocr_lab.fit_decoder_priors import (
    DECODER_DIR,
    DECODER_JSON,
    NUMBER_DIR,
    _dt,
    _int,
    context_for,
    load_predictions,
    read_manifest,
    truth_forms,
)
from ocr_lab.predictions import VOUCHER_NUMBER

NUMBER_JSON = paths.MODELS_DIR / "number_priors_v0.json"
PAIR_JSON = paths.MODELS_DIR / "pair_priors_v0.json"
OUT_DIR = paths.MODELS_DIR / "decoder_v1"
SOURCE = "decoder_v1"
TARGET = 0.995
# Фиксированные пороги уверенности для отчёта: «принято / из них ошибок».
FIXED_THRESHOLDS: tuple[float, ...] = (0.99, 0.999)
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# История номеров
# ---------------------------------------------------------------------------


def tug_of(row: Mapping[str, str]) -> str:
    """Код буксира: из имени файла (первичный источник, T01), иначе из манифеста."""
    hint = filename_hint(row.get("voucher_file"))
    if hint is not None and hint.tug_code:
        return hint.tug_code
    return str(row.get("tug_code") or "").strip()


def history_from_manifest(rows: Sequence[Mapping[str, str]]) -> list[NumberHistoryItem]:
    """История подтверждённых номеров: номер из имени файла (иначе манифеста), дата выхода."""
    out: list[NumberHistoryItem] = []
    for row in rows:
        dt = _dt(row.get("left_base_dt"))
        year = _int(row.get("year"))
        hint = filename_hint(row.get("voucher_file"))
        number = hint.number if hint is not None else _int(row.get("voucher_number"))
        if dt is None or year is None or number is None:
            continue
        out.append(NumberHistoryItem(tug_of(row), year, dt, number, row["scan_id"]))
    return out


def number_context(
    row: Mapping[str, str],
    history: Sequence[NumberHistoryItem],
    before: datetime | None,
    *,
    use_filename: bool,
) -> NumberContext:
    """Контекст номера ваучера по истории строго раньше `before` (без самого ваучера)."""
    year = _int(row.get("year"))
    tug = tug_of(row)
    expected: int | None = None
    used: frozenset[int] = frozenset()
    pause: float | None = None
    if year is not None and tug and before is not None:
        expected, used, pause = history_context(
            history, tug_code=tug, year=year, before=before, exclude=row["scan_id"]
        )
    return NumberContext(
        expected=expected,
        used=used,
        pause_days=pause,
        filename=row.get("voucher_file") if use_filename else None,
        tug_code=tug or None,
    )


# ---------------------------------------------------------------------------
# Приоры номера
# ---------------------------------------------------------------------------


@dataclass
class NumberFitStats:
    n: int = 0
    n_long: int = 0
    offsets: dict[int, int] = field(default_factory=dict)
    long_offsets: dict[int, int] = field(default_factory=dict)
    outside: int = 0
    rate: float = float("nan")
    long_weight: float = float("nan")
    used_hits: int = 0
    file_strong: int = 0
    file_strong_mismatch: int = 0
    file_mismatch_ids: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["offsets"] = {str(k): v for k, v in sorted(self.offsets.items())}
        out["long_offsets"] = {str(k): v for k, v in sorted(self.long_offsets.items())}
        return out


def fit_number_priors(
    rows: Sequence[Mapping[str, str]],
    history: Sequence[NumberHistoryItem],
    *,
    radius: int = 15,
    alpha: float = 10.0,
    long_pause_days: float = 4.0,
    span_factor: float = 1.5,
) -> tuple[NumberPriors, NumberFitStats]:
    """Приоры номера по строкам `train`; «раньше» — по истинной дате выхода ваучера.

    Окно сдвигов — по ваучерам с короткой паузой, сглаживание к :func:`default_offset_probs`
    псевдосчётчиком `alpha`. Вес смеси режима «после паузы» — максимум правдоподобия по
    ваучерам с паузой `≥ long_pause_days` (сетка), темп — ваучеров в день по истории `train`.
    Штраф за занятый номер и доля расхождений имени файла — частоты со сглаживанием
    `+0,5 / +1`.
    """
    stats = NumberFitStats()
    counts = np.zeros(2 * radius + 1)
    long_d: list[tuple[int, float]] = []
    for row in rows:
        truth = _int(row.get("voucher_number"))
        before = _dt(row.get("left_base_dt"))
        if truth is None or before is None:
            continue
        ctx = number_context(row, history, before, use_filename=False)
        if ctx.expected is None:
            continue
        d = truth - ctx.expected
        stats.used_hits += truth in ctx.used
        hint = filename_hint(row.get("voucher_file"))
        if hint is not None and hint.tug_code is not None:
            stats.file_strong += 1
            if hint.number != truth:
                stats.file_strong_mismatch += 1
                stats.file_mismatch_ids.append(row["scan_id"])
        if ctx.pause_days is not None and ctx.pause_days >= long_pause_days:
            stats.n_long += 1
            stats.long_offsets[d] = stats.long_offsets.get(d, 0) + 1
            long_d.append((d, ctx.pause_days))
            continue
        stats.n += 1
        stats.offsets[d] = stats.offsets.get(d, 0) + 1
        if abs(d) <= radius:
            counts[d + radius] += 1
        else:
            stats.outside += 1
    base_window = default_offset_probs(radius)
    base = np.append(base_window, max(0.0, 1.0 - float(base_window.sum())))
    observed = np.append(counts, stats.outside)
    smoothed = (observed + alpha * base) / (stats.n + alpha)
    n_all = stats.n + stats.n_long
    used_logp = math.log((stats.used_hits + 0.5) / (n_all + 1.0))
    file_eps = (stats.file_strong_mismatch + 0.5) / (stats.file_strong + 1.0)
    stats.rate = voucher_rate(rows)
    priors = NumberPriors.from_offset_probs(
        smoothed[:-1],
        float(smoothed[-1]),
        used_logp=used_logp,
        file_eps=file_eps,
        long_pause_days=long_pause_days,
        rate=stats.rate,
        span_factor=span_factor,
        meta={"source": "ocr_lab.fit_number_pairs", "fit_on": "train", "alpha": alpha},
    )
    if long_d:
        grid = [round(0.05 * i, 2) for i in range(1, 20)]

        def loglik(w: float) -> float:
            p = replace(priors, long_weight=w)
            return sum(p.offset_value(d, pause) for d, pause in long_d)

        best = max(grid, key=loglik)
        priors = replace(priors, long_weight=best)
    stats.long_weight = priors.long_weight
    return priors, stats


def voucher_rate(rows: Sequence[Mapping[str, str]]) -> float:
    """Ваучеров в день на буксир: прирост номера за период (буксир, год) по истории."""
    spans: dict[tuple[str, int], list[tuple[datetime, int]]] = defaultdict(list)
    for item in history_from_manifest(rows):
        spans[(item.tug_code, item.year)].append((item.dt, item.number))
    numbers = days = 0.0
    for items in spans.values():
        if len(items) < 2:
            continue
        items.sort()
        days += (items[-1][0] - items[0][0]).total_seconds() / 86400.0
        numbers += max(n for _, n in items) - min(n for _, n in items)
    return numbers / days if days > 0 else 0.93


# ---------------------------------------------------------------------------
# Приоры пары
# ---------------------------------------------------------------------------


def manifest_pairs(rows: Sequence[Mapping[str, str]]) -> list[tuple[str, str]]:
    """Пары `(scan_id, scan_id)` по `pair_id` (группы ровно из двух ваучеров)."""
    groups: dict[str, list[str]] = defaultdict(list)
    for row in rows:
        pid = str(row.get("pair_id") or "").strip()
        if pid:
            groups[pid].append(row["scan_id"])
    return [(g[0], g[1]) for _, g in sorted(groups.items()) if len(g) == 2]


@dataclass
class PairFitStats:
    n: int = 0
    same: int = 0
    patterns: dict[str, int] = field(default_factory=dict)
    row_diffs: dict[str, int] = field(default_factory=dict)
    mult10: int = 0
    rest_link: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _delta_counts(values: Sequence[int]) -> np.ndarray:
    counts = np.zeros(len(DELTA_EDGES) - 1)
    if values:
        idx = np.searchsorted(np.asarray(DELTA_EDGES), np.asarray(values, float), side="right") - 1
        inside = (idx >= 0) & (idx < counts.size)
        np.add.at(counts, idx[inside], 1.0)
    return counts


def fit_pair_priors(
    rows: Sequence[Mapping[str, str]],
    *,
    alpha_same: float = 10.0,
    alpha_pattern: float = 5.0,
    alpha_delta: float = 10.0,
    alpha_row: float = 10.0,
) -> tuple[PairPriors, PairFitStats]:
    """Приоры пары по парам `train`: `π`, шаблоны расхождений и |Δ| по строкам.

    |Δ| строки сглаживается к общей по строкам плотности, та — к плотности по умолчанию.
    `rest_link` — медиана связи по парам `train`, где времена расходятся; доля |Δ|, кратных
    10, — частота со сглаживанием `+0,5 / +1`.
    """
    by_id = {row["scan_id"]: row for row in rows}
    stats = PairFitStats()
    pattern_counts: Counter[tuple[int, ...]] = Counter()
    row_values: dict[str, list[int]] = {r: [] for r in CHAIN}
    diff_deltas: list[tuple[int, ...]] = []
    for a_id, b_id in manifest_pairs(rows):
        fa, fb = truth_forms(by_id[a_id]), truth_forms(by_id[b_id])
        if fa is None or fb is None:
            continue
        deltas = time_deltas(record_times(fa), record_times(fb))
        stats.n += 1
        pattern = tuple(int(d != 0) for d in deltas)
        if not any(pattern):
            stats.same += 1
            continue
        pattern_counts[pattern] += 1
        diff_deltas.append(deltas)
        for row, d in zip(CHAIN, deltas, strict=True):
            if d != 0:
                row_values[row].append(abs(d))
    stats.patterns = {"".join(map(str, p)): c for p, c in pattern_counts.most_common()}
    stats.row_diffs = {r: len(v) for r, v in row_values.items()}
    all_diffs = [d for v in row_values.values() for d in v]
    stats.mult10 = sum(d % 10 == 0 for d in all_diffs)
    mult10_share = (stats.mult10 + 0.5) / (len(all_diffs) + 1.0)

    same_share = (stats.same + alpha_same * DEFAULT_SAME_SHARE) / (stats.n + alpha_same)
    patterns = diff_patterns()
    known = sum(DEFAULT_DIFF_PATTERNS.values())
    other = (1.0 - known) / (len(patterns) - len(DEFAULT_DIFF_PATTERNS))
    base_pat = np.asarray([DEFAULT_DIFF_PATTERNS.get(p, other) for p in patterns])
    obs_pat = np.asarray([pattern_counts.get(p, 0) for p in patterns], dtype=float)
    pat_p = (obs_pat + alpha_pattern * base_pat) / (obs_pat.sum() + alpha_pattern)
    pattern_logp = {p: float(math.log(v)) for p, v in zip(patterns, pat_p, strict=True)}

    default = np.asarray(DEFAULT_DELTA_MASSES) / sum(DEFAULT_DELTA_MASSES)
    pooled_counts = sum((_delta_counts(v) for v in row_values.values()), np.zeros(default.size))
    pooled = (pooled_counts + alpha_delta * default) / (pooled_counts.sum() + alpha_delta)
    delta: dict[str, BinnedLogDensity] = {}
    for row in CHAIN:
        c = _delta_counts(row_values[row])
        masses = (c + alpha_row * pooled) / (c.sum() + alpha_row)
        delta[row] = BinnedLogDensity.from_masses(
            DELTA_EDGES, masses.tolist(), DELTA_FLOOR_PER_MINUTE
        )
    priors = PairPriors(
        same_logp=math.log(same_share),
        pattern_logp=pattern_logp,
        pattern_floor=float(min(pattern_logp.values())),
        delta=delta,
        rest_link=0.0,
        mult10_share=mult10_share,
        meta={"source": "ocr_lab.fit_number_pairs", "fit_on": "train"},
    )
    links = [priors.link_deltas(d) for d in diff_deltas]
    rest = float(np.median(links)) if links else PairPriors.default().rest_link
    stats.rest_link = rest
    return replace(priors, rest_link=rest), stats


# ---------------------------------------------------------------------------
# Оценка вариантов
# ---------------------------------------------------------------------------


def load_number_outputs(path: Path) -> tuple[dict[str, np.ndarray], str, Any]:
    """Сырые выходы модели номера (`number_v0/<split>_outputs.npz`): `{scan_id: логиты}`."""
    if not path.exists():
        return {}, "ctc", 1.0
    data = np.load(path, allow_pickle=False)
    ids = [str(s).strip() for s in data["scan_id"]]
    temp = data["temperature"] if "temperature" in data.files else np.asarray([1.0])
    temperature: Any = float(temp.reshape(-1)[0]) if temp.size == 1 else temp
    return dict(zip(ids, data["output"], strict=True)), str(data["kind"]), temperature


@dataclass
class VoucherEval:
    """Итог одного ваучера в одном варианте."""

    rows_ok: bool
    number_ok: bool
    p_rows: float
    p_number: float

    @property
    def p_joint(self) -> float:
        return self.p_rows * self.p_number


def variant_metrics(items: Sequence[VoucherEval], target: float = TARGET) -> dict[str, Any]:
    """Точности и покрытие при целевой точности (точечная оценка и Уилсон)."""
    from ocr_lab.evaluate import threshold_for_precision

    n = len(items)
    if n == 0:
        return {"n": 0}

    def accept(conf: list[float], ok: list[bool]) -> dict[str, Any]:
        thr = threshold_for_precision(conf, ok, target, n_total=n)
        return {
            "threshold": thr.threshold,
            "n_accepted": thr.n_accepted,
            "coverage": thr.coverage,
            "precision": thr.precision,
            "ci_low": thr.ci_low,
            "confirmed": thr.confirmed,
        }

    def fixed(conf: list[float], ok: list[bool]) -> dict[str, list[int]]:
        out = {}
        for thr in FIXED_THRESHOLDS:
            acc = [o for c, o in zip(conf, ok, strict=True) if c >= thr]
            out[f"{thr:g}"] = [len(acc), len(acc) - sum(acc)]
        return out

    rows_ok = [v.rows_ok for v in items]
    rn_ok = [v.rows_ok and v.number_ok for v in items]
    p_rows = [v.p_rows for v in items]
    p_joint = [v.p_joint for v in items]
    return {
        "n": n,
        "rows": sum(rows_ok) / n,
        "rows_number": sum(rn_ok) / n,
        "number": sum(v.number_ok for v in items) / n,
        "accept_rows": accept(p_rows, rows_ok),
        "accept_rows_number": accept(p_joint, rn_ok),
        "fixed_rows": fixed(p_rows, rows_ok),
        "fixed_rows_number": fixed(p_joint, rn_ok),
    }


@dataclass(frozen=True)
class SplitData:
    rows: list[dict[str, str]]
    by_id: dict[str, dict[str, str]]
    inputs: dict[str, DecoderInputs]
    forms: dict[str, dict[str, FormRow]]
    contexts: dict[str, DecodeContext]
    numbers: dict[str, np.ndarray]
    number_kind: str
    number_temperature: Any
    pairs: list[tuple[str, str]]


def load_split(
    split: str, manifest: Path, inputs_path: Path, number_outputs: Path
) -> tuple[SplitData, list[dict[str, str]]]:
    all_rows = read_manifest(manifest)
    rows = [r for r in all_rows if r.get("split") == split]
    inputs = load_predictions(inputs_path)
    by_id: dict[str, dict[str, str]] = {}
    forms: dict[str, dict[str, FormRow]] = {}
    contexts: dict[str, DecodeContext] = {}
    for row in rows:
        f = truth_forms(row)
        if f is None or row["scan_id"] not in inputs:
            continue
        by_id[row["scan_id"]] = row
        forms[row["scan_id"]] = f
        contexts[row["scan_id"]] = context_for(row, f)
    numbers, kind, temperature = load_number_outputs(number_outputs)
    pairs = [(a, b) for a, b in manifest_pairs(rows) if a in by_id and b in by_id]
    data = SplitData(
        rows=[r for r in rows if r["scan_id"] in by_id],
        by_id=by_id,
        inputs={k: v for k, v in inputs.items() if k in by_id},
        forms=forms,
        contexts=contexts,
        numbers=numbers,
        number_kind=kind,
        number_temperature=temperature,
        pairs=pairs,
    )
    return data, all_rows


def _ref_date(result: DecodeResult, context: DecodeContext) -> datetime | None:
    """Дата для «раньше» в истории номеров: прочитанный декодером выход, иначе заявка."""
    if result.records:
        return result.records[0].rows[CHAIN[0]].dt
    return context.app_dt


def evaluate_split(
    data: SplitData,
    history: Sequence[NumberHistoryItem],
    model: DecoderModel,
    number_priors: NumberPriors,
    pair_priors: PairPriors,
    *,
    top_n: int = 5,
    log: Any = None,
) -> dict[str, Any]:
    """Все варианты на сплите: метрики, ошибки, время, и решения итогового варианта."""
    model = replace(model, search=replace(model.search, top_n=top_n))
    ids = [r["scan_id"] for r in data.rows]
    partner: dict[str, str] = {}
    for a, b in data.pairs:
        partner[a], partner[b] = b, a

    def truth_number(sid: str) -> int | None:
        return _int(data.by_id[sid].get("voucher_number"))

    # 1. Ядро.
    t0 = time.perf_counter()
    core = {sid: decode(data.inputs[sid], data.contexts[sid], model) for sid in ids}
    t_core = (time.perf_counter() - t0) / max(1, len(ids))

    # 2. Номер: картинка, + порядок, + имя файла.
    num: dict[str, dict[str, NumberResult]] = {"image": {}, "seq": {}, "seq_file": {}}
    t_num: list[float] = []
    no_prior = NumberContext()
    for sid in ids:
        out = data.numbers.get(sid)
        kw = {"kind": data.number_kind, "temperature": data.number_temperature}
        num["image"][sid] = decode_number_output(out, no_prior, number_priors, **kw)
        before = _ref_date(core[sid], data.contexts[sid])
        row = data.by_id[sid]
        t1 = time.perf_counter()
        ctx = number_context(row, history, before, use_filename=False)
        num["seq"][sid] = decode_number_output(out, ctx, number_priors, **kw)
        t_num.append(time.perf_counter() - t1)
        ctx_f = number_context(row, history, before, use_filename=True)
        num["seq_file"][sid] = decode_number_output(out, ctx_f, number_priors, **kw)

    # 3. Пары: обе стороны ожидают проверки; и вариант «партнёр подтверждён».
    paired: dict[str, DecodeResult] = {}
    confirmed: dict[str, DecodeResult] = {}
    p_same: dict[str, float] = {}
    t_pair: list[float] = []
    for a, b in data.pairs:
        t1 = time.perf_counter()
        res = decode_pair(
            data.inputs[a],
            data.inputs[b],
            PairContext(a=data.contexts[a], b=data.contexts[b]),
            model,
            pair_priors,
        )
        t_pair.append(time.perf_counter() - t1)
        assert res.b is not None
        paired[a], paired[b] = res.a, res.b
        p_same[a] = p_same[b] = res.p_same
        for x, y in ((a, b), (b, a)):
            conf = decode_pair(
                data.inputs[x],
                None,
                PairContext(a=data.contexts[x], b_confirmed=data.forms[y]),
                model,
                pair_priors,
            )
            confirmed[x] = conf.a

    def ev(rec_res: DecodeResult, nres: NumberResult, sid: str) -> VoucherEval:
        top = rec_res.records[0] if rec_res.records else None
        rows_ok = top is not None and top.forms() == data.forms[sid]
        tn = truth_number(sid)
        number_ok = tn is not None and nres.value == tn
        return VoucherEval(rows_ok, number_ok, rec_res.confidence, nres.confidence)

    variants: dict[str, dict[str, VoucherEval]] = {
        "core": {s: ev(core[s], num["image"][s], s) for s in ids},
        "core_number": {s: ev(core[s], num["seq"][s], s) for s in ids},
        "core_number_file": {s: ev(core[s], num["seq_file"][s], s) for s in ids},
        "core_number_pairs": {s: ev(paired.get(s, core[s]), num["seq"][s], s) for s in ids},
        "core_number_file_pairs": {
            s: ev(paired.get(s, core[s]), num["seq_file"][s], s) for s in ids
        },
        "partner_confirmed": {s: ev(confirmed[s], num["seq"][s], s) for s in confirmed},
    }
    pair_ids = set(partner)
    metrics: dict[str, Any] = {}
    errors: dict[str, Any] = {}
    for name, items in variants.items():
        metrics[name] = {
            "all": variant_metrics(list(items.values())),
            "pairs": variant_metrics([v for s, v in items.items() if s in pair_ids]),
        }
        errors[name] = {
            "rows": sorted(s for s, v in items.items() if not v.rows_ok),
            "number": sorted(s for s, v in items.items() if not v.number_ok),
        }
    number_detail = {
        sid: {
            "truth": truth_number(sid),
            "expected": num["seq"][sid].expected,
            "image": num["image"][sid].value,
            "seq": num["seq"][sid].value,
            "seq_file": num["seq_file"][sid].value,
            "p_seq": num["seq"][sid].confidence,
        }
        for sid in ids
        if not all(num[k][sid].value == truth_number(sid) for k in num)
    }
    pair_detail = {
        sid: {
            "partner": partner[sid],
            "core_ok": variants["core"][sid].rows_ok,
            "pair_ok": variants["core_number_pairs"][sid].rows_ok,
            "p_core": core[sid].confidence,
            "p_pair": paired[sid].confidence,
            "p_same": p_same[sid],
            "truth_same": data.forms[sid] == data.forms[partner[sid]]
            or record_times(data.forms[sid]) == record_times(data.forms[partner[sid]]),
        }
        for sid in sorted(pair_ids)
        if variants["core"][sid].rows_ok != variants["core_number_pairs"][sid].rows_ok
        or not variants["core_number_pairs"][sid].rows_ok
    }
    truth_same = sum(
        record_times(data.forms[a]) == record_times(data.forms[b]) for a, b in data.pairs
    )
    timing = {
        "core_ms_mean": 1000 * t_core,
        "number_ms_median": 1000 * float(np.median(t_num)) if t_num else None,
        "pair_ms_median": 1000 * float(np.median(t_pair)) if t_pair else None,
        "pair_ms_max": 1000 * float(np.max(t_pair)) if t_pair else None,
    }
    if log:
        for name, cell in metrics.items():
            summary: Mapping[str, Any] = cell["all"]
            if summary.get("n"):
                log(
                    f"{name}: запись {summary['rows']:.4f}, с номером "
                    f"{summary['rows_number']:.4f}, номер {summary['number']:.4f} "
                    f"(N = {summary['n']})"
                )
    return {
        "n_vouchers": len(ids),
        "n_pairs": len(data.pairs),
        "n_pairs_truth_same": int(truth_same),
        "metrics": metrics,
        "errors": errors,
        "number_detail": number_detail,
        "pair_detail": pair_detail,
        "timing": timing,
        "_final": {
            sid: (paired.get(sid, core[sid]), num["seq"][sid], num["seq_file"][sid]) for sid in ids
        },
    }


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

VARIANT_TITLES: dict[str, str] = {
    "core": "1. Ядро T19 (номер — top-1 `number_v0`)",
    "core_number": "2. Ядро + номер (порядок, без имени файла)",
    "core_number_file": "2б. Ядро + номер (порядок + имя файла)",
    "core_number_pairs": "3. Ядро + номер + пары",
    "core_number_file_pairs": "3б. Ядро + номер (с именем) + пары",
    "partner_confirmed": "Пара: партнёр подтверждён (только ваучеры с парой)",
}


def _pct(v: Any) -> str:
    return "—" if v is None else f"{100 * float(v):.2f} %"


def _acc_cell(a: Mapping[str, Any]) -> str:
    if not a or a.get("n_accepted") is None:
        return "—"
    return f"{_pct(a['coverage'])} ({a['n_accepted']})"


def _fixed_cell(fixed: Mapping[str, Sequence[int]]) -> str:
    return " · ".join(f"{acc} / {err}" for acc, err in fixed.values())


def render_report(split: str, res: Mapping[str, Any], meta: Mapping[str, Any]) -> str:
    lines = [
        f"# T20: номер ваучера и пары k/p в декодере, сплит `{split}`",
        "",
        f"- Ваучеров {res['n_vouchers']}, пар k/p {res['n_pairs']} (все времена совпадают по "
        f"истине — {res['n_pairs_truth_same']}).",
        "- Запись верна, если все 4 строки совпали с истиной манифеста; «с номером» — и номер.",
        "- Покрытие — доля ваучеров, принятых при точности ≥ 99,5 % (точечная оценка по "
        "`confidence`; для «с номером» — `p(запись) · p(номер)`), в скобках — число принятых.",
        "- «p ≥ 0,99 · p ≥ 0,999» — принято при фиксированном пороге / из них ошибок (запись).",
        f"- Модель ядра: `{meta.get('decoder')}`; приоры номера и пары: "
        f"`{meta.get('number_priors')}`, `{meta.get('pair_priors')}`.",
        "",
    ]
    for subset, title in (("all", "Все ваучеры"), ("pairs", "Подмножество с парой")):
        lines += [
            f"## {title}",
            "",
            "| Вариант | N | Запись | Запись + номер | Номер | Покрытие 99,5 %: запись | "
            "Покрытие 99,5 %: запись + номер | p ≥ 0,99 · p ≥ 0,999 |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
        for name, m in res["metrics"].items():
            cell = m[subset]
            if not cell.get("n"):
                continue
            lines.append(
                f"| {VARIANT_TITLES[name]} | {cell['n']} | {_pct(cell['rows'])} | "
                f"{_pct(cell['rows_number'])} | {_pct(cell['number'])} | "
                f"{_acc_cell(cell['accept_rows'])} | {_acc_cell(cell['accept_rows_number'])} | "
                f"{_fixed_cell(cell['fixed_rows'])} |"
            )
        lines.append("")
    lines += ["## Ошибки", ""]
    for name, e in res["errors"].items():
        lines.append(
            f"- {VARIANT_TITLES[name]}: запись — {', '.join(e['rows']) or 'нет'}; "
            f"номер — {', '.join(e['number']) or 'нет'}"
        )
    lines += ["", "## Номер: ваучеры, где варианты расходятся или ошибаются", ""]
    if res["number_detail"]:
        lines += [
            "| scan_id | истина | ожидаемый | картинка | + порядок (p) | + имя |",
            "|---|---:|---:|---:|---:|---:|",
        ]
        for sid, d in sorted(res["number_detail"].items()):
            lines.append(
                f"| {sid} | {d['truth']} | {d['expected']} | {d['image']} | "
                f"{d['seq']} ({d['p_seq']:.3f}) | {d['seq_file']} |"
            )
    else:
        lines.append("Нет.")
    lines += ["", "## Пары: ваучеры, где пара меняет итог или ошибается", ""]
    if res["pair_detail"]:
        lines += [
            "| scan_id | партнёр | ядро верно | пара верна | p ядра | p пары | p(равны) | "
            "равны по истине |",
            "|---|---|---|---|---:|---:|---:|---|",
        ]
        for sid, d in res["pair_detail"].items():
            lines.append(
                f"| {sid} | {d['partner']} | {'да' if d['core_ok'] else 'нет'} | "
                f"{'да' if d['pair_ok'] else 'нет'} | {d['p_core']:.4f} | {d['p_pair']:.4f} | "
                f"{d['p_same']:.4f} | {'да' if d['truth_same'] else 'нет'} |"
            )
    else:
        lines.append("Нет.")
    t = res["timing"]
    lines += [
        "",
        "## Время (облако, CPU)",
        "",
        f"- Ядро: в среднем {t['core_ms_mean']:.1f} мс на ваучер; номер: медиана "
        f"{(t['number_ms_median'] or 0):.1f} мс; пара (оба ваучера): медиана "
        f"{(t['pair_ms_median'] or 0):.1f} мс, максимум {(t['pair_ms_max'] or 0):.1f} мс.",
        "",
    ]
    return "\n".join(lines)


def write_predictions(final: Mapping[str, Any], out: Path, *, with_file: bool) -> int:
    """JSONL T13 итогового варианта: записи пары/ядра и номер декодера."""
    lines = []
    for sid, (res, nres, nres_f) in sorted(final.items()):
        item: dict[str, Any] = {"scan_id": sid, "source": SOURCE, "fields": {}}
        number = nres_f if with_file else nres
        if number.candidates:
            item["fields"][VOUCHER_NUMBER] = number.to_json()
        item.update(res.to_json())
        if not res.records:
            item.pop("confidence")
            item.pop("margin")
        lines.append(json.dumps(item, ensure_ascii=False))
    paths.ensure_dir(out.parent)
    out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    return len(lines)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_fit(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        print(f"нет манифеста: {args.manifest}", file=sys.stderr)
        return 2
    rows = read_manifest(args.manifest)
    train = [r for r in rows if r.get("split") == "train"]
    history = history_from_manifest(rows)
    number_priors, nstats = fit_number_priors(train, history, radius=args.radius)
    pair_priors, pstats = fit_pair_priors(train)
    in_window = sum(v for d, v in nstats.offsets.items() if abs(d) <= args.radius)
    print(
        f"номер (train): с короткой паузой {nstats.n}, сдвиг 0 — {nstats.offsets.get(0, 0)}, "
        f"в окне ±{args.radius} — {in_window}, вне окна {nstats.outside}; после паузы "
        f"{nstats.n_long} (сдвиги {dict(sorted(nstats.long_offsets.items()))}, вес смеси "
        f"{nstats.long_weight:.2f}), темп {nstats.rate:.3f} в день; занятый номер у истины — "
        f"{nstats.used_hits}, имя файла ≠ истине — {nstats.file_strong_mismatch} из "
        f"{nstats.file_strong} ({', '.join(nstats.file_mismatch_ids) or '—'})"
    )
    print(
        f"пары (train): {pstats.n}, все времена равны {pstats.same} "
        f"(π = {math.exp(pair_priors.same_logp):.3f}), шаблоны {pstats.patterns}, |Δ| кратны 10: "
        f"{pstats.mult10} из {sum(pstats.row_diffs.values())}, "
        f"rest_link {pstats.rest_link:.2f}"
    )
    number_priors = replace(number_priors, meta={**number_priors.meta, "stats": nstats.as_dict()})
    pair_priors = replace(pair_priors, meta={**pair_priors.meta, "stats": pstats.as_dict()})
    number_priors.save(args.number_out)
    pair_priors.save(args.pair_out)
    print(f"записано: {args.number_out}, {args.pair_out}")
    return 0


def cmd_eval(args: argparse.Namespace) -> int:
    data, all_rows = load_split(
        args.split,
        args.manifest,
        args.inputs or DECODER_DIR / f"{args.split}_inputs.jsonl",
        args.number_outputs or NUMBER_DIR / f"{args.split}_outputs.npz",
    )
    model = DecoderModel.load(args.model)
    number_priors = NumberPriors.from_json(json.loads(args.number_priors.read_text("utf-8")))
    pair_priors = PairPriors.from_json(json.loads(args.pair_priors.read_text("utf-8")))
    if args.pair_weight is not None:
        pair_priors = replace(pair_priors, weight=args.pair_weight)
    if args.image_weight is not None:
        number_priors = replace(number_priors, image_weight=args.image_weight)
    res = evaluate_split(
        data, history_from_manifest(all_rows), model, number_priors, pair_priors, log=print
    )
    final = res.pop("_final")
    meta = {
        "decoder": args.model.name,
        "number_priors": args.number_priors.name,
        "pair_priors": args.pair_priors.name,
        "pair_weight": pair_priors.weight,
        "image_weight": number_priors.image_weight,
    }
    out = args.out or paths.REPORTS_DIR / f"decoder_t20_compare_{args.split}.md"
    paths.ensure_dir(out.parent)
    out.write_text(render_report(args.split, res, meta), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps({"meta": meta, **res}, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    if not args.no_predictions:
        pred = OUT_DIR / f"{args.split}_predictions.jsonl"
        n = write_predictions(final, pred, with_file=args.with_filename)
        print(f"предсказаний {n} → {pred}")
    print(f"отчёт: {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocr_lab.fit_number_pairs", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="приоры номера и пары по train")
    p_fit.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    p_fit.add_argument("--radius", type=int, default=15)
    p_fit.add_argument("--number-out", type=Path, default=NUMBER_JSON)
    p_fit.add_argument("--pair-out", type=Path, default=PAIR_JSON)
    p_fit.set_defaults(func=cmd_fit)

    p_ev = sub.add_parser("eval", help="варианты ядро / + номер / + пары на сплите")
    p_ev.add_argument("--split", choices=SPLITS, default="val")
    p_ev.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    p_ev.add_argument("--model", type=Path, default=DECODER_JSON)
    p_ev.add_argument("--number-priors", type=Path, default=NUMBER_JSON)
    p_ev.add_argument("--pair-priors", type=Path, default=PAIR_JSON)
    p_ev.add_argument("--inputs", type=Path, default=None)
    p_ev.add_argument("--number-outputs", type=Path, default=None)
    p_ev.add_argument("--pair-weight", type=float, default=None)
    p_ev.add_argument("--image-weight", type=float, default=None)
    p_ev.add_argument(
        "--with-filename",
        action="store_true",
        help="в JSONL — номер с приором имени файла (по умолчанию без него)",
    )
    p_ev.add_argument("--no-predictions", action="store_true")
    p_ev.add_argument("--out", type=Path, default=None)
    p_ev.set_defaults(func=cmd_eval)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
