"""Подбор приоров и весов совместного декодера и его прогон по сплитам (T19).

Запуск (пути — из `ocr_lab.paths`, то есть от `OCR_WORK_DIR`):

    python -m ocr_lab.fit_decoder_priors inputs --split val   # полные распределения цифр
    python -m ocr_lab.fit_decoder_priors fit                  # приоры (train) + веса (val)
    python -m ocr_lab.fit_decoder_priors predict --split val  # декодер → JSONL T13
    python -m ocr_lab.fit_decoder_priors compare --split val  # «top-1 по полям» против декодера
    python -m ocr_lab.evaluate --pred <models>/decoder_v0/val_predictions.jsonl --split val

- `inputs` — распределения подполей по кропам через рантайм T18 (`app.ocr.runtime`,
  onnxruntime, температуры из `meta.json`): по умолчанию все значения части (`k = 60`), как
  будет звать приложение. Пишет `models/decoder_v0/<split>_inputs.jsonl` (формат T13).
- `fit` — приоры по сплиту `train` манифеста (колонки T02): минуты, часы по строкам (час 24 —
  не дороже 00), длительности участков (общие и по виду работ, если данных хватает), шаблоны
  смещений дней, отклонение «Начало − время заявки», доля «Окончание позже Прихода» для
  мягкого правила. Эмпирика сглаживается к приорам по умолчанию из `app.ocr.decoder`, пол не
  даёт отрезать хвосты. Веса (`w_part` и множители приоров) — покоординатный поиск по сетке
  на `val`; цель — среднее `log p` истинной записи при softmax по перечисленным кандидатам
  (если истины среди них нет, она добавляется в нормировку). Пишет
  `models/decoder_priors_v0.json`.
- `predict` — декодер по сплиту: JSONL T13 (`records` top-N, `confidence`, `margin`, флаги),
  `fields` — top-5 входа (метрики подполей те же, что у `digits_v0`), номер ваучера — top-1
  `number_v0`, если есть его предсказания (совместно с номером декодирует T20).
- `compare` — таблица «top-1 по полям» (top-1 тех же входов) против декодера по метрикам
  `ocr_lab.evaluate`: запись целиком, строки, ошибка в минутах, автоприём.

В консоль и отчёты идут только агрегаты и `scan_id`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import sys
import time
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.ocr.decoder import (
    APP_EDGES,
    CHAIN,
    DEFAULT_APP_MASSES,
    DEFAULT_LATE_FINISH_SHARE,
    DURATION_EDGES,
    LEG_NAMES,
    PARTS,
    ROWS,
    WEIGHT_NAMES,
    BinnedLogDensity,
    DecodeContext,
    DecodeResult,
    DecoderInputs,
    DecoderModel,
    DecoderPriors,
    FormRow,
    _evidence,
    _score_scalar,
    candidate_space,
    chain_patterns,
    decode,
    default_duration_masses,
    default_minute_probs,
    late_finish_density,
    subfield,
)
from ocr_lab import paths
from ocr_lab.predictions import VOUCHER_NUMBER, read_predictions

LOG_P_CLIP = -30.0
DEFAULT_GRID: tuple[float, ...] = (0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
# Веса картинки не обнуляем: без них декодер перестаёт читать.
IMAGE_WEIGHTS = ("day", "month", "hour", "minute")
# Шаг покоординатного поиска принимается, если среднее log p растёт хотя бы на столько и
# прирост держится не на одном-двух ваучерах: в доле `DEFAULT_MIN_SUPPORT` бутстреп-выборок
# `val` средний прирост положителен. На 51 ваучере с 3 ошибками более мелкие или узкие
# улучшения — подгонка под конкретные ваучеры.
DEFAULT_MIN_GAIN = 0.01
DEFAULT_MIN_SUPPORT = 0.9
N_BOOTSTRAP = 2000

DECODER_JSON = paths.MODELS_DIR / "decoder_priors_v0.json"
DECODER_DIR = paths.MODELS_DIR / "decoder_v0"
DIGITS_DIR = paths.MODELS_DIR / "digits_v0"
NUMBER_DIR = paths.MODELS_DIR / "number_v0"
INDEX_CSV = paths.WORK_DIR / "crops_index.csv"
SOURCE = "decoder_v0"
INPUT_SOURCE = "digits_v0_full"
SPLITS = ("train", "val", "test")


# ---------------------------------------------------------------------------
# Манифест и истина
# ---------------------------------------------------------------------------


def read_manifest(path: Path) -> list[dict[str, str]]:
    """Строки манифеста; `scan_id` без пробелов по краям (как в `ocr_lab.evaluate`)."""
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for row in rows:
        row["scan_id"] = str(row.get("scan_id", "")).strip()
    return rows


def _int(value: str | None) -> int | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return int(float(value))
    except ValueError:
        return None


def _dt(value: str | None) -> datetime | None:
    if value is None or str(value).strip() == "":
        return None
    try:
        return datetime.fromisoformat(str(value).strip())
    except ValueError:
        return None


def _is_false(value: str | None) -> bool:
    return str(value).strip().lower() in {"false", "0", "no", "нет"}


def truth_forms(row: Mapping[str, str]) -> dict[str, FormRow] | None:
    """Истинная запись в написании бланка; `None`, если чего-то не хватает."""
    year = _int(row.get("year"))
    out: dict[str, FormRow] = {}
    for name in CHAIN:
        d = _int(row.get(f"{name}_day"))
        m = _int(row.get(f"{name}_month"))
        y = _int(row.get(f"{name}_year")) or year
        h = _int(row.get(f"{name}_hour"))
        mi = _int(row.get(f"{name}_minute"))
        if d is None or m is None or y is None or h is None or mi is None:
            return None
        try:
            out[name] = (date(y, m, d), h, mi)
        except ValueError:
            return None
    return out


def form_minutes(forms: Mapping[str, FormRow]) -> list[int]:
    """Фактическое время строк CHAIN в минутах от начала даты выхода."""
    base = forms[CHAIN[0]][0]
    return [(forms[r][0] - base).days * 1440 + forms[r][1] * 60 + forms[r][2] for r in CHAIN]


def chain_ok(row: Mapping[str, str], forms: Mapping[str, FormRow]) -> bool:
    if _is_false(row.get("chain_ok")):
        return False
    t = form_minutes(forms)
    return all(b >= a for a, b in zip(t[:-1], t[1:], strict=True))


def late_finish_minutes(forms: Mapping[str, FormRow]) -> int | None:
    """На сколько минут Приход раньше Окончания, если это единственное нарушение цепочки."""
    t = form_minutes(forms)
    if t[0] <= t[1] <= t[2] and t[3] < t[2]:
        return t[2] - t[3]
    return None


def context_for(row: Mapping[str, str], forms: Mapping[str, FormRow] | None) -> DecodeContext:
    year = _int(row.get("year"))
    if year is None and forms is not None:
        year = forms[CHAIN[0]][0].year
    if year is None:
        raise ValueError(f"нет года у {row.get('scan_id')}")
    work_type = (row.get("work_type") or "").strip() or None
    return DecodeContext(year=year, app_dt=_dt(row.get("app_dt")), work_type=work_type)


# ---------------------------------------------------------------------------
# Приоры
# ---------------------------------------------------------------------------


def _smoothed(counts: np.ndarray, base: np.ndarray, alpha: float) -> np.ndarray:
    """Сглаживание к базовому распределению псевдосчётчиком `alpha`."""
    base = base / base.sum()
    total = counts.sum()
    return (counts + alpha * base) / (total + alpha)


def _binned(values: Iterable[float], edges: Sequence[float]) -> tuple[np.ndarray, int]:
    arr = np.asarray(list(values), dtype=float)
    counts = np.zeros(len(edges) - 1)
    if arr.size == 0:
        return counts, 0
    idx = np.searchsorted(np.asarray(edges), arr, side="right") - 1
    inside = (idx >= 0) & (idx < counts.size)
    np.add.at(counts, idx[inside], 1.0)
    return counts, int((~inside).sum())


@dataclass
class PriorStats:
    """Агрегаты подбора приоров для журнала."""

    train_rows: int = 0
    train_vouchers: int = 0
    chain_ok_vouchers: int = 0
    minutes_mult10_share: float = float("nan")
    crosses_midnight_share: float = float("nan")
    patterns: dict[str, int] | None = None
    bad_patterns: int = 0
    work_types: dict[str, int] | None = None
    app_pairs: int = 0
    app_outside_bins: int = 0
    app_median_hours: float = float("nan")
    durations_outside_bins: int = 0
    hour24_count: int = 0
    late_finish: int = 0
    late_finish_beyond: int = 0
    late_finish_share: float = float("nan")

    def as_dict(self) -> dict[str, Any]:
        return {k: v for k, v in self.__dict__.items()}


def fit_priors(
    rows: Sequence[Mapping[str, str]],
    base: DecoderPriors | None = None,
    *,
    min_work_type: int = 60,
    alpha_minute: float = 30.0,
    alpha_hour: float = 25.0,
    alpha_duration: float = 20.0,
    alpha_pattern: float = 5.0,
    alpha_app: float = 10.0,
    alpha_late: float = 200.0,
) -> tuple[DecoderPriors, PriorStats]:
    """Приоры декодера по строкам `train` манифеста."""
    base = base or DecoderPriors.default()
    stats = PriorStats(train_rows=len(rows))
    samples: list[tuple[Mapping[str, str], dict[str, FormRow]]] = []
    for row in rows:
        forms = truth_forms(row)
        if forms is not None:
            samples.append((row, forms))
    stats.train_vouchers = len(samples)

    minute_counts = np.zeros(60)
    hour_counts = {name: np.zeros(25) for name in CHAIN}
    for _, forms in samples:
        for name in CHAIN:
            _, h, mi = forms[name]
            if 0 <= mi <= 59:
                minute_counts[mi] += 1
            if 0 <= h <= 24:
                hour_counts[name][h] += 1
    stats.hour24_count = int(sum(c[24] for c in hour_counts.values()))
    if minute_counts.sum():
        stats.minutes_mult10_share = float(minute_counts[::10].sum() / minute_counts.sum())
    minute_p = _smoothed(minute_counts, default_minute_probs(), alpha_minute)
    hour_logp: dict[str, tuple[float, ...]] = {}
    for name in CHAIN:
        p = _smoothed(hour_counts[name], np.ones(25), alpha_hour)
        # `24:00` — то же время, что 00:00 следующих суток; выгрузка его не различает, а на
        # бланках T04 видела только 00:00. Поэтому приор не даёт 24 преимущества перед 00
        # (и не штрафует его сильнее): выбор между ними — за картинкой.
        p[24] = p[0]
        p = p / p.sum()
        hour_logp[name] = tuple(float(v) for v in np.log(p))

    good = [(row, forms) for row, forms in samples if chain_ok(row, forms)]
    stats.chain_ok_vouchers = len(good)

    # Мягкое правило «Окончание ≤ Приход»: доля бланков, где Приход раньше Окончания не
    # больше чем на `late_finish_max` минут (остальная цепочка в порядке).
    late = [m for _, forms in samples if (m := late_finish_minutes(forms)) is not None]
    stats.late_finish = sum(m <= base.late_finish_max for m in late)
    stats.late_finish_beyond = len(late) - stats.late_finish
    share = (stats.late_finish + alpha_late * DEFAULT_LATE_FINISH_SHARE) / (
        len(samples) + alpha_late
    )
    stats.late_finish_share = float(share)

    # Шаблоны смещений дней.
    pattern_counts: Counter[tuple[int, ...]] = Counter()
    for _, forms in good:
        b = forms[CHAIN[0]][0]
        pattern = tuple((forms[r][0] - b).days for r in CHAIN)
        if pattern[0] == 0 and all(y >= x for x, y in zip(pattern[:-1], pattern[1:], strict=True)):
            pattern_counts[pattern] += 1
        else:
            stats.bad_patterns += 1
    observed_max = max((p[-1] for p in pattern_counts), default=0)
    max_off = min(2, max(base.max_day_offset, observed_max))
    allowed = chain_patterns(max_off)
    base_pattern = np.asarray(
        [math.exp(base.pattern_value(p)) if p in base.pattern_logp else 1e-4 for p in allowed]
    )
    counts = np.asarray([pattern_counts.get(p, 0) for p in allowed], dtype=float)
    pattern_p = _smoothed(counts, base_pattern, alpha_pattern)
    stats.patterns = {",".join(map(str, p)): int(c) for p, c in pattern_counts.items()}
    if good:
        stats.crosses_midnight_share = 1.0 - pattern_counts.get((0, 0, 0, 0), 0) / len(good)

    # Длительности участков.
    def leg_durations(
        items: Sequence[tuple[Mapping[str, str], dict[str, FormRow]]],
    ) -> dict[str, list[float]]:
        out: dict[str, list[float]] = {name: [] for name in LEG_NAMES}
        for _, forms in items:
            t = form_minutes(forms)
            for i, name in enumerate(LEG_NAMES):
                out[name].append(float(t[i + 1] - t[i]))
        return out

    def densities(
        durs: Mapping[str, list[float]], base_masses: Mapping[str, np.ndarray]
    ) -> tuple[dict[str, BinnedLogDensity], dict[str, np.ndarray]]:
        dens: dict[str, BinnedLogDensity] = {}
        masses: dict[str, np.ndarray] = {}
        for name in LEG_NAMES:
            c, outside = _binned(durs[name], DURATION_EDGES)
            stats.durations_outside_bins += outside
            masses[name] = _smoothed(c, base_masses[name], alpha_duration)
            floor = math.exp(base.durations[name].floor)
            dens[name] = BinnedLogDensity.from_masses(DURATION_EDGES, masses[name].tolist(), floor)
        return dens, masses

    default_masses = {name: _masses_of(base.durations[name], DURATION_EDGES) for name in LEG_NAMES}
    all_durs = leg_durations(good)
    durations, overall_masses = densities(all_durs, default_masses)
    by_type: dict[str, list[tuple[Mapping[str, str], dict[str, FormRow]]]] = {}
    for row, forms in good:
        wt = (row.get("work_type") or "").strip()
        if wt:
            by_type.setdefault(wt, []).append((row, forms))
    stats.work_types = {wt: len(items) for wt, items in sorted(by_type.items())}
    durations_by_type: dict[str, dict[str, BinnedLogDensity]] = {}
    for wt, items in by_type.items():
        if len(items) >= min_work_type:
            durations_by_type[wt], _ = densities(leg_durations(items), overall_masses)

    # Отклонение «Начало − время заявки».
    devs: list[float] = []
    for row, forms in good:
        app = _dt(row.get("app_dt"))
        if app is None:
            continue
        start = datetime.combine(forms[CHAIN[0]][0], datetime.min.time())
        start_min = form_minutes(forms)[1]
        devs.append((start - app).total_seconds() / 60.0 + start_min)
    stats.app_pairs = len(devs)
    if devs:
        stats.app_median_hours = float(np.median(devs) / 60.0)
    app_counts, stats.app_outside_bins = _binned(devs, APP_EDGES)
    app_masses = _smoothed(app_counts, np.asarray(DEFAULT_APP_MASSES), alpha_app)
    app_density = BinnedLogDensity.from_masses(
        APP_EDGES, app_masses.tolist(), math.exp(base.app_deviation.floor)
    )

    priors = DecoderPriors(
        minute_logp=tuple(float(v) for v in np.log(minute_p)),
        hour_logp=hour_logp,
        durations=durations,
        durations_by_work_type=durations_by_type,
        pattern_logp={p: float(math.log(v)) for p, v in zip(allowed, pattern_p, strict=True)},
        pattern_floor=base.pattern_floor,
        max_day_offset=max_off,
        app_deviation=app_density,
        other_year_logp=base.other_year_logp,
        late_finish_max=base.late_finish_max,
        late_finish_logp=late_finish_density(share, base.late_finish_max),
    )
    return priors, stats


def _masses_of(density: BinnedLogDensity, edges: Sequence[float]) -> np.ndarray:
    """Массы бинов плотности (обратно к `from_masses`)."""
    widths = np.diff(np.asarray(edges, dtype=float))
    masses = np.exp(np.asarray(density.logp, dtype=float)) * widths
    if density.edges != tuple(edges) or masses.sum() <= 0:
        # Другая сетка — берём приор по умолчанию с медианой 60 минут.
        masses = np.asarray(default_duration_masses(60.0))
    return masses / masses.sum()


# ---------------------------------------------------------------------------
# Входы декодера и выборка с истиной
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    scan_id: str
    inputs: DecoderInputs
    context: DecodeContext
    forms: dict[str, FormRow]


def load_predictions(path: Path) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Подполя дат из JSONL T13 (`ocr_lab.predictions`): `{scan_id: {подполе: [(v, p), …]}}`."""
    out: dict[str, dict[str, list[tuple[int, float]]]] = {}
    for pred in read_predictions(path):
        out[pred.scan_id.strip()] = {
            name: [(int(v), float(p)) for v, p in c if p is not None]
            for name, c in pred.fields.items()
            if name != VOUCHER_NUMBER
        }
    return out


def build_samples(
    rows: Sequence[Mapping[str, str]],
    predictions: Mapping[str, DecoderInputs],
    *,
    only_chain_ok: bool = False,
) -> list[Sample]:
    """Ваучеры сплита с предсказаниями и полной истиной.

    По умолчанию берутся и ваучеры с нарушенной цепочкой (как в `ocr_lab.evaluate`): у мягкого
    правила «Окончание ≤ Приход» их истина допустима, у прочих она получит `log p = LOG_P_CLIP`.
    """
    samples: list[Sample] = []
    for row in rows:
        scan_id = str(row.get("scan_id", "")).strip()
        if scan_id not in predictions:
            continue
        forms = truth_forms(row)
        if forms is None or (only_chain_ok and not chain_ok(row, forms)):
            continue
        samples.append(Sample(scan_id, predictions[scan_id], context_for(row, forms), forms))
    return samples


def fields_top1_correct(sample: Sample) -> bool:
    """Верна ли запись, собранная из top-1 каждого подполя (базовая линия)."""
    for name in CHAIN:
        written, h, mi = sample.forms[name]
        truth = {"day": written.day, "month": written.month, "hour": h, "minute": mi}
        for part in PARTS:
            dist = sample.inputs.get(subfield(name, part))
            if not dist or int(dist[0][0]) != truth[part]:
                return False
    return True


def truth_log_prob(
    model: DecoderModel, sample: Sample, result: DecodeResult | None = None
) -> tuple[float, bool]:
    """`log p` истины при softmax по перечисленным кандидатам и достижима ли она.

    Если истины нет среди перечисленных, она добавляется в нормировку — так цель гладкая и
    не упирается в `LOG_P_CLIP`, когда истинный час вне top-k картинки.
    """
    result = result or decode(sample.inputs, sample.context, model)
    ev = _evidence(sample.inputs, model.search.prob_floor)
    score = _score_scalar(sample.forms, ev, sample.context, model)
    if not math.isfinite(score):
        return LOG_P_CLIP, False
    reachable = candidate_space(sample.inputs, sample.context, model).contains(sample.forms)
    log_z = result.log_z if reachable else float(np.logaddexp(result.log_z, score))
    return max(LOG_P_CLIP, min(0.0, score - log_z)), reachable


@dataclass(frozen=True)
class FitScore:
    n: int
    mean_logp: float
    top1: float
    reachable: float
    fields_top1: float
    errors: tuple[str, ...] = field(default=())
    logps: tuple[float, ...] = field(default=(), repr=False)

    def as_dict(self) -> dict[str, Any]:
        out = dict(self.__dict__)
        out["errors"] = list(self.errors)
        out.pop("logps")
        return out


def bootstrap_support(gain: np.ndarray, n_boot: int = N_BOOTSTRAP, seed: int = 0) -> float:
    """Доля бутстреп-выборок, где средний прирост по ваучерам положителен."""
    if gain.size == 0:
        return 0.0
    idx = np.random.default_rng(seed).integers(0, gain.size, size=(n_boot, gain.size))
    return float((gain[idx].mean(axis=1) > 0).mean())


def evaluate_model(model: DecoderModel, samples: Sequence[Sample]) -> FitScore:
    """Среднее `log p` истинной записи и точность top-1 декодера."""
    fit_model = replace(model, search=replace(model.search, top_n=1))
    logps: list[float] = []
    hits = reach = base_hits = 0
    errors: list[str] = []
    for s in samples:
        result = decode(s.inputs, s.context, fit_model)
        logp, reachable = truth_log_prob(fit_model, s, result)
        logps.append(logp)
        reach += reachable
        ok = bool(result.records) and result.records[0].forms() == s.forms
        hits += ok
        if not ok:
            errors.append(s.scan_id)
        base_hits += fields_top1_correct(s)
    n = len(samples)
    if n == 0:
        return FitScore(0, float("nan"), float("nan"), float("nan"), float("nan"))
    return FitScore(
        n,
        float(np.mean(logps)),
        hits / n,
        reach / n,
        base_hits / n,
        tuple(sorted(errors)),
        tuple(logps),
    )


def fit_weights(
    model: DecoderModel,
    samples: Sequence[Sample],
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    rounds: int = 2,
    min_gain: float = DEFAULT_MIN_GAIN,
    min_support: float = DEFAULT_MIN_SUPPORT,
    log: Any = None,
) -> tuple[DecoderModel, FitScore, FitScore]:
    """Покоординатный поиск весов по сетке; цель — среднее `log p` истины.

    Шаг принимается, если цель растёт больше чем на `min_gain`, прирост поддержан бутстрепом
    (`bootstrap_support ≥ min_support`), а точность top-1 не падает.
    """
    start = evaluate_model(model, samples)
    best, best_score = model, start
    for rnd in range(rounds):
        improved = False
        for name in WEIGHT_NAMES:
            for value in grid:
                if name in IMAGE_WEIGHTS and value <= 0:
                    continue
                if math.isclose(getattr(best.weights, name), value):
                    continue
                cand = best.with_weights(**{name: value})
                score = evaluate_model(cand, samples)
                if (
                    score.mean_logp > best_score.mean_logp + min_gain
                    and score.top1 >= best_score.top1
                    and bootstrap_support(np.subtract(score.logps, best_score.logps)) >= min_support
                ):
                    best, best_score, improved = cand, score, True
            if log:
                log(
                    f"раунд {rnd + 1}: {name}={getattr(best.weights, name):g} "
                    f"logp={best_score.mean_logp:.4f} top1={best_score.top1:.4f}"
                )
        if not improved:
            break
    return best, start, best_score


# ---------------------------------------------------------------------------
# Полные распределения подполей через рантайм T18
# ---------------------------------------------------------------------------


def split_crops(split: str, *, manifest: Path, index_csv: Path) -> list[tuple[str, str, Path]]:
    """Кропы двузначных подполей сплита: `(scan_id, подполе, путь)`; `align_ok=False` — мимо."""
    ids = {r["scan_id"] for r in read_manifest(manifest) if r.get("split") == split}
    wanted = {subfield(r, p) for r in ROWS for p in PARTS}
    out: list[tuple[str, str, Path]] = []
    with index_csv.open(encoding="utf-8", newline="") as fh:
        for row in csv.DictReader(fh):
            scan_id = row["scan_id"].strip()  # `2026_223k ` в индексе T10 — с пробелом
            if scan_id not in ids or row["subfield"] not in wanted:
                continue
            if (row.get("align_ok") or "True") != "True":
                continue
            out.append((scan_id, row["subfield"], index_csv.parent / row["path"]))
    return out


def write_inputs(
    split: str,
    out: Path,
    *,
    model_dir: Path = DIGITS_DIR,
    manifest: Path = paths.MANIFEST,
    index_csv: Path = INDEX_CSV,
    k: int = 60,
    batch: int = 128,
) -> tuple[int, int]:
    """Распределения подполей сплита через `app.ocr.runtime` → JSONL T13. `(кропов, сканов)`."""
    from app.ocr import runtime as rt
    from ocr_lab.cut_crops import read_gray

    model = rt.OnnxModel(model_dir)
    crops = split_crops(split, manifest=manifest, index_csv=index_csv)
    by_scan: dict[str, dict[str, list[list[float]]]] = {}
    for s in range(0, len(crops), batch):
        chunk = crops[s : s + batch]
        parts = [rt.part_of(name) for _, name, _ in chunk]
        tens, units = rt.digit_logits(model, [read_gray(p) for _, _, p in chunk], parts)
        tops = rt.digit_top_values(tens, units, parts, model.meta.temperatures, k)
        for (scan_id, name, _), cands in zip(chunk, tops, strict=True):
            by_scan.setdefault(scan_id, {})[name] = [[v, p] for v, p in cands]
    paths.ensure_dir(out.parent)
    with out.open("w", encoding="utf-8") as fh:
        for scan_id in sorted(by_scan):
            item = {"scan_id": scan_id, "source": INPUT_SOURCE, "fields": by_scan[scan_id]}
            fh.write(json.dumps(item, ensure_ascii=False) + "\n")
    return len(crops), len(by_scan)


# ---------------------------------------------------------------------------
# Прогон декодера по сплиту
# ---------------------------------------------------------------------------


@dataclass
class PredictStats:
    split: str
    n_scans: int = 0
    n_records: int = 0
    n_no_candidates: int = 0
    ms_median: float = float("nan")
    ms_p95: float = float("nan")
    ms_max: float = float("nan")
    flags: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def _top_fields(dist: Iterable[Sequence[float]], k: int) -> list[list[float]]:
    ranked = sorted(((int(v), float(p)) for v, p in dist), key=lambda vp: (-vp[1], vp[0]))[:k]
    return [[v, p] for v, p in ranked]


def predict_split(
    split: str,
    model: DecoderModel,
    inputs: Mapping[str, DecoderInputs],
    rows: Sequence[Mapping[str, str]],
    out: Path,
    *,
    numbers: Mapping[str, list[list[float]]] | None = None,
    top_n: int = 5,
    fields_k: int = 5,
) -> PredictStats:
    """Декодер по ваучерам сплита → JSONL T13. Контекст (год, заявка, вид работ) — из манифеста."""
    model = replace(model, search=replace(model.search, top_n=top_n))
    stats = PredictStats(split=split)
    flags: Counter[str] = Counter()
    times: list[float] = []
    lines: list[str] = []
    for row in rows:
        scan_id = row["scan_id"]
        if row.get("split") != split or scan_id not in inputs:
            continue
        context = context_for(row, None)
        t0 = time.perf_counter()
        result = decode(inputs[scan_id], context, model)
        times.append(time.perf_counter() - t0)
        stats.n_scans += 1
        stats.n_records += bool(result.records)
        stats.n_no_candidates += not result.records
        flags.update(result.flags)
        fields = {name: _top_fields(d, fields_k) for name, d in inputs[scan_id].items() if d}
        if numbers and scan_id in numbers:
            fields[VOUCHER_NUMBER] = numbers[scan_id]
        item: dict[str, Any] = {"scan_id": scan_id, "source": SOURCE, "fields": fields}
        item.update(result.to_json())
        if not result.records:
            item.pop("confidence")
            item.pop("margin")
        lines.append(json.dumps(item, ensure_ascii=False))
    paths.ensure_dir(out.parent)
    out.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    if times:
        ms = np.asarray(times) * 1000.0
        stats.ms_median = float(np.median(ms))
        stats.ms_p95 = float(np.percentile(ms, 95))
        stats.ms_max = float(ms.max())
    stats.flags = dict(sorted(flags.items()))
    return stats


def load_numbers(path: Path) -> dict[str, list[list[float]]]:
    """Подполе `voucher_number` из JSONL T13 (например, `number_v0`)."""
    if not path.exists():
        return {}
    out: dict[str, list[list[float]]] = {}
    for pred in read_predictions(path):
        cands = pred.fields.get(VOUCHER_NUMBER)
        if cands:
            out[pred.scan_id.strip()] = [[v, p] for v, p in cands if p is not None]
    return out


# ---------------------------------------------------------------------------
# Сравнение «top-1 по полям» и декодера
# ---------------------------------------------------------------------------


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f} %"


def compare_metrics(base: Mapping[str, Any], dec: Mapping[str, Any]) -> list[str]:
    """Markdown-таблица ключевых метрик двух прогонов `ocr_lab.evaluate.evaluate`."""

    def ts(m: Mapping[str, Any], key: str) -> Any:
        dim = "overall" if key == "all" else "row"
        return m["timestamps"].get(dim, {}).get(key, {}).get("accuracy")

    def vouch(m: Mapping[str, Any], crit: str) -> tuple[Any, Any]:
        cell = m["vouchers"].get(crit, {}).get("all", {})
        return cell.get("accuracy"), cell.get("n")

    lines = [
        "| Метрика | Top-1 по полям | Декодер |",
        "|---|---:|---:|",
    ]
    for crit, title in (("rows", "ваучер: 4 строки"), ("rows_number", "ваучер: 4 строки + номер")):
        (a, n), (b, _) = vouch(base, crit), vouch(dec, crit)
        lines.append(f"| {title} (N = {n}) | {_pct(a)} | {_pct(b)} |")
    lines.append(f"| метки времени, все | {_pct(ts(base, 'all'))} | {_pct(ts(dec, 'all'))} |")
    for row in ROWS:
        lines.append(f"| строка {row} | {_pct(ts(base, row))} | {_pct(ts(dec, row))} |")
    for key, title in (
        ("abs_delta_work_minutes", r"\|Δ work_minutes\|"),
        ("abs_delta_busy_minutes", r"\|Δ busy_minutes\|"),
    ):
        a, b = base["money"][key], dec["money"][key]

        def fmt(d: Mapping[str, Any]) -> str:
            if not d["n"]:
                return "—"
            return (
                f"среднее {d['mean']:.1f}, p95 {d['p95']}, макс. {d['max']}, "
                f"> 0 у {_pct(d['share_nonzero'])} (N = {d['n']})"
            )

        lines.append(f"| {title}, мин | {fmt(a)} | {fmt(b)} |")
    lines.append(
        f"| без интервала работ | {base['money']['n_missing_work_interval']} | "
        f"{dec['money']['n_missing_work_interval']} |"
    )
    return lines


def row_minute_errors(
    truth: Mapping[str, Any], predictions: Iterable[Any]
) -> dict[str, dict[str, Any]]:
    """|предсказание − истина| меток времени в минутах по строкам (и по всем вместе)."""
    from ocr_lab.predictions import best_record

    preds = {p.scan_id: p for p in predictions}
    errs: dict[str, list[int]] = {r: [] for r in ("all", *ROWS)}
    for sid, t in truth.items():
        pred = preds.get(sid)
        if pred is None:
            continue
        rec = best_record(pred, t.row_years)
        for r in ROWS:
            tdt, pdt = t.rows.get(r), rec.rows.get(r)
            if tdt is None or pdt is None:
                continue
            err = int(abs((pdt - tdt).total_seconds()) // 60)
            errs[r].append(err)
            errs["all"].append(err)
    out: dict[str, dict[str, Any]] = {}
    for key, values in errs.items():
        arr = np.asarray(values, dtype=float)
        out[key] = {
            "n": int(arr.size),
            "mean": float(arr.mean()) if arr.size else None,
            "p95": float(np.percentile(arr, 95)) if arr.size else None,
            "max": float(arr.max()) if arr.size else None,
            "share_nonzero": float((arr > 0).mean()) if arr.size else None,
        }
    return out


def curve_points(
    curve: Sequence[Mapping[str, Any]], coverages: Sequence[float] = (0.25, 0.5, 0.6, 0.7, 0.75)
) -> list[Mapping[str, Any]]:
    """Сжатая кривая «покрытие — точность»: первая точка не ниже каждого уровня покрытия."""
    levels = sorted({*coverages, *(i / 20 for i in range(16, 21))})
    out: list[Mapping[str, Any]] = []
    for level in levels:
        point = next((p for p in curve if float(p["coverage"]) >= level - 1e-12), None)
        if point is not None and point not in out:
            out.append(point)
    return out


def write_compare(
    split: str,
    base_pred: Path,
    dec_pred: Path,
    out: Path,
    *,
    manifest: Path = paths.MANIFEST,
    stats: Mapping[str, Any] | None = None,
    numbers: Mapping[str, list[list[float]]] | None = None,
) -> dict[str, Any]:
    """Отчёт сравнения (Markdown + JSON рядом). Возвращает метрики обоих прогонов.

    В базовую линию подставляется номер ваучера из `numbers` (top-1 `number_v0`), как и в
    JSONL декодера: тогда «4 строки + номер» сравнимы.
    """
    from ocr_lab.evaluate import evaluate, load_printed_flags, load_truth

    truth = load_truth(manifest, split)
    flags = load_printed_flags(paths.PRINTED_FLAGS_CSV)
    base_preds = read_predictions(base_pred)
    for pred in base_preds:
        if numbers and VOUCHER_NUMBER not in pred.fields and pred.scan_id in numbers:
            pred.fields[VOUCHER_NUMBER] = [(int(v), float(p)) for v, p in numbers[pred.scan_id]]
    dec_preds = read_predictions(dec_pred)
    base = evaluate(truth, base_preds, printed_flags=flags)
    dec = evaluate(truth, dec_preds, printed_flags=flags)
    base["row_minute_errors"] = row_minute_errors(truth, base_preds)
    dec["row_minute_errors"] = row_minute_errors(truth, dec_preds)
    lines = [
        f"# Декодер T19 против «top-1 по полям», сплит `{split}`",
        "",
        f"- Top-1 по полям: `{base_pred.name}` ({', '.join(base['sources'])}); декодер: "
        f"`{dec_pred.name}`. Истина — манифест, ваучеров {len(truth)}.",
        "- Запись верна, если все 4 строки совпали по datetime (`24:00` = `00:00` след. суток).",
        "- Номер ваучера в обоих столбцах — top-1 `number_v0` (совместно с номером — T20).",
        "",
        *compare_metrics(base, dec),
        "",
        "## Ошибка меток времени, минуты",
        "",
        "| Строка | N | Top-1 по полям: среднее / p95 / макс. / доля > 0 | Декодер |",
        "|---|---:|---:|---:|",
    ]

    def err_cell(d: Mapping[str, Any]) -> str:
        if not d["n"]:
            return "—"
        return f"{d['mean']:.1f} / {d['p95']:.0f} / {d['max']:.0f} / {_pct(d['share_nonzero'])}"

    for key in ("all", *ROWS):
        a, b = base["row_minute_errors"][key], dec["row_minute_errors"][key]
        lines.append(f"| {key} | {a['n']} | {err_cell(a)} | {err_cell(b)} |")
    lines.append("")
    accept = dec.get("auto_accept")
    if accept:
        t = accept["threshold"]
        lines += [
            "## Автоприём декодера по `confidence`",
            "",
            f"- Цель {_pct(t['target'])}: порог {t['threshold']}, принято {t['n_accepted']} из "
            f"{t['n_total']} (покрытие {_pct(t['coverage'])}), точность {_pct(t['precision'])}, "
            f"95 % ДИ Уилсона [{_pct(t['ci_low'])}; {_pct(t['ci_high'])}].",
            f"- Подтверждено: {'да' if t['confirmed'] else 'нет'} — {t['note']}.",
            "",
            "Кривая «покрытие — точность» (сжатая; полная — в отчёте `ocr_lab.evaluate`):",
            "",
            "| Порог | Принято | Покрытие | Точность |",
            "|---:|---:|---:|---:|",
        ]
        for p in curve_points(accept["curve"]):
            lines.append(
                f"| {float(p['threshold']):.4f} | {p['n_accepted']} | {_pct(p['coverage'])} | "
                f"{_pct(p['precision'])} |"
            )
        lines.append("")
    if stats:
        lines += [
            "## Прогон декодера",
            "",
            f"- Ваучеров {stats['n_scans']}, без кандидатов {stats['n_no_candidates']}; время "
            f"на ваучер: медиана {stats['ms_median']:.1f} мс, p95 {stats['ms_p95']:.1f} мс, "
            f"максимум {stats['ms_max']:.1f} мс.",
            "- Флаги: " + ", ".join(f"`{k}` {v}" for k, v in stats["flags"].items()),
            "",
        ]
    paths.ensure_dir(out.parent)
    out.write_text("\n".join(lines), encoding="utf-8")
    out.with_suffix(".json").write_text(
        json.dumps(
            {"fields_top1": base, "decoder": dec, "stats": stats}, ensure_ascii=False, indent=1
        ),
        encoding="utf-8",
    )
    return {"fields_top1": base, "decoder": dec}


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def cmd_fit(args: argparse.Namespace) -> int:
    if not args.manifest.exists():
        print(f"нет манифеста: {args.manifest}", file=sys.stderr)
        return 2
    rows = read_manifest(args.manifest)
    train = [r for r in rows if r.get("split") == "train"]
    priors, stats = fit_priors(train, min_work_type=args.min_work_type)
    print(
        f"train: ваучеров {stats.train_vouchers}, цепочка ок {stats.chain_ok_vouchers}, "
        f"минуты кратны 10: {stats.minutes_mult10_share:.4f}, "
        f"через полночь: {stats.crosses_midnight_share:.4f}, "
        f"Окончание позже Прихода: {stats.late_finish} (+{stats.late_finish_beyond} дальше окна), "
        f"пар с заявкой {stats.app_pairs} (медиана {stats.app_median_hours:.2f} ч), "
        f"вид работ с отдельными длительностями: {len(priors.durations_by_work_type)}"
    )
    model = DecoderModel(priors=priors)
    meta: dict[str, Any] = {
        "source": "ocr_lab.fit_decoder_priors",
        "created": datetime.now().isoformat(timespec="seconds"),
        "manifest_sha256": _sha256(args.manifest),
        "prior_stats": stats.as_dict(),
    }
    pred: Path | None = args.pred
    if pred is None and not args.no_weights and DECODER_DIR.joinpath("val_inputs.jsonl").exists():
        pred = DECODER_DIR / "val_inputs.jsonl"
    if pred is not None and not args.no_weights:
        val = [r for r in rows if r.get("split") == args.val_split]
        samples = build_samples(val, load_predictions(pred))
        print(f"{args.val_split}: ваучеров с предсказаниями и истиной {len(samples)}")
        grid = tuple(float(v) for v in args.grid.split(",") if v.strip())
        model, before, after = fit_weights(
            model,
            samples,
            grid=grid,
            rounds=args.rounds,
            min_gain=args.min_gain,
            min_support=args.min_support,
            log=print,
        )
        print(f"до: {before.as_dict()}")
        print(f"после: {after.as_dict()}")
        meta.update(
            {
                "weights_fit_on": args.val_split,
                "predictions": pred.name,
                "predictions_sha256": _sha256(pred),
                "fit_before": before.as_dict(),
                "fit_after": after.as_dict(),
            }
        )
    else:
        meta["weights_fit_on"] = None
    model = replace(model, meta=meta)
    model.save(args.out)
    print(f"записано: {args.out}")
    return 0


def cmd_inputs(args: argparse.Namespace) -> int:
    out = args.out or DECODER_DIR / f"{args.split}_inputs.jsonl"
    n_crops, n_scans = write_inputs(
        args.split, out, model_dir=args.model_dir, manifest=args.manifest, k=args.k
    )
    print(f"{args.split}: {n_crops} кропов, {n_scans} сканов → {out}")
    return 0


def cmd_predict(args: argparse.Namespace) -> int:
    model = DecoderModel.load(args.model)
    if model.meta.get("source") == "defaults_from_plan":
        print(f"нет {args.model}: приоры по умолчанию", file=sys.stderr)
    inputs_path = args.inputs or DECODER_DIR / f"{args.split}_inputs.jsonl"
    inputs = load_predictions(inputs_path)
    numbers = load_numbers(args.number_pred or NUMBER_DIR / f"{args.split}_predictions.jsonl")
    out = args.out or DECODER_DIR / f"{args.split}_predictions.jsonl"
    stats = predict_split(
        args.split,
        model,
        inputs,
        read_manifest(args.manifest),
        out,
        numbers=numbers,
        top_n=args.top_n,
    )
    out.with_name(f"{args.split}_summary.json").write_text(
        json.dumps(stats.as_dict(), ensure_ascii=False, indent=1), encoding="utf-8"
    )
    print(
        f"{args.split}: ваучеров {stats.n_scans}, без кандидатов {stats.n_no_candidates}, "
        f"время медиана {stats.ms_median:.1f} мс, p95 {stats.ms_p95:.1f} мс, "
        f"макс. {stats.ms_max:.1f} мс → {out}"
    )
    return 0


def cmd_compare(args: argparse.Namespace) -> int:
    # Базовая линия — top-1 тех же распределений, что получил декодер (`inputs`); у JSONL
    # `digits_v0` на `test` нет скана `2026_223k ` (пробел в индексе кропов, T17).
    base = args.base or DECODER_DIR / f"{args.split}_inputs.jsonl"
    if not base.exists():
        base = DIGITS_DIR / f"{args.split}_predictions.jsonl"
    dec = args.pred or DECODER_DIR / f"{args.split}_predictions.jsonl"
    out = args.out or paths.REPORTS_DIR / f"decoder_v0_compare_{args.split}.md"
    summary = dec.with_name(f"{args.split}_summary.json")
    stats = json.loads(summary.read_text(encoding="utf-8")) if summary.exists() else None
    numbers = load_numbers(NUMBER_DIR / f"{args.split}_predictions.jsonl")
    metrics = write_compare(
        args.split, base, dec, out, manifest=args.manifest, stats=stats, numbers=numbers
    )
    for name, m in metrics.items():
        cell = m["vouchers"].get("rows", {}).get("all", {})
        print(f"{name}: ваучер целиком {_pct(cell.get('accuracy'))} (N = {cell.get('n')})")
    print(f"отчёт: {out}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocr_lab.fit_decoder_priors", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_fit = sub.add_parser("fit", help="приоры по train и веса по val")
    p_fit.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    p_fit.add_argument(
        "--pred",
        type=Path,
        default=None,
        help="входы val (JSONL T13); по умолчанию models/decoder_v0/val_inputs.jsonl",
    )
    p_fit.add_argument("--no-weights", action="store_true", help="только приоры")
    p_fit.add_argument("--out", type=Path, default=DECODER_JSON)
    p_fit.add_argument("--val-split", default="val")
    p_fit.add_argument("--min-work-type", type=int, default=60)
    p_fit.add_argument("--grid", default=",".join(f"{v:g}" for v in DEFAULT_GRID))
    p_fit.add_argument("--rounds", type=int, default=2)
    p_fit.add_argument("--min-gain", type=float, default=DEFAULT_MIN_GAIN)
    p_fit.add_argument("--min-support", type=float, default=DEFAULT_MIN_SUPPORT)
    p_fit.set_defaults(func=cmd_fit)

    p_in = sub.add_parser("inputs", help="распределения подполей через рантайм T18")
    p_in.add_argument("--split", choices=SPLITS, default="val")
    p_in.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    p_in.add_argument("--model-dir", type=Path, default=DIGITS_DIR)
    p_in.add_argument("--k", type=int, default=60, help="значений на подполе (60 = все)")
    p_in.add_argument("--out", type=Path, default=None)
    p_in.set_defaults(func=cmd_inputs)

    p_pr = sub.add_parser("predict", help="декодер по сплиту → JSONL T13")
    p_pr.add_argument("--split", choices=SPLITS, default="val")
    p_pr.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    p_pr.add_argument("--model", type=Path, default=DECODER_JSON)
    p_pr.add_argument("--inputs", type=Path, default=None)
    p_pr.add_argument("--number-pred", type=Path, default=None)
    p_pr.add_argument("--top-n", type=int, default=5)
    p_pr.add_argument("--out", type=Path, default=None)
    p_pr.set_defaults(func=cmd_predict)

    p_cmp = sub.add_parser("compare", help="«top-1 по полям» против декодера")
    p_cmp.add_argument("--split", choices=SPLITS, default="val")
    p_cmp.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    p_cmp.add_argument(
        "--base", type=Path, default=None, help="JSONL top-1 по полям (по умолчанию — входы)"
    )
    p_cmp.add_argument("--pred", type=Path, default=None, help="JSONL декодера")
    p_cmp.add_argument("--out", type=Path, default=None)
    p_cmp.set_defaults(func=cmd_compare)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
