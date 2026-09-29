"""Подбор приоров и весов совместного декодера (T19).

Запуск:

    python -m ocr_lab.fit_decoder_priors [--manifest data/ocr/manifest.csv]
        [--pred data/ocr/models/digits_v0/val_predictions.jsonl]
        [--out data/ocr/models/decoder_priors_v0.json]

- Приоры считаются по сплиту `train` манифеста (колонки T02): минуты, часы по строкам,
  длительности участков цепочки (общие и по виду работ, если данных хватает), шаблоны
  смещений дней и отклонение «Начало − время заявки». Эмпирика сглаживается к приорам
  по умолчанию из `app.ocr.decoder` (цифры плана), пол не даёт отрезать хвосты.
- Веса (`w_part` и множители приоров) подбираются на `val` по откалиброванным
  предсказаниям в формате T13: покоординатный поиск по сетке, цель — среднее
  лог-правдоподобие истинной записи при softmax по перечисленным кандидатам.
  Без `--pred` веса остаются по умолчанию.

В консоль идут только агрегаты.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import math
import sys
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np

from app.ocr.decoder import (
    APP_EDGES,
    CHAIN,
    DEFAULT_APP_MASSES,
    DURATION_EDGES,
    LEG_NAMES,
    PARTS,
    WEIGHT_NAMES,
    BinnedLogDensity,
    DecodeContext,
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
    subfield,
)
from ocr_lab import paths
from ocr_lab.predictions import read_predictions

LOG_P_CLIP = -30.0
DEFAULT_GRID: tuple[float, ...] = (0.0, 0.25, 0.5, 0.75, 1.0, 1.5, 2.0, 3.0)
# Веса картинки не обнуляем: без них декодер перестаёт читать.
IMAGE_WEIGHTS = ("day", "month", "hour", "minute")


# ---------------------------------------------------------------------------
# Манифест и истина
# ---------------------------------------------------------------------------


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


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
        if None in (d, m, y, h, mi):
            return None
        try:
            out[name] = (date(y, m, d), h, mi)  # type: ignore[arg-type]
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
    if minute_counts.sum():
        stats.minutes_mult10_share = float(minute_counts[::10].sum() / minute_counts.sum())
    minute_p = _smoothed(minute_counts, default_minute_probs(), alpha_minute)
    hour_logp = {
        name: tuple(float(v) for v in np.log(_smoothed(hour_counts[name], np.ones(25), alpha_hour)))
        for name in CHAIN
    }

    good = [(row, forms) for row, forms in samples if chain_ok(row, forms)]
    stats.chain_ok_vouchers = len(good)

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
    def leg_durations(items: Sequence[tuple[Mapping[str, str], dict[str, FormRow]]]):
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
            dens[name] = BinnedLogDensity.from_masses(DURATION_EDGES, masses[name], floor)
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
        APP_EDGES, app_masses, math.exp(base.app_deviation.floor)
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
# Веса по val
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    scan_id: str
    inputs: DecoderInputs
    context: DecodeContext
    forms: dict[str, FormRow]


def load_predictions(path: Path) -> dict[str, dict[str, list[tuple[int, float]]]]:
    """Подполя из JSONL T13 (`ocr_lab.predictions`): `{scan_id: {подполе: [(v, p), …]}}`."""
    return {
        pred.scan_id: {name: [(int(v), float(p)) for v, p in c] for name, c in pred.fields.items()}
        for pred in read_predictions(path)
    }


def build_samples(
    rows: Sequence[Mapping[str, str]], predictions: Mapping[str, DecoderInputs]
) -> list[Sample]:
    samples: list[Sample] = []
    for row in rows:
        scan_id = str(row.get("scan_id", ""))
        if scan_id not in predictions:
            continue
        forms = truth_forms(row)
        if forms is None or not chain_ok(row, forms):
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


@dataclass(frozen=True)
class FitScore:
    n: int
    mean_logp: float
    top1: float
    reachable: float
    fields_top1: float

    def as_dict(self) -> dict[str, float]:
        return dict(self.__dict__)


def evaluate_model(model: DecoderModel, samples: Sequence[Sample]) -> FitScore:
    """Среднее лог-правдоподобие истинной записи и точность top-1 декодера."""
    fit_model = replace(model, search=replace(model.search, top_n=1))
    logps: list[float] = []
    hits = reach = base_hits = 0
    for s in samples:
        result = decode(s.inputs, s.context, fit_model)
        space = candidate_space(s.inputs, s.context, fit_model)
        if result.records and space.contains(s.forms):
            reach += 1
            ev = _evidence(s.inputs, fit_model.search.prob_floor)
            logp = _score_scalar(s.forms, ev, s.context, fit_model) - result.log_z
            logps.append(max(LOG_P_CLIP, min(0.0, logp)))
        else:
            logps.append(LOG_P_CLIP)
        if result.records and result.records[0].forms() == s.forms:
            hits += 1
        base_hits += fields_top1_correct(s)
    n = len(samples)
    if n == 0:
        return FitScore(0, float("nan"), float("nan"), float("nan"), float("nan"))
    return FitScore(n, float(np.mean(logps)), hits / n, reach / n, base_hits / n)


def fit_weights(
    model: DecoderModel,
    samples: Sequence[Sample],
    *,
    grid: Sequence[float] = DEFAULT_GRID,
    rounds: int = 2,
    log: Any = None,
) -> tuple[DecoderModel, FitScore, FitScore]:
    """Покоординатный поиск весов по сетке; цель — среднее лог-правдоподобие истины."""
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
                if score.mean_logp > best_score.mean_logp + 1e-6:
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
# CLI
# ---------------------------------------------------------------------------


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    parser.add_argument("--pred", type=Path, default=None, help="val-предсказания T13 (JSONL)")
    parser.add_argument("--out", type=Path, default=paths.MODELS_DIR / "decoder_priors_v0.json")
    parser.add_argument("--val-split", default="val")
    parser.add_argument("--min-work-type", type=int, default=60)
    parser.add_argument("--grid", default=",".join(f"{v:g}" for v in DEFAULT_GRID))
    parser.add_argument("--rounds", type=int, default=2)
    args = parser.parse_args(argv)

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
    if args.pred is not None:
        predictions = load_predictions(args.pred)
        val = [r for r in rows if r.get("split") == args.val_split]
        samples = build_samples(val, predictions)
        print(f"{args.val_split}: ваучеров с предсказаниями и истиной {len(samples)}")
        grid = tuple(float(v) for v in args.grid.split(",") if v.strip())
        model, before, after = fit_weights(model, samples, grid=grid, rounds=args.rounds, log=print)
        print(f"до: {before.as_dict()}")
        print(f"после: {after.as_dict()}")
        meta.update(
            {
                "weights_fit_on": args.val_split,
                "predictions": args.pred.name,
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


if __name__ == "__main__":
    raise SystemExit(main())
