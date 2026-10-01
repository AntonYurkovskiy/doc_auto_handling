"""Оценка распознавателей дат по единому формату предсказаний.

Запуск:

    python -m ocr_lab.evaluate --pred <file.jsonl> --split val [--out <dir>]
    python -m ocr_lab.evaluate --write-perfect <file.jsonl> --split val

Истина — `data/ocr/manifest.csv` (уже с поправками), читает её только `load_truth`.
Метрики: подполя (top-1/top-3, NLL, ECE, покрытие, разрезы), метки времени, ваучер целиком,
деньги (Δ минут работы и занятости, ночной флаг, дата курса) и автоприём по `confidence`.
В отчёт попадают только агрегаты.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

from app.services.calculation import _is_night, minutes_between
from ocr_lab import paths
from ocr_lab.predictions import (
    PARTS,
    ROWS,
    SUBFIELDS,
    VOUCHER_NUMBER,
    Candidate,
    RecordCandidate,
    ScanPrediction,
    best_record,
    read_predictions,
    row_datetime,
    write_predictions,
)

SPLITS = ("train", "val", "test", "all")
# Нижняя граница вероятности истины для NLL, если истины нет среди top-k.
NLL_FLOOR = 1e-6
ECE_BINS = 10
WILSON_Z = 1.96
DEFAULT_TARGET = 0.995


# --- Истина -------------------------------------------------------------------------------


@dataclass
class ScanTruth:
    """Истина по одному скану из манифеста."""

    scan_id: str
    split: str
    tug_code: str
    year: int | None
    voucher_number: int | None
    rows: dict[str, datetime | None]
    row_years: dict[str, int | None]
    parts: dict[str, int | None]
    has_hour24: bool = False


def _to_int(value: str | None) -> int | None:
    text = (value or "").strip()
    if not text or text.lower() in {"nan", "none", "null"}:
        return None
    return int(float(text))


def _to_bool(value: str | None) -> bool:
    return (value or "").strip().lower() in {"1", "true", "yes", "да"}


def _to_dt(value: str | None) -> datetime | None:
    text = (value or "").strip()
    if not text or text.lower() in {"nan", "nat", "none"}:
        return None
    return datetime.fromisoformat(text)


def load_truth(manifest: Path, split: str = "all") -> dict[str, ScanTruth]:
    """Прочитать истину из манифеста T02, при `split != "all"` — только этот сплит.

    Колонки: `scan_id`, `split`, `tug_code`, `year`, `voucher_number`, `has_hour24`
    и для каждой строки `r`: `r_dt`, `r_day`, `r_month`, `r_year`, `r_hour`, `r_minute`.
    `r_hour` может быть 24, тогда `r_dt` — 00:00 следующих суток.
    """
    if split not in SPLITS:
        raise ValueError(f"Неизвестный сплит: {split}")
    out: dict[str, ScanTruth] = {}
    with manifest.open(encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            row_split = (rec.get("split") or "").strip()
            if split != "all" and row_split != split:
                continue
            scan_id = rec["scan_id"].strip()
            year = _to_int(rec.get("year"))
            parts: dict[str, int | None] = {}
            rows: dict[str, datetime | None] = {}
            row_years: dict[str, int | None] = {}
            for r in ROWS:
                for p in PARTS:
                    parts[f"{r}.{p}"] = _to_int(rec.get(f"{r}_{p}"))
                row_year = _to_int(rec.get(f"{r}_year"))
                row_years[r] = row_year if row_year is not None else year
                dt = _to_dt(rec.get(f"{r}_dt"))
                vals = [parts[f"{r}.{p}"] for p in PARTS]
                if dt is None and row_years[r] is not None and None not in vals:
                    d, m, h, mi = (int(v) for v in vals if v is not None)
                    dt = row_datetime(int(row_years[r] or 0), m, d, h, mi)
                rows[r] = dt
            number = _to_int(rec.get("voucher_number"))
            parts[VOUCHER_NUMBER] = number
            out[scan_id] = ScanTruth(
                scan_id=scan_id,
                split=row_split,
                tug_code=(rec.get("tug_code") or "").strip() or "?",
                year=year,
                voucher_number=number,
                rows=rows,
                row_years=row_years,
                parts=parts,
                has_hour24=_to_bool(rec.get("has_hour24")),
            )
    return out


def load_printed_flags(path: Path | None) -> dict[tuple[str, str], bool]:
    """Флаги «печатное/рукописное» из CSV `scan_id, field, printed`.

    `field` — подполе (`left_base.hour`), строка бланка (`left_base`, действует на все её
    части) или `voucher_number`. Нет файла — пустой словарь.
    """
    if path is None or not path.exists():
        return {}
    out: dict[tuple[str, str], bool] = {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        for rec in csv.DictReader(fh):
            out[(rec["scan_id"].strip(), rec["field"].strip())] = _to_bool(rec.get("printed"))
    return out


def _printed_key(flags: Mapping[tuple[str, str], bool], scan_id: str, subfield: str) -> str:
    flag = flags.get((scan_id, subfield))
    if flag is None:
        flag = flags.get((scan_id, subfield.split(".")[0]))
    if flag is None:
        return "unknown"
    return "printed" if flag else "handwritten"


def perfect_predictions(
    truth: Mapping[str, ScanTruth], *, with_records: bool = False, source: str = "truth"
) -> list[ScanPrediction]:
    """«Идеальные» предсказания из самой истины — для проверки модуля оценки."""
    out: list[ScanPrediction] = []
    for t in truth.values():
        fields: dict[str, list[Candidate]] = {
            k: [(v, 1.0)] for k, v in t.parts.items() if v is not None and k in SUBFIELDS
        }
        records: list[RecordCandidate] = []
        if with_records:
            records.append(
                RecordCandidate(
                    rows=dict(t.rows),
                    hour24=tuple(r for r in ROWS if t.parts.get(f"{r}.hour") == 24),
                    voucher_number=t.voucher_number,
                    p=1.0,
                )
            )
        out.append(
            ScanPrediction(
                scan_id=t.scan_id,
                source=source,
                fields=fields,
                records=records,
                confidence=1.0,
                margin=1.0,
            )
        )
    return out


# --- Метрики подполей ---------------------------------------------------------------------


@dataclass
class _FieldAcc:
    n: int = 0
    covered: int = 0
    top1: int = 0
    top3: int = 0
    nll_sum: float = 0.0
    nll_n: int = 0
    conf: list[float] = field(default_factory=list)
    hit: list[bool] = field(default_factory=list)

    def add(self, truth: int, cands: Sequence[tuple[int, float | None]] | None) -> None:
        self.n += 1
        if not cands:
            return
        self.covered += 1
        values = [v for v, _ in cands]
        ok1 = values[0] == truth
        self.top1 += ok1
        self.top3 += truth in values[:3]
        probs = [p for _, p in cands]
        if all(p is not None for p in probs):
            p_true = next((p for v, p in cands if v == truth), None) or 0.0
            self.nll_sum += -math.log(max(p_true, NLL_FLOOR))
            self.nll_n += 1
            self.conf.append(float(probs[0] or 0.0))
            self.hit.append(ok1)

    def summary(self) -> dict[str, Any]:
        return {
            "n": self.n,
            "coverage": _ratio(self.covered, self.n),
            "top1": _ratio(self.top1, self.n),
            "top3": _ratio(self.top3, self.n),
            "n_with_prob": self.nll_n,
            "nll": self.nll_sum / self.nll_n if self.nll_n else None,
            "ece": expected_calibration_error(self.conf, self.hit) if self.conf else None,
        }


def _ratio(num: int, den: int) -> float | None:
    return num / den if den else None


def expected_calibration_error(
    confidences: Sequence[float], correct: Sequence[bool], bins: int = ECE_BINS
) -> float:
    """ECE по top-1 вероятности: равные по ширине корзины на [0, 1]."""
    if not confidences:
        return 0.0
    buckets: dict[int, list[tuple[float, bool]]] = defaultdict(list)
    for c, ok in zip(confidences, correct, strict=True):
        buckets[min(int(c * bins), bins - 1)].append((c, ok))
    total = len(confidences)
    ece = 0.0
    for items in buckets.values():
        avg_conf = sum(c for c, _ in items) / len(items)
        acc = sum(ok for _, ok in items) / len(items)
        ece += abs(acc - avg_conf) * len(items) / total
    return ece


# --- Автоприём ----------------------------------------------------------------------------


def wilson_interval(k: int, n: int, z: float = WILSON_Z) -> tuple[float, float]:
    """95 % доверительный интервал Уилсона для доли k/n."""
    if n <= 0:
        return (0.0, 1.0)
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def min_n_to_confirm(target: float, z: float = WILSON_Z) -> int:
    """Сколько принятых без ошибок нужно, чтобы нижняя граница Уилсона достигла `target`."""
    if target >= 1:
        raise ValueError("Цель 100 % не подтверждается конечной выборкой")
    # При k = n нижняя граница равна n / (n + z²).
    return math.ceil(target * z * z / (1 - target) - 1e-9)


@dataclass
class AcceptThreshold:
    """Порог автоприёма для заданной точности."""

    target: float
    threshold: float | None
    n_total: int
    n_accepted: int
    n_correct: int
    coverage: float | None
    precision: float | None
    ci_low: float | None
    ci_high: float | None
    confirmed: bool
    min_n_to_confirm: int
    note: str


def coverage_precision_curve(
    confidences: Sequence[float], correct: Sequence[bool], n_total: int | None = None
) -> list[dict[str, float | int]]:
    """Точки «порог → покрытие, точность» по уникальным значениям уверенности.

    Принимается скан с `confidence >= порог`. Покрытие — доля от `n_total` (по умолчанию
    от числа сканов с уверенностью).
    """
    total = n_total if n_total is not None else len(confidences)
    pairs = sorted(zip(confidences, correct, strict=True), key=lambda x: -x[0])
    points: list[dict[str, float | int]] = []
    n_acc = n_ok = 0
    i = 0
    while i < len(pairs):
        t = pairs[i][0]
        while i < len(pairs) and pairs[i][0] == t:
            n_acc += 1
            n_ok += pairs[i][1]
            i += 1
        points.append(
            {
                "threshold": t,
                "n_accepted": n_acc,
                "n_correct": n_ok,
                "coverage": n_acc / total if total else 0.0,
                "precision": n_ok / n_acc,
            }
        )
    return points


def threshold_for_precision(
    confidences: Sequence[float],
    correct: Sequence[bool],
    target: float = DEFAULT_TARGET,
    *,
    n_total: int | None = None,
    z: float = WILSON_Z,
) -> AcceptThreshold:
    """Наименьший порог (наибольшее покрытие), при котором точность не ниже `target`.

    Сначала ищется порог, где цель подтверждена нижней границей Уилсона. Если такого нет,
    берётся порог по точечной оценке, а `confirmed=False` и `note` честно говорят, что
    выборка цель не подтверждает.
    """
    total = n_total if n_total is not None else len(confidences)
    need = min_n_to_confirm(target, z)
    points = coverage_precision_curve(confidences, correct, total)

    def build(point: dict[str, float | int] | None, confirmed: bool, note: str) -> AcceptThreshold:
        if point is None:
            return AcceptThreshold(
                target, None, total, 0, 0, 0.0 if total else None, None, None, None,
                False, need, note,
            )
        n_acc, n_ok = int(point["n_accepted"]), int(point["n_correct"])
        low, high = wilson_interval(n_ok, n_acc, z)
        return AcceptThreshold(
            target=target,
            threshold=float(point["threshold"]),
            n_total=total,
            n_accepted=n_acc,
            n_correct=n_ok,
            coverage=float(point["coverage"]),
            precision=float(point["precision"]),
            ci_low=low,
            ci_high=high,
            confirmed=confirmed,
            min_n_to_confirm=need,
            note=note,
        )

    if not points:
        return build(None, False, "нет предсказаний с confidence")
    confirmed = [
        p for p in points
        if wilson_interval(int(p["n_correct"]), int(p["n_accepted"]), z)[0] >= target
    ]
    if confirmed:
        return build(confirmed[-1], True, "цель подтверждена нижней границей Уилсона")
    by_point = [p for p in points if float(p["precision"]) >= target]
    small = (
        f"выборка слишком мала, чтобы подтвердить {target:.1%}: даже без ошибок нужно "
        f"не меньше {need} принятых ваучеров"
    )
    if by_point:
        best = by_point[-1]
        if int(best["n_accepted"]) < need:
            return build(best, False, small)
        return build(
            best, False, "точечная оценка ≥ цели, но нижняя граница Уилсона ниже цели"
        )
    note = f"ни один порог не даёт точность ≥ {target:.1%}"
    if len(confidences) < need:
        note += f"; {small}"
    return build(None, False, note)


# --- Деньги -------------------------------------------------------------------------------


def _distribution(values: Sequence[int]) -> dict[str, Any]:
    if not values:
        return {"n": 0, "mean": None, "p95": None, "max": None, "share_nonzero": None}
    s = sorted(values)
    rank = max(1, math.ceil(0.95 * len(s)))
    return {
        "n": len(s),
        "mean": sum(s) / len(s),
        "p95": s[rank - 1],
        "max": s[-1],
        "share_nonzero": sum(v > 0 for v in s) / len(s),
    }


# --- Оценка -------------------------------------------------------------------------------


def evaluate(
    truth: Mapping[str, ScanTruth],
    predictions: Iterable[ScanPrediction],
    *,
    printed_flags: Mapping[tuple[str, str], bool] | None = None,
    target: float = DEFAULT_TARGET,
) -> dict[str, Any]:
    """Посчитать все метрики. Сканы истины без предсказания считаются непокрытыми."""
    flags = printed_flags or {}
    preds = {p.scan_id: p for p in predictions}
    sources = sorted({p.source for p in preds.values()})
    matched = [sid for sid in truth if sid in preds]

    field_slices: dict[str, dict[str, _FieldAcc]] = defaultdict(lambda: defaultdict(_FieldAcc))
    ts_slices: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))
    vouchers: dict[str, dict[str, list[int]]] = defaultdict(lambda: defaultdict(lambda: [0, 0]))
    d_work: list[int] = []
    d_busy: list[int] = []
    night = [0, 0]
    fx_date = [0, 0]
    money_missing = 0
    acc_conf: list[float] = []
    acc_ok: list[bool] = []

    for sid, t in truth.items():
        pred = preds.get(sid)
        year_key = str(t.year) if t.year is not None else "?"

        # Подполя.
        for sub in SUBFIELDS:
            tv = t.parts.get(sub)
            if tv is None:
                continue
            cands = pred.fields.get(sub) if pred else None
            part = sub.split(".")[1] if "." in sub else "number"
            for dim, key in (
                ("overall", "all"),
                ("subfield", sub),
                ("part", part),
                ("tug", t.tug_code),
                ("printed", _printed_key(flags, sid, sub)),
                ("year", year_key),
            ):
                field_slices[dim][key].add(tv, cands)

        # Метки времени и ваучер.
        rec = best_record(pred, t.row_years) if pred else None
        rows_ok = True
        n_rows = 0
        for r in ROWS:
            tdt = t.rows.get(r)
            if tdt is None:
                continue
            n_rows += 1
            pdt = rec.rows.get(r) if rec else None
            ok = pdt is not None and pdt == tdt
            rows_ok &= ok
            for dim, key in (("overall", "all"), ("row", r), ("tug", t.tug_code)):
                cell = ts_slices[dim][key]
                cell[0] += 1
                cell[1] += pdt is not None
                cell[2] += ok
        if n_rows:
            number_ok = (
                rec is not None
                and t.voucher_number is not None
                and rec.voucher_number == t.voucher_number
            )
            for key in ("all", t.tug_code):
                vouchers["rows"][key][0] += 1
                vouchers["rows"][key][1] += rows_ok
                if t.voucher_number is not None:
                    vouchers["rows_number"][key][0] += 1
                    vouchers["rows_number"][key][1] += rows_ok and number_ok
            if pred is not None and pred.confidence is not None:
                acc_conf.append(pred.confidence)
                acc_ok.append(rows_ok)

        # Деньги.
        ts, tf = t.rows.get("started_work"), t.rows.get("finished_work")
        tl, ta = t.rows.get("left_base"), t.rows.get("arrived_base")
        ps = rec.rows.get("started_work") if rec else None
        pf = rec.rows.get("finished_work") if rec else None
        pl = rec.rows.get("left_base") if rec else None
        pa = rec.rows.get("arrived_base") if rec else None
        if ts and tf:
            if ps and pf:
                d_work.append(abs(minutes_between(ps, pf) - minutes_between(ts, tf)))
            else:
                money_missing += 1
        if tl and ta and pl and pa:
            d_busy.append(abs(minutes_between(pl, pa) - minutes_between(tl, ta)))
        if ts and ps:
            night[0] += 1
            night[1] += _is_night(ts) != _is_night(ps)
        if tf and pf:
            fx_date[0] += 1
            fx_date[1] += tf.date() != pf.date()

    def _ts_summary(cell: list[int]) -> dict[str, Any]:
        return {
            "n": cell[0],
            "coverage": _ratio(cell[1], cell[0]),
            "accuracy": _ratio(cell[2], cell[0]),
        }

    def _v_summary(cell: list[int]) -> dict[str, Any]:
        return {"n": cell[0], "accuracy": _ratio(cell[1], cell[0])}

    accept: dict[str, Any] | None = None
    if acc_conf:
        n_vouchers = vouchers["rows"]["all"][0]
        thr = threshold_for_precision(acc_conf, acc_ok, target, n_total=n_vouchers)
        accept = {
            "criterion": "все 4 строки верны",
            "n_with_confidence": len(acc_conf),
            "curve": coverage_precision_curve(acc_conf, acc_ok, n_vouchers),
            "threshold": asdict(thr),
        }

    return {
        "sources": sources,
        "n_truth_scans": len(truth),
        "n_pred_scans": len(preds),
        "n_matched": len(matched),
        "n_pred_not_in_split": len(set(preds) - set(truth)),
        "subfields": {
            dim: {k: acc.summary() for k, acc in sorted(sl.items())}
            for dim, sl in field_slices.items()
        },
        "timestamps": {
            dim: {k: _ts_summary(c) for k, c in sorted(sl.items())}
            for dim, sl in ts_slices.items()
        },
        "vouchers": {
            crit: {k: _v_summary(c) for k, c in sorted(sl.items())}
            for crit, sl in vouchers.items()
        },
        "money": {
            "abs_delta_work_minutes": _distribution(d_work),
            "abs_delta_busy_minutes": _distribution(d_busy),
            "n_missing_work_interval": money_missing,
            "night_flag_changed": {"n": night[0], "share": _ratio(night[1], night[0])},
            "finish_date_changed": {"n": fx_date[0], "share": _ratio(fx_date[1], fx_date[0])},
        },
        "auto_accept": accept,
    }


# --- Отчёт --------------------------------------------------------------------------------


def _pct(value: Any) -> str:
    return "—" if value is None else f"{100 * float(value):.2f} %"


def _num(value: Any, digits: int = 3) -> str:
    if value is None:
        return "—"
    if isinstance(value, float):
        return f"{value:.{digits}f}"
    return str(value)


def render_report(metrics: Mapping[str, Any], *, split: str) -> str:
    """Отчёт в Markdown (только агрегаты)."""
    src = ", ".join(metrics["sources"]) or "—"
    lines = [
        f"# Оценка распознавателя: {src}, сплит `{split}`",
        "",
        f"- Сканов в истине: {metrics['n_truth_scans']}; в предсказаниях: "
        f"{metrics['n_pred_scans']}; совпало: {metrics['n_matched']}; "
        f"вне сплита: {metrics['n_pred_not_in_split']}.",
        "",
        "## Подполя",
        "",
    ]
    names = {
        "overall": "Всего",
        "subfield": "По подполю",
        "part": "По части",
        "tug": "По буксиру",
        "printed": "Печатное / рукописное",
        "year": "По году",
    }
    for dim, title in names.items():
        sl = metrics["subfields"].get(dim)
        if not sl:
            continue
        lines += [
            f"### {title}",
            "",
            "| Срез | N | Покрытие | Top-1 | Top-3 | NLL | ECE |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
        for key, s in sl.items():
            lines.append(
                f"| {key} | {s['n']} | {_pct(s['coverage'])} | {_pct(s['top1'])} | "
                f"{_pct(s['top3'])} | {_num(s['nll'])} | {_num(s['ece'])} |"
            )
        lines.append("")

    lines += [
        "## Метки времени",
        "",
        "Строка верна, если совпадает datetime; `24:00` дня D равно `00:00` дня D+1.",
        "",
        "| Срез | N | Покрытие | Точность |",
        "|---|---:|---:|---:|",
    ]
    for dim in ("overall", "row", "tug"):
        for key, s in metrics["timestamps"].get(dim, {}).items():
            label = "всего" if dim == "overall" else f"{dim}: {key}"
            lines.append(
                f"| {label} | {s['n']} | {_pct(s['coverage'])} | {_pct(s['accuracy'])} |"
            )
    lines += [
        "",
        "## Ваучер целиком",
        "",
        "| Критерий | Срез | N | Точность |",
        "|---|---|---:|---:|",
    ]
    crit_names = {"rows": "4 строки", "rows_number": "4 строки + номер"}
    for crit, sl in metrics["vouchers"].items():
        for key, s in sl.items():
            lines.append(
                f"| {crit_names.get(crit, crit)} | {key} | {s['n']} | {_pct(s['accuracy'])} |"
            )

    money = metrics["money"]
    lines += [
        "",
        "## Деньги",
        "",
        "| Величина | N | Среднее | p95 | Максимум | Доля > 0 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for key, title in (
        ("abs_delta_work_minutes", r"\|Δ work_minutes\|"),
        ("abs_delta_busy_minutes", r"\|Δ busy_minutes\|"),
    ):
        d = money[key]
        lines.append(
            f"| {title} | {d['n']} | {_num(d['mean'], 1)} | {_num(d['p95'])} | "
            f"{_num(d['max'])} | {_pct(d['share_nonzero'])} |"
        )
    lines += [
        "",
        f"- Без предсказанного интервала работ: {money['n_missing_work_interval']}.",
        f"- Сменился ночной флаг начала работ: {_pct(money['night_flag_changed']['share'])} "
        f"(N = {money['night_flag_changed']['n']}).",
        f"- Сменилась дата окончания (курс): {_pct(money['finish_date_changed']['share'])} "
        f"(N = {money['finish_date_changed']['n']}).",
        "- Выходные и праздники не оцениваются: для них нужен внешний календарь.",
        "",
        "## Автоприём",
        "",
    ]
    accept = metrics.get("auto_accept")
    if not accept:
        lines.append("Нет предсказаний с `confidence` — автоприём не оценивается.")
    else:
        t = accept["threshold"]
        lines += [
            f"Критерий верности: {accept['criterion']}. Сканов с confidence: "
            f"{accept['n_with_confidence']}.",
            "",
            f"- Цель: {_pct(t['target'])}; порог: {_num(t['threshold'], 4)}.",
            f"- Принято: {t['n_accepted']} из {t['n_total']} (покрытие {_pct(t['coverage'])}); "
            f"точность {_pct(t['precision'])}, 95 % ДИ Уилсона "
            f"[{_pct(t['ci_low'])}; {_pct(t['ci_high'])}].",
            f"- Цель подтверждена: {'да' if t['confirmed'] else 'нет'} — {t['note']}.",
            "",
            "| Порог | Принято | Покрытие | Точность |",
            "|---:|---:|---:|---:|",
        ]
        for p in accept["curve"]:
            lines.append(
                f"| {_num(float(p['threshold']), 4)} | {p['n_accepted']} | "
                f"{_pct(p['coverage'])} | {_pct(p['precision'])} |"
            )
    lines.append("")
    return "\n".join(lines)


def write_report(metrics: Mapping[str, Any], out_dir: Path, *, split: str) -> None:
    """Записать `metrics.json` и `report.md` в `out_dir`."""
    paths.ensure_dir(out_dir)
    (out_dir / "metrics.json").write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    (out_dir / "report.md").write_text(render_report(metrics, split=split), encoding="utf-8")


# --- CLI ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ocr_lab.evaluate", description=__doc__)
    mode = parser.add_mutually_exclusive_group(required=True)
    mode.add_argument("--pred", type=Path, help="JSONL с предсказаниями")
    mode.add_argument(
        "--write-perfect", type=Path, help="записать «идеальные» предсказания из истины"
    )
    parser.add_argument("--split", choices=SPLITS, default="val")
    parser.add_argument("--out", type=Path, help="каталог отчёта")
    parser.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    parser.add_argument("--printed-flags", type=Path, default=paths.PRINTED_FLAGS_CSV)
    parser.add_argument("--target", type=float, default=DEFAULT_TARGET)
    parser.add_argument(
        "--with-records", action="store_true", help="для --write-perfect: добавить records"
    )
    args = parser.parse_args(argv)

    truth = load_truth(args.manifest, args.split)
    if args.write_perfect:
        n = write_predictions(
            args.write_perfect, perfect_predictions(truth, with_records=args.with_records)
        )
        print(f"Записано идеальных предсказаний: {n} ({args.split})")
        return 0

    preds = read_predictions(args.pred)
    metrics = evaluate(
        truth, preds, printed_flags=load_printed_flags(args.printed_flags), target=args.target
    )
    source = "+".join(metrics["sources"]) or "empty"
    out_dir = args.out or paths.REPORTS_DIR / f"eval_{source}_{args.split}"
    write_report(metrics, out_dir, split=args.split)
    overall = metrics["timestamps"].get("overall", {}).get("all", {})
    vouch = metrics["vouchers"].get("rows", {}).get("all", {})
    print(
        f"{source} / {args.split}: сканов {metrics['n_matched']}/{metrics['n_truth_scans']}, "
        f"метки времени {_pct(overall.get('accuracy'))}, ваучеры {_pct(vouch.get('accuracy'))}"
    )
    print(f"Отчёт: {out_dir}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
