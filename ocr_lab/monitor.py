"""T28: ежемесячный отчёт мониторинга качества OCR (план, «Этап 6»).

Смотрит на ваучеры, подтверждённые оператором в заданном календарном месяце
(`Voucher.reviewed_at`), и считает:

- долю автоприёма (`voucher_fields.voucher_auto_accept`);
- долю правок оператором по полям, где предсказание (`predicted_value`) не совпало с
  подтверждением (`confirmed_value`) — номер ваучера и 4 строки дат (`app.ocr.decoder.ROWS`);
- тот же разрез по буксирам, по слотам вахт (формула T23: смена в 09:00, ротация 4 суток) и
  по вариантам бланка (переопределение выравнивания, независимое от того, что было сохранено
  при распознавании — см. :func:`_detect_variant`);
- предупреждения: всплеск правок в одном слоте вахты одного буксира (возможно, новый человек)
  и превышение порога отказов выравнивания 5 % (критерий T10) в одном варианте бланка
  (возможно, новый бланк).

Запуск: `.venv/Scripts/python -m ocr_lab.monitor --month 2026-09`.

В отчёт и консоль идут только агрегаты, коды буксиров (`k`/`p`) и `id` ваучера — без имён
судов, агентов и текстов (правило `_common.md`, п. 3).
"""

from __future__ import annotations

import argparse
import calendar
import sys
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.database import SessionLocal
from app.models import Voucher, VoucherFieldPrediction
from app.ocr.decoder import ROWS
from app.ocr.io import ScanLoadError, load_scan, normalize_width
from app.ocr.layouts import VOUCHER_NUMBER
from app.ocr.pipeline import STATUS_ALIGN_FAILED
from app.ocr.variants import Layout as VariantLayout
from app.ocr.variants import detect_variant
from app.ocr.variants import load_layouts as load_variant_layouts
from app.services.voucher_fields import voucher_auto_accept
from ocr_lab.export_confirmed import REVIEWED_STATUSES, _tug_code
from ocr_lab.paths import LAYOUTS_DIR, REPORTS_DIR, ensure_dir

#: Поля карточки, по которым считаем правки — номер + 4 строки дат (как в `export_confirmed`).
OCR_FIELDS: tuple[str, ...] = (VOUCHER_NUMBER, *ROWS)
#: Сид ГСЧ OpenCV перед выравниванием — воспроизводимость RANSAC (как в `app.ocr.pipeline`).
CV_SEED = 0
#: Минимальный размер группы, при котором предупреждение не считается шумом малой выборки.
MIN_GROUP_N = 5
#: Группа «подозрительна», если её доля правок на столько выше среднего по месяцу.
SLOT_SPIKE_MARGIN = 0.2
#: Порог доли отказов выравнивания — тот же критерий, что в T10 («0,4 % при пороге 5 %»).
ALIGN_FAILURE_RATE_THRESHOLD = 0.05
UNKNOWN_VARIANT = "unknown"


def month_range(month: str) -> tuple[date, date]:
    """`(первый день, первый день следующего месяца)` для `"YYYY-MM"`."""
    year, mon = (int(p) for p in month.split("-", 1))
    start = date(year, mon, 1)
    days = calendar.monthrange(year, mon)[1]
    return start, start + timedelta(days=days)


def shift_slot(dt: datetime) -> int:
    """Слот вахты 0..3 по формуле T23: смена в 09:00, ротация «сутки через трое».

    Слот осмыслен только в паре с буксиром (у каждого буксира свой состав вахт) —
    используется только внутри разреза `(буксир, слот)`.
    """
    return (dt - timedelta(hours=9)).toordinal() % 4


@dataclass
class FieldStats:
    """Правки по одному полю внутри группы: `attempted` — где была подсказка модели."""

    attempted: int = 0
    corrected: int = 0

    @property
    def rate(self) -> float | None:
        return self.corrected / self.attempted if self.attempted else None


@dataclass
class GroupStats:
    """Агрегаты одной группы ваучеров (весь месяц / один буксир / буксир+слот / вариант)."""

    n_vouchers: int = 0
    n_auto_accept: int = 0
    n_align_failed: int = 0
    fields: dict[str, FieldStats] = field(
        default_factory=lambda: {f: FieldStats() for f in OCR_FIELDS}
    )

    @property
    def auto_accept_rate(self) -> float | None:
        return self.n_auto_accept / self.n_vouchers if self.n_vouchers else None

    @property
    def align_failed_rate(self) -> float | None:
        return self.n_align_failed / self.n_vouchers if self.n_vouchers else None

    @property
    def correction_rate(self) -> float | None:
        """Доля правок по всем OCR-полям группы вместе (не по одному полю)."""
        attempted = sum(f.attempted for f in self.fields.values())
        corrected = sum(f.corrected for f in self.fields.values())
        return corrected / attempted if attempted else None


@dataclass
class MonitorReport:
    """Результат мониторинга одного месяца — только агрегаты и `id` ваучеров."""

    month: str
    n_confirmed: int = 0
    overall: GroupStats = field(default_factory=GroupStats)
    by_tug: dict[str, GroupStats] = field(default_factory=dict)
    by_tug_slot: dict[tuple[str, int], GroupStats] = field(default_factory=dict)
    by_variant: dict[str, GroupStats] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    skipped_no_rows: list[str] = field(default_factory=list)  # нет left_base_dt — нет слота


def _predictions_by_field(db: Session, voucher_id: int) -> dict[str, VoucherFieldPrediction]:
    rows = db.scalars(
        select(VoucherFieldPrediction).where(VoucherFieldPrediction.voucher_id == voucher_id)
    ).all()
    return {r.field_name: r for r in rows}


def _detect_variant(
    voucher: Voucher, variants: dict[str, VariantLayout]
) -> str:
    """Лучший вариант бланка по выравниванию — независимо от `Voucher.ocr_status`.

    Решение без человека: вариант бланка нигде не сохраняется в БД (T26 это обсуждал —
    `app.ocr.pipeline.VoucherRecognition.variant` живёт только в памяти одного вызова), а
    для разреза «по вариантам бланка» он нужен отдельно от авторитетного `ocr_status`
    (тот берём из БД как есть, не пересчитываем). Пересчёт — только выравнивание (cv2/numpy,
    без моделей цифр/номера), поэтому дешёвый и не требует весов. Любая ошибка чтения файла
    или отсутствие эталонов — `"unknown"`, не падение всего отчёта.
    """
    if not variants or not voucher.file_path:
        return UNKNOWN_VARIANT
    path = Path(voucher.file_path)
    if not path.is_file():
        return UNKNOWN_VARIANT
    try:
        cv2.setRNGSeed(CV_SEED)
        page = normalize_width(load_scan(path))
        match = detect_variant(page, variants, prefer=_tug_code(voucher))
    except (ScanLoadError, ValueError, OSError):
        return UNKNOWN_VARIANT
    return match.variant


def _update_group(
    stats: GroupStats, voucher: Voucher, predictions: dict[str, VoucherFieldPrediction]
) -> None:
    stats.n_vouchers += 1
    if voucher_auto_accept(voucher):
        stats.n_auto_accept += 1
    if voucher.ocr_status == STATUS_ALIGN_FAILED:
        stats.n_align_failed += 1
    for name in OCR_FIELDS:
        pred = predictions.get(name)
        if pred is None or not pred.predicted_value:
            continue
        fs = stats.fields[name]
        fs.attempted += 1
        if pred.confirmed_value != pred.predicted_value:
            fs.corrected += 1


def build(db: Session, month: str, *, layouts_dir: Path = LAYOUTS_DIR) -> MonitorReport:
    """Собрать отчёт мониторинга за месяц `"YYYY-MM"`."""
    start, end = month_range(month)
    report = MonitorReport(month=month)
    variants = load_variant_layouts(layouts_dir) if layouts_dir.is_dir() else {}

    vouchers = db.scalars(
        select(Voucher).where(
            Voucher.status.in_(REVIEWED_STATUSES),
            Voucher.reviewed_at >= start,
            Voucher.reviewed_at < end,
        )
    ).all()
    report.n_confirmed = len(vouchers)

    for voucher in vouchers:
        predictions = _predictions_by_field(db, voucher.id)
        _update_group(report.overall, voucher, predictions)

        tug_code = _tug_code(voucher) or "unknown"
        _update_group(report.by_tug.setdefault(tug_code, GroupStats()), voucher, predictions)

        if voucher.left_base_dt is not None:
            slot = shift_slot(voucher.left_base_dt)
            key = (tug_code, slot)
            _update_group(report.by_tug_slot.setdefault(key, GroupStats()), voucher, predictions)
        else:
            report.skipped_no_rows.append(str(voucher.id))

        variant = _detect_variant(voucher, variants)
        _update_group(report.by_variant.setdefault(variant, GroupStats()), voucher, predictions)

    report.warnings = _warnings(report)
    return report


def _warnings(report: MonitorReport) -> list[str]:
    warnings: list[str] = []
    overall_rate = report.overall.correction_rate or 0.0
    for (tug_code, slot), stats in sorted(report.by_tug_slot.items()):
        attempted = sum(f.attempted for f in stats.fields.values())
        rate = stats.correction_rate
        spike = rate is not None and rate - overall_rate >= SLOT_SPIKE_MARGIN
        if attempted >= MIN_GROUP_N and spike:
            warnings.append(
                f"буксир {tug_code}, слот {slot}: доля правок {rate:.0%} "
                f"при среднем {overall_rate:.0%} по месяцу (n={attempted}) — "
                "возможно, в вахте новый человек"
            )
    for variant, stats in sorted(report.by_variant.items()):
        rate = stats.align_failed_rate
        exceeded = rate is not None and rate > ALIGN_FAILURE_RATE_THRESHOLD
        if stats.n_vouchers >= MIN_GROUP_N and exceeded:
            warnings.append(
                f"вариант бланка {variant}: доля отказов выравнивания {rate:.0%} "
                f"выше порога {ALIGN_FAILURE_RATE_THRESHOLD:.0%} (n={stats.n_vouchers}) — "
                "возможно, новый бланк"
            )
    return warnings


def _group_table(groups: dict[Any, GroupStats], *, header: str, key_label: str) -> list[str]:
    lines = [header, "", f"| {key_label} | n | автоприём | правки |", "|---|---:|---:|---:|"]
    for key, stats in sorted(groups.items(), key=lambda kv: str(kv[0])):
        acc = f"{stats.auto_accept_rate:.0%}" if stats.auto_accept_rate is not None else "—"
        corr = f"{stats.correction_rate:.0%}" if stats.correction_rate is not None else "—"
        lines.append(f"| {key} | {stats.n_vouchers} | {acc} | {corr} |")
    lines.append("")
    return lines


def render_report(report: MonitorReport) -> str:
    lines = [
        f"# Мониторинг качества OCR — {report.month}",
        "",
        f"- подтверждённых ваучеров за месяц: {report.n_confirmed}",
        f"- доля автоприёма: "
        f"{report.overall.auto_accept_rate:.1%}" if report.overall.auto_accept_rate is not None
        else "- доля автоприёма: —",
        "",
        "## Правки по полям",
        "",
        "| поле | попыток | правок | доля |",
        "|---|---:|---:|---:|",
    ]
    for name, fs in report.overall.fields.items():
        rate = f"{fs.rate:.1%}" if fs.rate is not None else "—"
        lines.append(f"| {name} | {fs.attempted} | {fs.corrected} | {rate} |")
    lines.append("")
    lines += _group_table(report.by_tug, header="## По буксирам", key_label="буксир")
    lines += _group_table(
        {f"{t}/{s}": v for (t, s), v in report.by_tug_slot.items()},
        header="## По слотам вахт (буксир/слот)", key_label="буксир/слот",
    )
    lines += _group_table(report.by_variant, header="## По вариантам бланка", key_label="вариант")
    lines.append("## Предупреждения")
    lines.append("")
    if report.warnings:
        lines += [f"- {w}" for w in report.warnings]
    else:
        lines.append("нет")
    lines.append("")
    if report.skipped_no_rows:
        lines.append(
            f"Без `left_base_dt` (пропущены в разрезе по слотам): {len(report.skipped_no_rows)}"
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Ежемесячный мониторинг качества OCR (доля автоприёма и правок)"
    )
    parser.add_argument("--month", required=True, help="ГГГГ-ММ, например 2026-09")
    parser.add_argument("--layouts-dir", type=Path, default=LAYOUTS_DIR)
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)

    with SessionLocal() as db:
        report = build(db, args.month, layouts_dir=args.layouts_dir)

    out = args.out or (REPORTS_DIR / f"monitor_{args.month}.md")
    ensure_dir(out.parent)
    out.write_text(render_report(report), encoding="utf-8")
    print(f"мониторинг {args.month}: {report.n_confirmed} ваучеров, "
         f"предупреждений {len(report.warnings)} -> {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
