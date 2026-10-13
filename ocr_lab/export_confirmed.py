"""T28: выгрузка подтверждённых оператором ваучеров в формате манифеста (`source="app"`).

Каждый подтверждённый в приложении ваучер — потенциальный новый обучающий пример (план,
«Этап 6»): он прошёл ручную проверку оператором так же, как замороженная история прошла
ручную проверку истины в T02/T04/T21. Этот модуль только **выгружает** такие ваучеры в тот
же табличный формат, что и `ocr_lab.manifest.build()` (плюс колонки `source`/`sha256`) —
дедуп с историей и месячный holdout считает `ocr_lab.manifest.combine()` (вызывается из
`ocr_lab.retrain`).

Запуск: `.venv/Scripts/python -m ocr_lab.export_confirmed --out data/ocr/manifest_app.csv`.

В консоль и отчёт идут только агрегаты и `scan_id`/`id` ваучера (правило `_common.md`, п. 3).
"""

from __future__ import annotations

import argparse
import hashlib
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import agent_group
from app.database import SessionLocal
from app.models import DocStatus, Voucher, VoucherFieldPrediction
from app.services.calculation import normalize_work_type
from app.services.voucher_ocr import application_reference_datetime
from ocr_lab.manifest import (
    PARTS,
    ROWS,
    TUG_CODE_BY_NAME,
    _column_order,
    _row_chain_ok,
    extract_voucher_number,
    make_scan_id,
)
from ocr_lab.paths import REPORTS_DIR, WORK_DIR, ensure_dir

SOURCE_APP = "app"
#: Формат `VoucherFieldPrediction.confirmed_value` для дат — совпадает с
#: `app.services.voucher.DATETIME_FORMAT` (продублирован, как в `voucher_fields.py`/
#: `voucher_ocr.py`, чтобы не замыкать модули друг на друга).
CONFIRMED_DATETIME_FORMAT = "%Y-%m-%d %H:%M"
EXPORT_CSV = WORK_DIR / "manifest_app.csv"
EXPORT_REPORT_MD = REPORTS_DIR / "export_confirmed_report.md"


@dataclass
class ExportReport:
    """Агрегаты выгрузки — без имён судов/агентов, только счётчики и `id` ваучера."""

    total_confirmed: int = 0
    skipped_no_file: list[str] = field(default_factory=list)
    skipped_unknown_tug: list[str] = field(default_factory=list)
    skipped_incomplete: list[str] = field(default_factory=list)
    n_exported: int = 0


def _confirmed_values(db: Session, voucher_id: int) -> dict[str, str]:
    """`field_name -> confirmed_value` непустых подтверждённых полей ваучера."""
    rows = db.scalars(
        select(VoucherFieldPrediction).where(VoucherFieldPrediction.voucher_id == voucher_id)
    ).all()
    return {r.field_name: r.confirmed_value for r in rows if r.confirmed_value}


def _parse_confirmed_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    try:
        return datetime.strptime(value, CONFIRMED_DATETIME_FORMAT)
    except ValueError:
        return None


def file_sha256(path: Path) -> str | None:
    """sha256 файла по пути; `None`, если файл не читается."""
    try:
        digest = hashlib.sha256()
        with Path(path).open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                digest.update(chunk)
        return digest.hexdigest()
    except OSError:
        return None


def _tug_code(voucher: Voucher) -> str | None:
    tug = voucher.tug
    if tug is None:
        return None
    return tug.code or TUG_CODE_BY_NAME.get(tug.name)


def export_confirmed_records(db: Session) -> tuple[list[dict[str, Any]], ExportReport]:
    """Подтверждённые ваучеры БД -> строки в формате манифеста (`source="app"`).

    Ваучер пропускается (и попадает в отчёт), если: у него нет файла скана на диске;
    буксир не входит в `ocr_lab.manifest.TUG_CODE_BY_NAME` (третий буксир или не указан);
    не подтверждены все 4 строки дат или номер ваучера (частичное подтверждение не даёт
    полноценный обучающий пример). Выгружаются только `VoucherFieldPrediction.
    confirmed_value` — то, что реально подтвердил оператор, а не последнее предсказание.
    """
    report = ExportReport()
    records: list[dict[str, Any]] = []
    vouchers = db.scalars(select(Voucher).where(Voucher.status == DocStatus.confirmed)).all()
    report.total_confirmed = len(vouchers)

    for voucher in vouchers:
        voucher_tag = str(voucher.id)
        if not voucher.file_path or not Path(voucher.file_path).is_file():
            report.skipped_no_file.append(voucher_tag)
            continue
        tug_code = _tug_code(voucher)
        if tug_code is None:
            report.skipped_unknown_tug.append(voucher_tag)
            continue

        confirmed = _confirmed_values(db, voucher.id)
        dts: dict[str, datetime | None] = {r: _parse_confirmed_dt(confirmed.get(r)) for r in ROWS}
        if any(dt is None for dt in dts.values()):
            report.skipped_incomplete.append(voucher_tag)
            continue
        number = extract_voucher_number(confirmed.get("voucher_number") or voucher.number or "")
        if number is None:
            report.skipped_incomplete.append(voucher_tag)
            continue

        left_dt = dts["left_base"]
        assert left_dt is not None  # проверено выше (any(... is None) -> continue)
        year = left_dt.year
        voucher_file = voucher.original_filename or Path(voucher.file_path).name
        scan_id = make_scan_id(voucher_file, year)
        scan_path = str(Path(voucher.file_path).resolve())
        sha256 = voucher.sha256 or file_sha256(Path(voucher.file_path))

        work_type_raw = voucher.work_type
        work_type = normalize_work_type(work_type_raw)
        application = voucher.application
        # Синтетический ключ пары (не настоящий путь файла заявки истории) — пары T28
        # внутри `source="app"` находятся по той же заявке, но не путаются с ключами
        # `application_file` истории (реальные пути писем T02).
        application_file = f"app:{voucher.application_id}" if voucher.application_id else ""
        app_dt = application_reference_datetime(application)

        chain_ok, chain_violation = _row_chain_ok(dts)
        dates_present = {dt.date() for dt in dts.values() if dt is not None}
        crosses_midnight = len(dates_present) > 1
        minutes = [dt.minute for dt in dts.values() if dt is not None]
        minutes_mult10 = all(m % 10 == 0 for m in minutes)

        rec: dict[str, Any] = {
            "scan_id": scan_id,
            "tug_code": tug_code,
            "year": year,
            "voucher_number": number,
            "voucher_file": voucher_file,
            "scan_path": scan_path,
            "scan_exists": True,
            "ext": Path(scan_path).suffix.lstrip(".").lower(),
            "application_file": application_file,
            "work_type_raw": work_type_raw or "",
            "work_type": work_type,
            "agent_group": agent_group(voucher.agent),
            "app_dt": app_dt.isoformat() if app_dt else "",
            "truth_corrected": "",
            "dedup_conflict": False,
            "chain_ok": chain_ok,
            "chain_violation": chain_violation,
            "crosses_midnight": crosses_midnight,
            "minutes_mult10": minutes_mult10,
            # Подтверждённое в приложении время — уже обычный календарный datetime
            # (Python datetime не умеет час 24): переход через полночь, написанный на
            # бланке как «24:00», в этих данных неотличим от «00:00 новых суток» — в
            # отличие от истории, где исходная запись бланка сохранена отдельно (T02).
            # `has_hour24` для строк `source=app` поэтому всегда False (решение без
            # человека, см. журнал T28).
            "has_hour24": False,
            "pair_id": "",
            "split": "",  # заполняет ocr_lab.manifest.combine (месячный holdout)
            "source": SOURCE_APP,
            "sha256": sha256,
        }
        for r in ROWS:
            dt = dts[r]
            assert dt is not None
            rec[f"{r}_dt"] = dt.isoformat()
            values = {"day": dt.day, "month": dt.month, "hour": dt.hour, "minute": dt.minute}
            for p in PARTS:
                rec[f"{r}_{p}"] = values[p]
            rec[f"{r}_year"] = dt.year
        records.append(rec)

    report.n_exported = len(records)
    return records, report


def build(db: Session) -> tuple[pd.DataFrame, ExportReport]:
    """Выгрузка как `DataFrame` в формате манифеста + колонки `source`/`sha256`."""
    records, report = export_confirmed_records(db)
    columns = [*_column_order(), "source", "sha256"]
    return pd.DataFrame(records, columns=columns), report


def render_report(report: ExportReport) -> str:
    lines = [
        "# Выгрузка подтверждённых ваучеров (T28)",
        "",
        f"- подтверждённых ваучеров в БД: {report.total_confirmed}",
        f"- выгружено строк (`source=app`): {report.n_exported}",
        f"- пропущено, нет файла скана: {len(report.skipped_no_file)}",
        f"- пропущено, буксир не из набора OCR: {len(report.skipped_unknown_tug)}",
        f"- пропущено, не все поля подтверждены (4 строки дат + номер): "
        f"{len(report.skipped_incomplete)}",
    ]
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Выгрузка подтверждённых ваучеров БД в формате манифеста (source=app)"
    )
    parser.add_argument("--out", type=Path, default=EXPORT_CSV)
    parser.add_argument("--report", type=Path, default=EXPORT_REPORT_MD)
    args = parser.parse_args(argv)

    with SessionLocal() as db:
        df, report = build(db)

    ensure_dir(args.out.parent)
    df.to_csv(args.out, index=False, encoding="utf-8-sig")
    ensure_dir(args.report.parent)
    args.report.write_text(render_report(report), encoding="utf-8")
    print(f"выгрузка: {report.n_exported} строк -> {args.out}")
    print(f"отчёт -> {args.report}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
