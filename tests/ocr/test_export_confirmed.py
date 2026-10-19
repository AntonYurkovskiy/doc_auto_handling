"""Тесты выгрузки подтверждённых ваучеров (T28, `ocr_lab.export_confirmed`) на синтетике.

Реальная БД/сканы не используются — только временная SQLite-сессия (как в
`tests/test_voucher_fields.py`) и пустые временные файлы вместо сканов (выгрузке нужен
только `Path.is_file()` и байты для sha256, не содержимое)."""

from __future__ import annotations

from pathlib import Path

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import DocStatus, Tug, Voucher, VoucherFieldPrediction
from ocr_lab.export_confirmed import build

ROWS = ("left_base", "arrived_base", "started_work", "finished_work")


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _scan_file(tmp_path: Path, name: str) -> str:
    path = tmp_path / name
    path.write_bytes(b"%PDF-fake-scan")
    return str(path)


def _confirmed_voucher(
    tmp_path: Path, *, tug: Tug, number: str = "243",
    times: dict[str, str] | None = None, scan_name: str = "243k.pdf",
) -> Voucher:
    times = times or {
        "left_base": "2026-07-05 08:00",
        "started_work": "2026-07-05 08:10",
        "finished_work": "2026-07-05 08:50",
        "arrived_base": "2026-07-05 09:00",
    }
    voucher = Voucher(
        status=DocStatus.confirmed,
        tug=tug,
        work_type="Швартовка",
        file_path=_scan_file(tmp_path, scan_name),
        original_filename=scan_name,
    )
    voucher.predictions = [
        VoucherFieldPrediction(field_name="voucher_number", confirmed_value=number),
        *(
            VoucherFieldPrediction(field_name=row, confirmed_value=times[row])
            for row in ROWS
            if row in times
        ),
    ]
    return voucher


def test_build_exports_fully_confirmed_voucher(tmp_path):
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()
    voucher = _confirmed_voucher(tmp_path, tug=tug)
    db.add(voucher)
    db.commit()

    df, report = build(db)

    assert report.total_confirmed == 1
    assert report.n_exported == 1
    assert len(df) == 1
    row = df.iloc[0]
    assert row["source"] == "app"
    assert row["tug_code"] == "k"
    assert row["year"] == 2026
    assert row["voucher_number"] == 243
    assert row["scan_id"] == "2026_243k"
    assert row["chain_ok"]
    assert row["left_base_day"] == 5
    assert row["left_base_month"] == 7
    assert row["left_base_hour"] == 8
    assert row["left_base_minute"] == 0
    assert row["sha256"]  # файл реально прочитан


def test_build_skips_voucher_without_file(tmp_path):
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()
    voucher = _confirmed_voucher(tmp_path, tug=tug)
    voucher.file_path = None
    db.add(voucher)
    db.commit()

    df, report = build(db)

    assert report.n_exported == 0
    assert report.skipped_no_file == [str(voucher.id)]


def test_build_skips_unknown_tug(tmp_path):
    db = _session()
    tug = Tug(name="МБ Лигер", code=None)  # третий буксир — вне набора OCR
    db.add(tug)
    db.commit()
    voucher = _confirmed_voucher(tmp_path, tug=tug, scan_name="1l.pdf")
    db.add(voucher)
    db.commit()

    df, report = build(db)

    assert report.n_exported == 0
    assert report.skipped_unknown_tug == [str(voucher.id)]


@pytest.mark.parametrize(
    "missing",
    ["left_base", "arrived_base", "started_work", "finished_work", "voucher_number"],
)
def test_build_skips_partially_confirmed_voucher(tmp_path, missing):
    db = _session()
    tug = Tug(name="БК Пионер", code="p")
    db.add(tug)
    db.commit()
    voucher = _confirmed_voucher(tmp_path, tug=tug, scan_name="5p.pdf")
    predictions = [p for p in voucher.predictions if p.field_name != missing]
    voucher.predictions = predictions
    db.add(voucher)
    db.commit()

    df, report = build(db)

    assert report.n_exported == 0
    assert report.skipped_incomplete == [str(voucher.id)]


@pytest.mark.parametrize("status", [DocStatus.matched, DocStatus.calculated, DocStatus.done])
def test_build_exports_vouchers_after_matching_with_application_d2(tmp_path, status):
    """D2: после `/match` статус `matched`, выгрузка для дообучения не должна их терять."""
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()
    voucher = _confirmed_voucher(tmp_path, tug=tug)
    voucher.status = status
    db.add(voucher)
    db.commit()

    df, report = build(db)

    assert report.total_confirmed == 1
    assert report.n_exported == 1


def test_build_ignores_non_confirmed_vouchers(tmp_path):
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()
    voucher = _confirmed_voucher(tmp_path, tug=tug)
    voucher.status = DocStatus.needs_review
    db.add(voucher)
    db.commit()

    df, report = build(db)

    assert report.total_confirmed == 0
    assert report.n_exported == 0


def test_render_report_has_only_aggregates():
    from ocr_lab.export_confirmed import ExportReport, render_report

    report = ExportReport(total_confirmed=5, n_exported=3)
    report.skipped_no_file = ["7"]
    report.skipped_unknown_tug = ["8"]
    text = render_report(report)

    assert "5" in text
    assert "3" in text
    # Только счётчики и заголовок — ни судов, ни агентов.
    assert "БК" not in text
