"""Тесты `ocr_lab.monitor` (T28) на синтетике: формула слота, границы месяца и агрегаты.

Детекция варианта бланка (`_detect_variant`) не тестируется здесь на реальных сканах
(правило `_common.md`, п. 3: только синтетика) — проверяется только деградация на
`"unknown"` при отсутствии файла/эталонов, остальное покрыто `tests/ocr/test_variants.py`.
"""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import DocStatus, Tug, Voucher, VoucherFieldPrediction
from app.ocr.pipeline import STATUS_ALIGN_FAILED, STATUS_OK
from ocr_lab.monitor import (
    OCR_FIELDS,
    build,
    month_range,
    shift_slot,
)


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_month_range_handles_month_boundaries():
    start, end = month_range("2026-02")
    assert start.isoformat() == "2026-02-01"
    assert end.isoformat() == "2026-03-01"


def test_shift_slot_rotates_every_four_days():
    # В 09:00 начинается новая вахта — время до 09:00 относится к предыдущим суткам.
    before = shift_slot(datetime(2026, 7, 5, 8, 59))
    after = shift_slot(datetime(2026, 7, 5, 9, 0))
    assert before != after
    # Ротация «сутки через трое»: слот повторяется через 4 суток от смены в 09:00.
    assert shift_slot(datetime(2026, 7, 9, 10, 0)) == shift_slot(datetime(2026, 7, 5, 10, 0))
    assert len({shift_slot(datetime(2026, 7, d, 10, 0)) for d in range(5, 9)}) == 4


def _voucher(
    *, tug: Tug, reviewed_at: datetime, left_base_dt: datetime,
    ocr_status: str = STATUS_OK, ocr_confidence: float = 0.9999,
    predicted_number: str = "243", confirmed_number: str = "243",
) -> Voucher:
    voucher = Voucher(
        status=DocStatus.confirmed,
        tug=tug,
        reviewed_at=reviewed_at,
        left_base_dt=left_base_dt,
        ocr_status=ocr_status,
        ocr_confidence=ocr_confidence,
    )
    voucher.predictions = [
        VoucherFieldPrediction(
            field_name="voucher_number",
            predicted_value=predicted_number,
            confirmed_value=confirmed_number,
        ),
    ]
    return voucher


def test_build_counts_vouchers_after_matching_with_application_d2():
    """D2: после `/match` статус `matched`, ваучер должен оставаться в мониторинге."""
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()
    for status in (DocStatus.confirmed, DocStatus.matched, DocStatus.calculated, DocStatus.done):
        voucher = _voucher(
            tug=tug, reviewed_at=datetime(2026, 9, 15, 12, 0),
            left_base_dt=datetime(2026, 9, 15, 8, 0),
        )
        voucher.status = status
        db.add(voucher)
    unreviewed = _voucher(
        tug=tug, reviewed_at=datetime(2026, 9, 15, 12, 0), left_base_dt=datetime(2026, 9, 15, 8, 0),
    )
    unreviewed.status = DocStatus.needs_review
    db.add(unreviewed)
    db.commit()

    report = build(db, "2026-09", layouts_dir=Path("нет такого каталога"))

    assert report.n_confirmed == 4


def test_build_counts_confirmed_vouchers_within_month(tmp_path):
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()

    in_month = _voucher(
        tug=tug, reviewed_at=datetime(2026, 9, 15, 12, 0), left_base_dt=datetime(2026, 9, 15, 8, 0),
    )
    out_of_month = _voucher(
        tug=tug, reviewed_at=datetime(2026, 8, 15, 12, 0), left_base_dt=datetime(2026, 8, 15, 8, 0),
    )
    db.add_all([in_month, out_of_month])
    db.commit()

    report = build(db, "2026-09", layouts_dir=tmp_path / "no-such-layouts")

    assert report.n_confirmed == 1
    assert report.overall.n_vouchers == 1
    assert report.by_tug["k"].n_vouchers == 1


def test_build_counts_field_corrections(tmp_path):
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()
    corrected = _voucher(
        tug=tug, reviewed_at=datetime(2026, 9, 1, 9, 0), left_base_dt=datetime(2026, 9, 1, 8, 0),
        predicted_number="243", confirmed_number="244",
    )
    db.add(corrected)
    db.commit()

    report = build(db, "2026-09", layouts_dir=tmp_path / "no-such-layouts")

    number_stats = report.overall.fields["voucher_number"]
    assert number_stats.attempted == 1
    assert number_stats.corrected == 1
    assert number_stats.rate == 1.0


def test_build_tracks_align_failures_without_crashing_on_missing_file(tmp_path):
    db = _session()
    tug = Tug(name="БК Пионер", code="p")
    db.add(tug)
    db.commit()
    voucher = _voucher(
        tug=tug, reviewed_at=datetime(2026, 9, 2, 9, 0), left_base_dt=datetime(2026, 9, 2, 8, 0),
        ocr_status=STATUS_ALIGN_FAILED, ocr_confidence=None,
    )
    voucher.file_path = None
    db.add(voucher)
    db.commit()

    report = build(db, "2026-09", layouts_dir=tmp_path / "no-such-layouts")

    assert report.by_tug["p"].n_align_failed == 1
    # нет эталонов/файла -> "unknown", отчёт не падает
    assert "unknown" in report.by_variant


def test_ocr_fields_match_voucher_number_and_four_rows():
    assert set(OCR_FIELDS) == {
        "voucher_number", "left_base", "arrived_base", "started_work", "finished_work",
    }
