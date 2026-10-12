"""Тесты автоприёма высокой уверенности (T26): `voucher_auto_accept`/`apply_auto_accept`."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import settings
from app.database import Base
from app.models import Application, Direction, DocStatus, Tug, Voucher, VoucherFieldPrediction
from app.ocr.pipeline import STATUS_ALIGN_FAILED, STATUS_OK
from app.services.voucher_fields import apply_auto_accept, voucher_auto_accept
from app.services.voucher_template import ensure_default_template


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


@pytest.fixture(autouse=True)
def _reset_auto_confirm_setting():
    # Тесты не должны зависеть друг от друга значением флага автоприёма.
    original = settings.ocr_auto_confirm
    yield
    settings.ocr_auto_confirm = original


def test_voucher_auto_accept_requires_ok_status_and_threshold(monkeypatch):
    monkeypatch.setattr(settings, "ocr_auto_accept_threshold", 0.9)

    not_ok = Voucher(ocr_status=STATUS_ALIGN_FAILED, ocr_confidence=0.99)
    assert voucher_auto_accept(Voucher()) is False
    assert voucher_auto_accept(not_ok) is False
    assert voucher_auto_accept(Voucher(ocr_status=STATUS_OK, ocr_confidence=0.5)) is False
    assert voucher_auto_accept(Voucher(ocr_status=STATUS_OK, ocr_confidence=0.95)) is True


def test_apply_auto_accept_noop_below_threshold(monkeypatch):
    monkeypatch.setattr(settings, "ocr_auto_accept_threshold", 0.9)
    db = _session()
    template = ensure_default_template(db)
    voucher = Voucher(template=template, ocr_status=STATUS_OK, ocr_confidence=0.5)
    db.add(voucher)
    db.commit()

    apply_auto_accept(db, voucher)

    assert voucher.number is None
    assert voucher.status is DocStatus.new


def test_apply_auto_accept_prefills_without_changing_status_by_default(monkeypatch):
    """DECISIONS Q3: APP_OCR_AUTO_CONFIRM=false — только предзаполнение."""
    monkeypatch.setattr(settings, "ocr_auto_accept_threshold", 0.9)
    settings.ocr_auto_confirm = False
    db = _session()
    template = ensure_default_template(db)
    voucher = Voucher(
        template=template,
        status=DocStatus.needs_review,
        ocr_status=STATUS_OK,
        ocr_confidence=0.999,
    )
    db.add(voucher)
    db.flush()
    voucher.predictions = [
        VoucherFieldPrediction(field_name="voucher_number", predicted_value="243"),
    ]
    db.commit()

    apply_auto_accept(db, voucher)

    assert voucher.number == "243"  # предзаполнено
    assert voucher.status is DocStatus.needs_review  # статус не изменился
    assert voucher.reviewed_at is None


def test_apply_auto_accept_confirms_when_flag_enabled(monkeypatch):
    monkeypatch.setattr(settings, "ocr_auto_accept_threshold", 0.9)
    settings.ocr_auto_confirm = True
    db = _session()
    template = ensure_default_template(db)
    tug = Tug(name="БК Коммунар", code="k")
    application = Application(
        direction=Direction.entry,
        vessel_name="ARIES",
        agent="Транс-Агро",
        entry_datetime=datetime(2026, 7, 20, 9, 0),
    )
    db.add_all([tug, application])
    db.commit()

    voucher = Voucher(
        template=template,
        status=DocStatus.needs_review,
        application=application,
        tug=tug,
        vessel_name="ARIES",
        ocr_status=STATUS_OK,
        ocr_confidence=0.999,
    )
    db.add(voucher)
    db.flush()
    voucher.predictions = [
        VoucherFieldPrediction(field_name="voucher_number", predicted_value="243"),
    ]
    db.commit()

    apply_auto_accept(db, voucher)

    assert voucher.number == "243"
    assert voucher.status is DocStatus.confirmed
    assert voucher.reviewed_at is not None
    assert voucher.application_id == application.id
    confirmed = {p.field_name: p.confirmed_value for p in voucher.predictions}
    assert confirmed["voucher_number"] == "243"
