"""Тесты модели ваучера и эталонного шаблона регионов."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import DocStatus, Voucher, VoucherFieldPrediction
from app.services.voucher_template import DEFAULT_VOUCHER_REGIONS, ensure_default_template


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def test_default_template_contains_all_labeled_regions():
    db = _session()

    template = ensure_default_template(db)

    assert template.name == "baltiyskie_buksiry"
    assert len(template.regions) == len(DEFAULT_VOUCHER_REGIONS) == 12
    assert {region.name for region in template.regions} == {
        definition.name for definition in DEFAULT_VOUCHER_REGIONS
    }
    assert all(0 <= value <= 1 for region in template.regions for value in (
        region.center_x,
        region.center_y,
        region.width,
        region.height,
    ))


def test_voucher_stores_source_links_and_field_prediction():
    db = _session()
    template = ensure_default_template(db)
    region = template.regions[0]
    voucher = Voucher(
        original_filename="243k.pdf",
        file_path="data/incoming/vouchers/243k.pdf",
        content_type="application/pdf",
        sha256="a" * 64,
        template=template,
        status=DocStatus.needs_review,
    )
    db.add(voucher)
    db.flush()
    prediction = VoucherFieldPrediction(
        voucher=voucher,
        region=region,
        field_name="tugboat",
        predicted_value="БК Коммунар",
        predicted_normalized_value="коммунар",
        confidence=0.98,
        source="classification",
        confirmed_value="БК Коммунар",
        confirmed_at=datetime(2026, 7, 20, 10, 0),
    )
    db.add(prediction)
    db.commit()

    loaded = db.get(Voucher, voucher.id)
    assert loaded is not None
    assert loaded.original_filename == "243k.pdf"
    assert loaded.predictions[0].predicted_normalized_value == "коммунар"
    assert loaded.predictions[0].confirmed_value == "БК Коммунар"
