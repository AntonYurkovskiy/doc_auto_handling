"""Тесты OCR по регионам ваучера."""

from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Application, Direction, Tug, Voucher, VoucherFieldPrediction
from app.services.voucher import (
    PREDICTION_SOURCE_APPLICATION,
    PREDICTION_SOURCE_FILENAME,
    PREDICTION_SOURCE_OCR,
    PREDICTION_SOURCE_PRIOR,
    VoucherHistory,
    predict_and_store,
    predict_fields,
)
from app.services.voucher_fields import apply_predictions_to_voucher
from app.services.voucher_ocr import (
    HANDWRITTEN_FIELDS,
    crop_region,
    load_voucher_image,
    ocr_image,
    ocr_voucher_regions,
)
from app.services.voucher_template import ensure_default_template
from app.services.voucher_trocr import trocr_image


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _make_voucher_with_template(db) -> Voucher:
    template = ensure_default_template(db)
    voucher = Voucher(template=template, original_filename="scan.pdf")
    db.add(voucher)
    db.commit()
    return voucher


def test_crop_region_uses_normalized_coordinates():
    image = Image.new("RGB", (1000, 2000), color="white")
    region = MagicMock()
    region.name = "vessel"
    region.center_x = 0.5
    region.center_y = 0.5
    region.width = 0.2
    region.height = 0.1

    cropped = crop_region(image, region)

    assert cropped.size == (200, 200)


def test_crop_region_clamps_to_image_bounds():
    image = Image.new("RGB", (100, 100))
    region = MagicMock()
    region.name = "test"
    region.center_x = 1.0
    region.center_y = 1.0
    region.width = 0.5
    region.height = 0.5

    cropped = crop_region(image, region)

    assert cropped.size[0] <= 100
    assert cropped.size[1] <= 100


def test_load_voucher_image_returns_none_when_no_file():
    voucher = Voucher(file_path=None)
    assert load_voucher_image(voucher) is None


def test_ocr_image_returns_text_and_confidence(monkeypatch):
    fake_image = Image.new("RGB", (100, 50))
    fake_data = {
        "text": ["Hello", "", "World"],
        "conf": [80, -1, 90],
    }

    def fake_image_to_data(image, lang, output_type):
        return fake_data

    def fake_image_to_string(image, lang):
        return ""

    with patch("app.services.voucher_ocr.pytesseract") as mock_tesseract:
        mock_tesseract.image_to_data = fake_image_to_data
        mock_tesseract.image_to_string = fake_image_to_string
        mock_tesseract.Output.DICT = "dict"
        text, conf = ocr_image(fake_image)

    assert text == "Hello World"
    assert conf == pytest.approx(0.85)


def test_ocr_image_falls_back_to_image_to_string(monkeypatch):
    fake_image = Image.new("RGB", (100, 50))
    fake_data = {"text": ["", ""], "conf": [-1, -1]}

    with patch("app.services.voucher_ocr.pytesseract") as mock_tesseract:
        mock_tesseract.image_to_data.return_value = fake_data
        mock_tesseract.image_to_string.return_value = "Fallback"
        mock_tesseract.Output.DICT = "dict"
        text, conf = ocr_image(fake_image)

    assert text == "Fallback"
    assert conf is not None


def test_ocr_voucher_regions_returns_empty_without_template():
    voucher = Voucher(template=None)
    assert ocr_voucher_regions(voucher) == {}


def test_ocr_voucher_regions_runs_ocr_for_each_region(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db)
    assert voucher.template is not None

    fake_image = Image.new("RGB", (1000, 1414))

    def fake_load(_voucher):
        return fake_image

    def fake_ocr(image, lang="rus+eng"):
        return ("test-text", 0.75)

    monkeypatch.setattr("app.services.voucher_ocr.load_voucher_image", fake_load)
    monkeypatch.setattr("app.services.voucher_ocr.ocr_image", fake_ocr)

    result = ocr_voucher_regions(voucher)

    assert len(result) == len(voucher.template.regions)
    for region in voucher.template.regions:
        assert region.name in result
        assert result[region.name] == ("test-text", 0.75)


def test_predict_fields_prefers_ocr_over_prior():
    voucher = Voucher()
    application = Application(
        direction=Direction.entry,
        vessel_name="MERIDIAN",
        agent="Транс-Агро",
        entry_datetime=datetime(2026, 7, 20, 9, 30),
    )
    history = VoucherHistory()
    ocr_values = {
        "agent": ("  Другой Агент  ", 0.9),
    }

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, application, history, ocr_values)
    }

    assert predictions["agent"].predicted_value == "Другой Агент"
    assert predictions["agent"].predicted_normalized_value == "другой агент"
    assert predictions["agent"].source == PREDICTION_SOURCE_OCR
    assert predictions["agent"].confidence == pytest.approx(0.9)
    assert predictions["tugboat"].source == PREDICTION_SOURCE_PRIOR


def test_predict_fields_filename_beats_ocr_for_number_and_tug():
    voucher = Voucher(original_filename="323p.pdf")
    ocr_values = {
        "voucher_number": ("9999", 0.9),
        "tugboat": ("распознанный мусор", 0.9),
        "vessel": ("GARBAGE", 0.9),
    }
    application = Application(vessel_name="MERIDIAN")

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, application, VoucherHistory(), ocr_values)
    }

    assert predictions["voucher_number"].predicted_value == "323"
    assert predictions["voucher_number"].source == PREDICTION_SOURCE_FILENAME
    assert predictions["voucher_number"].confidence == pytest.approx(1.0)
    assert predictions["tugboat"].predicted_value == "БК Пионер"
    assert predictions["tugboat"].source == PREDICTION_SOURCE_FILENAME
    assert predictions["vessel"].predicted_value == "MERIDIAN"
    assert predictions["vessel"].source == PREDICTION_SOURCE_APPLICATION


def test_predict_fields_ocr_postprocesses_tugboat_name():
    voucher = Voucher()
    ocr_values = {"tugboat": ("  бк пионер  ", 0.85)}

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["tugboat"].predicted_value == "БК Пионер"
    assert predictions["tugboat"].predicted_normalized_value == "пионер"


def test_predict_fields_ocr_maps_work_type_to_contract_service():
    voucher = Voucher()
    ocr_values = {"work_type": ("Отшвартовка судна CUMBRIAN", 0.9)}

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["work_type"].predicted_value == "Отшвартовка"
    assert predictions["work_type"].predicted_normalized_value == "отшвартовка"


def test_predict_fields_ocr_maps_composite_work_type():
    voucher = Voucher()
    ocr_values = {"work_type": ("Отшвартовка + Сопровождение судна", 0.9)}

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["work_type"].predicted_value == "Отшвартовка + Сопровождение"


def test_ocr_voucher_regions_uses_trocr_for_handwritten_fields(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db)
    assert voucher.template is not None

    fake_image = Image.new("RGB", (1000, 1414))
    monkeypatch.setattr(
        "app.services.voucher_ocr.load_voucher_image", lambda _v: fake_image
    )
    monkeypatch.setattr(
        "app.services.voucher_ocr.ocr_image", lambda _img: ("tesseract", 0.5)
    )
    monkeypatch.setattr(
        "app.services.voucher_ocr.trocr_image", lambda _img: ("trocr", 0.7)
    )

    result = ocr_voucher_regions(voucher)

    for region in voucher.template.regions:
        if region.name in HANDWRITTEN_FIELDS:
            assert result[region.name] == ("trocr", 0.7), region.name
        else:
            assert result[region.name] == ("tesseract", 0.5), region.name


def test_ocr_voucher_regions_falls_back_to_tesseract_when_trocr_empty(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db)
    assert voucher.template is not None

    fake_image = Image.new("RGB", (1000, 1414))
    monkeypatch.setattr(
        "app.services.voucher_ocr.load_voucher_image", lambda _v: fake_image
    )
    monkeypatch.setattr(
        "app.services.voucher_ocr.ocr_image", lambda _img: ("tesseract", 0.5)
    )
    monkeypatch.setattr(
        "app.services.voucher_ocr.trocr_image", lambda _img: (None, None)
    )

    result = ocr_voucher_regions(voucher)

    for region in voucher.template.regions:
        assert result[region.name] == ("tesseract", 0.5), region.name


def test_trocr_image_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr("app.services.voucher_trocr.settings.trocr_enabled", False)
    assert trocr_image(Image.new("RGB", (10, 10))) == (None, None)


def test_trocr_image_returns_none_when_model_unavailable(monkeypatch):
    monkeypatch.setattr("app.services.voucher_trocr._load_model", lambda: None)
    assert trocr_image(Image.new("RGB", (10, 10))) == (None, None)


def test_predict_fields_ocr_parses_handwritten_datetime():
    voucher = Voucher()
    ocr_values = {
        "started_work": ("2O.O7.2O26 O9-3О", 0.8),
        "left_base": ("20/07/26 9.30", 0.8),
    }

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["started_work"].predicted_value == "2026-07-20 09:30"
    assert predictions["left_base"].predicted_value == "2026-07-20 09:30"


def test_predict_fields_ocr_parses_datetime():
    voucher = Voucher()
    ocr_values = {"started_work": ("20.07.2026 09:30", 0.88)}

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["started_work"].predicted_value == "2026-07-20 09:30"


def test_predict_and_store_runs_ocr_and_saves_predictions(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db)

    fake_ocr_values = {
        "vessel": ("ARIES", 0.9),
        "tugboat": ("БК Пионер", 0.85),
    }
    monkeypatch.setattr("app.services.voucher.ocr_voucher_regions", lambda v: fake_ocr_values)

    stored = predict_and_store(db, voucher, None, VoucherHistory())

    by_name = {row.field_name: row for row in stored}
    assert by_name["vessel"].predicted_value == "ARIES"
    assert by_name["vessel"].source == PREDICTION_SOURCE_OCR
    assert by_name["tugboat"].predicted_value == "БК Пионер"
    assert by_name["tugboat"].source == PREDICTION_SOURCE_OCR


def test_apply_predictions_to_voucher_fills_fields():
    db = _session()
    template = ensure_default_template(db)
    tug = Tug(name="БК Пионер", code="p")
    db.add(tug)
    db.commit()

    voucher = Voucher(template=template)
    db.add(voucher)
    db.flush()
    voucher.predictions = [
        VoucherFieldPrediction(field_name="voucher_number", predicted_value="123"),
        VoucherFieldPrediction(field_name="tugboat", predicted_value="БК Пионер"),
        VoucherFieldPrediction(field_name="vessel", predicted_value="ARIES"),
        VoucherFieldPrediction(field_name="started_work", predicted_value="2026-07-20 10:00"),
    ]
    db.commit()

    apply_predictions_to_voucher(db, voucher)
    db.commit()

    assert voucher.number == "123"
    assert voucher.tug_id == tug.id
    assert voucher.vessel_name == "ARIES"
    assert voucher.started_dt == datetime(2026, 7, 20, 10, 0)
