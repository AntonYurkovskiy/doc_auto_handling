"""Тесты OCR по регионам ваучера: Tesseract (печатные поля) + пайплайн T25 (даты/номер)."""

from __future__ import annotations

from datetime import date, datetime
from unittest.mock import MagicMock, patch

import pytest
from PIL import Image
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Application, Direction, DocStatus, Tug, Voucher, VoucherFieldPrediction
from app.ocr.decoder import RowOption, RowValue
from app.ocr.pipeline import STATUS_ALIGN_FAILED, STATUS_OK, ConfirmedPartner, VoucherRecognition
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
    PRINTED_OCR_FIELDS,
    OcrFieldValue,
    VoucherOcrResult,
    build_recognition_context,
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


def _make_voucher_with_template(db, **kwargs) -> Voucher:
    template = ensure_default_template(db)
    kwargs.setdefault("original_filename", "scan.pdf")
    voucher = Voucher(template=template, **kwargs)
    db.add(voucher)
    db.commit()
    return voucher


def _fake_recognition(**overrides) -> VoucherRecognition:
    base: dict = dict(
        status=STATUS_OK,
        reason="",
        variant="kommunar_v1",
        alignment=None,
        fields={
            "left_base": (
                RowOption(RowValue(date(2026, 7, 20), 8, 0), 0.9),
                RowOption(RowValue(date(2026, 7, 20), 8, 10), 0.1),
            ),
        },
        voucher_number=((243, 0.95), (244, 0.05)),
        records=(),
        confidence=0.9,
        margin=0.8,
        auto_accept=False,
        crop_paths={},
        timing={},
        flags=(),
    )
    base.update(overrides)
    return VoucherRecognition(**base)


# --- печатные поля: crop/Tesseract, без изменений поведения -------------------------------


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


def test_trocr_image_returns_none_when_disabled(monkeypatch):
    monkeypatch.setattr("app.services.voucher_trocr.settings.trocr_enabled", False)
    assert trocr_image(Image.new("RGB", (10, 10))) == (None, None)


def test_trocr_image_returns_none_when_model_unavailable(monkeypatch):
    monkeypatch.setattr("app.services.voucher_trocr._load_model", lambda: None)
    assert trocr_image(Image.new("RGB", (10, 10))) == (None, None)


# --- ocr_voucher_regions: печатные поля (Tesseract) + пайплайн T25 (даты/номер) -----------


def test_ocr_voucher_regions_returns_empty_without_template():
    db = _session()
    voucher = Voucher(template=None)
    result = ocr_voucher_regions(db, voucher)
    assert result.fields == {}
    assert result.recognition is None


def test_ocr_voucher_regions_runs_tesseract_only_for_printed_fields(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db)
    assert voucher.template is not None

    fake_image = Image.new("RGB", (1000, 1414))
    monkeypatch.setattr("app.services.voucher_ocr.load_voucher_image", lambda _v: fake_image)
    monkeypatch.setattr("app.services.voucher_ocr.ocr_image", lambda _img: ("test-text", 0.75))

    result = ocr_voucher_regions(db, voucher)

    # Нет file_path -> пайплайн T25 не вызывается, остаются только печатные поля.
    assert set(result.fields) == PRINTED_OCR_FIELDS
    for value in result.fields.values():
        assert value == OcrFieldValue("test-text", 0.75)
    assert result.recognition is None
    # Строки дат/номер больше не идут через Tesseract (убраны из рабочего пути T26).
    assert not HANDWRITTEN_FIELDS & set(result.fields)
    assert "voucher_number" not in result.fields


def test_ocr_voucher_regions_merges_pipeline_fields(monkeypatch):
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()
    voucher = _make_voucher_with_template(
        db, tug=tug, file_path="/scans/243k.pdf", original_filename="243k.pdf"
    )

    monkeypatch.setattr("app.services.voucher_ocr.load_voucher_image", lambda _v: None)
    captured: dict = {}

    def fake_recognize_voucher(path, **kwargs):
        captured["path"] = path
        captured["kwargs"] = kwargs
        return _fake_recognition()

    monkeypatch.setattr(
        "app.ocr.pipeline.get_resources", lambda **kwargs: object()
    )
    monkeypatch.setattr("app.ocr.pipeline.recognize_voucher", fake_recognize_voucher)

    result = ocr_voucher_regions(db, voucher)

    assert captured["path"] == voucher.file_path
    assert captured["kwargs"]["tug_hint"] == "k"
    assert result.recognition is not None and result.recognition.ok
    assert result.fields["left_base"] == OcrFieldValue(
        "2026-07-20 08:00",
        0.9,
        (("2026-07-20 08:00", 0.9), ("2026-07-20 08:10", 0.1)),
    )
    assert result.fields["voucher_number"] == OcrFieldValue(
        "243", 0.95, (("243", 0.95), ("244", 0.05))
    )


def test_ocr_voucher_regions_skips_pipeline_fields_when_not_ok(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db, file_path="/scans/243k.pdf")
    monkeypatch.setattr("app.services.voucher_ocr.load_voucher_image", lambda _v: None)
    monkeypatch.setattr("app.ocr.pipeline.get_resources", lambda **kwargs: object())
    monkeypatch.setattr(
        "app.ocr.pipeline.recognize_voucher",
        lambda path, **kwargs: _fake_recognition(status=STATUS_ALIGN_FAILED, reason="noise"),
    )

    result = ocr_voucher_regions(db, voucher)

    assert "left_base" not in result.fields
    assert "voucher_number" not in result.fields
    assert result.recognition is not None and not result.recognition.ok


def test_ocr_voucher_regions_pipeline_unavailable_without_models(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db, file_path="/scans/243k.pdf")
    monkeypatch.setattr("app.services.voucher_ocr.load_voucher_image", lambda _v: None)
    monkeypatch.setattr(
        "app.ocr.pipeline.get_resources",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("нет моделей")),
    )

    result = ocr_voucher_regions(db, voucher)

    assert result.fields == {}
    assert result.recognition is None


def test_ocr_voucher_regions_respects_ocr_enabled_flag(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db, file_path="/scans/243k.pdf")
    monkeypatch.setattr("app.services.voucher_ocr.load_voucher_image", lambda _v: None)
    monkeypatch.setattr("app.services.voucher_ocr.settings.ocr_enabled", False)
    called = False

    def fail_if_called(**kwargs):
        nonlocal called
        called = True
        raise AssertionError("пайплайн не должен вызываться при ocr_enabled=False")

    monkeypatch.setattr("app.ocr.pipeline.get_resources", fail_if_called)

    result = ocr_voucher_regions(db, voucher)

    assert called is False
    assert result.fields == {}
    assert result.recognition is None


# --- build_recognition_context: год, вид работ, история номеров, пара --------------------


def test_build_recognition_context_year_from_voucher_fields():
    db = _session()
    voucher = Voucher(started_dt=datetime(2025, 3, 1, 10, 0))
    ctx = build_recognition_context(db, voucher, None)
    assert ctx.year == 2025


def test_build_recognition_context_year_from_application_then_today():
    db = _session()
    application = Application(direction=Direction.entry, received_at=datetime(2024, 5, 1))
    ctx = build_recognition_context(db, Voucher(), application)
    assert ctx.year == 2024

    ctx_default = build_recognition_context(db, Voucher(), None)
    assert ctx_default.year == datetime.utcnow().year


def test_build_recognition_context_work_type_from_voucher_then_application():
    db = _session()
    application = Application(direction=Direction.exit)
    ctx = build_recognition_context(db, Voucher(work_type="Швартовка"), application)
    assert ctx.work_type == "швартовка"

    ctx_from_app = build_recognition_context(db, Voucher(), application)
    assert ctx_from_app.work_type == "отшвартовка"


def test_build_recognition_context_number_history_expected_and_used():
    db = _session()
    tug = Tug(name="БК Коммунар", code="k")
    db.add(tug)
    db.commit()
    db.add_all(
        [
            Voucher(tug=tug, number="10", started_dt=datetime(2026, 1, 5)),
            Voucher(tug=tug, number="11", started_dt=datetime(2026, 2, 1)),
            Voucher(tug=tug, number="5", started_dt=datetime(2025, 1, 1)),  # другой год
        ]
    )
    db.commit()

    current = Voucher(tug=tug, started_dt=datetime(2026, 3, 1))
    db.add(current)
    db.commit()

    ctx = build_recognition_context(db, current, None)
    assert ctx.number_history is not None
    assert ctx.number_history.expected == 12
    assert ctx.number_history.used == frozenset({10, 11})


def test_build_recognition_context_number_history_none_without_tug():
    db = _session()
    ctx = build_recognition_context(db, Voucher(started_dt=datetime(2026, 1, 1)), None)
    assert ctx.number_history is None


def test_build_recognition_context_partner_confirmed():
    db = _session()
    tug_k = Tug(name="БК Коммунар", code="k")
    tug_p = Tug(name="БК Пионер", code="p")
    application = Application(direction=Direction.entry)
    db.add_all([tug_k, tug_p, application])
    db.commit()

    partner = Voucher(
        tug=tug_p,
        application=application,
        status=DocStatus.confirmed,
        left_base_dt=datetime(2026, 7, 20, 7, 0),
        arrived_base_dt=datetime(2026, 7, 20, 9, 0),
        started_dt=datetime(2026, 7, 20, 7, 30),
        finished_dt=datetime(2026, 7, 20, 8, 30),
    )
    voucher = Voucher(tug=tug_k, application=application)
    db.add_all([partner, voucher])
    db.commit()

    ctx = build_recognition_context(db, voucher, application)

    assert isinstance(ctx.partner, ConfirmedPartner)
    assert ctx.partner.forms["left_base"] == (date(2026, 7, 20), 7, 0)


def test_build_recognition_context_partner_pending_has_file():
    db = _session()
    tug_k = Tug(name="БК Коммунар", code="k")
    tug_p = Tug(name="БК Пионер", code="p")
    application = Application(direction=Direction.entry)
    db.add_all([tug_k, tug_p, application])
    db.commit()

    partner = Voucher(
        tug=tug_p,
        application=application,
        status=DocStatus.needs_review,
        file_path="/scans/12p.pdf",
        original_filename="12p.pdf",
    )
    voucher = Voucher(tug=tug_k, application=application)
    db.add_all([partner, voucher])
    db.commit()

    ctx = build_recognition_context(db, voucher, application)

    assert ctx.partner is not None
    assert not isinstance(ctx.partner, ConfirmedPartner)
    assert ctx.partner.tug_hint == "p"
    assert ctx.partner.context.partner is None  # без рекурсии


def test_build_recognition_context_no_partner_without_application():
    db = _session()
    ctx = build_recognition_context(db, Voucher(), None)
    assert ctx.partner is None


# --- predict_fields: OCR (Tesseract и пайплайн) важнее приоров -----------------------------


def test_predict_fields_prefers_ocr_over_prior():
    voucher = Voucher()
    application = Application(
        direction=Direction.entry,
        vessel_name="MERIDIAN",
        agent="Транс-Агро",
        entry_datetime=datetime(2026, 7, 20, 9, 30),
    )
    history = VoucherHistory()
    ocr_values = {"agent": OcrFieldValue("  Другой Агент  ", 0.9)}

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
        "voucher_number": OcrFieldValue("9999", 0.9),
        "tugboat": OcrFieldValue("распознанный мусор", 0.9),
        "vessel": OcrFieldValue("GARBAGE", 0.9),
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
    ocr_values = {"tugboat": OcrFieldValue("  бк пионер  ", 0.85)}

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["tugboat"].predicted_value == "БК Пионер"
    assert predictions["tugboat"].predicted_normalized_value == "пионер"


def test_predict_fields_ocr_maps_work_type_to_contract_service():
    voucher = Voucher()
    ocr_values = {"work_type": OcrFieldValue("Отшвартовка судна CUMBRIAN", 0.9)}

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["work_type"].predicted_value == "Отшвартовка"
    assert predictions["work_type"].predicted_normalized_value == "отшвартовка"


def test_predict_fields_ocr_maps_composite_work_type():
    voucher = Voucher()
    ocr_values = {"work_type": OcrFieldValue("Отшвартовка + Сопровождение судна", 0.9)}

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["work_type"].predicted_value == "Отшвартовка + Сопровождение"


def test_predict_fields_ocr_parses_handwritten_datetime():
    """Разбор OCR-строк дат: сохранился и работает для источника Tesseract (fallback)."""
    voucher = Voucher()
    ocr_values = {
        "started_work": OcrFieldValue("2O.O7.2O26 O9-3О", 0.8),
        "left_base": OcrFieldValue("20/07/26 9.30", 0.8),
    }

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["started_work"].predicted_value == "2026-07-20 09:30"
    assert predictions["left_base"].predicted_value == "2026-07-20 09:30"


def test_predict_fields_ocr_parses_datetime():
    voucher = Voucher()
    ocr_values = {"started_work": OcrFieldValue("20.07.2026 09:30", 0.88)}

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["started_work"].predicted_value == "2026-07-20 09:30"


def test_predict_fields_stores_topk_candidates():
    voucher = Voucher()
    ocr_values = {
        "started_work": OcrFieldValue(
            "2026-07-20 09:30",
            0.9,
            (("2026-07-20 09:30", 0.9), ("2026-07-20 09:40", 0.1)),
        )
    }

    predictions = {
        p.field_name: p
        for p in predict_fields(voucher, None, VoucherHistory(), ocr_values)
    }

    assert predictions["started_work"].topk == (
        ("2026-07-20 09:30", 0.9),
        ("2026-07-20 09:40", 0.1),
    )
    assert predictions["tugboat"].topk == ()


# --- predict_and_store: пайплайн T25, topk и статус ваучера --------------------------------


def test_predict_and_store_runs_ocr_and_saves_predictions(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db)

    fake_ocr_result = VoucherOcrResult(
        fields={
            "vessel": OcrFieldValue("ARIES", 0.9),
            "tugboat": OcrFieldValue("БК Пионер", 0.85),
        }
    )
    monkeypatch.setattr(
        "app.services.voucher.ocr_voucher_regions", lambda db, v, application: fake_ocr_result
    )

    stored = predict_and_store(db, voucher, None, VoucherHistory())

    by_name = {row.field_name: row for row in stored}
    assert by_name["vessel"].predicted_value == "ARIES"
    assert by_name["vessel"].source == PREDICTION_SOURCE_OCR
    assert by_name["tugboat"].predicted_value == "БК Пионер"
    assert by_name["tugboat"].source == PREDICTION_SOURCE_OCR


def test_predict_and_store_persists_topk_as_json():
    db = _session()
    voucher = _make_voucher_with_template(db)
    ocr_values = {
        "voucher_number": OcrFieldValue("243", 0.95, (("243", 0.95), ("244", 0.05))),
    }

    stored = predict_and_store(db, voucher, None, VoucherHistory(), ocr_values)

    row = next(r for r in stored if r.field_name == "voucher_number")
    assert row.topk == '[["243", 0.95], ["244", 0.05]]'


def test_predict_and_store_sets_voucher_ocr_status_from_recognition(monkeypatch):
    db = _session()
    voucher = _make_voucher_with_template(db, file_path="/scans/243k.pdf")

    fake_ocr_result = VoucherOcrResult(fields={}, recognition=_fake_recognition(confidence=0.42))
    monkeypatch.setattr(
        "app.services.voucher.ocr_voucher_regions", lambda db, v, application: fake_ocr_result
    )

    predict_and_store(db, voucher, None, VoucherHistory())

    assert voucher.ocr_status == STATUS_OK
    assert voucher.ocr_confidence == pytest.approx(0.42)


def test_predict_and_store_keeps_ocr_status_when_pipeline_not_invoked():
    db = _session()
    voucher = _make_voucher_with_template(db)
    voucher.ocr_status = "ok"
    voucher.ocr_confidence = 0.7
    db.commit()

    # ocr_values передан явно -> ocr_voucher_regions не зовётся, recognition=None,
    # прежние значения ocr_status/ocr_confidence не трогаются.
    predict_and_store(db, voucher, None, VoucherHistory(), {})

    assert voucher.ocr_status == "ok"
    assert voucher.ocr_confidence == pytest.approx(0.7)


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
