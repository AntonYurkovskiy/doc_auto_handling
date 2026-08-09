"""Тесты сервиса предсказаний полей ваучера."""

from __future__ import annotations

from datetime import datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Application, Direction, Tug, Voucher, VoucherFieldPrediction
from app.services.voucher import (
    DATE_WINDOW_DAYS,
    DEFAULT_TUG_NAMES,
    MINUTE_STEP,
    VOUCHER_FIELDS,
    VoucherHistory,
    candidate_values,
    normalize_prediction,
    predict_and_store,
    predict_fields,
)
from app.services.voucher_template import ensure_default_template


def _session():
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine)()


def _application() -> Application:
    return Application(
        direction=Direction.entry,
        vessel_name="MERIDIAN",
        agent="Транс-Агро",
        entry_datetime=datetime(2026, 7, 20, 9, 30),
    )


@pytest.mark.parametrize(
    ("field_name", "raw", "expected"),
    [
        ("tugboat", " БК  Коммунар ", "коммунар"),
        ("tugboat", "bk Пионер", "пионер"),
        ("vessel", "  MERIDIAN  ", "meridian"),
        ("work_type", "Вход", "швартовка"),
        ("work_type", "перешвартовка", "перестановка"),
        ("voucher_number", "243k(2)", "243"),
        ("voucher_number", "№ 007", "7"),
        ("left_base", "20.07.2026 09:30", "2026-07-20 09:30"),
        ("started_work", "2026-07-20T10:00", "2026-07-20 10:00"),
        ("finished_work", "не дата", None),
        ("remarks", "  Лёд  ", "лёд"),
        ("joint_with_line_1", "БК Пионер", "бк пионер"),
        ("agent", None, None),
        ("agent", "   ", None),
    ],
)
def test_normalize_prediction(field_name: str, raw: str | None, expected: str | None):
    assert normalize_prediction(field_name, raw) == expected


def test_normalize_prediction_rejects_unknown_field():
    with pytest.raises(ValueError):
        normalize_prediction("unknown_field", "x")


def test_printed_field_candidates_come_from_application_and_history():
    history = VoucherHistory(vessel_names=("ARIES",), agents=("Терминал",))

    assert candidate_values("tugboat", _application(), history) == list(DEFAULT_TUG_NAMES)
    assert candidate_values("vessel", _application(), history) == ["MERIDIAN", "ARIES"]
    assert candidate_values("agent", _application(), history) == ["Транс-Агро", "Терминал"]
    assert candidate_values("work_type", _application(), history) == ["швартовка"]


def test_datetime_candidates_are_bounded_grid_around_application():
    candidates = candidate_values("left_base", _application(), VoucherHistory())

    days = 2 * DATE_WINDOW_DAYS + 1
    assert len(candidates) == days * 24 * (60 // MINUTE_STEP)
    assert candidates[0] == "2026-07-17 00:00"
    assert candidates[-1] == "2026-07-23 23:50"
    assert "2026-07-20 09:30" in candidates
    assert "2026-07-20 09:35" not in candidates
    assert "2026-07-16 12:00" not in candidates


def test_datetime_candidates_empty_without_application_time():
    assert candidate_values("started_work", None, VoucherHistory()) == []


def test_predict_fields_uses_prior_source_and_transparent_confidence():
    voucher = Voucher()
    history = VoucherHistory(vessel_names=("ARIES",))

    predictions = {p.field_name: p for p in predict_fields(voucher, _application(), history)}

    assert set(predictions) == set(VOUCHER_FIELDS)
    assert all(p.source == "prior" for p in predictions.values())

    tugboat = predictions["tugboat"]
    assert tugboat.predicted_value == "БК Коммунар"
    assert tugboat.predicted_normalized_value == "коммунар"
    assert tugboat.confidence == pytest.approx(0.5)

    vessel = predictions["vessel"]
    assert vessel.predicted_value == "MERIDIAN"
    assert vessel.confidence == pytest.approx(0.5)

    assert predictions["remarks"].predicted_value is None
    assert predictions["remarks"].confidence is None


def test_predictions_are_saved_and_confirmed_values_survive_rerun():
    db = _session()
    template = ensure_default_template(db)
    tug = Tug(name="БК Пионер", code="p")
    application = _application()
    voucher = Voucher(template=template, tug=tug, application=application)
    db.add_all([tug, application, voucher])
    db.commit()

    stored = predict_and_store(db, voucher, application, VoucherHistory())
    assert {row.field_name for row in stored} == set(VOUCHER_FIELDS)
    by_name = {row.field_name: row for row in stored}
    assert by_name["tugboat"].predicted_value == "БК Пионер"
    assert by_name["tugboat"].region is not None
    assert by_name["tugboat"].region.name == "tugboat"
    assert voucher.predicted_at is not None

    by_name["vessel"].confirmed_value = "ARIES"
    by_name["vessel"].confirmed_at = datetime(2026, 7, 21, 8, 0)
    db.commit()

    history = VoucherHistory(vessel_names=("OTHER SHIP",))
    reran = {row.field_name: row for row in predict_and_store(db, voucher, application, history)}

    assert db.query(VoucherFieldPrediction).count() == len(VOUCHER_FIELDS)
    assert reran["vessel"].confirmed_value == "ARIES"
    assert reran["vessel"].predicted_value == "MERIDIAN"
    assert reran["work_type"].predicted_value == "швартовка"
