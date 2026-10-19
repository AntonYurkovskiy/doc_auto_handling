"""Тесты решений по печатным полям и дефектам приёмки T29 (T30).

Печатные поля (судно, агент, вид работ): ввод человека важнее заявки, заявка важнее чтения,
прочитанная строка пишется в форму только после сверки со списками. Дефекты: D1 (автозаполнение
не сохранялось), D3 (подтверждение перераспознавало ваучер), D4 (нет моделей: пустой статус).
"""

from __future__ import annotations

from collections.abc import Iterator
from datetime import date, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.models import Application, Direction, DocStatus, Tug, Voucher, VoucherFieldPrediction
from app.ocr.decoder import RowOption, RowValue
from app.ocr.pipeline import STATUS_NO_MODELS, STATUS_OK, VoucherRecognition
from app.services.voucher import (
    PREDICTION_SOURCE_APPLICATION,
    PREDICTION_SOURCE_OCR,
    PREDICTION_SOURCE_PRIOR,
    VoucherHistory,
    load_history,
    predict_and_store,
    predict_fields,
)
from app.services.voucher_fields import (
    apply_predictions_to_voucher,
    human_value,
    printed_hints,
)
from app.services.voucher_ocr import OcrFieldValue, VoucherOcrResult
from app.services.voucher_template import ensure_default_template

PDF_BYTES = b"%PDF-1.4\ntest voucher scan\n%%EOF\n"


@pytest.fixture
def db() -> Iterator[Session]:
    engine = create_engine("sqlite://")
    Base.metadata.create_all(engine)
    session = sessionmaker(bind=engine)()
    try:
        yield session
    finally:
        session.close()


def _prediction(name: str, value: str | None, *, source: str = "prior", **kw: object):
    return VoucherFieldPrediction(field_name=name, predicted_value=value, source=source, **kw)


def _voucher(db: Session, **kwargs: object) -> Voucher:
    voucher = Voucher(template=ensure_default_template(db), **kwargs)
    db.add(voucher)
    db.commit()
    return voucher


# --- кто записал значение: человек или приложение ---------------------------------------------


def test_human_value_without_any_prediction_is_typed_by_a_person(db: Session) -> None:
    voucher = _voucher(db, vessel_name="MERIDIAN")

    assert human_value(voucher, "vessel") == "MERIDIAN"


def test_value_equal_to_last_prediction_was_written_by_the_app(db: Session) -> None:
    voucher = _voucher(db, vessel_name="MERIDIAN")
    voucher.predictions = [_prediction("vessel", "MERIDIAN", source="ocr")]
    db.commit()

    assert human_value(voucher, "vessel") is None


def test_value_changed_by_operator_is_human_even_if_a_prediction_exists(db: Session) -> None:
    voucher = _voucher(db, vessel_name="ORIOLE")
    voucher.predictions = [_prediction("vessel", "MERIDIAN", source="ocr")]
    db.commit()

    assert human_value(voucher, "vessel") == "ORIOLE"


def test_confirmed_value_is_always_human(db: Session) -> None:
    voucher = _voucher(db, agent="ТБК")
    voucher.predictions = [_prediction("agent", "ТБК", confirmed_value="ТБК")]
    db.commit()

    assert human_value(voucher, "agent") == "ТБК"


def test_human_value_is_none_for_empty_and_unsupported_fields(db: Session) -> None:
    voucher = _voucher(db, vessel_name="   ")

    assert human_value(voucher, "vessel") is None
    assert human_value(voucher, "tugboat") is None
    assert human_value(voucher, "no_such_field") is None


# --- приоритет: человек > заявка > чтение ---------------------------------------------------


def _by_name(predictions):
    return {p.field_name: p for p in predictions}


def test_machine_filled_vessel_does_not_override_the_application(db: Session) -> None:
    """Прежний мусор в поле (записало приложение) не должен считаться «истиной 100 %»."""
    voucher = _voucher(db, vessel_name="Tug «Kommunar» / «Буксир»")
    voucher.predictions = [_prediction("vessel", "Tug «Kommunar» / «Буксир»")]
    db.commit()
    application = Application(direction=Direction.entry, vessel_name="MERIDIAN")

    predictions = _by_name(predict_fields(voucher, application, VoucherHistory()))

    assert predictions["vessel"].predicted_value == "MERIDIAN"
    assert predictions["vessel"].source == PREDICTION_SOURCE_APPLICATION


def test_typed_vessel_beats_application_and_ocr(db: Session) -> None:
    voucher = _voucher(db, vessel_name="ORIOLE")
    application = Application(direction=Direction.entry, vessel_name="MERIDIAN")
    ocr = {"vessel": OcrFieldValue("MERIDIAN", 0.9)}

    predictions = _by_name(predict_fields(voucher, application, VoucherHistory(), ocr))

    assert predictions["vessel"].predicted_value == "ORIOLE"
    assert predictions["vessel"].source == PREDICTION_SOURCE_PRIOR


def test_typed_agent_and_work_type_are_kept(db: Session) -> None:
    voucher = _voucher(db, agent="ТБК", work_type="Перестановка")
    ocr = {
        "agent": OcrFieldValue("Транс-Агро", 0.9),
        "work_type": OcrFieldValue("Швартовка", 0.9),
    }

    predictions = _by_name(
        predict_fields(voucher, None, VoucherHistory(agents=("Транс-Агро",)), ocr)
    )

    assert predictions["agent"].predicted_value == "ТБК"
    assert predictions["work_type"].predicted_value == "Перестановка"


def test_history_ignores_values_that_nobody_reviewed(db: Session) -> None:
    """В списки для сверки не попадают значения непроверенных ваучеров (там мог быть мусор)."""
    garbage = _voucher(db, vessel_name="Tug «Kommunar» / «Буксир»", agent="me: m/v 106")
    reviewed = _voucher(db, vessel_name="MERIDIAN", agent="ТБК", reviewed_at=datetime(2026, 7, 1))
    db.add(Application(direction=Direction.entry, vessel_name="ORIOLE", agent="Транс-Агро"))
    db.commit()
    assert garbage.id != reviewed.id

    history = load_history(db)

    assert set(history.vessel_names) == {"MERIDIAN", "ORIOLE"}
    assert set(history.agents) == {"ТБК", "Транс-Агро"}


# --- запись в форму ---------------------------------------------------------------------------


def test_apply_writes_only_verified_printed_values(db: Session) -> None:
    voucher = _voucher(db)
    voucher.predictions = [
        _prediction("vessel", "MERIDIAN", source=PREDICTION_SOURCE_OCR),
        _prediction("agent", "Транс-Агро", source=PREDICTION_SOURCE_APPLICATION),
        _prediction("work_type", "ОТШВАРТОВКА", source="prior"),  # догадка из истории
        _prediction("remarks", "прежнее примечание", source="prior"),
        _prediction("joint_with_line_1", "БК Пионер", source="prior"),
    ]
    db.commit()

    apply_predictions_to_voucher(db, voucher)

    assert voucher.vessel_name == "MERIDIAN"
    assert voucher.agent == "Транс-Агро"
    assert voucher.work_type is None
    assert voucher.remarks is None
    assert voucher.joint_with is None


def test_apply_never_fills_remarks_and_joint_with_even_from_ocr_source(db: Session) -> None:
    voucher = _voucher(db)
    voucher.predictions = [
        _prediction("remarks", "что-то", source=PREDICTION_SOURCE_OCR),
        _prediction("joint_with_line_1", "БК Пионер, БК Коммунар", source=PREDICTION_SOURCE_OCR),
    ]
    db.commit()

    apply_predictions_to_voucher(db, voucher)

    assert voucher.remarks is None and voucher.joint_with is None


def test_apply_does_not_overwrite_typed_value(db: Session) -> None:
    voucher = _voucher(db, vessel_name="ORIOLE")
    voucher.predictions = [_prediction("vessel", "MERIDIAN", source=PREDICTION_SOURCE_OCR)]
    db.commit()

    apply_predictions_to_voucher(db, voucher)

    assert voucher.vessel_name == "ORIOLE"


def test_rerecognition_replaces_machine_value_and_clears_old_garbage(db: Session) -> None:
    """«Распознать» заменяет то, что записало приложение, и убирает прежний мусор чтения."""
    voucher = _voucher(db, vessel_name="мусор", agent="мусор 2", joint_with="a, b; c + d")
    voucher.predictions = [  # прежнее чтение: значения в полях записало приложение
        _prediction("vessel", "мусор"),
        _prediction("agent", "мусор 2"),
        _prediction("joint_with_line_1", "a, b; c + d"),
    ]
    db.commit()
    history = VoucherHistory(vessel_names=("MERIDIAN",))

    predict_and_store(db, voucher, None, history, {"vessel": OcrFieldValue("m/v MERIDIAN", 0.8)})
    apply_predictions_to_voucher(db, voucher)

    assert voucher.vessel_name == "MERIDIAN"
    assert voucher.agent is None  # прежний мусор убран, нового проверенного значения нет
    assert voucher.joint_with is None  # число буксиров (tug_count) не портится мусором


def test_value_typed_after_recognition_survives_the_next_recognition(db: Session) -> None:
    voucher = _voucher(db)
    history = VoucherHistory(vessel_names=("MERIDIAN", "ORIOLE"))
    predict_and_store(db, voucher, None, history, {"vessel": OcrFieldValue("MERIDIAN", 0.8)})
    apply_predictions_to_voucher(db, voucher)
    assert voucher.vessel_name == "MERIDIAN"
    voucher.vessel_name = "ORIOLE"  # оператор исправил

    predict_and_store(db, voucher, None, history, {"vessel": OcrFieldValue("MERIDIAN", 0.8)})
    apply_predictions_to_voucher(db, voucher)

    assert voucher.vessel_name == "ORIOLE"


def test_apply_still_transfers_dates_and_number(db: Session) -> None:
    tug = Tug(name="БК Пионер", code="p")
    db.add(tug)
    db.commit()
    voucher = _voucher(db, left_base_dt=datetime(2026, 1, 1, 1, 0))
    voucher.predictions = [
        _prediction("voucher_number", "123"),
        _prediction("left_base", "2026-07-20 10:00", source=PREDICTION_SOURCE_OCR),
        _prediction("tugboat", "БК Пионер"),
    ]
    db.commit()

    apply_predictions_to_voucher(db, voucher)

    assert voucher.number == "123"
    assert voucher.left_base_dt == datetime(2026, 7, 20, 10, 0)  # даты, как раньше, заменяются
    assert voucher.tug_id == tug.id


# --- подсказки --------------------------------------------------------------------------------


def test_unmatched_ocr_line_is_stored_as_a_hint_and_shown_for_empty_field(db: Session) -> None:
    voucher = _voucher(db)
    ocr = {"vessel": OcrFieldValue("Tug «Kommunar» / «Буксир»", 0.9)}

    predict_and_store(db, voucher, None, VoucherHistory(vessel_names=("MERIDIAN",)), ocr)

    assert printed_hints(voucher) == {"vessel_name": "Tug «Kommunar» / «Буксир»"}


def test_hint_disappears_when_field_is_filled(db: Session) -> None:
    voucher = _voucher(db)
    predict_and_store(db, voucher, None, VoucherHistory(), {"agent": OcrFieldValue("мусор", 0.5)})
    assert "agent" in printed_hints(voucher)

    voucher.agent = "ТБК"

    assert printed_hints(voucher) == {}


def test_matched_value_is_not_a_hint(db: Session) -> None:
    voucher = _voucher(db)
    history = VoucherHistory(agents=("ТБК",))

    predict_and_store(db, voucher, None, history, {"agent": OcrFieldValue("ТБК", 0.9)})

    assert printed_hints(voucher) == {}
    assert voucher.predictions[-1].topk is None or "ТБК" not in (voucher.predictions[-1].topk or "")


# --- Веб: D1 и D3 ----------------------------------------------------------------------------


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def client(
    session_factory: sessionmaker[Session], tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> Iterator[TestClient]:
    monkeypatch.setattr(settings, "incoming_vouchers_dir", tmp_path / "vouchers")
    (tmp_path / "vouchers").mkdir()

    def override_get_db() -> Iterator[Session]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _confident_recognition() -> VoucherRecognition:
    option = RowOption(RowValue(date(2026, 7, 20), 10, 0), 0.99999)
    return VoucherRecognition(
        status=STATUS_OK,
        reason="",
        variant="kommunar_v1",
        alignment=None,
        fields={"left_base": (option,)},
        voucher_number=((262, 0.99),),
        records=(),
        confidence=0.99999,
        margin=0.9,
        auto_accept=True,
        crop_paths={},
        timing={},
        flags=(),
    )


def _upload(client: TestClient) -> int:
    response = client.post(
        "/vouchers/upload",
        files={"file": ("262k.pdf", PDF_BYTES, "application/pdf")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return int(response.headers["location"].rsplit("/", 1)[-1])


def test_upload_commits_auto_accept_prefill_d1(
    client: TestClient, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """D1: после автозаполнения нужен commit, иначе поля пустые при зелёной плашке."""
    result = VoucherOcrResult(
        fields={
            "left_base": OcrFieldValue("2026-07-20 10:00", 0.99999),
            "agent": OcrFieldValue("ТБК", 0.9),
            "vessel": OcrFieldValue("мусор мусор", 0.8),
        },
        recognition=_confident_recognition(),
    )
    monkeypatch.setattr(
        "app.services.voucher.ocr_voucher_regions", lambda db, v, application: result
    )
    db = session_factory()
    db.add(Application(direction=Direction.entry, vessel_name="MERIDIAN", agent="ТБК"))
    db.commit()
    db.close()

    voucher_id = _upload(client)

    db = session_factory()
    try:
        voucher = db.get(Voucher, voucher_id)
        assert voucher is not None
        assert voucher.ocr_status == STATUS_OK
        assert voucher.left_base_dt == datetime(2026, 7, 20, 10, 0)  # предзаполнение сохранилось
        assert voucher.agent == "ТБК"  # печатное, найденное в списках
        assert voucher.vessel_name is None  # нераспознанное в форму не попало
        assert voucher.status is DocStatus.needs_review  # статус по-прежнему за оператором
    finally:
        db.close()


def test_confirm_and_manual_save_do_not_rerecognize_d3(
    client: TestClient, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    """D3: «Подтвердить» и «Сохранить» не запускают распознавание заново (решение В6)."""
    voucher_id = _upload(client)
    db = session_factory()
    application = Application(direction=Direction.entry, vessel_name="MERIDIAN")
    db.add(application)
    voucher = db.get(Voucher, voucher_id)
    assert voucher is not None
    voucher.ocr_status = STATUS_OK
    voucher.ocr_confidence = 0.42
    db.commit()
    application_id = application.id
    db.close()

    def forbidden(*args: object, **kwargs: object) -> None:
        raise AssertionError("распознавание не должно запускаться")

    monkeypatch.setattr("app.main.predict_and_store", forbidden)

    form = {"number": "262", "vessel_name": "MERIDIAN", "application_id": str(application_id)}
    assert client.post("/vouchers", data={"voucher_id": str(voucher_id), **form}).status_code == 200
    response = client.post(f"/vouchers/{voucher_id}/confirm", data=form, follow_redirects=False)
    assert response.status_code == 303

    db = session_factory()
    try:
        voucher = db.get(Voucher, voucher_id)
        assert voucher is not None
        assert voucher.status is DocStatus.confirmed
        assert voucher.ocr_confidence == pytest.approx(0.42)  # уверенность не затёрта
    finally:
        db.close()


def test_recognize_button_still_runs_recognition(
    client: TestClient, session_factory: sessionmaker[Session], monkeypatch: pytest.MonkeyPatch
) -> None:
    voucher_id = _upload(client)
    calls: list[int] = []

    def fake_predict_and_store(db: Session, voucher: Voucher, *args: object, **kwargs: object):
        calls.append(voucher.id)
        return []

    monkeypatch.setattr("app.main.predict_and_store", fake_predict_and_store)

    response = client.post(f"/vouchers/{voucher_id}/recognize", follow_redirects=False)

    assert response.status_code == 303 and calls == [voucher_id]


def test_card_shows_hint_for_unmatched_printed_line(
    client: TestClient, session_factory: sessionmaker[Session]
) -> None:
    voucher_id = _upload(client)
    db = session_factory()
    voucher = db.get(Voucher, voucher_id)
    assert voucher is not None
    predict_and_store(db, voucher, None, VoucherHistory(), {"vessel": OcrFieldValue("ABRA", 0.5)})
    db.commit()
    db.close()

    html = client.get(f"/vouchers/{voucher_id}").text

    assert "На бланке прочитано «ABRA»" in html


# --- D4: нет моделей -------------------------------------------------------------------------


def test_missing_models_set_ocr_status_no_models_d4(
    db: Session, monkeypatch: pytest.MonkeyPatch
) -> None:
    voucher = _voucher(db, file_path="/scans/262k.pdf")
    monkeypatch.setattr(
        "app.ocr.pipeline.get_resources",
        lambda **kwargs: (_ for _ in ()).throw(FileNotFoundError("нет моделей")),
    )
    monkeypatch.setattr("app.services.voucher_ocr.load_voucher_image", lambda _v: None)

    predict_and_store(db, voucher)

    assert voucher.ocr_status == STATUS_NO_MODELS
    assert voucher.ocr_confidence is None
