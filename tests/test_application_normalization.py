"""Регрессионные тесты нормализации заявок: агент, даты, осадка, raw_html."""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Agent, Application
from app.services.application_parser import (
    ParsedApplication,
    _apply_fields,
    fields_from_html,
    normalize_agent,
    parse_eml,
)
from app.services.html_sanitize import sanitize_email_html
from app.services.imap_ingest import ingest_messages

KNOWN_AGENTS = ["Транс-Агро", "Содружество - Соя", "МореСервис"]

REAL_HTML = """
<html><body>
<p>ЗАЯВКА НА ВХОД</p>
<table>
  <tr><td>№ п/п</td><td>Перечень сведений о судне</td><td>Сведения</td></tr>
  <tr><td>1</td><td>Название судна</td><td>CUMBRIAN</td></tr>
  <tr><td>2</td><td>Морской агент</td><td>ООО &quot;Транс-Агро&quot;</td></tr>
  <tr><td>3</td><td>Осадка носом/кормой</td><td>8,711 m/8,714 m</td></tr>
  <tr><td>4</td><td>Планируемое время входа (ETA)</td><td>20.07.2026 в 04:00</td></tr>
  <tr><td>5</td><td>Планируемое время выхода (ETD)</td><td>22/07/2026 18-30</td></tr>
  <tr><td>6</td><td>Пункт назначения/№ причала</td><td>Терминал Содружество № 7</td></tr>
</table>
</body></html>
"""


def _session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _parse(html: str = REAL_HTML) -> ParsedApplication:
    parsed = ParsedApplication()
    _apply_fields(parsed, fields_from_html(html), KNOWN_AGENTS)
    return parsed


def test_agent_normalized_to_reference_value():
    assert _parse().agent == "Транс-Агро"


def test_agent_normalization_ignores_legal_form_and_quotes():
    assert normalize_agent('ООО «Содружество-Соя»', KNOWN_AGENTS) == "Содружество - Соя"
    assert normalize_agent("Неизвестный агент", KNOWN_AGENTS) == "Неизвестный агент"
    assert normalize_agent("", KNOWN_AGENTS) is None


def test_entry_and_exit_datetimes_from_real_labels():
    parsed = _parse()
    assert parsed.entry_datetime is not None
    assert (parsed.entry_datetime.day, parsed.entry_datetime.hour) == (20, 4)
    assert parsed.exit_datetime is not None
    assert parsed.exit_datetime.day == 22
    assert (parsed.exit_datetime.hour, parsed.exit_datetime.minute) == (18, 30)


def test_destination_label_is_not_treated_as_exit_datetime():
    parsed = _parse()
    assert parsed.destination == "Терминал Содружество № 7"


def test_draft_takes_maximum_of_pair():
    assert _parse().draft_m == 8.714


def test_single_draft_value_is_unchanged():
    html = "<table><tr><td>Осадка</td><td>9,4 м</td></tr></table>"
    assert _parse(html).draft_m == 9.4


def test_agent_and_dates_from_plain_text_outside_table(tmp_path: Path):
    msg = EmailMessage()
    msg["Subject"] = "Заявка на вход"
    msg["From"] = "agency@example.com"
    msg.set_content(
        "Агент: ООО Транс-Агро\n"
        "Дата и время прихода: 01.02.2026 07:05\n"
        "Дата и время отхода: 02.02.2026 09:15\n"
        "Осадка: 8,711 m/8,714 m\n"
    )
    path = tmp_path / "plain.eml"
    path.write_bytes(msg.as_bytes())

    parsed = parse_eml(path, known_agents=KNOWN_AGENTS)

    assert parsed.agent == "Транс-Агро"
    assert parsed.entry_datetime is not None and parsed.entry_datetime.hour == 7
    assert parsed.exit_datetime is not None and parsed.exit_datetime.minute == 15
    assert parsed.draft_m == 8.714


def test_raw_html_kept_separately_from_raw_text(tmp_path: Path):
    msg = EmailMessage()
    msg["Subject"] = "Заявка на вход"
    msg["From"] = "agency@example.com"
    msg.set_content("Заявка на вход")
    msg.add_alternative(REAL_HTML, subtype="html")
    path = tmp_path / "html.eml"
    path.write_bytes(msg.as_bytes())

    parsed = parse_eml(path, known_agents=KNOWN_AGENTS)

    assert parsed.raw_html is not None
    assert "<table" in parsed.raw_html
    assert "<table" not in parsed.raw_text
    assert "CUMBRIAN" in parsed.raw_text


def test_ingest_stores_raw_html_and_agent(tmp_path: Path):
    db = _session()
    db.add(Agent(name="Транс-Агро"))
    db.commit()

    msg = EmailMessage()
    msg["Message-ID"] = "<raw-html@example.com>"
    msg["Subject"] = "Заявка на вход"
    msg["From"] = "agency@example.com"
    msg.set_content("Заявка на вход")
    msg.add_alternative(REAL_HTML, subtype="html")

    ingest_messages(
        db,
        [msg.as_bytes()],
        applications_dir=tmp_path / "apps",
        vouchers_dir=tmp_path / "vouchers",
    )

    row = db.query(Application).one()
    assert row.agent == "Транс-Агро"
    assert row.raw_html is not None and "<table" in row.raw_html
    assert row.raw_text and "<table" not in row.raw_text
    assert row.draft_m == 8.714


def test_application_form_renders_email_safely():
    from collections.abc import Iterator

    from fastapi.testclient import TestClient
    from sqlalchemy.pool import StaticPool

    from app.database import get_db
    from app.main import app as web_app

    engine = create_engine(
        "sqlite://", connect_args={"check_same_thread": False}, poolclass=StaticPool
    )
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)

    def override_get_db() -> Iterator:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    web_app.dependency_overrides[get_db] = override_get_db
    try:
        db = session_factory()
        db.add(
            Application(
                vessel_name="CUMBRIAN",
                agent="Транс-Агро",
                raw_html=REAL_HTML + '<script>alert(1)</script>',
            )
        )
        db.commit()
        app_id = db.query(Application).one().id
        db.close()

        response = TestClient(web_app).get(f"/applications/{app_id}")
        assert response.status_code == 200
        assert "srcdoc=" in response.text
        assert "alert(1)" not in response.text
        assert "&lt;table" in response.text  # письмо экранировано внутри srcdoc
    finally:
        web_app.dependency_overrides.clear()
        Base.metadata.drop_all(engine)
        engine.dispose()


def test_sanitizer_strips_active_content_but_keeps_table():
    dirty = (
        '<div onclick="alert(1)"><script>alert(2)</script>'
        '<img src="https://evil.example/track.gif" alt="лого">'
        '<a href="javascript:alert(3)">click</a>'
        '<a href="https://ok.example">ok</a>'
        '<table><tr><td style="color:red">CUMBRIAN</td></tr></table></div>'
    )
    clean = sanitize_email_html(dirty)

    assert "script" not in clean
    assert "onclick" not in clean
    assert "evil.example" not in clean
    assert "javascript:" not in clean
    assert "https://ok.example" in clean
    assert "<td style=\"color:red\">CUMBRIAN</td>" in clean
    assert sanitize_email_html(None) == ""
