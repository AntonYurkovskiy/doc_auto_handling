"""Тесты парсера заявок (HTML-таблица письма)."""

from __future__ import annotations

from app.services.application_parser import fields_from_html

SAMPLE_HTML = """
<html><body>
<p>ЗАЯВКА НА ВХОД</p>
<table>
  <tr><td>№ п/п</td><td>Перечень сведений о судне</td><td>Сведения</td></tr>
  <tr><td>1</td><td>Название судна</td><td>CUMBRIAN</td></tr>
  <tr><td>2</td><td>№ ИМО</td><td>9298404</td></tr>
  <tr><td>5</td><td>Дата/время входа</td><td>20.07.2026 в 04:00</td></tr>
  <tr><td>10</td><td>Брутто/нетто</td><td>8446 / 4053</td></tr>
  <tr><td>9</td><td>Пункт назначения/№ причала</td><td>Терминал Содружество № 7</td></tr>
</table>
</body></html>
"""


def test_fields_from_html_extracts_labels():
    fields = fields_from_html(SAMPLE_HTML)
    assert fields["Название судна"] == "CUMBRIAN"
    assert fields["№ ИМО"] == "9298404"
    assert "Брутто/нетто" in fields


def test_apply_fields_via_parse():
    from app.services.application_parser import ParsedApplication, _apply_fields

    parsed = ParsedApplication()
    _apply_fields(parsed, fields_from_html(SAMPLE_HTML))
    assert parsed.vessel_name == "CUMBRIAN"
    assert parsed.imo == "9298404"
    assert parsed.gross_tonnage == 8446
    assert parsed.net_tonnage == 4053
    assert parsed.entry_datetime is not None
    assert parsed.entry_datetime.hour == 4
    assert parsed.entry_datetime.day == 20
