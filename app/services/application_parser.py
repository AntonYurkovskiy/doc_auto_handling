"""Парсер заявок.

Основной формат (≈90%): HTML-письмо от морского агентства с таблицей
«№ п/п | Перечень сведений о судне | Сведения». Парсится напрямую из HTML,
без OCR. Резервно — извлечение из текстового PDF (pdfplumber) или тела письма.
"""

from __future__ import annotations

import email
import re
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from pathlib import Path

from bs4 import BeautifulSoup

_WS = re.compile(r"[\s\u00a0]+")


@dataclass
class ParsedApplication:
    vessel_name: str | None = None
    imo: str | None = None
    gross_tonnage: int | None = None
    net_tonnage: int | None = None
    entry_datetime: datetime | None = None
    exit_datetime: datetime | None = None
    destination: str | None = None
    tugs_text: str | None = None
    direction: str = "прочее"  # вход / выход / прочее
    sender: str | None = None
    subject: str | None = None
    received_at: datetime | None = None
    raw_text: str = ""
    fields: dict[str, str] = field(default_factory=dict)


def _clean(value: str) -> str:
    return _WS.sub(" ", value).strip()


def _parse_dt(value: str) -> datetime | None:
    """Достать дату и время из текста вида «20.07.2026 в 04:00» (с любыми пробелами)."""
    compact = _WS.sub("", value)
    m = re.search(r"(\d{2})\.(\d{2})\.(\d{4}).{0,3}?(\d{1,2}):(\d{2})", compact)
    if not m:
        return None
    d, mo, y, hh, mm = (int(x) for x in m.groups())
    try:
        return datetime(y, mo, d, hh, mm)
    except ValueError:
        return None


def _extract_int_pair(value: str) -> tuple[int | None, int | None]:
    compact = _WS.sub("", value)
    m = re.search(r"(\d+)\s*/\s*(\d+)", compact)
    if m:
        return int(m.group(1)), int(m.group(2))
    m = re.search(r"(\d+)", compact)
    return (int(m.group(1)), None) if m else (None, None)


def fields_from_html(html: str) -> dict[str, str]:
    """Собрать словарь {метка: значение} из таблиц заявки."""
    soup = BeautifulSoup(html, "html.parser")
    result: dict[str, str] = {}
    for table in soup.find_all("table"):
        for row in table.find_all("tr"):
            cells = [c.get_text(" ", strip=True) for c in row.find_all(["td", "th"])]
            cells = [_clean(c) for c in cells if _clean(c)]
            if len(cells) >= 2:
                # Значение — последняя ячейка, метка — предпоследняя.
                label = cells[-2]
                value = cells[-1]
                if label and value and not label.isdigit():
                    result[label] = value
    return result


def _apply_fields(parsed: ParsedApplication, fields: dict[str, str]) -> None:
    parsed.fields = fields
    for label, value in fields.items():
        low = label.lower()
        if "название судна" in low:
            parsed.vessel_name = value
        elif "имо" in low:
            parsed.imo = _WS.sub("", value)
        elif "брутто" in low:
            gross, net = _extract_int_pair(value)
            parsed.gross_tonnage, parsed.net_tonnage = gross, net
        elif "входа" in low:
            parsed.entry_datetime = _parse_dt(value)
        elif "выхода" in low and "пункт" not in low:
            parsed.exit_datetime = _parse_dt(value)
        elif "назначения" in low:
            parsed.destination = value


def _detect_direction(text: str, subject: str | None) -> str:
    haystack = f"{subject or ''} {text[:400]}".lower()
    if "на вход" in haystack or re.search(r"\bвход\b", haystack):
        return "вход"
    if "на выход" in haystack or re.search(r"\bвыход\b", haystack):
        return "выход"
    return "прочее"


def parse_eml(path: str | Path) -> ParsedApplication:
    """Разобрать сохранённое письмо (.eml)."""
    from email.message import EmailMessage
    from typing import cast

    with open(path, "rb") as fh:
        msg = cast(
            EmailMessage,
            email.message_from_binary_file(fh, policy=policy.default),  # type: ignore[arg-type]
        )

    html: str | None = None
    text: str | None = None
    for part in msg.walk():
        ctype = part.get_content_type()
        if ctype == "text/html" and html is None:
            html = cast(EmailMessage, part).get_content()
        elif ctype == "text/plain" and text is None:
            try:
                text = cast(EmailMessage, part).get_content()
            except Exception:  # noqa: BLE001
                text = None

    parsed = ParsedApplication()
    parsed.subject = msg.get("subject")
    parsed.sender = msg.get("from")
    date_hdr = msg.get("date")
    if date_hdr:
        try:
            parsed.received_at = email.utils.parsedate_to_datetime(date_hdr).replace(tzinfo=None)
        except (TypeError, ValueError):
            parsed.received_at = None

    if html:
        soup = BeautifulSoup(html, "html.parser")
        parsed.raw_text = soup.get_text("\n", strip=True)
        _apply_fields(parsed, fields_from_html(html))
    elif text:
        parsed.raw_text = text
        _apply_fields(parsed, fields_from_text(text))

    parsed.direction = _detect_direction(parsed.raw_text, parsed.subject)
    return parsed


def fields_from_text(text: str) -> dict[str, str]:
    """Резервный парсер по строкам (для текстовых PDF / тела письма).

    Ищет известные метки и берёт ближайшее значение справа или на след. строке.
    """
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    result: dict[str, str] = {}
    labels = [
        "Название судна",
        "№ ИМО",
        "Брутто/нетто",
        "Дата/время входа",
        "Дата/время выхода",
        "Пункт назначения",
    ]
    joined = "\n".join(lines)
    for label in labels:
        idx = joined.find(label)
        if idx == -1:
            continue
        tail = joined[idx + len(label): idx + len(label) + 60]
        value = _clean(tail.split("\n", 2)[0]) or _clean(
            tail.split("\n", 2)[1] if "\n" in tail else ""
        )
        if value:
            result[label] = value
    return result


def parse_pdf(path: str | Path) -> ParsedApplication:
    """Разобрать текстовый PDF заявки."""
    import pdfplumber

    parsed = ParsedApplication()
    with pdfplumber.open(path) as pdf:
        parsed.raw_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    _apply_fields(parsed, fields_from_text(parsed.raw_text))
    parsed.direction = _detect_direction(parsed.raw_text, None)
    return parsed


def parse_application(path: str | Path) -> ParsedApplication:
    """Определить тип файла и разобрать заявку."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return parse_pdf(p)
    # .eml или файл без расширения, начинающийся с email-заголовков.
    return parse_eml(p)
