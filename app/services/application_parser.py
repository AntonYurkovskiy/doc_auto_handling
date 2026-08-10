"""Парсер заявок.

Основной формат (≈90%): HTML-письмо от морского агентства с таблицей
«№ п/п | Перечень сведений о судне | Сведения». Парсится напрямую из HTML,
без OCR. Резервно — извлечение из текстового PDF (pdfplumber) или тела письма.
"""

from __future__ import annotations

import email
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from pathlib import Path

from bs4 import BeautifulSoup

_WS = re.compile(r"[\s\u00a0]+")
_NUMBER = re.compile(r"\d+(?:\.\d+)?")
_ORG_PREFIX = re.compile(
    r"^(?:ооо|зао|оао|пао|ао|ип|llc|ltd|jsc|co)\b\.?",
    re.IGNORECASE,
)
_QUOTES = "\"'«»“”„‘’"

# Метки таблицы/текста, из которых берём даты и агента (реальные письма
# отличаются формулировками: «Дата/время входа», «Планируемый вход (ETA)» и т.п.).
_ENTRY_HINTS = ("вход", "приход", "прибыти", "швартовк", "eta")
_EXIT_HINTS = ("выход", "отход", "убыти", "отшвартовк", "etd")
_PLACE_HINTS = ("пункт", "порт", "причал", "место", "назначени")


@dataclass
class ParsedApplication:
    vessel_name: str | None = None
    imo: str | None = None
    gross_tonnage: int | None = None
    net_tonnage: int | None = None
    loa_m: float | None = None
    draft_m: float | None = None
    entry_datetime: datetime | None = None
    exit_datetime: datetime | None = None
    destination: str | None = None
    agent: str | None = None
    tugs_text: str | None = None
    direction: str = "прочее"  # вход / выход / прочее
    sender: str | None = None
    subject: str | None = None
    received_at: datetime | None = None
    raw_text: str = ""
    raw_html: str | None = None
    fields: dict[str, str] = field(default_factory=dict)


def _clean(value: str) -> str:
    return _WS.sub(" ", value).strip()


def _parse_dt(value: str) -> datetime | None:
    """Достать дату/время из текста.

    Поддерживаются формы реальных писем: «20.07.2026 в 04:00», «20.07.26 04-00»,
    «20/07/2026 04.00», «04:00 20.07.2026» и дата без времени (тогда 00:00).
    """
    compact = _WS.sub("", value)
    date_m = re.search(r"(\d{1,2})[.\-/](\d{1,2})[.\-/](\d{2,4})", compact)
    if not date_m:
        return None
    d, mo, y = (int(x) for x in date_m.groups())
    if y < 100:
        y += 2000

    hh = mm = 0
    tail = compact[date_m.end():]
    head = compact[: date_m.start()]
    time_m = re.search(r"(\d{1,2})[:.\-](\d{2})", tail) or re.search(
        r"(\d{1,2})[:.\-](\d{2})", head
    )
    if time_m:
        hh, mm = int(time_m.group(1)), int(time_m.group(2))
    try:
        return datetime(y, mo, d, hh, mm)
    except ValueError:
        return None


def _floats(value: str) -> list[float]:
    compact = _WS.sub("", value).replace(",", ".")
    return [float(x) for x in _NUMBER.findall(compact)]


def _extract_float(value: str) -> float | None:
    """Достать число с плавающей точкой (запятая или точка) из текста, напр. «9,5 м»."""
    numbers = _floats(value)
    return numbers[0] if numbers else None


def _extract_max_float(value: str) -> float | None:
    """Максимальное число из текста: осадка «8,711 m/8,714 m» -> 8.714."""
    numbers = _floats(value)
    return max(numbers) if numbers else None


def normalize_agent(value: str | None, known: Iterable[str] = ()) -> str | None:
    """Привести агента к значению справочника (чтобы сработал select в форме).

    Сравнение без регистра, кавычек, правовой формы и разделителей:
    «ООО "Содружество-Соя"» -> «Содружество - Соя». Если совпадения нет — возвращает
    очищенное исходное значение.
    """
    if not value:
        return None
    cleaned = _clean(value).strip(_QUOTES + " .,;:")
    if not cleaned:
        return None
    key = _agent_key(cleaned)
    for name in known:
        if _agent_key(name) == key:
            return name
    return cleaned


def _agent_key(value: str) -> str:
    text = _clean(value).lower()
    for ch in _QUOTES:
        text = text.replace(ch, " ")
    text = _ORG_PREFIX.sub(" ", text.strip())
    return re.sub(r"[^\w]+", "", text)


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
            elif len(cells) == 1:
                # В части писем метка и значение стоят в одной ячейке: «Агент: ООО …».
                label, value = _split_label_value(cells[0])
                if label and value:
                    result.setdefault(label, value)
    return result


def _split_label_value(text: str) -> tuple[str | None, str | None]:
    if ":" not in text:
        return None, None
    label, _, value = text.partition(":")
    label, value = _clean(label), _clean(value)
    return (label or None), (value or None)


def _is_datetime_label(low: str, hints: tuple[str, ...]) -> bool:
    """Метка описывает момент времени (а не место или пункт назначения)."""
    if not any(hint in low for hint in hints):
        return False
    return not any(place in low for place in _PLACE_HINTS)


def _apply_fields(
    parsed: ParsedApplication,
    fields: dict[str, str],
    known_agents: Iterable[str] = (),
) -> None:
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
        elif "длина" in low or "loa" in low:
            parsed.loa_m = _extract_float(value)
        elif "осадка" in low or "draft" in low:
            # В письмах встречается пара значений «8,711 m/8,714 m» — берём максимум.
            parsed.draft_m = _extract_max_float(value)
        elif "агент" in low or "agent" in low:
            parsed.agent = normalize_agent(value, known_agents)
        elif _is_datetime_label(low, _ENTRY_HINTS):
            parsed.entry_datetime = _parse_dt(value)
        elif _is_datetime_label(low, _EXIT_HINTS):
            parsed.exit_datetime = _parse_dt(value)
        elif "назначения" in low:
            parsed.destination = value


def _fill_from_text(
    parsed: ParsedApplication, text: str, known_agents: Iterable[str] = ()
) -> None:
    """Добрать недостающие поля из плоского текста письма (вне таблиц)."""
    for line in text.splitlines():
        label, value = _split_label_value(line)
        if not label or not value:
            continue
        low = label.lower()
        if parsed.agent is None and ("агент" in low or "agent" in low):
            parsed.agent = normalize_agent(value, known_agents)
        elif parsed.entry_datetime is None and _is_datetime_label(low, _ENTRY_HINTS):
            parsed.entry_datetime = _parse_dt(value)
        elif parsed.exit_datetime is None and _is_datetime_label(low, _EXIT_HINTS):
            parsed.exit_datetime = _parse_dt(value)
        elif parsed.draft_m is None and ("осадка" in low or "draft" in low):
            parsed.draft_m = _extract_max_float(value)


def _detect_direction(text: str, subject: str | None) -> str:
    haystack = f"{subject or ''} {text[:400]}".lower()
    if "на вход" in haystack or re.search(r"\bвход\b", haystack):
        return "вход"
    if "на выход" in haystack or re.search(r"\bвыход\b", haystack):
        return "выход"
    return "прочее"


def parse_eml(
    path: str | Path, *, known_agents: Iterable[str] = ()
) -> ParsedApplication:
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

    known_agents = list(known_agents)
    if html:
        soup = BeautifulSoup(html, "html.parser")
        parsed.raw_html = html
        parsed.raw_text = soup.get_text("\n", strip=True)
        _apply_fields(parsed, fields_from_html(html), known_agents)
    elif text:
        parsed.raw_text = text
        _apply_fields(parsed, fields_from_text(text), known_agents)

    # Часть сведений (агент, даты) в реальных письмах идёт текстом вне таблицы.
    _fill_from_text(parsed, parsed.raw_text, known_agents)
    if parsed.agent is None and parsed.sender:
        parsed.agent = _agent_from_sender(parsed.sender, known_agents)

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
        "Длина наибольшая",
        "Осадка",
        "Дата/время входа",
        "Дата/время выхода",
        "Пункт назначения",
        "Агент",
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


def _agent_from_sender(sender: str, known: Iterable[str]) -> str | None:
    """Сопоставить отображаемое имя отправителя со справочником агентов."""
    key = _agent_key(sender)
    for name in known:
        agent_key = _agent_key(name)
        if agent_key and agent_key in key:
            return name
    return None


def parse_pdf(path: str | Path, *, known_agents: Iterable[str] = ()) -> ParsedApplication:
    """Разобрать текстовый PDF заявки."""
    import pdfplumber

    parsed = ParsedApplication()
    with pdfplumber.open(path) as pdf:
        parsed.raw_text = "\n".join((page.extract_text() or "") for page in pdf.pages)
    _apply_fields(parsed, fields_from_text(parsed.raw_text), known_agents)
    _fill_from_text(parsed, parsed.raw_text, known_agents)
    parsed.direction = _detect_direction(parsed.raw_text, None)
    return parsed


def parse_application(
    path: str | Path, *, known_agents: Iterable[str] = ()
) -> ParsedApplication:
    """Определить тип файла и разобрать заявку."""
    p = Path(path)
    if p.suffix.lower() == ".pdf":
        return parse_pdf(p, known_agents=known_agents)
    # .eml или файл без расширения, начинающийся с email-заголовков.
    return parse_eml(p, known_agents=known_agents)
