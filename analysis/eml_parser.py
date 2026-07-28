"""Parse vessel-call request emails and their HTML vessel-information tables."""

from __future__ import annotations

import argparse
import csv
import email
import email.policy
import re
from collections.abc import Iterable
from pathlib import Path

from bs4 import BeautifulSoup

FIELDS = [
    "path",
    "subject",
    "from",
    "date",
    "direction",
    "vessel",
    "date_raw",
    "time_raw",
    "berth",
    "agent",
    "vessel_name",
    "imo",
    "flag",
    "loa_m",
    "beam_m",
    "draft_fore_m",
    "draft_aft_m",
    "grt",
    "nrt",
    "port_from",
    "port_to",
    "cargo",
    "purpose",
    "isps_cert",
    "security_level",
    "restrictions",
    "contacts",
]


def iter_files(path: Path) -> Iterable[Path]:
    if path.is_file():
        yield path
    elif path.is_dir():
        yield from sorted(item for item in path.rglob("*") if item.is_file())


def parse_message(path: Path) -> dict[str, str]:
    message = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    subject = str(message.get("Subject", "") or "").strip()
    sender = str(message.get("From", "") or "").strip()
    result = {
        "path": str(path),
        "subject": subject,
        "from": sender,
        "date": str(message.get("Date", "") or "").strip(),
    }
    result.update(parse_subject(subject))
    body_part = message.get_body(preferencelist=("html",))
    body = body_part.get_content() if body_part else ""
    result.update(parse_html_body(body))
    result["agent"] = infer_agent(body, sender)
    return result


def is_email_file(path: Path) -> bool:
    try:
        message = email.message_from_bytes(path.read_bytes(), policy=email.policy.default)
    except (OSError, ValueError):
        return False
    return bool(message.get("Subject") or message.get("From"))


def parse_subject(subject: str) -> dict[str, str]:
    direction = ""
    if re.search(r"перешвартовк", subject, re.IGNORECASE):
        direction = "Перешвартовка"
    elif re.search(r"\bвход", subject, re.IGNORECASE):
        direction = "Вход"
    elif re.search(r"\bвыход", subject, re.IGNORECASE):
        direction = "Выход"

    date_match = re.search(r"(?<!\d)(\d{1,2}\.\d{1,2})(?!\d)", subject)
    date_raw = date_match.group(1) if date_match else ""
    time_match = re.search(r"(?<!\d)(\d{1,2})[_:](\d{2})(?!\d)", subject)
    time_raw = f"{time_match.group(1)}:{time_match.group(2)}" if time_match else ""
    berths = re.findall(r"(?i)ТСС\s*№?\s*\d+", subject)
    berth = " - ".join(berths)

    vessel = ""
    if date_match:
        before_date = subject[: date_match.start()]
        vessel_match = re.search(r"(?i)(?:тх|т/х)\s+(.+?)\s*$", before_date)
        if vessel_match:
            vessel = re.sub(r"\s+", " ", vessel_match.group(1)).strip(" -_")
    if not vessel:
        vessel_match = re.search(r"([A-ZА-ЯЁ]{2,}(?:[\s-]+[A-ZА-ЯЁ0-9]+)*)", subject)
        if vessel_match:
            vessel = re.sub(r"\s+", " ", vessel_match.group(1)).strip()
    vessel = re.sub(r"\s+\d+$", "", vessel).strip()
    return {
        "direction": direction,
        "vessel": vessel,
        "date_raw": date_raw,
        "time_raw": time_raw,
        "berth": berth,
    }


def clean_text(value: str) -> str:
    return re.sub(r"\s+", " ", value.replace("\xa0", " ")).strip()


def normalized_label(value: str) -> str:
    return re.sub(r"[^a-zа-яё]", "", clean_text(value).lower())


def normalize_number(value: str) -> str:
    value = clean_text(value)
    match = re.search(r"[-+]?\d[\d\s]*(?:[,.]\s*\d+)?", value)
    if not match:
        return ""
    number = re.sub(r"\s+", "", match.group(0)).replace(",", ".")
    return number


def split_pair(value: str) -> tuple[str, str]:
    values = re.findall(r"[-+]?\d[\d\s]*(?:[,.]\s*\d+)?", value)
    values = [re.sub(r"\s+", "", item).replace(",", ".") for item in values]
    return (values + ["", ""])[:2]


def table_rows(body: str) -> list[tuple[str, str]]:
    soup = BeautifulSoup(body, "html.parser")
    rows: list[tuple[str, str]] = []
    for row in soup.find_all("tr"):
        cells = [clean_text(cell.get_text(" ", strip=True)) for cell in row.find_all(["td", "th"])]
        if len(cells) >= 3 and cells[1] and cells[2]:
            rows.append((cells[1], cells[2]))
    return rows


def find_value(rows: list[tuple[str, str]], *needles: str) -> str:
    for label, value in rows:
        normalized = normalized_label(label)
        if any(needle in normalized for needle in needles):
            return value
    return ""


def parse_html_body(body: str) -> dict[str, str]:
    rows = table_rows(body)
    length_width = find_value(rows, "длинаширина")
    draft = find_value(rows, "осадканосомкормой", "осадканосомкормойвпреснойводе")
    gross_net = find_value(rows, "бруттонетто")
    loa, beam = split_pair(length_width)
    draft_fore, draft_aft = split_pair(draft)
    grt, nrt = split_pair(gross_net)
    return {
        "vessel_name": find_value(rows, "названиесудна"),
        "imo": normalize_number(find_value(rows, "имо")),
        "flag": find_value(rows, "флаг"),
        "loa_m": loa,
        "beam_m": beam,
        "draft_fore_m": draft_fore,
        "draft_aft_m": draft_aft,
        "grt": grt,
        "nrt": nrt,
        "port_from": find_value(rows, "пунктвыхода"),
        "port_to": find_value(rows, "пунктназначения"),
        "cargo": find_value(rows, "грузколичество"),
        "purpose": find_value(rows, "цельперешвартовки"),
        "isps_cert": find_value(rows, "международногосвидетельстваоспс", "свидетельстваоспс"),
        "security_level": find_value(rows, "уровняохраны"),
        "restrictions": find_value(rows, "ограничения"),
        "contacts": find_value(rows, "контактныетелефоны"),
    }


def infer_agent(body: str, sender: str) -> str:
    if "транс-агро" in body.lower():
        return "Транс-Агро"
    domain = re.search(r"@([A-Za-z0-9.-]+)", sender)
    return domain.group(1) if domain else ""


def write_csv(records: list[dict[str, str]], output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=FIELDS)
        writer.writeheader()
        writer.writerows(records)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("path", type=Path, help="Письмо или папка с письмами")
    parser.add_argument("-o", "--output", type=Path, help="Путь для CSV")
    args = parser.parse_args()
    records = [parse_message(path) for path in iter_files(args.path) if is_email_file(path)]
    if args.output:
        write_csv(records, args.output)
    print(f"Parsed emails: {len(records)}")
    for record in records:
        keys = (
            "path",
            "subject",
            "direction",
            "vessel",
            "agent",
            "vessel_name",
            "imo",
            "loa_m",
            "grt",
        )
        print({key: record[key] for key in keys})


if __name__ == "__main__":
    main()
