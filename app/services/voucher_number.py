"""Предсказание следующего номера ваучера и разбор имени файла ваучера.

Имя файла ваучера — первичный источник полей (OCR ненадёжен): цифры —
номер ваучера, литера — код буксира («p» — ПИОНЕР, «k» — КОММУНАР).
Например, файл «323p.pdf» означает ваучер № 323 буксира ПИОНЕР.
"""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Tug, Voucher

_VOUCHER_NAME_RE = re.compile(
    r"^(\d+)([kp])?(?:\(\d+\)|_\d+)?(?:\.[^.]+)?$",
    re.IGNORECASE,
)

# Префикс вида «82c16a358e69653b_», который store_upload добавляет к имени файла.
_SHA_PREFIX_RE = re.compile(r"^[0-9a-f]{16}_", re.IGNORECASE)


def parse_voucher_number(name: str) -> tuple[int | None, str | None]:
    """Извлечь номер ваучера и код буксира из имени файла."""
    match = _VOUCHER_NAME_RE.fullmatch(Path(name).name.strip())
    if match is None:
        return None, None
    return int(match.group(1)), match.group(2).lower() if match.group(2) else None


def voucher_filename_fields(voucher: Voucher) -> tuple[int | None, str | None]:
    """Номер ваучера и код буксира (k/p) из имени файла ваучера.

    Проверяются и оригинальное имя, и сохранённое (с sha-префиксом).
    """
    for raw_name in (voucher.original_filename, voucher.file_path):
        if not raw_name:
            continue
        name = Path(raw_name).name.strip()
        while name:
            number, code = parse_voucher_number(name)
            if number is not None:
                return number, code
            stripped = _SHA_PREFIX_RE.sub("", name)
            if stripped == name:
                break
            name = stripped
    return None, None


def apply_filename_fields(db: Session, voucher: Voucher) -> None:
    """Заполнить номер ваучера и буксира из имени файла, если они не заданы."""
    number, code = voucher_filename_fields(voucher)
    if number is not None and not voucher.number:
        voucher.number = str(number)
    if code and voucher.tug_id is None:
        tug = db.scalar(select(Tug).where(Tug.code == code))
        if tug is not None:
            voucher.tug_id = tug.id


def predict_next(db: Session, tug_id: int, year: int) -> int:
    """Вернуть следующий номер ваучера для буксира в указанном году."""
    max_number = 0
    vouchers = db.scalars(select(Voucher).where(Voucher.tug_id == tug_id))
    for voucher in vouchers:
        voucher_date = voucher.started_dt or voucher.left_base_dt
        if voucher_date is None or voucher_date.year != year:
            continue
        number, _ = parse_voucher_number(voucher.number or "")
        if number is not None:
            max_number = max(max_number, number)
    return max_number + 1
