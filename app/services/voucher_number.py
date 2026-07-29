"""Предсказание следующего номера ваучера."""

from __future__ import annotations

import re
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Voucher

_VOUCHER_NAME_RE = re.compile(
    r"^(\d+)([kp])?(?:\(\d+\))?(?:\.[^.]+)?$",
    re.IGNORECASE,
)


def parse_voucher_number(name: str) -> tuple[int | None, str | None]:
    """Извлечь номер ваучера и код буксира из имени файла."""
    match = _VOUCHER_NAME_RE.fullmatch(Path(name).name.strip())
    if match is None:
        return None, None
    return int(match.group(1)), match.group(2).lower() if match.group(2) else None


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
