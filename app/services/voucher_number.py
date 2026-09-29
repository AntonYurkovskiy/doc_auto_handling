"""Предсказание следующего номера ваучера."""

from __future__ import annotations

import re
from dataclasses import dataclass
from pathlib import Path

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Voucher

# Кириллические двойники латинских букв в именах файлов: «к», «р», «а».
_HOMOGLYPHS = str.maketrans({"к": "k", "К": "k", "р": "p", "Р": "p", "а": "a", "А": "a"})

# Номер, необязательная «a» (дубль номера), код буксира (k/p/l/s, в части имён удвоен:
# 397kk, 410pp), маркер копии «(N)» и хвост вроде «_» или «_кор».
_VOUCHER_NAME_RE = re.compile(
    r"^(?P<number>\d+)"
    r"(?P<dup>a)?"
    r"(?:(?P<code>[kpls])(?P=code)?)?"
    r"(?:\((?P<copy>\d+)\))?"
    r"(?:[ _][^.]*)?"
    r"(?:\.[^.]+)?$",
    re.IGNORECASE,
)


@dataclass(frozen=True)
class VoucherName:
    """Разобранное имя файла ваучера."""

    number: int
    tug_code: str | None
    duplicate: bool = False
    copy: int | None = None


def parse_voucher_name(name: str) -> VoucherName | None:
    """Разобрать имя файла ваучера.

    «12ap» — второй ваучер с тем же номером 12 на Пионере (номер задублировали);
    «262k(2)» — копия того же ваучера. В обоих случаях номер — 12 и 262.
    """
    text = Path(name.strip()).name.strip().translate(_HOMOGLYPHS)
    match = _VOUCHER_NAME_RE.fullmatch(text)
    if match is None:
        return None
    code = match.group("code")
    copy = match.group("copy")
    return VoucherName(
        number=int(match.group("number")),
        tug_code=code.lower() if code else None,
        duplicate=match.group("dup") is not None,
        copy=int(copy) if copy else None,
    )


def parse_voucher_number(name: str) -> tuple[int | None, str | None]:
    """Извлечь номер ваучера и код буксира из имени файла."""
    parsed = parse_voucher_name(name)
    if parsed is None:
        return None, None
    return parsed.number, parsed.tug_code


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
