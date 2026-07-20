"""Расчёт стоимости выполненной работы.

Логика построена на константах из app.config (ПЛЕЙСХОЛДЕРЫ — замени на реальные).
Функции чистые и принимают провайдеры курса/календаря аргументами — удобно тестировать.

Проверено на примерах matching.xls:
  * почасовая тарификация пропорциональна минутам: rate * (минуты / 60);
  * выручка в рублях = сумма * курс ЦБ на дату завершения работ;
  * для рублёвых договоров курс = 1.0.
"""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from datetime import date, datetime

from app.config import (
    RATES,
    RATES_PER_OPERATION,
    WORK_TYPE_ALIASES,
    settings,
)
from app.services.calendar_ru import is_day_off
from app.services.cbr import get_cbr_rate

FxProvider = Callable[[str, date], float]
DayOffProvider = Callable[[date], bool]


@dataclass
class CalcResult:
    amount: float
    currency: str
    cbr_rate: float
    revenue_rub: float
    calc_note: str
    work_minutes: int
    busy_minutes: int


def normalize_work_type(raw: str | None) -> str | None:
    """Привести вид работ из документа к нормализованному ключу тарифа."""
    if not raw:
        return None
    text = raw.strip().lower()
    if text in WORK_TYPE_ALIASES:
        return WORK_TYPE_ALIASES[text]
    for alias, canonical in WORK_TYPE_ALIASES.items():
        if alias in text:
            return canonical
    return text


def tug_count_from_joint(joint_with: str | None) -> int:
    """Число буксиров, работавших совместно, из строки ваучера «совместно с …».

    Пусто → 1 буксир (без деления). 1 упомянутый буксир → 2 (делим на 2),
    2 буксира → 3 (делим на 3) и т.д. — так стоимость за тонну делится поровну.
    """
    if not joint_with or not joint_with.strip():
        return 1
    parts = re.split(r"[,;/]|\bи\b|\+", joint_with)
    others = [p for p in (x.strip() for x in parts) if p]
    return len(others) + 1


def minutes_between(start: datetime | None, end: datetime | None) -> int:
    if start is None or end is None:
        return 0
    return max(0, int((end - start).total_seconds() // 60))


def format_hm(total_minutes: int) -> str:
    return f"{total_minutes // 60:02d}:{total_minutes % 60:02d}"


def _is_night(dt: datetime) -> bool:
    h = dt.hour
    return h >= settings.night_start_hour or h < settings.night_end_hour


def calculate(
    *,
    agent: str,
    work_type: str | None,
    gross_tonnage: int | None,
    started_dt: datetime | None,
    finished_dt: datetime | None,
    left_base_dt: datetime | None = None,
    arrived_base_dt: datetime | None = None,
    is_ice: bool = False,
    escort_hours: float | None = None,
    tug_count: int = 1,
    fx_provider: FxProvider = get_cbr_rate,
    dayoff_provider: DayOffProvider = is_day_off,
) -> CalcResult:
    """Рассчитать сумму, курс и выручку в рублях для одной работы."""
    canonical = normalize_work_type(work_type)
    if canonical is None:
        raise ValueError("Не указан вид работ")

    work_minutes = minutes_between(started_dt, finished_dt)
    busy_minutes = minutes_between(left_base_dt, arrived_base_dt)
    work_hours = work_minutes / 60.0

    ref_dt = started_dt or finished_dt
    if ref_dt is None:
        raise ValueError("Не указаны даты работ")
    night_or_holiday = _is_night(ref_dt) or dayoff_provider(ref_dt.date())

    small_ship = (
        gross_tonnage is not None
        and gross_tonnage < settings.gross_tonnage_threshold
    )

    per_op_key = (agent, canonical)
    use_per_operation = small_ship and per_op_key in RATES_PER_OPERATION

    if use_per_operation:
        amount, currency, note = _calc_per_operation(
            per_op_key, is_ice=is_ice, night_or_holiday=night_or_holiday
        )
    else:
        amount, currency, note = _calc_by_rule(
            agent=agent,
            canonical=canonical,
            gross_tonnage=gross_tonnage,
            work_minutes=work_minutes,
            work_hours=work_hours,
            is_ice=is_ice,
            tug_count=tug_count,
        )

    # Доп. компонент «Сопровождение», если указаны часы сопровождения.
    if escort_hours and escort_hours > 0:
        escort_rule = RATES.get(agent, {}).get("сопровождение")
        if escort_rule:
            escort_rate = escort_rule["rate"]
            escort_amount = escort_rate * escort_hours
            amount += escort_amount
            note += (
                f"; Сопровождение: {format_hm(int(escort_hours * 60))} x "
                f"{escort_rate:.2f} = {escort_amount:.2f}"
            )

    amount = round(amount, 2)

    fx_date = (finished_dt or ref_dt).date()
    cbr_rate = 1.0 if currency.upper() == "RUB" else fx_provider(currency, fx_date)
    revenue_rub = round(amount * cbr_rate, 2)

    return CalcResult(
        amount=amount,
        currency=currency,
        cbr_rate=round(cbr_rate, 4),
        revenue_rub=revenue_rub,
        calc_note=note,
        work_minutes=work_minutes,
        busy_minutes=busy_minutes,
    )


def _calc_per_operation(
    key: tuple[str, str], *, is_ice: bool, night_or_holiday: bool
) -> tuple[float, str, str]:
    rule = RATES_PER_OPERATION[key]
    currency = rule["currency"]
    if night_or_holiday:
        field = "holiday_night_ice" if is_ice else "holiday_night"
        label = "праздник/ночь" + (" (лёд)" if is_ice else "")
    else:
        field = "weekday_ice" if is_ice else "weekday"
        label = "будни" + (" (лёд)" if is_ice else "")
    amount = float(rule[field])
    note = f"{key[1]} <2000 GRT, {label}: {amount:.2f} {currency} за операцию"
    return amount, currency, note


def _calc_by_rule(
    *,
    agent: str,
    canonical: str,
    gross_tonnage: int | None,
    work_minutes: int,
    work_hours: float,
    is_ice: bool,
    tug_count: int = 1,
) -> tuple[float, str, str]:
    agent_rates = RATES.get(agent)
    if agent_rates is None:
        raise ValueError(f"Нет тарифов для агента: {agent}")
    rule = agent_rates.get(canonical)
    if rule is None:
        raise ValueError(f"Нет тарифа для вида работ '{canonical}' у агента '{agent}'")

    currency = rule["currency"]
    rate = float(rule["rate"])
    if is_ice and rule.get("rate_ice"):
        rate = float(rule["rate_ice"])
    ice_label = " (лёд)" if is_ice else ""

    if rule["unit"] == "per_ton":
        if gross_tonnage is None:
            raise ValueError("Для тарифа за тонну нужен GRT из заявки")
        # Стоимость за тонну делится на число буксиров, работавших совместно.
        divisor = tug_count if tug_count and tug_count > 0 else 1
        amount = gross_tonnage * rate / divisor
        divisor_note = f" / {divisor} букс." if divisor != 1 else ""
        note = f"{gross_tonnage} x {rate:.2f} {currency}{divisor_note}{ice_label} = {amount:.2f}"
    elif rule["unit"] == "per_hour":
        amount = rate * work_hours
        note = (
            f"{rate:.2f} {currency} x {format_hm(work_minutes)}{ice_label} = {amount:.2f}"
        )
    else:
        raise ValueError(f"Неизвестная единица тарификации: {rule['unit']}")

    return amount, currency, note
