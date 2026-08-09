"""Расчёт стоимости работ по прайсу группы A."""

from __future__ import annotations

import re
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from datetime import date, datetime

from app.config import (
    WORK_TYPE_ALIASES,
    TariffsGroupA,
    agent_group,
    settings,
    tariffs_a,
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
    """Число буксиров, работавших совместно, из строки ваучера «совместно с …»."""
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
    tariffs: TariffsGroupA | None = None,
) -> CalcResult:
    """Рассчитать сумму, курс и выручку в рублях для одной работы."""
    group = agent_group(agent)
    if group != "A":
        raise NotImplementedError(f"Тариф группы {group} ещё не задан (только группа A)")

    tariffs = tariffs or tariffs_a
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
    currency = settings.ue_currency
    period = "праздник/ночь" if night_or_holiday else "будни"
    ice_label = ", лёд" if is_ice else ""

    if canonical in {"швартовка", "отшвартовка"}:
        if gross_tonnage is None:
            raise ValueError("Для тарифа швартовки нужен GRT из заявки")
        if small_ship:
            rate = _small_ship_rate(
                tariffs,
                "mooring_unmooring_gt_below_2000",
                is_ice=is_ice,
                night_or_holiday=night_or_holiday,
            )
            amount = rate
            note = (
                f"{canonical} <2000 GRT, {period}{ice_label}: "
                f"{rate:.2f} {currency} за операцию"
            )
        else:
            rate = (
                tariffs.mooring_unmooring_gt_2000_above_ice
                if is_ice
                else tariffs.mooring_unmooring_gt_2000_above
            )
            divisor = tug_count if tug_count > 0 else 1
            amount = gross_tonnage * rate / divisor
            divisor_note = f" / {divisor} букс." if divisor != 1 else ""
            note = (
                f"{canonical}: {gross_tonnage} т x {rate:.2f} {currency}"
                f"{divisor_note}{ice_label} = {amount:.2f}"
            )
    elif canonical == "перестановка":
        if gross_tonnage is None:
            raise ValueError("Для тарифа перестановки нужен GRT из заявки")
        if small_ship:
            rate = _small_ship_rate(
                tariffs,
                "vessel_repositioning_gt_below_2000",
                is_ice=is_ice,
                night_or_holiday=night_or_holiday,
            )
            amount = rate
            note = (
                f"{canonical} <2000 GRT, {period}{ice_label}: "
                f"{rate:.2f} {currency} за операцию"
            )
        else:
            rate = (
                tariffs.vessel_repositioning_gt_2000_above_ice
                if is_ice
                else tariffs.vessel_repositioning_gt_2000_above
            )
            amount = rate * work_hours
            note = (
                f"{canonical}: {rate:.2f} {currency}/ч x "
                f"{format_hm(work_minutes)}{ice_label} = {amount:.2f}"
            )
    elif canonical == "сопровождение":
        if small_ship and is_ice:
            rate = _small_ship_rate(
                tariffs,
                "escort_towing_gt_below_2000",
                is_ice=True,
                night_or_holiday=night_or_holiday,
            )
            source = "сопровождение буксировкой <2000 GRT"
        else:
            rate = (
                tariffs.escort_vessel_meeting_departure_cargo_canal_ice
                if is_ice
                else tariffs.escort_vessel_meeting_departure_cargo_canal
            )
            source = "сопровождение по каналу"
        amount = rate * work_hours
        note = (
            f"{source}: {rate:.2f} {currency}/ч x "
            f"{format_hm(work_minutes)}{ice_label} = {amount:.2f}"
        )
    elif canonical == "буксировка баржи":
        rate = tariffs.barge_towing_canal_ice if is_ice else tariffs.barge_towing_canal
        amount = rate * work_hours
        note = (
            f"{canonical}: {rate:.2f} {currency}/ч x "
            f"{format_hm(work_minutes)}{ice_label} = {amount:.2f}"
        )
    elif canonical == "околка льда":
        rate = tariffs.ice_breaking_tug_vessel_approach_departure
        amount = rate * work_hours
        note = (
            f"{canonical}: {rate:.2f} {currency}/ч x "
            f"{format_hm(work_minutes)} = {amount:.2f}"
        )
    elif canonical == "обслуживание судна":
        rate = tariffs.vessel_maintenance_services
        amount = rate * work_hours
        note = (
            f"{canonical}: {rate:.2f} {currency}/ч x "
            f"{format_hm(work_minutes)} = {amount:.2f}"
        )
    elif canonical == "обслуживание морских сооружений":
        rate = tariffs.offshore_facilities_maintenance_services
        amount = rate * work_hours
        note = (
            f"{canonical}: {rate:.2f} {currency}/ч x "
            f"{format_hm(work_minutes)} = {amount:.2f}"
        )
    else:
        raise ValueError(f"Нет тарифа для вида работ '{canonical}' (группа A)")

    # escort_hours сохраняется для совместимости, но больше не добавляет компонент.
    del escort_hours
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


@dataclass
class ParticipantInput:
    """Фактический участник операции (наш буксир либо сторонний подрядчик)."""

    name: str = "буксир"
    is_external: bool = False
    work_start: datetime | None = None
    work_end: datetime | None = None
    escort: bool = False


def calculate_participants(
    *,
    agent: str,
    work_type: str | None,
    gross_tonnage: int | None,
    participants: Sequence[ParticipantInput],
    started_dt: datetime | None = None,
    finished_dt: datetime | None = None,
    participants_total: int | None = None,
    is_ice: bool = False,
    fx_provider: FxProvider = get_cbr_rate,
    dayoff_provider: DayOffProvider = is_day_off,
    tariffs: TariffsGroupA | None = None,
) -> CalcResult:
    """Рассчитать операцию с учётом фактических участников.

    Правила: делитель швартовки/отшвартовки (GRT>=2000) — все фактические участники,
    включая сторонних; тарифицируются только наши буксиры; перестановка считается
    отдельно по каждому нашему буксиру (его собственные времена) и суммируется.
    """
    canonical = normalize_work_type(work_type)
    own = [p for p in participants if not p.is_external]
    total = participants_total if participants_total is not None else len(participants)
    if total <= 0:
        total = 1

    common = {
        "agent": agent,
        "work_type": work_type,
        "gross_tonnage": gross_tonnage,
        "is_ice": is_ice,
        "fx_provider": fx_provider,
        "dayoff_provider": dayoff_provider,
        "tariffs": tariffs,
    }

    if canonical == "перестановка" and own:
        results = [
            calculate(
                started_dt=p.work_start or started_dt,
                finished_dt=p.work_end or finished_dt,
                tug_count=1,
                **common,  # type: ignore[arg-type]
            )
            for p in own
        ]
        amount = round(sum(r.amount for r in results), 2)
        note = "; ".join(
            f"{p.name}: {r.calc_note}" for p, r in zip(own, results, strict=True)
        )
        if len(own) != len(participants):
            note += f" (участников всего {total}, тарифицируются {len(own)})"
        base = results[0]
        return CalcResult(
            amount=amount,
            currency=base.currency,
            cbr_rate=base.cbr_rate,
            revenue_rub=round(amount * base.cbr_rate, 2),
            calc_note=note,
            work_minutes=max(r.work_minutes for r in results),
            busy_minutes=max(r.busy_minutes for r in results),
        )

    base = calculate(
        started_dt=started_dt,
        finished_dt=finished_dt,
        tug_count=total,
        **common,  # type: ignore[arg-type]
    )
    if not participants:
        return base

    # Ставка за тонну делится на всех фактических участников, а тарифицируются
    # только наши буксиры: сумма операции — их доли.
    per_ton = canonical in {"швартовка", "отшвартовка"} and (
        gross_tonnage is not None and gross_tonnage >= settings.gross_tonnage_threshold
    )
    if not per_ton:
        if len(own) == len(participants):
            return base
        return CalcResult(
            amount=base.amount,
            currency=base.currency,
            cbr_rate=base.cbr_rate,
            revenue_rub=base.revenue_rub,
            calc_note=f"{base.calc_note} (участников всего {total}, наших {len(own)})",
            work_minutes=base.work_minutes,
            busy_minutes=base.busy_minutes,
        )

    amount = round(base.amount * len(own), 2)
    note = f"{base.calc_note} x {len(own)} наших букс. из {total} = {amount:.2f}"
    return CalcResult(
        amount=amount,
        currency=base.currency,
        cbr_rate=base.cbr_rate,
        revenue_rub=round(amount * base.cbr_rate, 2),
        calc_note=note,
        work_minutes=base.work_minutes,
        busy_minutes=base.busy_minutes,
    )


def _small_ship_rate(
    tariffs: TariffsGroupA,
    prefix: str,
    *,
    is_ice: bool,
    night_or_holiday: bool,
) -> float:
    hn = night_or_holiday
    if prefix == "mooring_unmooring_gt_below_2000":
        if is_ice:
            return (
                tariffs.mooring_unmooring_gt_below_2000_ice_holiday_night
                if hn
                else tariffs.mooring_unmooring_gt_below_2000_ice_weekdays
            )
        return (
            tariffs.mooring_unmooring_gt_below_2000_holiday_night
            if hn
            else tariffs.mooring_unmooring_gt_below_2000_weekdays
        )
    if prefix == "vessel_repositioning_gt_below_2000":
        if is_ice:
            return (
                tariffs.vessel_repositioning_gt_below_2000_ice_holiday_night
                if hn
                else tariffs.vessel_repositioning_gt_below_2000_ice_weekdays
            )
        return (
            tariffs.vessel_repositioning_gt_below_2000_holiday_night
            if hn
            else tariffs.vessel_repositioning_gt_below_2000_weekdays
        )
    if prefix == "escort_towing_gt_below_2000":
        # В прайсе группы A только ледовые варианты этой услуги.
        return (
            tariffs.escort_towing_gt_below_2000_ice_holiday_night
            if hn
            else tariffs.escort_towing_gt_below_2000_ice_weekdays
        )
    raise ValueError(f"Неизвестный префикс тарифа: {prefix}")
