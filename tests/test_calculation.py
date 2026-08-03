"""Тесты расчётного модуля (без сети: тарифы и провайдеры подменяются)."""

from __future__ import annotations

from datetime import date, datetime

import pytest

from app.config import TariffsGroupA
from app.services.calculation import (
    calculate,
    format_hm,
    minutes_between,
    tug_count_from_joint,
)


def _fx(_currency: str, _on: date) -> float:
    return 2.5


def _not_dayoff(_d: date) -> bool:
    return False


def _tariffs() -> TariffsGroupA:
    return TariffsGroupA(
        escort_vessel_meeting_departure_cargo_canal=30,
        escort_vessel_meeting_departure_cargo_canal_ice=31,
        barge_towing_canal=50,
        barge_towing_canal_ice=51,
        mooring_unmooring_gt_2000_above=2,
        mooring_unmooring_gt_2000_above_ice=3,
        vessel_repositioning_gt_2000_above=120,
        vessel_repositioning_gt_2000_above_ice=130,
        mooring_unmooring_gt_below_2000_weekdays=10,
        mooring_unmooring_gt_below_2000_holiday_night=11,
        mooring_unmooring_gt_below_2000_ice_weekdays=12,
        mooring_unmooring_gt_below_2000_ice_holiday_night=13,
        vessel_repositioning_gt_below_2000_weekdays=20,
        vessel_repositioning_gt_below_2000_holiday_night=21,
        vessel_repositioning_gt_below_2000_ice_weekdays=22,
        vessel_repositioning_gt_below_2000_ice_holiday_night=23,
        escort_towing_gt_below_2000_ice_weekdays=40,
        escort_towing_gt_below_2000_ice_holiday_night=41,
        ice_breaking_tug_vessel_approach_departure=60,
        vessel_maintenance_services=70,
        offshore_facilities_maintenance_services=80,
    )


def test_per_ton_mooring_divides_by_tug_count() -> None:
    res = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=8446,
        started_dt=datetime(2026, 7, 20, 6, 0),
        finished_dt=datetime(2026, 7, 20, 6, 50),
        tug_count=2,
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    assert res.amount == 8446.0
    assert res.currency == "USD"
    assert "/ 2 букс." in res.calc_note


def test_per_hour_is_proportional_to_minutes() -> None:
    res = calculate(
        agent="Транс-Агро",
        work_type="перестановка",
        gross_tonnage=2500,
        started_dt=datetime(2026, 7, 20, 10, 0),
        finished_dt=datetime(2026, 7, 20, 11, 30),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    assert res.amount == 180.0
    assert res.work_minutes == 90


def test_small_ship_operation_rates_use_period_and_ice() -> None:
    weekday = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=1500,
        started_dt=datetime(2026, 7, 20, 10, 0),
        finished_dt=datetime(2026, 7, 20, 11, 0),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    holiday_night = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=1500,
        started_dt=datetime(2026, 7, 20, 23, 0),
        finished_dt=datetime(2026, 7, 21, 0, 0),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    ice_weekday = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=1500,
        started_dt=datetime(2026, 7, 20, 10, 0),
        finished_dt=datetime(2026, 7, 20, 11, 0),
        is_ice=True,
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    ice_holiday_night = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=1500,
        started_dt=datetime(2026, 7, 20, 23, 0),
        finished_dt=datetime(2026, 7, 21, 0, 0),
        is_ice=True,
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    assert [r.amount for r in (
        weekday,
        holiday_night,
        ice_weekday,
        ice_holiday_night,
    )] == [10.0, 11.0, 12.0, 13.0]


def test_revenue_rub_uses_fx_and_currency_is_usd() -> None:
    res = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=2000,
        started_dt=datetime(2026, 7, 20, 10, 0),
        finished_dt=datetime(2026, 7, 20, 11, 0),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    assert res.currency == "USD"
    assert res.cbr_rate == 2.5
    assert res.revenue_rub == 10000.0


def test_escort_rates_are_selected_correctly() -> None:
    channel = calculate(
        agent="Транс-Агро",
        work_type="сопровождение",
        gross_tonnage=2500,
        started_dt=datetime(2026, 7, 20, 10, 0),
        finished_dt=datetime(2026, 7, 20, 12, 0),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    small_ice_night = calculate(
        agent="Транс-Агро",
        work_type="сопровождение",
        gross_tonnage=1500,
        started_dt=datetime(2026, 7, 20, 23, 0),
        finished_dt=datetime(2026, 7, 21, 0, 0),
        is_ice=True,
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    assert channel.amount == 60.0
    assert small_ice_night.amount == 41.0


def test_escort_hours_is_ignored_as_separate_component() -> None:
    res = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=2500,
        started_dt=datetime(2026, 7, 20, 10, 0),
        finished_dt=datetime(2026, 7, 20, 11, 0),
        escort_hours=10,
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    assert res.amount == 5000.0


def test_group_b_is_not_implemented() -> None:
    with pytest.raises(NotImplementedError, match="Тариф группы B"):
        calculate(
            agent="Терминал",
            work_type="швартовка",
            gross_tonnage=2500,
            started_dt=datetime(2026, 7, 20, 10, 0),
            finished_dt=datetime(2026, 7, 20, 11, 0),
            tariffs=_tariffs(),
        )


def test_barge_towing_alias_and_unknown_generic_towing() -> None:
    res = calculate(
        agent="Транс-Агро",
        work_type="буксировка баржи",
        gross_tonnage=None,
        started_dt=datetime(2026, 7, 20, 10, 0),
        finished_dt=datetime(2026, 7, 20, 11, 0),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
        tariffs=_tariffs(),
    )
    assert res.amount == 50.0
    with pytest.raises(ValueError, match="Нет тарифа"):
        calculate(
            agent="Транс-Агро",
            work_type="буксировка",
            gross_tonnage=None,
            started_dt=datetime(2026, 7, 20, 10, 0),
            finished_dt=datetime(2026, 7, 20, 11, 0),
            tariffs=_tariffs(),
        )


def test_tug_count_from_joint() -> None:
    assert tug_count_from_joint(None) == 1
    assert tug_count_from_joint("") == 1
    assert tug_count_from_joint("БК Пионер") == 2
    assert tug_count_from_joint("БК Пионер, БК Коммунар") == 3
    assert tug_count_from_joint("Пионер и Коммунар") == 3


def test_helpers() -> None:
    assert format_hm(70) == "01:10"
    assert minutes_between(
        datetime(2026, 1, 1, 8, 0), datetime(2026, 1, 1, 11, 50)
    ) == 230
