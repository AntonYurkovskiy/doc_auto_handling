"""Тесты расчётного модуля (без сети: провайдеры курса/календаря подменяются)."""

from __future__ import annotations

from datetime import date, datetime

from app.services.calculation import calculate, format_hm, minutes_between


def _fx(_currency: str, _on: date) -> float:
    return 78.3987  # фиксированный курс USD для теста


def _not_dayoff(_d: date) -> bool:
    return False


def test_per_hour_is_proportional_to_minutes():
    # 590 USD/час * 70 мин / 60 = 688.33 (проверено на примере из matching.xls)
    res = calculate(
        agent="Транс-Агро",
        work_type="обслуживание морских сооружений",
        gross_tonnage=None,
        started_dt=datetime(2026, 7, 17, 10, 10),
        finished_dt=datetime(2026, 7, 17, 11, 20),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
    )
    assert res.amount == 688.33
    assert res.currency == "USD"
    assert res.work_minutes == 70


def test_per_ton_mooring():
    res = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=8446,
        started_dt=datetime(2026, 7, 20, 6, 0),
        finished_dt=datetime(2026, 7, 20, 6, 50),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
    )
    assert res.amount == 4223.00
    assert res.currency == "USD"


def test_revenue_rub_uses_fx():
    res = calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=8446,
        started_dt=datetime(2026, 7, 20, 6, 0),
        finished_dt=datetime(2026, 7, 20, 6, 50),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
    )
    assert res.cbr_rate == 78.3987
    assert res.revenue_rub == round(4223.00 * 78.3987, 2)


def test_rub_contract_rate_is_one():
    res = calculate(
        agent="МореСервис",
        work_type="отшвартовка",
        gross_tonnage=None,
        started_dt=datetime(2026, 7, 16, 8, 0),
        finished_dt=datetime(2026, 7, 16, 11, 50),
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
    )
    # 64000 * 230/60 = 245333.33
    assert res.currency == "RUB"
    assert res.cbr_rate == 1.0
    assert res.amount == 245333.33
    assert res.revenue_rub == 245333.33


def test_escort_component_added():
    res = calculate(
        agent="Транс-Агро",
        work_type="отшвартовка",
        gross_tonnage=24199,
        started_dt=datetime(2026, 7, 15, 17, 50),
        finished_dt=datetime(2026, 7, 15, 20, 40),
        is_ice=False,
        escort_hours=1.0,
        fx_provider=_fx,
        dayoff_provider=_not_dayoff,
    )
    base = 24199 * 0.50
    assert res.amount == round(base + 1508.0 * 1.0, 2)


def test_helpers():
    assert format_hm(70) == "01:10"
    assert minutes_between(datetime(2026, 1, 1, 8, 0), datetime(2026, 1, 1, 11, 50)) == 230
