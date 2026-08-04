"""Тесты расчёта стоимости на уровне операции."""

from __future__ import annotations

from datetime import date, datetime

import pytest
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import TariffsGroupA
from app.database import Base
from app.models import Operation, OperationKind, OperationTug, PortCall, Tug, Vessel
from app.services.operations import calculate_operation


def _session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _fake_fx(currency: str, on: date) -> float:
    return 90.0


def _no_dayoff(on: date) -> bool:
    return False


def _make(db, *, agent: str, grt: int, kind: OperationKind, tugs: int = 0) -> Operation:
    vessel = Vessel(name="Test", grt=grt, loa_m=150)
    db.add(vessel)
    db.flush()
    portcall = PortCall(
        vessel_id=vessel.id,
        agent=agent,
        eta=datetime(2024, 6, 1, 10, 0),
        etd=datetime(2024, 6, 1, 12, 0),
    )
    db.add(portcall)
    db.flush()
    operation = Operation(portcall_id=portcall.id, kind=kind)
    db.add(operation)
    db.flush()
    for i in range(tugs):
        tug = Tug(name=f"Tug {i}")
        db.add(tug)
        db.flush()
        db.add(OperationTug(operation_id=operation.id, tug_id=tug.id))
    db.commit()
    return operation


def test_calculate_operation_group_a_stores_result():
    db = _session()
    op = _make(db, agent="Транс-Агро", grt=2500, kind=OperationKind.mooring, tugs=2)

    result = calculate_operation(
        db, op, fx_provider=_fake_fx, dayoff_provider=_no_dayoff
    )

    assert result.currency == "USD"
    assert op.currency == "USD"
    assert op.cbr_rate == 90.0
    assert op.amount is not None
    assert op.revenue_rub is not None
    assert op.calculated_at is not None
    assert "швартовка" in op.calc_note


def test_calculate_operation_uses_portcall_times_as_fallback():
    db = _session()
    op = _make(db, agent="Транс-Агро", grt=2500, kind=OperationKind.mooring)
    # У операции нет собственных времён — берём ETA/ETD судозахода, ошибки нет.
    result = calculate_operation(
        db, op, fx_provider=_fake_fx, dayoff_provider=_no_dayoff
    )
    assert result.amount is not None


def test_calculate_operation_group_b_not_implemented():
    db = _session()
    op = _make(db, agent="Терминал", grt=2500, kind=OperationKind.mooring)
    with pytest.raises(NotImplementedError):
        calculate_operation(db, op, fx_provider=_fake_fx, dayoff_provider=_no_dayoff)


def test_calculate_operation_per_ton_with_tariff():
    db = _session()
    op = _make(db, agent="Транс-Агро", grt=1000, kind=OperationKind.mooring, tugs=2)
    # 1000 GRT < 2000 -> фиксированная ставка за операцию (из тарифа).
    tariffs = TariffsGroupA(mooring_unmooring_gt_below_2000_weekdays=500.0)
    from app.services import calculation

    result = calculation.calculate(
        agent="Транс-Агро",
        work_type="швартовка",
        gross_tonnage=1000,
        started_dt=op.portcall.eta,
        finished_dt=op.portcall.etd,
        tug_count=2,
        fx_provider=_fake_fx,
        dayoff_provider=_no_dayoff,
        tariffs=tariffs,
    )
    assert result.amount == 500.0
    assert result.revenue_rub == 45000.0
