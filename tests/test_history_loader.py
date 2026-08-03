"""Тесты группировки исторических строк в судозаходы."""

from __future__ import annotations

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import (
    Direction,
    Operation,
    OperationKind,
    OperationTug,
    PortCall,
    Vessel,
)
from app.services.history_loader import (
    group_rows_into_portcalls,
    load_history,
    normalize_work_type,
)


def _session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def _row(
    *,
    vessel: str = "Aurora",
    work_type: str = "Швартовка",
    start: str = "2024-01-01 10:00:00",
    end: str = "2024-01-01 11:00:00",
    tug: str = "БК Коммунар",
    voucher: str = "1",
    order_key: str = "entry-aurora-1",
    **extra,
) -> dict:
    return {
        "vessel": vessel,
        "work_type": work_type,
        "work_start": start,
        "work_end": end,
        "tug": tug,
        "voucher_number": voucher,
        "voucher_key": f"{voucher}k|2024",
        "order_key": order_key,
        "agent": "Транс-Агро",
        **extra,
    }


def test_normalize_work_type_handles_composite_work_types() -> None:
    assert normalize_work_type("Отшвартовка + Сопровождение") == (
        OperationKind.unmooring,
        True,
        False,
    )
    assert normalize_work_type("Буксировка + Швартовка") == (
        OperationKind.mooring,
        False,
        True,
    )
    assert normalize_work_type("Проводка каравана") == (
        OperationKind.other,
        False,
        False,
    )


def test_two_rows_form_one_operation_with_two_tugs() -> None:
    rows = [
        _row(tug="БК Коммунар", voucher="1"),
        _row(
            tug="БК Пионер",
            voucher="2",
            start="2024-01-01 10:05:00",
        ),
    ]

    calls = group_rows_into_portcalls(rows)

    assert len(calls) == 1
    assert len(calls[0].operations) == 1
    operation = calls[0].operations[0]
    assert operation.kind is OperationKind.mooring
    assert {tug.tug_name for tug in operation.tugs} == {
        "БК Коммунар",
        "БК Пионер",
    }
    assert operation.work_start.isoformat() == "2024-01-01T10:00:00"


def test_composite_escort_and_plain_unmooring_have_one_escort_link() -> None:
    rows = [
        _row(
            work_type="Отшвартовка + Сопровождение",
            start="2024-01-02 10:00:00",
            end="2024-01-02 11:00:00",
            tug="БК Коммунар",
            voucher="3",
            order_key="exit-aurora-1",
        ),
        _row(
            work_type="Отшвартовка",
            start="2024-01-02 10:05:00",
            end="2024-01-02 11:00:00",
            tug="БК Пионер",
            voucher="4",
            order_key="exit-aurora-1",
        ),
    ]

    operation = group_rows_into_portcalls(rows)[0].operations[0]

    assert operation.kind is OperationKind.unmooring
    assert [tug.escort for tug in operation.tugs] == [True, False]


def test_sequence_is_one_portcall_with_ordered_operations() -> None:
    rows = [
        _row(
            work_type="Швартовка",
            start="2024-01-01 10:00:00",
            end="2024-01-01 11:00:00",
            order_key="entry-1",
        ),
        _row(
            work_type="Перестановка",
            start="2024-01-02 10:00:00",
            end="2024-01-02 11:00:00",
            order_key="reshift-1",
        ),
        _row(
            work_type="Отшвартовка",
            start="2024-01-03 10:00:00",
            end="2024-01-03 11:00:00",
            order_key="exit-1",
        ),
    ]

    calls = group_rows_into_portcalls(rows)

    assert len(calls) == 1
    assert calls[0].direction is Direction.entry
    assert [operation.kind for operation in calls[0].operations] == [
        OperationKind.mooring,
        OperationKind.reshift,
        OperationKind.unmooring,
    ]
    assert [operation.seq for operation in calls[0].operations] == [1, 2, 3]
    assert calls[0].eta.isoformat() == "2024-01-01T10:00:00"
    assert calls[0].etd.isoformat() == "2024-01-03T11:00:00"


def test_new_entry_closes_unfinished_call_and_opens_another() -> None:
    rows = [
        _row(
            work_type="Швартовка",
            start="2024-01-01 10:00:00",
            end="2024-01-01 11:00:00",
            order_key="entry-1",
        ),
        _row(
            work_type="Перестановка",
            start="2024-01-02 10:00:00",
            end="2024-01-02 11:00:00",
            order_key="reshift-1",
        ),
        _row(
            work_type="Швартовка",
            start="2024-01-03 10:00:00",
            end="2024-01-03 11:00:00",
            order_key="entry-2",
        ),
    ]

    calls = group_rows_into_portcalls(rows)

    assert len(calls) == 2
    assert [len(call.operations) for call in calls] == [2, 1]
    assert calls[0].operations[0].seq == 1
    assert calls[0].operations[1].seq == 2
    assert calls[1].operations[0].seq == 1


def test_draft_is_copied_and_missing_draft_stays_none() -> None:
    calls = group_rows_into_portcalls(
        [
            _row(draft_aft_m="7.4"),
            _row(
                vessel="Baltic",
                order_key="entry-baltic",
                draft_aft_m="",
                draft_fore_m="",
            ),
        ]
    )

    assert calls[0].operations[0].draft_m == 7.4
    assert calls[1].operations[0].draft_m is None


def test_dash_tug_does_not_create_tug_link() -> None:
    calls = group_rows_into_portcalls([_row(tug="-")])

    assert calls[0].operations[0].tugs == []


def test_rows_without_vessel_or_work_start_are_skipped() -> None:
    calls = group_rows_into_portcalls(
        [
            _row(vessel="", vessel_name="", start=""),
            _row(vessel="NoStart", start=""),
        ]
    )

    assert calls == []


def test_load_history_is_idempotent_and_persists_operation_tugs() -> None:
    db = _session()
    rows = [
        _row(tug="БК Коммунар", voucher="1"),
        _row(
            tug="БК Пионер",
            voucher="2",
            start="2024-01-01 10:05:00",
        ),
        _row(
            work_type="Перестановка",
            start="2024-01-02 10:00:00",
            end="2024-01-02 11:00:00",
            order_key="reshift-1",
            tug="БК Коммунар",
            voucher="3",
        ),
        _row(
            work_type="Отшвартовка",
            start="2024-01-03 10:00:00",
            end="2024-01-03 11:00:00",
            order_key="exit-1",
            tug="БК Пионер",
            voucher="4",
        ),
    ]

    first = load_history(db, rows)

    assert first["vessels_created"] == 1
    assert first["portcalls_created"] == 1
    assert first["operations_created"] == 3
    assert first["tug_links_created"] == 4
    assert db.query(Vessel).count() == 1
    assert db.query(PortCall).count() == 1
    assert db.query(Operation).count() == 3
    assert db.query(OperationTug).count() == 4

    second = load_history(db, rows)

    assert second["vessels_created"] == 0
    assert second["portcalls_created"] == 0
    assert second["operations_created"] == 0
    assert second["tug_links_created"] == 0
    assert second["skipped"] == 3
    assert db.query(PortCall).count() == 1
    assert db.query(Operation).count() == 3
    assert db.query(OperationTug).count() == 4
    portcall = db.scalar(select(PortCall))
    assert portcall is not None
    assert [operation.seq for operation in portcall.operations] == [1, 2, 3]
