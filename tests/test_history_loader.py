"""Тесты загрузчика исторических судозаходов."""

from __future__ import annotations

import csv
from io import BytesIO, TextIOWrapper

from sqlalchemy import create_engine, select
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Direction, DocStatus, Operation, OperationKind, PortCall, Vessel
from app.services.history_loader import _op_kind, load_history


def _session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


def test_load_history_upserts_vessel_and_portcall() -> None:
    db = _session()
    rows = [
        {
            "vessel_name": " Aurora ",
            "imo": "1234567",
            "flag": "RU",
            "loa_m": "123,5",
            "draft_aft_m": "",
            "draft_fore_m": "7,2",
            "direction": " ВХОД ",
            "work_start": "2024-01-02T03:04:05",
            "work_end": "02.01.2024 05:06",
            "port_from": "Внешний рейд",
            "berth": "Причал 1",
            "purpose": "Швартовка",
        },
        {
            "vessel": "Aurora",
            "imo": "1234567",
            "beam_m": "20",
            "draft_aft_m": "7,5",
            "grt": "1000",
            "nrt": "500",
            "work_start": "2024-01-03 03:04:05",
            "direction": "entry",
        },
    ]

    first = load_history(db, rows)
    vessel = db.scalar(select(Vessel).where(Vessel.imo == "1234567"))
    portcall = db.scalar(select(PortCall).order_by(PortCall.id))

    assert first == {
        "vessels_created": 1,
        "vessels_updated": 0,
        "portcalls_created": 2,
        "skipped": 0,
    }
    assert vessel is not None
    assert vessel.loa_m == 123.5
    assert vessel.beam_m == 20
    assert vessel.grt == 1000
    assert vessel.nrt == 500
    assert portcall is not None
    assert portcall.eta is not None
    assert portcall.eta.isoformat() == "2024-01-02T03:04:05"
    assert portcall.etd is not None
    assert portcall.etd.isoformat() == "2024-01-02T05:06:00"
    assert portcall.direction is Direction.entry
    assert portcall.source == "history"
    assert portcall.status is DocStatus.confirmed
    assert portcall.berth_from == "Внешний рейд"
    assert portcall.berth_to == "Причал 1"
    assert len(portcall.operations) == 1
    assert portcall.operations[0].kind is OperationKind.mooring
    assert portcall.operations[0].draft_m == 7.2

    second = load_history(db, rows)
    assert second["vessels_created"] == 0
    assert second["portcalls_created"] == 0
    assert second["skipped"] == 2
    assert db.query(Operation).count() == 2


def test_load_history_accepts_utf8_sig_csv() -> None:
    csv_text = "\ufeffvessel,imo,draft_aft_m,draft_fore_m,direction,base_departure\n"
    csv_text += "Baltic,7654321,8,9,выход,2024-02-03 04:05:06\n"
    rows = csv.DictReader(TextIOWrapper(BytesIO(csv_text.encode()), encoding="utf-8-sig"))
    db = _session()

    summary = load_history(db, rows)
    vessel = db.scalar(select(Vessel).where(Vessel.name == "Baltic"))
    portcall = db.scalar(select(PortCall))

    assert summary["portcalls_created"] == 1
    assert vessel is not None
    assert portcall is not None and portcall.direction is Direction.exit
    assert len(portcall.operations) == 1
    assert portcall.operations[0].kind is OperationKind.unmooring
    assert portcall.operations[0].draft_m == 8


def test_load_history_updates_existing_vessel_without_overwriting() -> None:
    db = _session()
    db.add(Vessel(name="Existing", imo="1111111", grt=900))
    db.commit()

    summary = load_history(
        db,
        [
            {
                "vessel_name": "Existing",
                "imo": "1111111",
                "flag": "RU",
                "loa_m": "110,5",
                "beam_m": "18",
                "draft_fore_m": "6,5",
                "grt": "1200",
                "nrt": "450",
                "work_start": "2024-03-01 10:00:00",
            }
        ],
    )
    vessel = db.scalar(select(Vessel).where(Vessel.imo == "1111111"))

    assert summary["vessels_updated"] == 1
    assert summary["vessels_created"] == 0
    assert vessel is not None
    assert vessel.grt == 900
    assert vessel.flag == "RU"
    assert vessel.loa_m == 110.5
    assert vessel.beam_m == 18
    assert vessel.nrt == 450
    portcall = db.scalar(select(PortCall))
    assert portcall is not None
    assert portcall.operations[0].draft_m == 6.5
    assert portcall.operations[0].kind is OperationKind.other


def test_load_history_skips_row_without_vessel_name() -> None:
    db = _session()

    summary = load_history(
        db,
        [
            {
                "vessel": " ",
                "vessel_name": "",
                "imo": "2222222",
                "work_start": "2024-03-01 10:00:00",
            }
        ],
    )

    assert summary["skipped"] == 1
    assert summary["portcalls_created"] == 0
    assert db.query(Vessel).count() == 0


def test_operation_kind_prioritizes_reshift_and_unmooring() -> None:
    assert _op_kind("перешвартовка", Direction.other) is OperationKind.reshift
    assert _op_kind("отшвартовка", Direction.other) is OperationKind.unmooring
