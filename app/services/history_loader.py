"""Загрузка исторических судозаходов из согласованного датасета."""

from __future__ import annotations

from collections.abc import Iterable
from datetime import datetime
from typing import Any

from dateutil import parser as date_parser
from sqlalchemy.orm import Session

from app.models import Direction, DocStatus, Operation, OperationKind, PortCall, Vessel


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    return result or None


def _float(value: Any) -> float | None:
    text = _text(value)
    if text is None or text.lower() == "nan":
        return None
    try:
        return float(text.replace(",", "."))
    except (TypeError, ValueError):
        return None


def _int(value: Any) -> int | None:
    text = _text(value)
    if text is None or text.lower() == "nan":
        return None
    try:
        return int(float(text.replace(",", ".")))
    except (TypeError, ValueError):
        return None


def _datetime(value: Any) -> datetime | None:
    text = _text(value)
    if text is None or text.lower() == "nan":
        return None
    for fmt in (
        None,
        "%Y-%m-%d %H:%M:%S",
        "%d.%m.%Y %H:%M",
        "%d.%m.%Y %H:%M:%S",
    ):
        try:
            if fmt is None:
                return datetime.fromisoformat(text)
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    try:
        return date_parser.parse(text)
    except (TypeError, ValueError, OverflowError):
        return None


def _direction(value: Any) -> Direction:
    normalized = (_text(value) or "").casefold()
    if normalized in {"вход", "entry", "in"}:
        return Direction.entry
    if normalized in {"выход", "exit", "out"}:
        return Direction.exit
    return Direction.other


def _op_kind(work_type_text: Any, direction: Direction) -> OperationKind:
    normalized = (_text(work_type_text) or "").casefold()
    if "перешвартов" in normalized:
        return OperationKind.reshift
    if "отшвартов" in normalized:
        return OperationKind.unmooring
    if "швартов" in normalized:
        return OperationKind.mooring
    if "сопровожд" in normalized:
        return OperationKind.escort
    if direction is Direction.entry:
        return OperationKind.mooring
    if direction is Direction.exit:
        return OperationKind.unmooring
    return OperationKind.other


def _first(row: dict, *keys: str) -> Any:
    for key in keys:
        value = _text(row.get(key))
        if value is not None:
            return value
    return None


def _berths(row: dict, direction: Direction) -> tuple[str | None, str | None]:
    """Определить начало и конец перехода по портам и причалу.

    Для входа переход идёт из `port_from` к `berth` (или `port_to`), а для
    выхода — от `berth` (или `port_from`) к `port_to`. Для прочих записей
    сохраняется наиболее полная последовательность из этих полей.
    """
    port_from = _text(row.get("port_from"))
    port_to = _text(row.get("port_to"))
    berth = _text(row.get("berth"))
    if direction is Direction.entry:
        return port_from, berth or port_to
    if direction is Direction.exit:
        return berth or port_from, port_to
    return port_from or berth, berth or port_to


def load_history(db: Session, rows: Iterable[dict]) -> dict[str, int]:
    """Загрузить историю, накапливая характеристики судов и судозаходы.

    Судно ищется по IMO, если он указан, иначе по имени. Непустые сведения
    никогда не заменяются пустыми или новыми значениями из истории. Для
    входного судозахода `berth_from` — порт отправления, а `berth_to` —
    причал (с запасным использованием `port_to`); для выходного направления
    это причал (или `port_from`) и порт назначения. Повторные судозаходы с
    ключом `(vessel_id, eta, direction)` пропускаются.
    """
    result = {
        "vessels_created": 0,
        "vessels_updated": 0,
        "portcalls_created": 0,
        "skipped": 0,
    }
    vessels_by_key: dict[tuple[str, str], Vessel] = {}
    created_vessel_ids: set[int] = set()
    portcall_keys: set[tuple[int, datetime | None, Direction]] = set()

    for row in rows:
        name = _first(row, "vessel_name", "vessel")
        if name is None:
            result["skipped"] += 1
            continue

        imo = _text(row.get("imo"))
        vessel_key = ("imo", imo) if imo is not None else ("name", name)
        vessel = vessels_by_key.get(vessel_key)
        vessel_created = False
        if vessel is None:
            if imo is not None:
                vessel = db.query(Vessel).filter(Vessel.imo == imo).first()
            if vessel is None:
                vessel = db.query(Vessel).filter(Vessel.name == name).first()
            if vessel is None:
                vessel = Vessel(name=name, imo=imo)
                db.add(vessel)
                db.flush()
                result["vessels_created"] += 1
                vessel_created = True
                created_vessel_ids.add(vessel.id)
            vessels_by_key[vessel_key] = vessel
            vessels_by_key[("name", name)] = vessel

        draft_aft = _float(row.get("draft_aft_m"))
        draft_fore = _float(row.get("draft_fore_m"))
        values = {
            "flag": _text(row.get("flag")),
            "loa_m": _float(row.get("loa_m")),
            "beam_m": _float(row.get("beam_m")),
            "grt": _int(row.get("grt")),
            "nrt": _int(row.get("nrt")),
            "imo": imo,
        }
        vessel_updated = False
        for field, value in values.items():
            if getattr(vessel, field) is None and value is not None:
                setattr(vessel, field, value)
                vessel_updated = True
        if vessel_updated and not vessel_created and vessel.id not in created_vessel_ids:
            result["vessels_updated"] += 1

        direction = _direction(row.get("direction"))
        eta = _datetime(_first(row, "work_start", "base_departure"))
        etd = _datetime(_first(row, "work_end", "base_arrival"))
        key = (vessel.id, eta, direction)
        if key in portcall_keys or db.query(PortCall).filter(
            PortCall.vessel_id == vessel.id,
            PortCall.eta == eta,
            PortCall.direction == direction,
        ).first():
            result["skipped"] += 1
            continue

        berth_from, berth_to = _berths(row, direction)
        portcall = PortCall(
            vessel_id=vessel.id,
            direction=direction,
            agent=_text(row.get("agent")),
            eta=eta,
            etd=etd,
            berth_from=berth_from,
            berth_to=berth_to,
            purpose=_first(row, "purpose", "work_type"),
            source="history",
            status=DocStatus.confirmed,
        )
        db.add(portcall)
        db.flush()
        db.add(
            Operation(
                portcall_id=portcall.id,
                kind=_op_kind(row.get("work_type"), direction),
                seq=1,
                draft_m=draft_aft if draft_aft is not None else draft_fore,
            )
        )
        portcall_keys.add(key)
        result["portcalls_created"] += 1

    db.commit()
    return result
