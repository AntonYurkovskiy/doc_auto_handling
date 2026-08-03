"""Загрузка исторических строк в судозаходы и последовательности операций."""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from typing import Any

from dateutil import parser as date_parser
from sqlalchemy.orm import Session

from app.models import (
    Direction,
    DocStatus,
    Operation,
    OperationKind,
    OperationTug,
    PortCall,
    Tug,
    Vessel,
)


@dataclass
class TugLinkSpec:
    """Буксир, указанный в строке исторической операции."""

    tug_name: str
    escort: bool
    voucher_number: str | None
    voucher_key: str | None


@dataclass
class OperationSpec:
    """Сгруппированная операция до сохранения в БД."""

    kind: OperationKind
    seq: int
    work_start: datetime | None
    work_end: datetime | None
    draft_m: float | None
    tugs: list[TugLinkSpec] = field(default_factory=list)
    agent: str | None = None
    order_key: str | None = None
    voucher_keys: list[str] = field(default_factory=list)


@dataclass
class PortCallSpec:
    """Сгруппированный судозаход до сохранения в БД."""

    vessel_name: str
    imo: str | None
    direction: Direction
    agent: str | None
    eta: datetime | None
    etd: datetime | None
    operations: list[OperationSpec] = field(default_factory=list)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    result = str(value).strip()
    if not result or result.casefold() == "nan":
        return None
    return result


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


def normalize_work_type(value: Any) -> tuple[OperationKind, bool, bool]:
    """Нормализовать вид работы и выделить сопровождение/буксировку."""

    text = (_text(value) or "").casefold()
    has_escort = "сопровожд" in text
    has_towing = "буксировк" in text
    if "перешвартов" in text or "перестанов" in text:
        kind = OperationKind.reshift
    elif "отшвартов" in text:
        kind = OperationKind.unmooring
    elif "швартов" in text:
        kind = OperationKind.mooring
    elif has_escort:
        kind = OperationKind.escort
    else:
        kind = OperationKind.other
    return kind, has_escort, has_towing


def _direction_for_kind(kind: OperationKind) -> Direction:
    if kind is OperationKind.mooring:
        return Direction.entry
    if kind is OperationKind.unmooring:
        return Direction.exit
    return Direction.other


def _first(row: dict, *keys: str) -> Any:
    for key in keys:
        value = _text(row.get(key))
        if value is not None:
            return value
    return None


def _vessel_key(row: dict) -> tuple[str, str] | None:
    name = _first(row, "vessel_name", "vessel")
    if name is None:
        return None
    imo = _text(row.get("imo"))
    return ("imo", imo) if imo is not None else ("name", name.casefold())


def _row_dates(row: dict) -> tuple[datetime | None, datetime | None]:
    start = _datetime(_first(row, "work_start", "base_departure"))
    end = _datetime(_first(row, "work_end", "base_arrival"))
    return start, end


def _date_key(value: datetime | None) -> str:
    return value.isoformat() if value is not None else ""


def _operation_sort_key(operation: OperationSpec) -> tuple[datetime, datetime]:
    start = operation.work_start or operation.work_end
    end = operation.work_end or operation.work_start
    assert start is not None
    assert end is not None
    return start, end


def _append_operation(call: PortCallSpec, operation: OperationSpec) -> None:
    operation.seq = len(call.operations) + 1
    call.operations.append(operation)


def _finish_call(calls: list[PortCallSpec], call: PortCallSpec | None) -> None:
    if call is None or not call.operations:
        return
    call.eta = min(
        operation.work_start
        for operation in call.operations
        if operation.work_start is not None
    )
    call.etd = max(
        operation.work_end
        for operation in call.operations
        if operation.work_end is not None
    )
    call.direction = _direction_for_kind(call.operations[0].kind)
    call.agent = next(
        (operation.agent for operation in call.operations if operation.agent),
        None,
    )
    calls.append(call)


def group_rows_into_portcalls(rows: Iterable[dict]) -> list[PortCallSpec]:
    """Сгруппировать исторические строки в операции и судозаходы."""

    operation_groups: dict[
        tuple[tuple[str, str], OperationKind, str, str], OperationSpec
    ] = {}
    operation_vessels: dict[
        tuple[tuple[str, str], OperationKind, str, str], tuple[str, str | None]
    ] = {}

    for row in rows:
        vessel_key = _vessel_key(row)
        work_start, work_end = _row_dates(row)
        if vessel_key is None or work_start is None:
            continue

        kind, has_escort, _has_towing = normalize_work_type(row.get("work_type"))
        order_key = _text(row.get("order_key")) or ""
        date_value = work_end or work_start
        operation_key = (vessel_key, kind, order_key, _date_key(date_value))
        operation = operation_groups.get(operation_key)
        if operation is None:
            operation = OperationSpec(
                kind=kind,
                seq=1,
                work_start=work_start,
                work_end=work_end,
                draft_m=_float(
                    _first(row, "draft_aft_m", "draft_fore_m", "draft_m")
                ),
                agent=_text(row.get("agent")),
                order_key=order_key or None,
            )
            operation_groups[operation_key] = operation
            operation_vessels[operation_key] = (
                _first(row, "vessel_name", "vessel") or "",
                _text(row.get("imo")),
            )
        else:
            if operation.work_start is None or work_start < operation.work_start:
                operation.work_start = work_start
            if work_end is not None and (
                operation.work_end is None or work_end > operation.work_end
            ):
                operation.work_end = work_end
            if operation.agent is None:
                operation.agent = _text(row.get("agent"))
            if operation.draft_m is None:
                operation.draft_m = _float(
                    _first(row, "draft_aft_m", "draft_fore_m", "draft_m")
                )

        voucher_key = _text(row.get("voucher_key"))
        if voucher_key is not None and voucher_key not in operation.voucher_keys:
            operation.voucher_keys.append(voucher_key)

        tug_name = _text(row.get("tug"))
        if tug_name is not None and tug_name != "-":
            escort = has_escort and not any(tug.escort for tug in operation.tugs)
            operation.tugs.append(
                TugLinkSpec(
                    tug_name=tug_name,
                    escort=escort,
                    voucher_number=_text(row.get("voucher_number")),
                    voucher_key=voucher_key,
                )
            )

    vessel_operations: dict[
        tuple[str, str], list[tuple[OperationSpec, tuple[str, str | None]]]
    ] = {}
    for key, operation in operation_groups.items():
        vessel_operations.setdefault(key[0], []).append(
            (operation, operation_vessels[key])
        )

    result: list[PortCallSpec] = []
    for _key, grouped_operations in vessel_operations.items():
        grouped_operations.sort(key=lambda item: _operation_sort_key(item[0]))
        call: PortCallSpec | None = None
        for operation, (vessel_name, imo) in grouped_operations:
            if operation.kind is OperationKind.mooring:
                _finish_call(result, call)
                call = PortCallSpec(
                    vessel_name=vessel_name,
                    imo=imo,
                    direction=Direction.entry,
                    agent=operation.agent,
                    eta=operation.work_start,
                    etd=operation.work_end,
                )
                _append_operation(call, operation)
            elif operation.kind is OperationKind.unmooring:
                if call is None:
                    call = PortCallSpec(
                        vessel_name=vessel_name,
                        imo=imo,
                        direction=Direction.exit,
                        agent=operation.agent,
                        eta=operation.work_start,
                        etd=operation.work_end,
                    )
                _append_operation(call, operation)
                _finish_call(result, call)
                call = None
            else:
                if call is None:
                    call = PortCallSpec(
                        vessel_name=vessel_name,
                        imo=imo,
                        direction=Direction.other,
                        agent=operation.agent,
                        eta=operation.work_start,
                        etd=operation.work_end,
                    )
                _append_operation(call, operation)
        _finish_call(result, call)

    return result


def _upsert_vessel(
    db: Session,
    row: dict,
    vessels_by_key: dict[tuple[str, str], Vessel],
    created_vessel_ids: set[int],
    result: dict[str, int],
) -> Vessel | None:
    name = _first(row, "vessel_name", "vessel")
    if name is None:
        return None
    imo = _text(row.get("imo"))
    key = _vessel_key(row)
    assert key is not None
    vessel = vessels_by_key.get(key)
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
        vessels_by_key[key] = vessel
        vessels_by_key[("name", name.casefold())] = vessel

    values = {
        "flag": _text(row.get("flag")),
        "loa_m": _float(row.get("loa_m")),
        "beam_m": _float(row.get("beam_m")),
        "grt": _int(row.get("grt")),
        "nrt": _int(row.get("nrt")),
        "imo": imo,
    }
    vessel_updated = False
    for field_name, value in values.items():
        if getattr(vessel, field_name) is None and value is not None:
            setattr(vessel, field_name, value)
            vessel_updated = True
    if vessel_updated and not vessel_created and vessel.id not in created_vessel_ids:
        result["vessels_updated"] += 1
    return vessel


def load_history(db: Session, rows: Iterable[dict]) -> dict[str, int]:
    """Сохранить сгруппированную историю с идемпотентным судозаходом."""

    source_rows = list(rows)
    result = {
        "vessels_created": 0,
        "vessels_updated": 0,
        "portcalls_created": 0,
        "operations_created": 0,
        "tug_links_created": 0,
        "skipped": 0,
    }
    vessels_by_key: dict[tuple[str, str], Vessel] = {}
    created_vessel_ids: set[int] = set()
    for row in source_rows:
        vessel = _upsert_vessel(
            db, row, vessels_by_key, created_vessel_ids, result
        )
        if vessel is None or _row_dates(row)[0] is None:
            result["skipped"] += 1

    specs = group_rows_into_portcalls(source_rows)
    portcall_keys: set[tuple[int, datetime | None, Direction]] = set()
    for spec in specs:
        vessel = vessels_by_key.get(
            ("imo", spec.imo)
            if spec.imo is not None
            else ("name", spec.vessel_name.casefold())
        )
        if vessel is None:
            result["skipped"] += len(spec.operations)
            continue

        key = (vessel.id, spec.eta, spec.direction)
        existing = db.query(PortCall).filter(
            PortCall.vessel_id == vessel.id,
            PortCall.eta == spec.eta,
            PortCall.direction == spec.direction,
        ).first()
        if key in portcall_keys or existing is not None:
            result["skipped"] += len(spec.operations)
            continue

        portcall = PortCall(
            vessel_id=vessel.id,
            direction=spec.direction,
            agent=spec.agent,
            eta=spec.eta,
            etd=spec.etd,
            source="history",
            status=DocStatus.confirmed,
        )
        db.add(portcall)
        db.flush()
        portcall_keys.add(key)
        result["portcalls_created"] += 1

        for operation_spec in spec.operations:
            operation = Operation(
                portcall_id=portcall.id,
                kind=operation_spec.kind,
                seq=operation_spec.seq,
                draft_m=operation_spec.draft_m,
            )
            db.add(operation)
            db.flush()
            result["operations_created"] += 1
            links_by_tug_id: dict[int, OperationTug] = {}
            for tug_spec in operation_spec.tugs:
                tug = db.query(Tug).filter_by(name=tug_spec.tug_name).first()
                if tug is None:
                    tug = Tug(name=tug_spec.tug_name)
                    db.add(tug)
                    db.flush()
                link = links_by_tug_id.get(tug.id)
                if link is None:
                    link = db.query(OperationTug).filter_by(
                        operation_id=operation.id,
                        tug_id=tug.id,
                    ).first()
                if link is None:
                    link = OperationTug(
                        operation_id=operation.id,
                        tug_id=tug.id,
                        escort=tug_spec.escort,
                    )
                    db.add(link)
                    result["tug_links_created"] += 1
                elif tug_spec.escort and not link.escort and not any(
                    existing_link.escort for existing_link in links_by_tug_id.values()
                ):
                    link.escort = True
                links_by_tug_id[tug.id] = link

    db.commit()
    return result
