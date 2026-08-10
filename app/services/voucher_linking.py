"""Связать подтверждённый ваучер с заявкой, судозаходом и операцией."""

from __future__ import annotations

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Application,
    Direction,
    DocStatus,
    Operation,
    OperationKind,
    OperationTug,
    PortCall,
    Voucher,
)
from app.services.matching import score_match
from app.services.vessels import ensure_vessel


def link_voucher(
    db: Session,
    voucher: Voucher,
    application: Application | None = None,
) -> Application | None:
    """Выбрать заявку и создать внутренние судозаход/операцию при наличии данных."""
    application = application or _best_application(db, voucher)
    if application is None:
        return None

    voucher.application = application
    portcall = _ensure_portcall(db, application)
    operation = _ensure_operation(db, voucher, application, portcall)
    voucher.operation = operation
    if voucher.tug_id is not None:
        participant = (
            db.query(OperationTug)
            .filter_by(operation_id=operation.id, tug_id=voucher.tug_id)
            .first()
        )
        if participant is None:
            participant = OperationTug(
                operation=operation,
                tug_id=voucher.tug_id,
                voucher=voucher,
                work_start=voucher.started_dt,
                work_end=voucher.finished_dt,
            )
            db.add(participant)
        else:
            participant.voucher = voucher
            participant.work_start = voucher.started_dt
            participant.work_end = voucher.finished_dt
    return application


def _best_application(db: Session, voucher: Voucher) -> Application | None:
    candidates = [
        (application, score_match(application, voucher))
        for application in db.scalars(select(Application)).all()
    ]
    candidates = [candidate for candidate in candidates if candidate[1].score >= 0.8]
    candidates.sort(key=lambda candidate: candidate[1].score, reverse=True)
    if not candidates:
        return None
    if len(candidates) > 1 and candidates[0][1].score == candidates[1][1].score:
        return None
    return candidates[0][0]


def _ensure_portcall(db: Session, application: Application) -> PortCall:
    if application.portcall is not None:
        return application.portcall

    vessel = ensure_vessel(db, application.vessel_name, application.imo, loa_m=application.loa_m)
    if vessel is None:
        raise ValueError("Для связи ваучера не найдено судно в заявке")

    portcall = (
        db.query(PortCall)
        .filter(
            PortCall.vessel_id == vessel.id,
            PortCall.direction == application.direction,
            PortCall.eta == application.entry_datetime,
            PortCall.etd == application.exit_datetime,
        )
        .first()
    )
    if portcall is None:
        portcall = PortCall(
            status=DocStatus.needs_review,
            vessel_id=vessel.id,
            direction=application.direction,
            agent=application.agent,
            eta=application.entry_datetime,
            etd=application.exit_datetime,
            source="derived",
        )
        db.add(portcall)
        db.flush()
    application.portcall = portcall
    return portcall


def _ensure_operation(
    db: Session,
    voucher: Voucher,
    application: Application,
    portcall: PortCall,
) -> Operation:
    kind = _operation_kind(voucher.work_type, application.direction)
    operation = (
        db.query(Operation)
        .filter_by(portcall_id=portcall.id, kind=kind)
        .order_by(Operation.seq)
        .first()
    )
    if operation is None:
        operation = Operation(
            portcall=portcall,
            kind=kind,
            draft_m=application.draft_m,
            work_start=voucher.started_dt or application.entry_datetime,
            work_end=voucher.finished_dt or application.exit_datetime,
            is_ice=voucher.is_ice,
        )
        db.add(operation)
        db.flush()
    else:
        operation.work_start = operation.work_start or voucher.started_dt
        operation.work_end = operation.work_end or voucher.finished_dt
        operation.is_ice = operation.is_ice or voucher.is_ice
    return operation


def _operation_kind(work_type: str | None, direction: Direction) -> OperationKind:
    text = (work_type or "").lower()
    if "перешварт" in text or "перестанов" in text:
        return OperationKind.reshift
    if "отшварт" in text or direction == Direction.exit:
        return OperationKind.unmooring
    if "шварт" in text or direction == Direction.entry:
        return OperationKind.mooring
    if "сопров" in text:
        return OperationKind.escort
    return OperationKind.other
