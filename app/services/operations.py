"""Правила, относящиеся к операциям судозахода."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING

from sqlalchemy.orm import Session

from app.services.calculation import CalcResult, DayOffProvider, FxProvider, calculate

if TYPE_CHECKING:
    from app.models import Operation


def escort_likely(draft_m: float | None) -> bool:
    """Вернуть подсказку сопровождения для осадки 9.3–9.6 м включительно."""
    return draft_m is not None and 9.3 <= draft_m <= 9.6


def recommended_tug_count(loa_m: float | None) -> int | None:
    """Рекомендация буксиров по LOA; пороги настраиваемы в будущем."""
    if loa_m is None:
        return None
    if loa_m < 120:
        return 1
    if loa_m <= 160:
        return 2
    return 3


def calculate_operation(
    db: Session,
    operation: Operation,
    *,
    fx_provider: FxProvider | None = None,
    dayoff_provider: DayOffProvider | None = None,
) -> CalcResult:
    """Рассчитать стоимость операции по её судозаходу/судну и сохранить результат.

    Источники параметров: агент — из судозахода; GRT — из судна; время — из
    операции (иначе ETA/ETD судозахода); число буксиров — из назначенных связей.
    Провайдеры курса/выходных можно подменить в тестах.
    """
    portcall = operation.portcall
    vessel = portcall.vessel if portcall else None

    started = operation.work_start or (portcall.eta if portcall else None)
    finished = operation.work_end or (portcall.etd if portcall else None)
    tug_count = len(operation.tug_links) or 1

    kwargs: dict[str, FxProvider | DayOffProvider] = {}
    if fx_provider is not None:
        kwargs["fx_provider"] = fx_provider
    if dayoff_provider is not None:
        kwargs["dayoff_provider"] = dayoff_provider

    result = calculate(
        agent=(portcall.agent if portcall else None) or "",
        work_type=operation.kind.value,
        gross_tonnage=vessel.grt if vessel else None,
        started_dt=started,
        finished_dt=finished,
        is_ice=bool(operation.is_ice),
        tug_count=tug_count,
        **kwargs,  # type: ignore[arg-type]
    )

    operation.amount = result.amount
    operation.currency = result.currency
    operation.cbr_rate = result.cbr_rate
    operation.revenue_rub = result.revenue_rub
    operation.calc_note = result.calc_note
    operation.calculated_at = datetime.utcnow()
    db.commit()
    return result
