"""Работа со справочником судов."""

from __future__ import annotations

from sqlalchemy.orm import Session

from app.models import Vessel


def ensure_vessel(
    db: Session,
    name: str | None,
    imo: str | None = None,
    *,
    loa_m: float | None = None,
) -> Vessel | None:
    """Создать судно при отсутствии; дозаполнить постоянные характеристики.

    LOA — постоянная величина: записывается только если у судна её ещё нет.
    Поиск по IMO (если задан), иначе по имени.
    """
    if not name:
        return None

    vessel: Vessel | None = None
    if imo:
        vessel = db.query(Vessel).filter_by(imo=imo).first()
    if vessel is None:
        vessel = db.query(Vessel).filter_by(name=name).first()

    if vessel is None:
        vessel = Vessel(name=name, imo=imo, loa_m=loa_m)
        db.add(vessel)
    else:
        if imo and not vessel.imo:
            vessel.imo = imo
        if loa_m is not None and vessel.loa_m is None:
            vessel.loa_m = loa_m

    db.commit()
    return vessel
