"""Шаблон регионов ваучера для фиксированного бланка Baltiyskie buksiry."""

from __future__ import annotations

from dataclasses import dataclass

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import VoucherRegion, VoucherTemplate


@dataclass(frozen=True)
class VoucherRegionDefinition:
    name: str
    label: str
    order: int
    center_x: float
    center_y: float
    width: float
    height: float


DEFAULT_VOUCHER_REGIONS = (
    VoucherRegionDefinition("tugboat", "Tugboat/Буксир", 1, 0.577564, 0.313522, 0.646226, 0.039623),
    VoucherRegionDefinition("vessel", "Vessel/Судно", 2, 0.586020, 0.360063, 0.656017, 0.037107),
    VoucherRegionDefinition("agent", "Agent/Агент", 3, 0.551305, 0.397484, 0.789535, 0.041509),
    VoucherRegionDefinition(
        "work_type", "Order/Вид работ", 4, 0.529052, 0.435220, 0.794876, 0.042767
    ),
    VoucherRegionDefinition(
        "voucher_number", "№ ваучера", 5, 0.570888, 0.261635, 0.141529, 0.046541
    ),
    VoucherRegionDefinition(
        "left_base", "Выход из базы", 6, 0.553530, 0.487421, 0.532291, 0.040252
    ),
    VoucherRegionDefinition(
        "arrived_base", "Приход в базу", 7, 0.556646, 0.524214, 0.534961, 0.037107
    ),
    VoucherRegionDefinition(
        "started_work", "Начало работ", 8, 0.556646, 0.560377, 0.536742, 0.035220
    ),
    VoucherRegionDefinition(
        "finished_work", "Окончание работ", 9, 0.553976, 0.599371, 0.534961, 0.038994
    ),
    VoucherRegionDefinition(
        "remarks", "Remarks/Примечания", 10, 0.544629, 0.643082, 0.712095, 0.048428
    ),
    VoucherRegionDefinition(
        "joint_with_line_1",
        "Совместно с строка 1",
        11,
        0.657674,
        0.677044,
        0.516269,
        0.044654,
    ),
    VoucherRegionDefinition(
        "joint_with_line_2",
        "Совместно с строка 2",
        12,
        0.496118,
        0.722327,
        0.899910,
        0.042138,
    ),
)


def ensure_default_template(db: Session) -> VoucherTemplate:
    """Создать эталонный шаблон один раз и вернуть его."""
    template = db.scalar(
        select(VoucherTemplate).where(VoucherTemplate.name == "baltiyskie_buksiry")
    )
    if template is not None:
        return template

    template = VoucherTemplate(name="baltiyskie_buksiry", version="1")
    template.regions = [
        VoucherRegion(
            name=definition.name,
            label=definition.label,
            order=definition.order,
            center_x=definition.center_x,
            center_y=definition.center_y,
            width=definition.width,
            height=definition.height,
        )
        for definition in DEFAULT_VOUCHER_REGIONS
    ]
    db.add(template)
    db.commit()
    db.refresh(template)
    return template
