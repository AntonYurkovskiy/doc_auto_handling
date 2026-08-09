"""Поля ваучера: связь колонок Voucher с регионами шаблона и подтверждение."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import Voucher, VoucherFieldPrediction, VoucherRegion


@dataclass(frozen=True)
class VoucherFieldDef:
    """Поле карточки ваучера: имя региона шаблона + колонка модели."""

    name: str
    label: str
    attr: str


# Имена совпадают с VoucherRegion.name эталонного шаблона (см. voucher_template).
VOUCHER_FIELDS: tuple[VoucherFieldDef, ...] = (
    VoucherFieldDef("voucher_number", "№ ваучера", "number"),
    VoucherFieldDef("tugboat", "Буксир", "tug_id"),
    VoucherFieldDef("vessel", "Судно / объект", "vessel_name"),
    VoucherFieldDef("agent", "Агент", "agent"),
    VoucherFieldDef("work_type", "Вид работ", "work_type"),
    VoucherFieldDef("left_base", "Выход из базы", "left_base_dt"),
    VoucherFieldDef("arrived_base", "Приход в базу", "arrived_base_dt"),
    VoucherFieldDef("started_work", "Начало работ", "started_dt"),
    VoucherFieldDef("finished_work", "Окончание работ", "finished_dt"),
    VoucherFieldDef("remarks", "Примечания", "remarks"),
    VoucherFieldDef("joint_with_line_1", "Совместно с", "joint_with"),
)


def field_value(voucher: Voucher, field: VoucherFieldDef) -> str | None:
    """Текущее значение поля ваучера в виде строки."""
    if field.attr == "tug_id":
        return voucher.tug.name if voucher.tug else None
    value = getattr(voucher, field.attr)
    if value is None:
        return None
    if isinstance(value, datetime):
        return value.strftime("%Y-%m-%d %H:%M")
    text = str(value).strip()
    return text or None


def predictions_by_field(voucher: Voucher) -> dict[str, VoucherFieldPrediction]:
    """Предсказания ваучера, индексированные по имени поля."""
    return {prediction.field_name: prediction for prediction in voucher.predictions}


def _region_ids(db: Session, voucher: Voucher) -> dict[str, int]:
    if voucher.template_id is None:
        return {}
    regions = db.scalars(
        select(VoucherRegion).where(VoucherRegion.template_id == voucher.template_id)
    ).all()
    return {region.name: region.id for region in regions}


def confirm_fields(db: Session, voucher: Voucher, confirmed_at: datetime) -> None:
    """Переносит текущие значения полей ваучера в confirmed_value предсказаний."""
    existing = predictions_by_field(voucher)
    region_ids = _region_ids(db, voucher)
    for field in VOUCHER_FIELDS:
        value = field_value(voucher, field)
        prediction = existing.get(field.name)
        if prediction is None:
            prediction = VoucherFieldPrediction(
                voucher_id=voucher.id,
                field_name=field.name,
                source="manual",
            )
            voucher.predictions.append(prediction)
        if prediction.region_id is None and field.name in region_ids:
            prediction.region_id = region_ids[field.name]
        prediction.confirmed_value = value
        prediction.confirmed_at = confirmed_at
