"""Поля ваучера: связь колонок Voucher с регионами шаблона и подтверждение."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DocStatus, Tug, Voucher, VoucherFieldPrediction, VoucherRegion
from app.ocr.pipeline import STATUS_OK
from app.services.voucher_linking import link_voucher


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
    VoucherFieldDef("joint_with_line_2", "Совместно с (строка 2)", "joint_with_line_2"),
)


def field_value(voucher: Voucher, field: VoucherFieldDef) -> str | None:
    """Текущее значение поля ваучера в виде строки."""
    if field.attr == "tug_id":
        return voucher.tug.name if voucher.tug else None
    if field.attr == "joint_with_line_2":
        return None
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


def apply_predictions_to_voucher(db: Session, voucher: Voucher) -> None:
    """Заполнить поля ваучера из сохранённых предсказаний (OCR/приоров)."""
    existing = predictions_by_field(voucher)
    tug_ids_by_name = {t.name: t.id for t in db.scalars(select(Tug)).all()}
    for field in VOUCHER_FIELDS:
        prediction = existing.get(field.name)
        if prediction is None or not prediction.predicted_value:
            continue
        value = prediction.predicted_value
        if field.attr == "tug_id":
            voucher.tug_id = tug_ids_by_name.get(value)
        elif field.attr == "joint_with_line_2":
            continue
        elif field.attr.endswith("_dt"):
            try:
                parsed = datetime.strptime(value, "%Y-%m-%d %H:%M")
                setattr(voucher, field.attr, parsed)
            except ValueError:
                pass
        else:
            setattr(voucher, field.attr, value or None)


def voucher_auto_accept(voucher: Voucher) -> bool:
    """Уверенность последнего распознавания (T25/T26) не ниже порога автоприёма."""
    return (
        voucher.ocr_status == STATUS_OK
        and voucher.ocr_confidence is not None
        and voucher.ocr_confidence >= settings.ocr_auto_accept_threshold
    )


def apply_auto_accept(db: Session, voucher: Voucher) -> None:
    """Автоприём высокой уверенности (T26, DECISIONS Q3) после `predict_and_store`.

    При `voucher_auto_accept(voucher)` предсказания всегда переносятся в поля
    ваучера (`apply_predictions_to_voucher`) — это и есть «предзаполнение» и
    пометка «высокая уверенность» (`Voucher.ocr_status`/`ocr_confidence`).
    Статус ваучера по умолчанию не меняется — его подтверждает оператор.
    Флаг `settings.ocr_auto_confirm` (выключен по умолчанию) дополнительно
    подтверждает ваучер автоматически, как ручная кнопка «Подтвердить» —
    поведение включённого флага не подтверждено человеком (DECISIONS Q3
    описывает только выключенное состояние), см. вопрос в PROGRESS.md (T26).
    """
    if not voucher_auto_accept(voucher):
        return
    apply_predictions_to_voucher(db, voucher)
    if not settings.ocr_auto_confirm:
        return
    confirm_fields(db, voucher, datetime.utcnow())
    link_voucher(db, voucher, voucher.application)
    voucher.status = DocStatus.confirmed
    voucher.reviewed_at = datetime.utcnow()
