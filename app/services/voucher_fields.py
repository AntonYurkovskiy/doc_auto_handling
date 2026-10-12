"""Поля ваучера: связь колонок Voucher с регионами шаблона и подтверждение."""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import datetime

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import DocStatus, Tug, Voucher, VoucherFieldPrediction, VoucherRegion
from app.ocr.layouts import DATE_PARTS, VOUCHER_NUMBER
from app.ocr.pipeline import STATUS_OK, crop_output_dir
from app.services.voucher_linking import link_voucher

#: Формат даты/времени в `VoucherFieldPrediction.topk` — совпадает с
#: `app.services.voucher.DATETIME_FORMAT`. Продублирован (не импортирован), чтобы не
#: замкнуть модули друг на друга (см. `voucher_ocr.py`, тот же приём).
_CANDIDATE_DATETIME_FORMAT = "%Y-%m-%d %H:%M"


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


# ---------------------------------------------------------------------------
# T27: карточка проверки — плашка уверенности, кропы и кнопки top-3
# ---------------------------------------------------------------------------


def voucher_ocr_badge(voucher: Voucher) -> tuple[str, str]:
    """Плашка уровня ваучера для карточки/списка: (css-класс, текст).

    Три состояния из промпта T27: «Распознано уверенно» (та же формула, что
    `voucher_auto_accept`), «Нужна проверка» (пайплайн отработал, но не прошёл
    порог автоприёма) и «Распознавание недоступно» (пайплайн не вызывался,
    не дал `status="ok"` или ваучер ещё не распознавался — `ocr_status is None`).
    """
    if voucher.ocr_status != STATUS_OK:
        return "unavailable", "Распознавание недоступно"
    if voucher_auto_accept(voucher):
        return "confident", "Распознано уверенно"
    return "review", "Нужна проверка"


def voucher_needs_review(voucher: Voucher) -> bool:
    """Фильтр списка `/vouchers` (T27): True — ваучер уходит в очередь проверки.

    Включает и «нужна проверка», и «распознавание недоступно» — оператору в
    обоих случаях нужно проверить ваучер вручную; строгое разделение на три
    состояния остаётся только в плашке (`voucher_ocr_badge`).
    """
    return not voucher_auto_accept(voucher)


def field_confidence_level(confidence: float | None) -> str:
    """Категория уверенности поля для подсветки (T27): high/medium/low/unknown.

    Пороги — `settings.ocr_confidence_high_threshold`/`ocr_confidence_low_threshold`
    (не путать с `settings.ocr_auto_accept_threshold` — тот про запись целиком).
    """
    if confidence is None:
        return "unknown"
    if confidence >= settings.ocr_confidence_high_threshold:
        return "high"
    if confidence >= settings.ocr_confidence_low_threshold:
        return "medium"
    return "low"


def _format_candidate(field_name: str, raw_value: str) -> tuple[str, str]:
    """(значение для `<input>`, подпись на кнопке) для одного кандидата top-k."""
    if field_name == VOUCHER_NUMBER:
        return raw_value, raw_value
    try:
        parsed = datetime.strptime(raw_value, _CANDIDATE_DATETIME_FORMAT)
    except ValueError:
        return raw_value, raw_value
    return parsed.strftime("%Y-%m-%dT%H:%M"), parsed.strftime("%d.%m.%Y %H:%M")


@dataclass(frozen=True)
class QuickFieldCandidate:
    """Один кандидат top-k — кнопка быстрого выбора значения поля (T27)."""

    input_value: str
    label: str
    probability: float | None


@dataclass(frozen=True)
class QuickField:
    """Поле карточки с кропами подполей и кнопками top-3 (T27): номер и 4 строки дат.

    `crop_names` — только те подполя (`app.ocr.layouts.SUBFIELD_NAMES`), для которых
    на диске реально нашёлся файл кропа (отдаются через `/vouchers/{id}/crops/<имя>`).
    """

    field_name: str
    input_name: str
    label: str
    crop_names: tuple[str, ...]
    confidence: float | None
    confidence_level: str
    candidates: tuple[QuickFieldCandidate, ...]


#: (имя поля в VoucherFieldPrediction, имя <input> формы, подпись).
_QUICK_FIELD_DEFS: tuple[tuple[str, str, str], ...] = (
    (VOUCHER_NUMBER, "number", "№ ваучера"),
    ("left_base", "left_base_dt", "Выход из базы"),
    ("arrived_base", "arrived_base_dt", "Приход в базу"),
    ("started_work", "started_dt", "Начало работ"),
    ("finished_work", "finished_dt", "Окончание работ"),
)


def _crop_names(field_name: str) -> tuple[str, ...]:
    if field_name == VOUCHER_NUMBER:
        return (VOUCHER_NUMBER,)
    return tuple(f"{field_name}.{part}" for part in DATE_PARTS)


def _topk_candidates(
    field_name: str, prediction: VoucherFieldPrediction | None
) -> tuple[QuickFieldCandidate, ...]:
    if prediction is None or not prediction.topk:
        return ()
    try:
        raw = json.loads(prediction.topk)
    except (TypeError, ValueError):
        return ()
    if not isinstance(raw, list):
        return ()
    candidates: list[QuickFieldCandidate] = []
    for item in raw[:3]:
        if not isinstance(item, list | tuple) or len(item) != 2:
            continue
        value, probability = item
        input_value, label = _format_candidate(field_name, str(value))
        candidates.append(
            QuickFieldCandidate(
                input_value=input_value,
                label=label,
                probability=float(probability) if isinstance(probability, int | float) else None,
            )
        )
    return tuple(candidates)


def quick_fields(voucher: Voucher) -> tuple[QuickField, ...]:
    """Кропы и кнопки top-3 для номера ваучера и 4 строк дат (T27)."""
    predictions = predictions_by_field(voucher)
    crop_dir = (
        crop_output_dir(voucher.file_path, settings.ocr_crops_dir) if voucher.file_path else None
    )
    result: list[QuickField] = []
    for field_name, input_name, label in _QUICK_FIELD_DEFS:
        prediction = predictions.get(field_name)
        confidence = prediction.confidence if prediction else None
        available_crops = tuple(
            name
            for name in _crop_names(field_name)
            if crop_dir is not None and (crop_dir / f"{name}.png").is_file()
        )
        result.append(
            QuickField(
                field_name=field_name,
                input_name=input_name,
                label=label,
                crop_names=available_crops,
                confidence=confidence,
                confidence_level=field_confidence_level(confidence),
                candidates=_topk_candidates(field_name, prediction),
            )
        )
    return tuple(result)
