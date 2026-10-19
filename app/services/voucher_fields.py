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


#: Печатные поля, которые приложение может заполнить само, но только проверенным значением:
#: из заявки, из имени файла или найденным в списках известных (`app.services.printed_match`).
AUTOFILL_PRINTED: tuple[str, ...] = ("vessel", "agent", "work_type")
#: Поля, которые автоматически не заполняются никогда: чтение Tesseract здесь давало мусор,
#: а «Совместно с» ещё и определяет число буксиров в расчёте (`tug_count_from_joint`).
NEVER_AUTOFILL: tuple[str, ...] = ("remarks", "joint_with_line_1")
#: Источники предсказания, значение которых можно записать в пустое печатное поле.
AUTOFILL_SOURCES: frozenset[str] = frozenset({"ocr", "application", "filename"})


def _squash(value: object) -> str:
    return " ".join(str(value).split()) if value else ""


def human_value(voucher: Voucher, field_name: str) -> str | None:
    """Значение печатного поля, которое ввёл или подтвердил человек; иначе `None`.

    Поле считается введённым человеком, если по нему нет предсказания, оно уже подтверждено
    (`confirmed_value`) или отличается от последнего предсказания. Совпадение с последним
    предсказанием означает, что значение записало приложение (автозаполнение или «Распознать»):
    такое значение не должно считаться «абсолютной истиной» и перекрывать заявку.
    """
    field = next((f for f in VOUCHER_FIELDS if f.name == field_name), None)
    if field is None or field.attr in ("tug_id", "joint_with_line_2"):
        return None
    current = _squash(getattr(voucher, field.attr, None))
    if not current:
        return None
    row = predictions_by_field(voucher).get(field_name)
    if row is None or row.confirmed_value is not None:
        return current
    machine = row.applied_value if row.applied_value is not None else row.predicted_value
    return None if current == _squash(machine) else current


def remember_machine_values(voucher: Voucher) -> None:
    """Запомнить, какие печатные поля заполнило приложение, до нового предсказания.

    Вызывается перед перезаписью предсказаний: пока `predicted_value` ещё прежнее, значение
    поля, совпадающее с ним, записало приложение (так в базе остаётся и прежний мусор чтения
    Tesseract), и это запоминается в `applied_value`. Иначе после нового предсказания такое
    значение выглядело бы введённым человеком.
    """
    predictions = predictions_by_field(voucher)
    for field in VOUCHER_FIELDS:
        if field.name not in AUTOFILL_PRINTED and field.name not in NEVER_AUTOFILL:
            continue
        row = predictions.get(field.name)
        if row is None or row.applied_value is not None or row.confirmed_value is not None:
            continue
        current = _squash(getattr(voucher, field.attr, None))
        if current and current == _squash(row.predicted_value):
            row.applied_value = current


def printed_hints(voucher: Voucher) -> dict[str, str]:
    """Подсказки для пустых печатных полей: прочитанная строка, не совпавшая с известными.

    Ключ — имя `<input>` формы (`vessel_name`, `agent`, `work_type`). Подсказка показывается
    рядом с пустым полем и в поле не записывается.
    """
    hints: dict[str, str] = {}
    predictions = predictions_by_field(voucher)
    for field in VOUCHER_FIELDS:
        if field.name not in AUTOFILL_PRINTED or _squash(getattr(voucher, field.attr, None)):
            continue
        prediction = predictions.get(field.name)
        if prediction is None or not prediction.topk:
            continue
        try:
            raw = json.loads(prediction.topk)
        except (TypeError, ValueError):
            continue
        if isinstance(raw, list) and raw and isinstance(raw[0], list) and raw[0]:
            text = _squash(raw[0][0])
            if text:
                hints[field.attr] = text
    return hints


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


def _apply_printed(
    voucher: Voucher, field: VoucherFieldDef, prediction: VoucherFieldPrediction | None
) -> None:
    """Печатное поле: только проверенное значение и только в поле, где нет ввода человека.

    Значение, которое прежде записало приложение, заменяется новым; если нового проверенного
    значения нет, прежнее машинное чистится (так уходит мусор прежнего чтения Tesseract).
    """
    if human_value(voucher, field.name) is not None:
        return
    value = None
    if (
        field.name not in NEVER_AUTOFILL
        and prediction is not None
        and prediction.predicted_value
        and prediction.source in AUTOFILL_SOURCES
    ):
        value = prediction.predicted_value
    setattr(voucher, field.attr, value or None)
    if prediction is not None:
        prediction.applied_value = value or None


def apply_predictions_to_voucher(db: Session, voucher: Voucher) -> None:
    """Заполнить поля ваучера из сохранённых предсказаний (OCR/приоров).

    Даты, номер и буксир переносятся как есть. Печатные поля (судно, агент, вид работ,
    примечания, «совместно с») — только проверенными значениями и не поверх ввода человека
    (`_apply_printed`).
    """
    existing = predictions_by_field(voucher)
    tug_ids_by_name = {t.name: t.id for t in db.scalars(select(Tug)).all()}
    for field in VOUCHER_FIELDS:
        prediction = existing.get(field.name)
        if field.name in AUTOFILL_PRINTED or field.name in NEVER_AUTOFILL:
            _apply_printed(voucher, field, prediction)
            continue
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
