"""Подготовка предсказаний полей ваучера.

Ваучеры приходят как скан-картинки фиксированного бланка, но свободный OCR здесь
не выполняется. Сервис готовит приоры: для каждого поля бланка собирается малое
множество кандидатов (из заявки, истории и регламента работы буксиров), из него
выбирается наиболее вероятное значение и сохраняется как предсказание. Картинка
позже уточняет выбор классификацией региона по этому же множеству.

Пока изображение не анализируется, поэтому source предсказаний — "prior", а
confidence прозрачно равна 1 / (число кандидатов).
"""

from __future__ import annotations

import re
from collections.abc import Iterable, Sequence
from dataclasses import dataclass, field
from datetime import datetime, timedelta

from sqlalchemy import select
from sqlalchemy.orm import Session

from app.models import (
    Application,
    Direction,
    Voucher,
    VoucherFieldPrediction,
    VoucherRegion,
)
from app.services.calculation import normalize_work_type

PRINTED_FIELDS: tuple[str, ...] = ("tugboat", "vessel", "agent", "work_type")
DATETIME_FIELDS: tuple[str, ...] = (
    "left_base",
    "arrived_base",
    "started_work",
    "finished_work",
)
TEXT_FIELDS: tuple[str, ...] = (
    "voucher_number",
    "remarks",
    "joint_with_line_1",
    "joint_with_line_2",
)
VOUCHER_FIELDS: tuple[str, ...] = PRINTED_FIELDS + ("voucher_number",) + DATETIME_FIELDS + (
    "remarks",
    "joint_with_line_1",
    "joint_with_line_2",
)

DEFAULT_TUG_NAMES: tuple[str, ...] = ("БК Коммунар", "БК Пионер")

#: Рукописные дата/время ищутся вокруг времени заявки: ±3 дня, шаг минут — 10.
DATE_WINDOW_DAYS = 3
MINUTE_STEP = 10
DATETIME_FORMAT = "%Y-%m-%d %H:%M"

PREDICTION_SOURCE_PRIOR = "prior"

_TUG_KEYS: tuple[tuple[str, str], ...] = (("коммунар", "коммунар"), ("пионер", "пионер"))
_NUMBER_RE = re.compile(r"\d+")
_DATETIME_INPUT_FORMATS: tuple[str, ...] = (
    "%Y-%m-%d %H:%M",
    "%Y-%m-%dT%H:%M",
    "%Y-%m-%d %H:%M:%S",
    "%d.%m.%Y %H:%M",
    "%d.%m.%y %H:%M",
    "%d/%m/%Y %H:%M",
    "%d.%m.%Y",
    "%Y-%m-%d",
)


@dataclass(frozen=True)
class VoucherHistory:
    """Известные из истории варианты значений полей бланка."""

    tug_names: tuple[str, ...] = DEFAULT_TUG_NAMES
    vessel_names: tuple[str, ...] = ()
    agents: tuple[str, ...] = ()
    work_types: tuple[str, ...] = ()
    voucher_numbers: tuple[str, ...] = ()
    remarks: tuple[str, ...] = ()
    joint_with: tuple[str, ...] = ()


@dataclass(frozen=True)
class FieldPrediction:
    """Предсказание одного поля до сохранения в БД."""

    field_name: str
    predicted_value: str | None
    predicted_normalized_value: str | None
    confidence: float | None
    candidates: tuple[str, ...] = field(default=())
    source: str = PREDICTION_SOURCE_PRIOR


def load_history(db: Session, limit: int = 500) -> VoucherHistory:
    """Собрать множества известных значений из ранее подтверждённых ваучеров."""
    vouchers = db.scalars(select(Voucher).order_by(Voucher.id.desc()).limit(limit)).all()
    applications = db.scalars(
        select(Application).order_by(Application.id.desc()).limit(limit)
    ).all()
    return VoucherHistory(
        tug_names=DEFAULT_TUG_NAMES,
        vessel_names=_unique(
            [v.vessel_name for v in vouchers] + [a.vessel_name for a in applications]
        ),
        agents=_unique([v.agent for v in vouchers] + [a.agent for a in applications]),
        work_types=_unique([v.work_type for v in vouchers]),
        voucher_numbers=_unique([v.number for v in vouchers]),
        remarks=_unique([v.remarks for v in vouchers]),
        joint_with=_unique([v.joint_with for v in vouchers]),
    )


def candidate_values(
    field_name: str,
    application: Application | None,
    history: VoucherHistory,
) -> list[str]:
    """Вернуть ограниченное множество вариантов значения поля.

    Печатные поля берут кандидатов из заявки и истории, рукописные дата/время —
    из сетки вокруг времени заявки (±3 дня, часы, минуты кратные 10).
    """
    if field_name == "tugboat":
        return list(history.tug_names or DEFAULT_TUG_NAMES)
    if field_name == "vessel":
        return _unique_list(
            [application.vessel_name if application else None, *history.vessel_names]
        )
    if field_name == "agent":
        return _unique_list([application.agent if application else None, *history.agents])
    if field_name == "work_type":
        return _unique_list([_work_type_from_application(application), *history.work_types])
    if field_name == "voucher_number":
        return _unique_list(history.voucher_numbers)
    if field_name in DATETIME_FIELDS:
        return [dt.strftime(DATETIME_FORMAT) for dt in candidate_datetimes(application)]
    if field_name == "remarks":
        return _unique_list(history.remarks)
    if field_name in ("joint_with_line_1", "joint_with_line_2"):
        return _unique_list(history.joint_with)
    raise ValueError(f"Неизвестное поле ваучера: {field_name}")


def candidate_datetimes(application: Application | None) -> list[datetime]:
    """Сетка возможных дат/времён вокруг времени заявки."""
    reference = _application_datetime(application)
    if reference is None:
        return []
    base_date = reference.date()
    result: list[datetime] = []
    for day_shift in range(-DATE_WINDOW_DAYS, DATE_WINDOW_DAYS + 1):
        day = base_date + timedelta(days=day_shift)
        for hour in range(24):
            for minute in range(0, 60, MINUTE_STEP):
                result.append(datetime(day.year, day.month, day.day, hour, minute))
    return result


def normalize_prediction(field_name: str, value: str | None) -> str | None:
    """Привести значение поля к сравнимому нормализованному виду."""
    if value is None:
        return None
    text = " ".join(value.split())
    if not text:
        return None

    if field_name == "tugboat":
        lowered = text.lower()
        for marker, canonical in _TUG_KEYS:
            if marker in lowered:
                return canonical
        return lowered
    if field_name in ("vessel", "agent"):
        return text.lower()
    if field_name == "work_type":
        return normalize_work_type(text)
    if field_name == "voucher_number":
        match = _NUMBER_RE.search(text)
        return str(int(match.group())) if match else None
    if field_name in DATETIME_FIELDS:
        parsed = _parse_datetime(text)
        return parsed.strftime(DATETIME_FORMAT) if parsed else None
    if field_name in ("remarks", "joint_with_line_1", "joint_with_line_2"):
        return text.lower()
    raise ValueError(f"Неизвестное поле ваучера: {field_name}")


def predict_fields(
    voucher: Voucher,
    application: Application | None,
    history: VoucherHistory,
) -> list[FieldPrediction]:
    """Построить предсказания всех полей бланка без анализа изображения."""
    predictions: list[FieldPrediction] = []
    for field_name in VOUCHER_FIELDS:
        candidates = candidate_values(field_name, application, history)
        predicted = _preferred_value(field_name, voucher, application, candidates)
        predictions.append(
            FieldPrediction(
                field_name=field_name,
                predicted_value=predicted,
                predicted_normalized_value=normalize_prediction(field_name, predicted),
                confidence=round(1.0 / len(candidates), 6) if candidates else None,
                candidates=tuple(candidates),
                source=PREDICTION_SOURCE_PRIOR,
            )
        )
    return predictions


def store_predictions(
    db: Session,
    voucher: Voucher,
    predictions: Sequence[FieldPrediction],
) -> list[VoucherFieldPrediction]:
    """Сохранить предсказания, не затрагивая подтверждённые оператором значения."""
    existing = {row.field_name: row for row in voucher.predictions}
    regions = _regions_by_name(voucher)
    stored: list[VoucherFieldPrediction] = []

    for prediction in predictions:
        row = existing.get(prediction.field_name)
        if row is None:
            row = VoucherFieldPrediction(voucher=voucher, field_name=prediction.field_name)
            db.add(row)
        elif row.confirmed_value is not None:
            stored.append(row)
            continue
        row.region = regions.get(prediction.field_name)
        row.predicted_value = prediction.predicted_value
        row.predicted_normalized_value = prediction.predicted_normalized_value
        row.confidence = prediction.confidence
        row.source = prediction.source
        stored.append(row)

    voucher.predicted_at = datetime.utcnow()
    db.commit()
    return stored


def predict_and_store(
    db: Session,
    voucher: Voucher,
    application: Application | None = None,
    history: VoucherHistory | None = None,
) -> list[VoucherFieldPrediction]:
    """Посчитать предсказания и сохранить их через переданную сессию."""
    application = application or voucher.application
    history = history if history is not None else load_history(db)
    return store_predictions(db, voucher, predict_fields(voucher, application, history))


def _preferred_value(
    field_name: str,
    voucher: Voucher,
    application: Application | None,
    candidates: Sequence[str],
) -> str | None:
    """Наиболее вероятное значение: уже известное по ваучеру, иначе первый кандидат."""
    known = _known_value(field_name, voucher, application)
    if known is not None:
        return known
    return candidates[0] if candidates else None


def _known_value(
    field_name: str, voucher: Voucher, application: Application | None
) -> str | None:
    if field_name == "tugboat":
        return voucher.tug.name if voucher.tug else None
    if field_name == "vessel":
        return voucher.vessel_name or (application.vessel_name if application else None)
    if field_name == "agent":
        return voucher.agent or (application.agent if application else None)
    if field_name == "work_type":
        return voucher.work_type or _work_type_from_application(application)
    if field_name == "voucher_number":
        return voucher.number
    if field_name in DATETIME_FIELDS:
        value = {
            "left_base": voucher.left_base_dt,
            "arrived_base": voucher.arrived_base_dt,
            "started_work": voucher.started_dt,
            "finished_work": voucher.finished_dt,
        }[field_name]
        return value.strftime(DATETIME_FORMAT) if value else None
    if field_name == "remarks":
        return voucher.remarks
    if field_name in ("joint_with_line_1", "joint_with_line_2"):
        return voucher.joint_with if field_name == "joint_with_line_1" else None
    return None


def _work_type_from_application(application: Application | None) -> str | None:
    if application is None:
        return None
    if application.direction == Direction.entry:
        return "швартовка"
    if application.direction == Direction.exit:
        return "отшвартовка"
    return None


def _application_datetime(application: Application | None) -> datetime | None:
    if application is None:
        return None
    return application.entry_datetime or application.exit_datetime or application.received_at


def _regions_by_name(voucher: Voucher) -> dict[str, VoucherRegion]:
    if voucher.template is None:
        return {}
    return {region.name: region for region in voucher.template.regions}


def _parse_datetime(text: str) -> datetime | None:
    for fmt in _DATETIME_INPUT_FORMATS:
        try:
            return datetime.strptime(text, fmt)
        except ValueError:
            continue
    return None


def _unique(values: Iterable[str | None]) -> tuple[str, ...]:
    return tuple(_unique_list(values))


def _unique_list(values: Iterable[str | None]) -> list[str]:
    seen: set[str] = set()
    result: list[str] = []
    for value in values:
        if not value:
            continue
        text = " ".join(value.split())
        if not text or text.lower() in seen:
            continue
        seen.add(text.lower())
        result.append(text)
    return result
