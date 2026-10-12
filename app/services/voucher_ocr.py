"""OCR полей ваучера: печатные поля — Tesseract, рукописные даты/номер — пайплайн T25.

Цепочка «строка → TrOCR», которая раньше распознавала рукописные даты/время
посимвольно, заменена на рантайм-пайплайн распознавания
:func:`app.ocr.pipeline.recognize_voucher` (T25): картинка + приоры + жёсткие
правила совместного декодера (T19/T20) вместо посимвольного OCR. Печатные поля
бланка (буксир, судно, агент, вид работ, примечания, «совместно с») по-прежнему
распознаются Tesseract (pytesseract) по размеченным регионам шаблона.

TrOCR (``voucher_trocr.py``) в рабочий путь больше не входит: модуль оставлен
нетронутым для ручного сравнения качества моделей, в предсказания он не
встроен (решение T26, см. ``docs/ocr_tasks/PROGRESS.md``).

Если пайплайн T25 недоступен в этой среде (нет моделей/эталонов) или
выравнивание скана не прошло пороги, строки дат и номер ваучера просто
отсутствуют в результате — ``predict_fields`` в этом случае берёт их из
приоров (сетка кандидатов вокруг времени заявки), как и до T26.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

from PIL import Image, ImageEnhance
from sqlalchemy import select
from sqlalchemy.orm import Session

from app.config import settings
from app.models import Application, Direction, DocStatus, Voucher, VoucherRegion
from app.ocr import pipeline as ocr_pipeline
from app.ocr.decoder import FormRow
from app.ocr.io import load_scan_pil
from app.services.calculation import normalize_work_type
from app.services.voucher_number import parse_voucher_number, predict_next, voucher_filename_fields

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]

logger = logging.getLogger(__name__)

OCR_LANG = "rus+eng"
RENDER_DPI = 300
_MIN_CONFIDENCE = 0.3

#: Формат значений даты/времени в предсказаниях — совпадает с
#: `app.services.voucher.DATETIME_FORMAT`. Продублирован здесь (не импортирован),
#: чтобы не замкнуть `voucher.py` <-> `voucher_ocr.py` друг на друга.
DATETIME_FORMAT = "%Y-%m-%d %H:%M"

#: Строки дат/времени ваучера — распознаются пайплайном T25, а не Tesseract.
HANDWRITTEN_FIELDS: frozenset[str] = frozenset(
    {"left_base", "arrived_base", "started_work", "finished_work"}
)
#: Печатные поля бланка — как и раньше, Tesseract по размеченному региону целиком.
PRINTED_OCR_FIELDS: frozenset[str] = frozenset(
    {"tugboat", "vessel", "agent", "work_type", "remarks", "joint_with_line_1", "joint_with_line_2"}
)


@dataclass(frozen=True)
class OcrFieldValue:
    """OCR-значение одного поля ваучера: выбранный текст, уверенность и топ-k.

    `topk` — кандидаты по убыванию вероятности в уже нормализованном для
    `predict_fields` виде (дата/время — `DATETIME_FORMAT`, номер — строка цифр).
    Для печатных полей (Tesseract) `topk` пуст — там только одно значение.
    """

    text: str | None
    confidence: float | None
    topk: tuple[tuple[str, float], ...] = ()


@dataclass(frozen=True)
class VoucherOcrResult:
    """Итог распознавания ваучера для `predict_and_store` (T26).

    `recognition` — сырой результат пайплайна T25 (`None`, если он не
    вызывался или недоступен в этой среде); по нему `predict_and_store`
    выставляет `Voucher.ocr_status`/`Voucher.ocr_confidence`.
    """

    fields: dict[str, OcrFieldValue] = field(default_factory=dict)
    recognition: ocr_pipeline.VoucherRecognition | None = None


def _configure_tesseract() -> None:
    """При необходимости переопределяем путь к tesseract из настроек."""
    if pytesseract is None:
        return
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def load_voucher_image(voucher: Voucher, *, dpi: int = RENDER_DPI) -> Image.Image | None:
    """Загрузить картинку ваучера: изображение или первая страница PDF."""
    if not voucher.file_path:
        return None
    path = Path(voucher.file_path)
    if not path.is_file():
        logger.warning("Файл ваучера не найден: %s", path)
        return None

    try:
        return load_scan_pil(path, dpi=dpi, page=0)
    except Exception as exc:  # pragma: no cover
        logger.warning("Не удалось загрузить изображение ваучера %s: %s", path, exc)
        return None


def _preprocess(image: Image.Image) -> Image.Image:
    """Подготовить регион к OCR: grayscale, контраст, масштаб."""
    gray = image.convert("L")
    # Увеличиваем контраст для печатного/рукописного текста
    enhancer = ImageEnhance.Contrast(gray)
    return enhancer.enhance(2.0)


def crop_region(image: Image.Image, region: VoucherRegion) -> Image.Image:
    """Вырезать регион из изображения по нормализованным координатам шаблона."""
    width, height = image.size
    left = int((region.center_x - region.width / 2) * width)
    top = int((region.center_y - region.height / 2) * height)
    right = int((region.center_x + region.width / 2) * width)
    bottom = int((region.center_y + region.height / 2) * height)

    left = max(0, min(left, width))
    top = max(0, min(top, height))
    right = max(0, min(right, width))
    bottom = max(0, min(bottom, height))

    if right <= left or bottom <= top:
        logger.warning(
            "Некорректные координаты региона %s (%s): вырожденный crop",
            region.name,
            region.label,
        )
        return image

    return image.crop((left, top, right, bottom))


def _mean_confidence(confidences: list[int]) -> float:
    """Средняя уверенность Tesseract (conf хранится как целое 0..100)."""
    if not confidences:
        return _MIN_CONFIDENCE
    return sum(confidences) / len(confidences) / 100.0


def ocr_image(image: Image.Image, lang: str = OCR_LANG) -> tuple[str | None, float | None]:
    """Распознать текст на одном изображении. Возвращает (текст, confidence)."""
    if pytesseract is None:
        return None, None
    _configure_tesseract()

    try:
        processed = _preprocess(image)
        # Пробуем получить распознанный текст и среднюю уверенность
        data = pytesseract.image_to_data(
            processed,
            lang=lang,
            output_type=pytesseract.Output.DICT,
        )
        texts: list[str] = []
        confidences: list[int] = []
        for text, conf in zip(data["text"], data["conf"], strict=False):
            if not isinstance(conf, int) or conf < 0:
                continue
            stripped = text.strip()
            if not stripped:
                continue
            # -1 — Tesseract-специфичный «нет текста»
            if conf == -1:
                continue
            texts.append(stripped)
            confidences.append(conf)

        if texts:
            return " ".join(texts), _mean_confidence(confidences)

        # Если image_to_data не вернул строк, пробуем image_to_string
        text = pytesseract.image_to_string(processed, lang=lang).strip()
        return text or None, _MIN_CONFIDENCE if text else None
    except Exception as exc:  # pragma: no cover
        logger.warning("OCR региона завершился с ошибкой: %s", exc)
        return None, None


# ---------------------------------------------------------------------------
# Контекст ваучера для пайплайна T25 (app.ocr.pipeline.RecognitionContext)
# ---------------------------------------------------------------------------


def application_reference_datetime(application: Application | None) -> datetime | None:
    """Время заявки — слабый приор начала работ (вход/выход/получение письма)."""
    if application is None:
        return None
    return application.entry_datetime or application.exit_datetime or application.received_at


def work_type_from_application(application: Application | None) -> str | None:
    """Вид работ по направлению заявки, если он не указан явно на ваучере."""
    if application is None:
        return None
    if application.direction == Direction.entry:
        return "швартовка"
    if application.direction == Direction.exit:
        return "отшвартовка"
    return None


def _context_year(voucher: Voucher, application: Application | None) -> int:
    """Год базовой даты для декодера: известные поля ваучера, затем заявка, затем сегодня."""
    reference = (
        voucher.started_dt or voucher.left_base_dt or voucher.finished_dt or voucher.arrived_base_dt
    )
    if reference is None:
        reference = application_reference_datetime(application)
    if reference is None:
        reference = datetime.utcnow()
    return reference.year


def _number_history(
    db: Session, tug_id: int | None, year: int, *, exclude_voucher_id: int | None
) -> ocr_pipeline.NumberHistory | None:
    """Ожидаемый и занятые номера (буксир, год) без текущего ваучера (приор T20).

    Выбор даты ваучера — как в `app.services.voucher_number.predict_next`
    (`started_dt` важнее `left_base_dt`). `pause_days` не считаем (`None`):
    приложение не знает фактическое время текущего ваучера до распознавания.
    """
    if tug_id is None:
        return None
    used: set[int] = set()
    for other in db.scalars(select(Voucher).where(Voucher.tug_id == tug_id)).all():
        if exclude_voucher_id is not None and other.id == exclude_voucher_id:
            continue
        voucher_date = other.started_dt or other.left_base_dt
        if voucher_date is None or voucher_date.year != year:
            continue
        number, _ = parse_voucher_number(other.number or "")
        if number is not None:
            used.add(number)
    expected = predict_next(db, tug_id, year)
    return ocr_pipeline.NumberHistory(expected=expected, used=frozenset(used), pause_days=None)


def _voucher_forms(voucher: Voucher) -> dict[str, FormRow]:
    """Подтверждённые времена ваучера в написании бланка (приближённо, без `24:00`)."""
    mapping = {
        "left_base": voucher.left_base_dt,
        "arrived_base": voucher.arrived_base_dt,
        "started_work": voucher.started_dt,
        "finished_work": voucher.finished_dt,
    }
    return {row: (dt.date(), dt.hour, dt.minute) for row, dt in mapping.items() if dt is not None}


def _find_partner_voucher(
    db: Session, voucher: Voucher, application: Application
) -> Voucher | None:
    """Второй ваучер той же заявки (другого буксира), если он есть."""
    candidates = db.scalars(
        select(Voucher).where(Voucher.application_id == application.id)
    ).all()
    for other in candidates:
        if other.id == voucher.id:
            continue
        if voucher.tug_id is not None and other.tug_id == voucher.tug_id:
            continue
        return other
    return None


def _build_partner(
    db: Session, voucher: Voucher, application: Application | None
) -> ocr_pipeline.ConfirmedPartner | ocr_pipeline.PendingPartner | None:
    """Пара (T20): второй ваучер той же заявки у другого буксира, если он есть.

    Подтверждённый партнёр со всеми четырьмя строками — сильный приор
    (`ConfirmedPartner`); партнёр, который ещё ждёт проверки, но у которого
    есть файл, — распознаётся вместе с текущим ваучером (`PendingPartner`).
    Без заявки, без партнёра или без его файла — пары нет (`None`).
    """
    if application is None:
        return None
    partner_voucher = _find_partner_voucher(db, voucher, application)
    if partner_voucher is None:
        return None

    forms = _voucher_forms(partner_voucher)
    if partner_voucher.status == DocStatus.confirmed and len(forms) == len(HANDWRITTEN_FIELDS):
        return ocr_pipeline.ConfirmedPartner(forms=forms)

    if not partner_voucher.file_path:
        return None
    partner_code = (
        partner_voucher.tug.code if partner_voucher.tug else None
    ) or voucher_filename_fields(partner_voucher)[1]
    partner_context = build_recognition_context(
        db, partner_voucher, partner_voucher.application, include_partner=False
    )
    return ocr_pipeline.PendingPartner(
        path=Path(partner_voucher.file_path), context=partner_context, tug_hint=partner_code
    )


def build_recognition_context(
    db: Session,
    voucher: Voucher,
    application: Application | None,
    *,
    include_partner: bool = True,
) -> ocr_pipeline.RecognitionContext:
    """Контекст ваучера для пайплайна T25: год, время заявки, вид работ, номер, пара.

    `include_partner=False` — не искать партнёра (используется при построении
    контекста самого партнёра, чтобы не уйти в рекурсию).
    """
    year = _context_year(voucher, application)
    work_type = normalize_work_type(voucher.work_type) or work_type_from_application(application)
    number_history = _number_history(db, voucher.tug_id, year, exclude_voucher_id=voucher.id)
    partner = _build_partner(db, voucher, application) if include_partner else None
    return ocr_pipeline.RecognitionContext(
        year=year,
        app_dt=application_reference_datetime(application),
        work_type=work_type,
        filename=voucher.original_filename,
        number_history=number_history,
        partner=partner,
    )


# ---------------------------------------------------------------------------
# Пайплайн T25: строки дат и номер ваучера
# ---------------------------------------------------------------------------


def _recognize_pipeline(
    db: Session, voucher: Voucher, application: Application | None
) -> ocr_pipeline.VoucherRecognition | None:
    """Вызов пайплайна T25 для одного ваучера; `None` — пайплайн недоступен в этой среде."""
    try:
        resources = ocr_pipeline.get_resources(
            models_dir=settings.ocr_models_dir,
            layouts_dir=settings.ocr_layouts_dir,
            model_version=settings.ocr_model_version,
        )
    except Exception as exc:  # нет моделей/эталонов на машине — работаем на приорах
        logger.warning("Пайплайн распознавания ваучеров недоступен: %s", exc)
        return None

    context = build_recognition_context(db, voucher, application)
    _, file_tug_code = voucher_filename_fields(voucher)
    tug_hint = (voucher.tug.code if voucher.tug else None) or file_tug_code
    assert voucher.file_path is not None
    recognition = ocr_pipeline.recognize_voucher(
        voucher.file_path,
        tug_hint=tug_hint,
        context=context,
        resources=resources,
        auto_accept_threshold=settings.ocr_auto_accept_threshold,
        crops_dir=settings.ocr_crops_dir,
    )
    if not recognition.ok:
        logger.info(
            "Распознавание ваучера %s: %s (%s)", voucher.id, recognition.status, recognition.reason
        )
    return recognition


def _pipeline_field_values(
    recognition: ocr_pipeline.VoucherRecognition,
) -> dict[str, OcrFieldValue]:
    """Строки дат (топ-3 маргиналов) и номер ваучера (топ-3) в виде `OcrFieldValue`."""
    values: dict[str, OcrFieldValue] = {}
    for row, options in recognition.fields.items():
        if not options:
            continue
        topk = tuple((option.value.dt.strftime(DATETIME_FORMAT), option.p) for option in options)
        values[row] = OcrFieldValue(topk[0][0], options[0].p, topk)
    if recognition.voucher_number:
        topk_number = tuple((str(number), p) for number, p in recognition.voucher_number)
        values["voucher_number"] = OcrFieldValue(
            topk_number[0][0], recognition.voucher_number[0][1], topk_number
        )
    return values


def ocr_voucher_regions(
    db: Session, voucher: Voucher, application: Application | None = None
) -> VoucherOcrResult:
    """Распознать все поля ваучера.

    Печатные поля (буксир, судно, агент, вид работ, примечания, «совместно с») —
    Tesseract по размеченным регионам шаблона, как и раньше. Четыре строки дат и
    номер ваучера — пайплайн T25 (`app.ocr.pipeline.recognize_voucher`); если он
    недоступен в этой среде или выравнивание скана не прошло пороги, эти поля в
    результате просто отсутствуют — `predict_fields` берёт их из приоров, как и
    до T26.
    """
    fields: dict[str, OcrFieldValue] = {}

    image = load_voucher_image(voucher)
    if voucher.template is not None and image is not None:
        for region in voucher.template.regions:
            if region.name in PRINTED_OCR_FIELDS:
                text, confidence = ocr_image(crop_region(image, region))
                fields[region.name] = OcrFieldValue(text, confidence)

    recognition: ocr_pipeline.VoucherRecognition | None = None
    if settings.ocr_enabled and voucher.file_path:
        recognition = _recognize_pipeline(db, voucher, application)
        if recognition is not None and recognition.ok:
            fields.update(_pipeline_field_values(recognition))

    return VoucherOcrResult(fields=fields, recognition=recognition)
