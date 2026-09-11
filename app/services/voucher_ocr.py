"""OCR полей ваучера по размеченным регионам шаблона.

Работает поверх Tesseract (pytesseract). Для PDF сначала рендерится первая
страница через pypdfium2, затем по нормализованным координатам
VoucherRegion вырезаются фрагменты и распознаются.

Если двоичный файл Tesseract недоступен, сервис логирует предупреждение
и возвращает пустой результат: приложение продолжает работать на приорах.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import TYPE_CHECKING

from PIL import Image, ImageEnhance

from app.config import settings

try:
    import pytesseract
except ImportError:  # pragma: no cover
    pytesseract = None  # type: ignore[assignment]

if TYPE_CHECKING:
    from app.models import Voucher, VoucherRegion

logger = logging.getLogger(__name__)

OCR_LANG = "rus+eng"
RENDER_DPI = 300
_MIN_CONFIDENCE = 0.3


def _configure_tesseract() -> None:
    """При необходимости переопределяем путь к tesseract из настроек."""
    if pytesseract is None:
        return
    if settings.tesseract_cmd:
        pytesseract.pytesseract.tesseract_cmd = settings.tesseract_cmd


def _load_image(path: Path) -> Image.Image:
    """Открыть изображение и привести к RGB."""
    image: Image.Image = Image.open(path)
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def _render_pdf_first_page(path: Path, dpi: int = RENDER_DPI) -> Image.Image:
    """Отрендерить первую страницу PDF в PIL Image заданным DPI."""
    import pypdfium2 as pdfium

    scale = dpi / 72.0
    pdf = pdfium.PdfDocument(str(path))
    try:
        page = pdf.get_page(0)
        bitmap = page.render(scale=scale)
        try:
            pil_image = bitmap.to_pil()
        finally:
            bitmap.close()
    finally:
        pdf.close()

    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def load_voucher_image(voucher: Voucher, *, dpi: int = RENDER_DPI) -> Image.Image | None:
    """Загрузить картинку ваучера: изображение или первая страница PDF."""
    if not voucher.file_path:
        return None
    path = Path(voucher.file_path)
    if not path.is_file():
        logger.warning("Файл ваучера не найден: %s", path)
        return None

    try:
        if path.suffix.lower() == ".pdf":
            return _render_pdf_first_page(path, dpi=dpi)
        return _load_image(path)
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


def ocr_voucher_regions(voucher: Voucher) -> dict[str, tuple[str | None, float | None]]:
    """Распознать все регионы ваучера. Возвращает {region.name: (text, confidence)}."""
    if voucher.template is None or not voucher.template.regions:
        return {}

    image = load_voucher_image(voucher)
    if image is None:
        return {}

    result: dict[str, tuple[str | None, float | None]] = {}
    for region in voucher.template.regions:
        cropped = crop_region(image, region)
        text, confidence = ocr_image(cropped)
        result[region.name] = (text, confidence)
    return result
