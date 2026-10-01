"""Загрузка сканов ваучеров: PDF (pypdfium2) и картинки (Pillow) -> серое uint8.

Единая точка входа для рантайма приложения (`app/services/voucher_ocr.py`) и
лаборатории (`ocr_lab`): все получают серое изображение `uint8` предсказуемого
масштаба — ширина A4 при 200 dpi равна 1654 px (`normalize_width`).

Только CPU и лёгкие зависимости: numpy, opencv-python-headless, pypdfium2, Pillow.
torch сюда не импортировать (см. `docs/ocr_tasks/_common.md`).
"""

from __future__ import annotations

import math
from pathlib import Path

import cv2
import numpy as np
import pypdfium2 as pdfium
from PIL import Image, ImageOps

#: DPI по умолчанию для рендера PDF в `load_scan`.
DEFAULT_DPI = 200

#: Ширина страницы A4 при 200 dpi — целевой масштаб кэша `data/ocr/pages/`.
PAGE_WIDTH_200DPI = 1654

#: Форматы картинок, которые читаем через Pillow (с учётом EXIF-поворота).
IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png", ".tif", ".tiff", ".webp", ".bmp"})


class ScanLoadError(RuntimeError):
    """Не удалось прочитать скан: формат, битый файл или страница вне диапазона."""


def _check_file(path: Path) -> str:
    """Проверить существование файла и расширение; вернуть вид скана."""
    if not path.is_file():
        raise FileNotFoundError(f"Файл скана не найден: {path.name}")
    suffix = path.suffix.lower()
    if suffix == ".pdf":
        return "pdf"
    if suffix in IMAGE_SUFFIXES:
        return "image"
    raise ScanLoadError(f"Неподдерживаемый формат скана: {path.name}")


def _render_pdf_page(path: Path, dpi: int, page: int) -> Image.Image:
    """Отрендерить страницу `page` PDF-документа в RGB-картинку заданного DPI."""
    try:
        pdf = pdfium.PdfDocument(str(path))
    except Exception as exc:
        raise ScanLoadError(f"Не удалось открыть PDF {path.name}: {exc}") from exc
    try:
        n_pages = len(pdf)
        if not 0 <= page < n_pages:
            raise ScanLoadError(
                f"Страница {page} вне диапазона 0..{n_pages - 1} в PDF {path.name}"
            )
        pdf_page = pdf.get_page(page)
        try:
            bitmap = pdf_page.render(scale=dpi / 72.0)
            try:
                pil_image = bitmap.to_pil()
            finally:
                bitmap.close()
        finally:
            pdf_page.close()
    except ScanLoadError:
        raise
    except Exception as exc:
        raise ScanLoadError(f"Не удалось отрендерить PDF {path.name}: {exc}") from exc
    finally:
        pdf.close()
    if pil_image.mode != "RGB":
        pil_image = pil_image.convert("RGB")
    return pil_image


def _load_image_file(path: Path, page: int) -> Image.Image:
    """Открыть картинку через Pillow, применить EXIF-поворот, вернуть RGB."""
    try:
        image: Image.Image = Image.open(path)
        if page:
            # Многостраничный TIFF: кадр `page`. У однокадровых форматов seek
            # за пределы бросит EOFError — превращаем в понятную ошибку.
            try:
                image.seek(page)
            except EOFError as exc:
                raise ScanLoadError(
                    f"Кадр {page} вне диапазона в изображении {path.name}"
                ) from exc
        image = ImageOps.exif_transpose(image)
        image.load()
    except ScanLoadError:
        raise
    except Exception as exc:
        raise ScanLoadError(f"Не удалось прочитать изображение {path.name}: {exc}") from exc
    if image.mode != "RGB":
        image = image.convert("RGB")
    return image


def load_scan_pil(path: str | Path, *, dpi: int = DEFAULT_DPI, page: int = 0) -> Image.Image:
    """Загрузить страницу скана как RGB-картинку PIL.

    PDF рендерится через pypdfium2 с масштабом `dpi` (по умолчанию страница 0),
    картинки (JPG, PNG, TIFF, WEBP, BMP) открываются через Pillow с применением
    EXIF-поворота. Ошибки — `FileNotFoundError` и `ScanLoadError` с именем файла,
    без его содержимого.
    """
    path = Path(path)
    kind = _check_file(path)
    if kind == "pdf":
        return _render_pdf_page(path, dpi=dpi, page=page)
    return _load_image_file(path, page=page)


def load_scan(path: str | Path, *, dpi: int = DEFAULT_DPI, page: int = 0) -> np.ndarray:
    """Загрузить страницу скана как серое изображение `uint8`, форма HxW."""
    image = load_scan_pil(path, dpi=dpi, page=page)
    gray = np.asarray(image.convert("L"), dtype=np.uint8)
    image.close()
    return gray


def page_count(path: str | Path) -> int:
    """Число страниц в скане: для PDF — страниц документа, для картинки — кадров."""
    path = Path(path)
    kind = _check_file(path)
    if kind == "pdf":
        try:
            pdf = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise ScanLoadError(f"Не удалось открыть PDF {path.name}: {exc}") from exc
        try:
            return len(pdf)
        finally:
            pdf.close()
    try:
        with Image.open(path) as image:
            return int(getattr(image, "n_frames", 1))
    except Exception as exc:
        raise ScanLoadError(f"Не удалось прочитать изображение {path.name}: {exc}") from exc


def page_size_px(path: str | Path, *, dpi: int = DEFAULT_DPI, page: int = 0) -> tuple[int, int]:
    """Размер страницы (ширина, высота) в px, который дал бы `load_scan`.

    Без рендера: для PDF — из размера страницы в пунктах (72 pt = 1 дюйм,
    с учётом флага поворота страницы), для картинки — из её собственного
    размера с учётом EXIF-поворота.
    """
    path = Path(path)
    kind = _check_file(path)
    if kind == "pdf":
        try:
            pdf = pdfium.PdfDocument(str(path))
        except Exception as exc:
            raise ScanLoadError(f"Не удалось открыть PDF {path.name}: {exc}") from exc
        try:
            n_pages = len(pdf)
            if not 0 <= page < n_pages:
                raise ScanLoadError(
                    f"Страница {page} вне диапазона 0..{n_pages - 1} в PDF {path.name}"
                )
            pdf_page = pdf.get_page(page)
            try:
                w_pt, h_pt = pdf_page.get_size()
                rotation = pdf_page.get_rotation()
            finally:
                pdf_page.close()
        finally:
            pdf.close()
        if rotation in (90, 270):
            w_pt, h_pt = h_pt, w_pt
        # pypdfium2 считает размер битмапа как ceil(pt * scale): 595pt при 200 dpi
        # даёт 1653 px, при 300 — 2480 px.
        scale = dpi / 72.0
        return math.ceil(w_pt * scale), math.ceil(h_pt * scale)

    try:
        image = Image.open(path)
        if page:
            try:
                image.seek(page)
            except EOFError as exc:
                raise ScanLoadError(
                    f"Кадр {page} вне диапазона в изображении {path.name}"
                ) from exc
        return ImageOps.exif_transpose(image).size
    except ScanLoadError:
        raise
    except Exception as exc:
        raise ScanLoadError(f"Не удалось прочитать изображение {path.name}: {exc}") from exc


def normalize_width(img: np.ndarray, width: int = PAGE_WIDTH_200DPI) -> np.ndarray:
    """Привести изображение к ширине `width` с сохранением пропорций.

    При уменьшении используется `INTER_AREA` (не мылит тонкие линии бланка),
    при увеличении — `INTER_LINEAR`. Ширина 1654 px соответствует A4 при 200 dpi.
    """
    if img.ndim not in (2, 3):
        raise ValueError(f"Ожидается изображение HxW или HxWxC, получено shape={img.shape}")
    h, w = img.shape[:2]
    if w == width:
        return img
    new_height = max(1, round(h * width / w))
    interpolation = cv2.INTER_AREA if width < w else cv2.INTER_LINEAR
    return cv2.resize(img, (width, new_height), interpolation=interpolation)
