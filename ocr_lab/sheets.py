"""Листы и склейки кропов для просмотра глазами (человеком или моделью).

Правила листов (`docs/ocr_tasks/_common.md`, § 3): длинная сторона ≤ 2000 px, не больше
100 кропов на лист, у каждого кропа видимая подпись-индекс, рядом с листом — CSV
«индекс → подпись». Страницу целиком сюда не подавать: только кропы подполей, строк дат
и номера ваучера (решение Q5).

- `make_sheet(images, captions, cols, cell_w, out_path)` — сетка кропов с подписями;
- `make_strip(crops, caption)` — склейка кропов друг под другом с подписью сверху.

Кропы принимаются как `PIL.Image` или `numpy` (серое HxW или BGR HxWx3, как у opencv).
"""

from __future__ import annotations

import csv
import math
from collections.abc import Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_lab.paths import ensure_dir

#: Ограничения листа из `_common.md`.
MAX_SIDE = 2000
MAX_CROPS = 100

#: Шрифты с кириллицей, по порядку предпочтения. Первый найденный используется везде.
_FONT_CANDIDATES = (
    "DejaVuSans.ttf",
    "arial.ttf",
    "C:/Windows/Fonts/arial.ttf",
    "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
    "/Library/Fonts/Arial.ttf",
)

ImageLike = Image.Image | np.ndarray


@lru_cache(maxsize=16)
def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    """TTF-шрифт с кириллицей; если не найден — встроенный шрифт Pillow."""
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def to_pil(image: ImageLike) -> Image.Image:
    """Привести кроп к `PIL.Image` в режиме `L` или `RGB` (numpy BGR → RGB)."""
    if isinstance(image, Image.Image):
        return image if image.mode in ("L", "RGB") else image.convert("RGB")
    arr = np.asarray(image)
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    if arr.ndim == 2:
        return Image.fromarray(arr, mode="L")
    if arr.ndim == 3 and arr.shape[2] == 3:
        return Image.fromarray(np.ascontiguousarray(arr[:, :, ::-1]), mode="RGB")
    if arr.ndim == 3 and arr.shape[2] == 4:
        return Image.fromarray(np.ascontiguousarray(arr[:, :, 2::-1]), mode="RGB")
    raise ValueError(f"Неподдерживаемая форма кропа: {arr.shape}")


def _mode_for(images: Sequence[Image.Image]) -> str:
    return "RGB" if any(img.mode == "RGB" for img in images) else "L"


def _resize_to_width(img: Image.Image, width: int) -> Image.Image:
    if img.width == width:
        return img
    height = max(1, round(img.height * width / img.width))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def _wrap(text: str, font: ImageFont.FreeTypeFont | ImageFont.ImageFont, width: int) -> list[str]:
    """Разбить подпись на строки по словам, чтобы каждая влезала в `width` px."""
    probe = ImageDraw.Draw(Image.new("L", (1, 1)))
    lines: list[str] = []
    for paragraph in text.split("\n"):
        current = ""
        for word in paragraph.split(" "):
            candidate = f"{current} {word}" if current else word
            if current and probe.textlength(candidate, font=font) > width:
                lines.append(current)
                current = word
            else:
                current = candidate
        lines.append(current)
    return lines


def _text_block(
    text: str, width: int, font_size: int, mode: str, pad: int = 4
) -> Image.Image:
    """Белая полоса шириной `width` с подписью (с переносами) чёрным текстом."""
    font = _font(font_size)
    lines = _wrap(text, font, width - 2 * pad)
    line_h = round(font_size * 1.25)
    block = Image.new(mode, (width, pad * 2 + line_h * len(lines)), "white")
    draw = ImageDraw.Draw(block)
    for i, line in enumerate(lines):
        draw.text((pad, pad + i * line_h), line, fill="black", font=font)
    return block


def _fit_max_side(img: Image.Image, max_side: int) -> Image.Image:
    longest = max(img.size)
    if longest <= max_side:
        return img
    scale = max_side / longest
    size = (max(1, round(img.width * scale)), max(1, round(img.height * scale)))
    return img.resize(size, Image.Resampling.LANCZOS)


def make_strip(
    crops: Sequence[ImageLike],
    caption: str,
    *,
    width: int | None = None,
    gap: int = 6,
    font_size: int = 28,
    separator: bool = True,
) -> Image.Image:
    """Склеить кропы друг под другом, сверху — подпись.

    Все кропы приводятся к одной ширине `width` (по умолчанию — ширина самого широкого
    кропа) с сохранением пропорций. Между кропами — белый зазор `gap` px и, если
    `separator`, тонкая серая линия, чтобы границы кропов были видны.
    """
    if not crops:
        raise ValueError("make_strip: пустой список кропов")
    pil = [to_pil(c) for c in crops]
    mode = _mode_for(pil)
    pil = [img.convert(mode) for img in pil]
    target_w = width or max(img.width for img in pil)
    pil = [_resize_to_width(img, target_w) for img in pil]

    header = _text_block(caption, target_w, font_size, mode)
    total_h = header.height + sum(img.height for img in pil) + gap * len(pil)
    strip = Image.new(mode, (target_w, total_h), "white")
    strip.paste(header, (0, 0))
    draw = ImageDraw.Draw(strip)
    y = header.height
    for img in pil:
        if separator:
            draw.line([(0, y + gap // 2), (target_w - 1, y + gap // 2)], fill="gray", width=1)
        y += gap
        strip.paste(img, (0, y))
        y += img.height
    return strip


def make_sheet(
    images: Sequence[ImageLike],
    captions: Sequence[str],
    cols: int,
    cell_w: int,
    out_path: Path | str,
    *,
    max_side: int = MAX_SIDE,
    font_size: int = 14,
    pad: int = 6,
    write_index: bool = True,
) -> Image.Image:
    """Сетка кропов с подписями `#<индекс> <подпись>`; сохраняется в `out_path`.

    Кропы масштабируются к ширине ячейки `cell_w`; высота ряда — по самому высокому кропу
    ряда. Если длинная сторона листа больше `max_side`, лист целиком уменьшается.
    Рядом пишется `<out_path без суффикса>.csv` с колонками `index, caption`.
    Больше `MAX_CROPS` кропов на лист — ошибка: делите выборку на несколько листов.
    """
    if len(images) != len(captions):
        raise ValueError("make_sheet: число кропов и подписей не совпадает")
    if not images:
        raise ValueError("make_sheet: пустой список кропов")
    if len(images) > MAX_CROPS:
        raise ValueError(f"make_sheet: больше {MAX_CROPS} кропов на лист ({len(images)})")
    if cols < 1 or cell_w < 1:
        raise ValueError("make_sheet: cols и cell_w должны быть положительными")

    pil = [to_pil(img) for img in images]
    mode = _mode_for(pil)
    cells: list[Image.Image] = []
    for i, (img, caption) in enumerate(zip(pil, captions, strict=True)):
        body = _resize_to_width(img.convert(mode), cell_w)
        label = _text_block(f"#{i} {caption}", cell_w, font_size, mode, pad=2)
        cell = Image.new(mode, (cell_w, label.height + body.height), "white")
        cell.paste(label, (0, 0))
        cell.paste(body, (0, label.height))
        cells.append(cell)

    cols = min(cols, len(cells))
    n_rows = math.ceil(len(cells) / cols)
    row_heights = [
        max(c.height for c in cells[r * cols : (r + 1) * cols]) for r in range(n_rows)
    ]
    sheet_w = pad + cols * (cell_w + pad)
    sheet_h = pad + sum(h + pad for h in row_heights)
    sheet = Image.new(mode, (sheet_w, sheet_h), "white")
    draw = ImageDraw.Draw(sheet)
    y = pad
    for r, row_h in enumerate(row_heights):
        for c in range(cols):
            i = r * cols + c
            if i >= len(cells):
                break
            x = pad + c * (cell_w + pad)
            sheet.paste(cells[i], (x, y))
            draw.rectangle([x - 1, y - 1, x + cell_w, y + cells[i].height], outline="gray")
        y += row_h + pad
    sheet = _fit_max_side(sheet, max_side)

    out = Path(out_path)
    ensure_dir(out.parent)
    sheet.save(out)
    if write_index:
        with out.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fh:
            writer = csv.writer(fh)
            writer.writerow(["index", "caption"])
            writer.writerows(enumerate(captions))
    return sheet
