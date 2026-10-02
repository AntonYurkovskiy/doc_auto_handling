"""Тесты сборщика листов `ocr_lab.sheets` — только синтетика."""

from __future__ import annotations

import csv

import numpy as np
import pytest
from PIL import Image

from ocr_lab.sheets import MAX_CROPS, make_sheet, make_strip, to_pil


def _crop(w: int, h: int, value: int = 0) -> np.ndarray:
    img = np.full((h, w), 255, np.uint8)
    img[h // 4 : 3 * h // 4, w // 4 : 3 * w // 4] = value
    return img


def test_to_pil_gray_and_bgr():
    gray = to_pil(_crop(10, 6))
    assert gray.mode == "L" and gray.size == (10, 6)

    bgr = np.zeros((4, 5, 3), np.uint8)
    bgr[..., 0] = 255  # синий канал в BGR
    rgb = to_pil(bgr)
    assert rgb.mode == "RGB"
    assert rgb.getpixel((0, 0)) == (0, 0, 255)


def test_make_strip_stacks_crops_with_caption():
    crops = [_crop(400, 50), _crop(200, 40), Image.new("L", (800, 60), 200)]
    strip = make_strip(crops, "№ 243k | Выход 02.07 09:10", width=800, gap=6)

    assert strip.width == 800
    # 400x50 → 800x100, 200x40 → 800x160, 800x60 без изменений; плюс подпись и зазоры.
    body = 100 + 160 + 60 + 3 * 6
    assert strip.height > body
    # Подпись сверху: в верхней полосе есть тёмные пиксели текста.
    header = np.asarray(strip)[: strip.height - body]
    assert (header < 128).any()


def test_make_strip_wraps_long_caption():
    short = make_strip([_crop(300, 30)], "коротко", width=300)
    long = make_strip([_crop(300, 30)], " ".join(["Окончание 12.12 23:50"] * 10), width=300)
    assert long.height > short.height


def test_make_strip_rejects_empty():
    with pytest.raises(ValueError):
        make_strip([], "x")


def test_make_sheet_writes_png_and_index(tmp_path):
    images = [_crop(120, 40, value=i * 10) for i in range(7)]
    captions = [f"2025_{i}k" for i in range(7)]
    out = tmp_path / "sheets" / "лист.png"

    sheet = make_sheet(images, captions, cols=3, cell_w=150, out_path=out)

    assert out.exists()
    with Image.open(out) as saved:
        assert saved.size == sheet.size
    with out.with_suffix(".csv").open(encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["caption"] for r in rows] == captions
    assert [r["index"] for r in rows] == [str(i) for i in range(7)]


def test_make_sheet_respects_max_side(tmp_path):
    images = [_crop(300, 300) for _ in range(30)]
    sheet = make_sheet(
        images, [str(i) for i in range(30)], cols=10, cell_w=300, out_path=tmp_path / "s.png"
    )
    assert max(sheet.size) <= 2000


def test_make_sheet_limits(tmp_path):
    with pytest.raises(ValueError):
        make_sheet([_crop(10, 10)] * (MAX_CROPS + 1), ["x"] * (MAX_CROPS + 1), 10, 20,
                   tmp_path / "s.png")
    with pytest.raises(ValueError):
        make_sheet([_crop(10, 10)], ["a", "b"], 2, 20, tmp_path / "s.png")
