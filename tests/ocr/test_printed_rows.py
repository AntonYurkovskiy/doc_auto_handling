"""Тесты геометрии печатных строк на выровненной странице (T30), только синтетика."""

from __future__ import annotations

import numpy as np
import pytest

from app.ocr.printed_rows import (
    PRINTED_LAYOUT,
    PRINTED_ROWS,
    ROW_ABOVE,
    ROW_BELOW,
    ROW_RIGHT,
    crop_printed_rows,
    erase_rules,
    has_printed_layout,
)

PAGE_H, PAGE_W = 2340, 1654


def _page() -> np.ndarray:
    return np.full((PAGE_H, PAGE_W), 255, dtype=np.uint8)


def test_every_variant_has_all_rows_in_reading_order() -> None:
    assert PRINTED_ROWS == ("tugboat", "vessel", "agent", "work_type")
    for variant, layout in PRINTED_LAYOUT.items():
        assert set(layout) == set(PRINTED_ROWS), variant
        ys = [layout[row].y for row in PRINTED_ROWS]
        assert ys == sorted(ys), f"{variant}: строки должны идти сверху вниз"
        # шаг между строками больше высоты кропа: соседние строки в кроп не попадают
        assert all(
            b - a > ROW_ABOVE + ROW_BELOW - 10 for a, b in zip(ys, ys[1:], strict=False)
        ), variant


def test_crop_cuts_the_band_around_the_row_right_of_labels() -> None:
    page = _page()
    layout = PRINTED_LAYOUT["kommunar_v1"]

    crops = crop_printed_rows(page, "kommunar_v1")

    assert set(crops) == set(PRINTED_ROWS)
    for row, crop in crops.items():
        assert crop.shape == (ROW_ABOVE + ROW_BELOW, ROW_RIGHT - layout[row].x0), row


def test_crop_contains_the_pixels_of_its_own_row_only() -> None:
    page = _page()
    layout = PRINTED_LAYOUT["pioneer_v2"]
    vessel = layout["vessel"]
    page[vessel.y - 5 : vessel.y + 5, vessel.x0 + 50 : vessel.x0 + 120] = 0  # «текст» судна

    crops = crop_printed_rows(page, "pioneer_v2")

    assert (crops["vessel"] < 100).sum() > 0
    for other in ("tugboat", "agent", "work_type"):
        assert (crops[other] < 100).sum() == 0, other


def test_crop_subset_of_rows() -> None:
    crops = crop_printed_rows(_page(), "pioneer_v1", ("vessel", "agent"))

    assert set(crops) == {"vessel", "agent"}


@pytest.mark.parametrize("variant", [None, "", "unknown_v9"])
def test_unknown_variant_gives_no_crops(variant: str | None) -> None:
    assert not has_printed_layout(variant)
    assert crop_printed_rows(_page(), variant) == {}


def test_crop_is_clipped_to_a_small_page() -> None:
    small = np.full((900, 800), 255, dtype=np.uint8)

    crops = crop_printed_rows(small, "kommunar_v1")

    # на странице меньше эталона часть строк за краем: пустых кропов нет
    assert all(crop.size and crop.shape[0] >= 8 and crop.shape[1] >= 8 for crop in crops.values())
    assert "work_type" not in crops


def test_color_page_is_rejected_not_cropped() -> None:
    assert crop_printed_rows(np.zeros((PAGE_H, PAGE_W, 3), dtype=np.uint8), "kommunar_v1") == {}


# --- стирание подчёркивания ----------------------------------------------------------------


def test_erase_rules_removes_a_long_underline_but_keeps_letters() -> None:
    crop = np.full((64, 900), 255, dtype=np.uint8)
    crop[50:52, 20:880] = 40  # подчёркивание под значением
    for x in range(100, 500, 25):  # «буквы»: короткие вертикальные штрихи
        crop[10:40, x : x + 4] = 0

    cleaned = erase_rules(crop)

    assert (cleaned[48:54] < 150).sum() == 0  # линии нет
    assert (cleaned[10:40, 100:104] < 150).sum() > 0  # штрих буквы остался
    assert cleaned is not crop and (crop[50:52] < 150).sum() > 0  # исходный кроп не тронут


def test_erase_rules_keeps_short_horizontal_strokes() -> None:
    crop = np.full((64, 400), 255, dtype=np.uint8)
    crop[30:33, 100:160] = 0  # перекладина буквы, короче порога линии

    assert (erase_rules(crop)[30:33, 100:160] < 150).all()


def test_erase_rules_is_a_noop_without_lines() -> None:
    crop = np.full((64, 400), 255, dtype=np.uint8)
    crop[10:30, 50:54] = 0

    assert erase_rules(crop) is crop
