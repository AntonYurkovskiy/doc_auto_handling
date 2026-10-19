"""Печатные строки бланка (буксир, судно, агент, вид работ) на выровненной странице.

Эти строки набраны в редакторе и стоят на бланке на своих местах, но **на скане** их положение
гуляет на ±1 строку (лист лёг на сканер иначе), и у Коммунара и Пионера оно разное. Поэтому
прежний «трафарет» из долей страницы (`app/services/voucher_template.py`) попадал мимо строки у
половины сканов. Здесь строки задаются в пикселях **эталона варианта бланка** (1654×2340), в
которые скан уже приведён выравниванием (`app.ocr.align`): после него разброс положения строк
не больше 25 px при шаге строк 90–110 px.

Каждая строка — блок из двух строк подписи (английской и русской слева) и значения, которое
набрано на строке английской подписи правее обеих подписей. Кроп — узкая полоса вокруг этой
строки правее подписей, чтобы Tesseract читал одну строку значения, без подписей.

Под значением бланк подчёркивает строку линией, и Tesseract на кропе с линией часто не читает
ничего, поэтому длинные горизонтальные линии стираются (`erase_rules`).

Координаты замерены по статичным эталонам `data/ocr/layouts/<вариант>/reference.png`
(положение подписей) и проверены на сканах (`ocr_lab.eval_printed_fields`).
"""

from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass

import cv2
import numpy as np

#: Печатные строки в порядке сверху вниз.
PRINTED_ROWS: tuple[str, ...] = ("tugboat", "vessel", "agent", "work_type")

#: Полоса выше/ниже центра строки подписи, px эталона.
ROW_ABOVE = 32
ROW_BELOW = 32
#: Правый край кропа: правее — поле бланка, на скане там уже край листа.
ROW_RIGHT = 1600
#: Горизонтальная линия длиннее этого (px эталона) — подчёркивание, а не часть букв.
RULE_MIN_LEN = 90
#: Порог тёмного: линии на скане серые, не чисто чёрные.
RULE_DARK = 150


@dataclass(frozen=True)
class RowBox:
    """Строка значения: центр строки английской подписи `y` и левый край кропа `x0`.

    `x0` — правее обеих подписей блока (английской и русской), чтобы в кроп не попадал текст
    подписей. Всё в пикселях эталона варианта бланка.
    """

    y: int
    x0: int


#: Вариант бланка -> строка -> положение.
PRINTED_LAYOUT: dict[str, dict[str, RowBox]] = {
    "kommunar_v1": {
        "tugboat": RowBox(803, 414),
        "vessel": RowBox(892, 370),
        "agent": RowBox(989, 196),
        "work_type": RowBox(1076, 285),
    },
    "pioneer_v1": {
        "tugboat": RowBox(775, 566),
        "vessel": RowBox(880, 523),
        "agent": RowBox(989, 330),
        "work_type": RowBox(1090, 417),
    },
    "pioneer_v2": {
        "tugboat": RowBox(665, 541),
        "vessel": RowBox(777, 502),
        "agent": RowBox(885, 315),
        "work_type": RowBox(984, 399),
    },
}


def erase_rules(crop: np.ndarray) -> np.ndarray:
    """Стереть длинные горизонтальные линии (подчёркивание значения), буквы не трогая.

    Перекладины букв короче `RULE_MIN_LEN`, поэтому открытие по горизонтальному ядру оставляет
    только линии. Маска линий чуть утолщается: по краям открытие теряет пиксель-другой, а у
    сканов край линии размыт.
    """
    dark = (crop < RULE_DARK).astype(np.uint8)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (RULE_MIN_LEN, 1))
    rules = cv2.morphologyEx(dark, cv2.MORPH_OPEN, kernel)
    if not rules.any():
        return crop
    rules = cv2.dilate(rules, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))
    cleaned = crop.copy()
    cleaned[rules > 0] = 255
    return cleaned


def has_printed_layout(variant: str | None) -> bool:
    """Есть ли для варианта бланка координаты печатных строк."""
    return variant in PRINTED_LAYOUT


def crop_printed_rows(
    aligned: np.ndarray, variant: str | None, rows: Iterable[str] = PRINTED_ROWS
) -> dict[str, np.ndarray]:
    """Вырезать строки значений из выровненной страницы (серой, `HxW`).

    Возвращает только строки, для которых есть координаты и непустой кроп. Вариант без
    координат — пустой словарь (вызывающий откатывается на прежний способ).
    """
    layout = PRINTED_LAYOUT.get(variant or "")
    if layout is None or aligned.ndim != 2:
        return {}
    height, width = aligned.shape
    crops: dict[str, np.ndarray] = {}
    for row in rows:
        box = layout.get(row)
        if box is None:
            continue
        top = max(0, box.y - ROW_ABOVE)
        bottom = min(height, box.y + ROW_BELOW)
        left = max(0, box.x0)
        right = min(width, ROW_RIGHT)
        if bottom - top < 8 or right - left < 8:
            continue
        crops[row] = erase_rules(aligned[top:bottom, left:right])
    return crops
