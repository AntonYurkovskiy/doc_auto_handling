"""Кропы подполей из выровненного скана по макету боксов (:mod:`app.ocr.layouts`).

Скан сначала выравнивается по эталону своего варианта (:func:`app.ocr.align.align`),
``warped`` имеет размер эталона. Дальше боксы подгоняются по подчёркиваниям (T08).

Почему подгонка нужна. Бланк заполняют в редакторе, поэтому печатный текст строки дат
(«__» __.2025 : Time ____ Hrs. ____ Min.) «плывёт» от скана к скану: на реальных
сканах Коммунара подчёркивания сдвинуты по горизонтали до 70 px (p95 ≈ 55 px), строка
бывает наклонена на 10–15 px, а гомография выравнивает страницу по
неподвижной печати и этого не исправляет. Подчёркивание «____» каждого подполя едет
вместе с текстом, и цифры пишут на нём, поэтому оно — надёжный якорь бокса.

Как подгоняется бокс с ``line = (x0, x1, y)`` (подчёркивание в эталоне):

1. В окне вокруг ``line`` ищутся горизонтальные линии (:func:`find_lines`). Линии короче
   :data:`MIN_LINE_FRACTION` от подчёркивания эталона не берутся: это горизонтальные
   штрихи рукописи (верх «5» и «7», овал «0»), а не подчёркивание.
2. Если есть линия, у которой оба конца не дальше :data:`EDGE_TOLERANCE_PX` от концов
   эталонного подчёркивания, берётся она (при нескольких — с наименьшей суммой отклонений
   концов и длины). Иначе левый край подчёркивания — ближайший к ``x0`` левый край
   какой-нибудь линии (не дальше :data:`EDGE_TOLERANCE_PX`), правый — так же. Края,
   упёршиеся в окно поиска, не считаются: это обрезанная линия, а не её конец.
3. Найдены оба края:

   - день и номер: левая сторона бокса едет за левым краем, правая — за правым. Цифры
     стоят на самом подчёркивании (день — между «»), и на коротком подчёркивании бокс
     сужается, не захватывая месяц;
   - месяц, часы, минуты (:data:`UNION_PARTS`): бокс покрывает обе гипотезы, «сдвиг по
     левому краю» и «сдвиг по правому краю», то есть расширяется, но не сужается. Здесь
     не всегда понятно, какой край держит цифры: на бланках 2026 г. с крупным печатным
     месяцем («07._2026») подчёркивание начинается после месяца, а у бледного скана
     подчёркивание рвётся.

   Найден один край — бокс едет за ним. Вертикальный сдвиг — по высоте найденной линии.
4. Если краёв рядом нет (подчёркивание уехало дальше :data:`EDGE_TOLERANCE_PX` — так
   бывает с часами и минутами, они плывут отдельно от дня и месяца), ищется целая линия
   длиной как подчёркивание эталона (±:data:`LENGTH_TOL`), ближайшая к нему по центру,
   в окне шире по горизонтали (:data:`SEARCH_DX_WIDE_PX`; ``source="line_len"``).
5. Если не нашлось ничего во всей строке (так бывает с шапкой: на единичных сканах
   она съезжает на 70 px по вертикали и 100 px по горизонтали), такой же поиск идёт и
   в окне выше по вертикали (:data:`SEARCH_DY_WIDE_PX`; ``source="line_wide"``).
6. Иначе бокс берёт медианный сдвиг той же части в других строках (``source="column"``:
   табуляция у строк общая), а высоту — от ближайшего найденного бокса своей строки.
   Если такой части нигде не нашлось — сдвиг ближайшего по горизонтали найденного бокса
   той же строки (``source="row"``), а если и в строке ничего — бокс остаётся на месте
   (``source="static"``).
7. Соседи по строке не залезают друг к другу: бокс не заходит левее найденного правого
   края подчёркивания левого соседа и правее найденного левого края правого соседа.
   Так в бокс минут не попадают часы, а в бокс месяца — день. Если от такой обрезки бокс
   стал бы уже :data:`MIN_WIDTH_FRACTION` эталонного, она не применяется.

Зависимости — только numpy и opencv.
"""

from __future__ import annotations

import math
import statistics
from collections.abc import Iterable
from dataclasses import dataclass, replace

import cv2
import numpy as np

from app.ocr.layouts import Box, Layout

#: Окно поиска подчёркивания вокруг эталонного: по горизонтали и вертикали, px.
SEARCH_DX_PX = 110
SEARCH_DY_PX = 26
#: Окно поиска целой линии по длине: по горизонтали; по вертикали — для строки, в которой
#: ничего не нашлось (меньше шага строк дат ≈ 92 px, чтобы не взять соседнюю строку).
SEARCH_DX_WIDE_PX = 150
SEARCH_DY_WIDE_PX = 75
#: Допуск длины линии при поиске по длине, доля длины подчёркивания эталона.
LENGTH_TOL = 0.2
#: Край найденной линии дальше этого от эталонного — чужая линия.
EDGE_TOLERANCE_PX = 71
#: Минимальная длина линии и порог «тёмного» для поиска линий.
MIN_LINE_PX = 50
DARK_THRESHOLD = 160
#: Ядро открытия (минимальный горизонтальный отрезок) и закрытия (мостик через разрыв).
#: Оба нечётные: у чётного ядра якорь не в центре, и открытие сдвигает линию на 1 px.
OPEN_PX = 41
CLOSE_PX = 9
#: Линии короче этой доли подчёркивания эталона — штрихи рукописи, а не подчёркивание.
MIN_LINE_FRACTION = 0.45
#: Части строки, у которых бокс по двум краям расширяется, но не сужается (см. п. 3).
UNION_PARTS = frozenset({"month", "hour", "minute"})
#: Обрезка по соседу не делает бокс уже этой доли эталонной ширины.
MIN_WIDTH_FRACTION = 0.6
#: Сдвиг бокса больше этого считается ошибкой поиска и не применяется.
MAX_SHIFT_PX = 90
#: Боксы с подчёркиваниями ближе этого по высоте — одна строка (запасные сдвиги, обрезка).
SAME_ROW_PX = 30


@dataclass(frozen=True)
class LineSegment:
    """Горизонтальная линия: ``[x0, x1)`` по горизонтали, ``y`` — середина по толщине."""

    x0: int
    x1: int
    y: float


@dataclass(frozen=True)
class Placement:
    """Бокс на конкретном скане и то, как он найден.

    ``source``: ``line`` — по обоим краям подчёркивания, ``line_left``/``line_right`` —
    по одному краю, ``line_len`` — по целой линии подходящей длины, ``line_wide`` — то же
    в широком окне, ``column`` — сдвиг той же части в других строках, ``row`` — сдвиг
    ближайшего найденного бокса строки, ``static`` — без
    сдвига (или бокс без ``line``). ``dx0``/``dx1`` — сдвиги левой и правой сторон бокса.
    """

    box: Box
    source: str
    dx0: int
    dx1: int
    dy: int


def find_lines(gray: np.ndarray, *, offset: tuple[int, int] = (0, 0)) -> list[LineSegment]:
    """Горизонтальные линии не короче :data:`MIN_LINE_PX` на сером изображении.

    ``offset`` — ``(x, y)`` левого верхнего угла ``gray`` в координатах страницы.
    """
    dark = (gray < DARK_THRESHOLD).astype(np.uint8)
    lines = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_PX, 1))
    )
    lines = cv2.morphologyEx(
        lines, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (CLOSE_PX, 1))
    )
    n, _, stats, centroids = cv2.connectedComponentsWithStats(lines, connectivity=8)
    ox, oy = offset
    out = []
    for i in range(1, n):
        x, _, w, _, _ = stats[i]
        if w >= MIN_LINE_PX:
            out.append(LineSegment(int(x) + ox, int(x + w) + ox, float(centroids[i][1]) + oy))
    return sorted(out, key=lambda s: s.x0)


def _nearest(values: list[tuple[int, float]], target: int) -> tuple[int, float] | None:
    """Ближайшее к ``target`` значение ``(x, y)`` не дальше :data:`EDGE_TOLERANCE_PX`."""
    best = min(values, key=lambda v: abs(v[0] - target), default=None)
    if best is None or abs(best[0] - target) > EDGE_TOLERANCE_PX:
        return None
    return best


def _window(
    gray: np.ndarray, box: Box, dx: int, dy: int
) -> tuple[list[LineSegment], int, int]:
    """Линии в окне вокруг подчёркивания бокса и левая/правая границы окна."""
    assert box.line is not None
    lx0, lx1, ly = box.line
    h, w = gray.shape[:2]
    wx0, wx1 = max(lx0 - dx, 0), min(lx1 + dx, w)
    wy0, wy1 = max(ly - dy, 0), min(ly + dy + 1, h)
    if wx1 <= wx0 or wy1 <= wy0:
        return [], wx0, wx1
    return find_lines(gray[wy0:wy1, wx0:wx1], offset=(wx0, wy0)), wx0, wx1


def _fit_line_len(gray: np.ndarray, box: Box, search_dy: int) -> LineSegment | None:
    """Целая линия длиной как подчёркивание эталона, ближайшая к нему, или ``None``."""
    assert box.line is not None
    lx0, lx1, ly = box.line
    segments, wx0, wx1 = _window(gray, box, SEARCH_DX_WIDE_PX, search_dy)
    length = lx1 - lx0
    whole = [
        s
        for s in segments
        if s.x0 > wx0 and s.x1 < wx1 and abs((s.x1 - s.x0) - length) <= LENGTH_TOL * length
    ]
    if not whole:
        return None
    cx, cy = (lx0 + lx1) / 2, ly
    return min(whole, key=lambda s: math.hypot((s.x0 + s.x1) / 2 - cx, s.y - cy))


def _fit_line(gray: np.ndarray, box: Box) -> tuple[int | None, int | None, int | None]:
    """Найденные края подчёркивания ``(левый x, правый x)`` и вертикальный сдвиг.

    Каждое значение — ``None``, если не найдено.
    """
    assert box.line is not None
    lx0, lx1, ly = box.line
    segments, wx0, wx1 = _window(gray, box, SEARCH_DX_PX, SEARCH_DY_PX)
    segments = [s for s in segments if s.x1 - s.x0 >= MIN_LINE_FRACTION * (lx1 - lx0)]
    whole = [
        s
        for s in segments
        if s.x0 > wx0
        and s.x1 < wx1
        and abs(s.x0 - lx0) <= EDGE_TOLERANCE_PX
        and abs(s.x1 - lx1) <= EDGE_TOLERANCE_PX
    ]
    if whole:
        # Одна линия с обоими концами рядом надёжнее краёв разных линий: короткий
        # горизонтальный штрих рукописи (низ «2») не перетянет на себя один край.
        best = min(
            whole,
            key=lambda s: abs(s.x0 - lx0) + abs(s.x1 - lx1) + abs((s.x1 - s.x0) - (lx1 - lx0)),
        )
        return best.x0, best.x1, round(best.y - ly)
    lefts = [(s.x0, s.y) for s in segments if s.x0 > wx0]
    rights = [(s.x1, s.y) for s in segments if s.x1 < wx1]
    left = _nearest(lefts, lx0)
    right = _nearest(rights, lx1)
    ys = [v[1] for v in (left, right) if v is not None]
    dy = None if not ys else round(statistics.fmean(ys) - ly)
    return (
        None if left is None else left[0],
        None if right is None else right[0],
        dy,
    )


def _center(box: Box) -> float:
    """Середина подчёркивания бокса по горизонтали (или самого бокса, если линии нет)."""
    x0, x1 = (box.line[0], box.line[1]) if box.line is not None else (box.x0, box.x1)
    return (x0 + x1) / 2


def _nearest_box(boxes: list[Box], box: Box) -> Box:
    """Ближайший к ``box`` по горизонтали бокс из ``boxes``."""
    return min(boxes, key=lambda other: abs(_center(other) - _center(box)))


def _shifted(box: Box, dx0: int, dx1: int, dy: int) -> Box:
    return replace(box, x0=box.x0 + dx0, x1=box.x1 + dx1, y0=box.y0 + dy, y1=box.y1 + dy)


def _edge_shifts(box: Box, left: int | None, right: int | None) -> tuple[int, int] | None:
    """Сдвиги левой и правой сторон бокса по найденным краям подчёркивания (п. 3)."""
    assert box.line is not None
    dxl = None if left is None else left - box.line[0]
    dxr = None if right is None else right - box.line[1]
    if dxl is not None and dxr is not None:
        if box.part in UNION_PARTS:
            return min(dxl, dxr), max(dxl, dxr)
        return dxl, dxr
    shift = dxl if dxl is not None else dxr
    return None if shift is None else (shift, shift)


def locate_boxes(warped: np.ndarray, layout: Layout) -> dict[str, Placement]:
    """Положение боксов макета на выровненном скане: ``{имя: Placement}``.

    Боксы без ``line`` остаются на месте. Координаты могут выйти за изображение —
    :func:`crop_box` зальёт недостающее белым.
    """
    gray = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    # Сдвиги (dx0, dx1, dy, source) боксов, найденных по линии, и найденные края линий.
    fits: dict[str, tuple[int, int, int, str]] = {}
    edges: dict[str, tuple[int | None, int | None]] = {}
    lined = [box for box in layout.boxes if box.line is not None]
    for box in lined:
        left, right, dy = _fit_line(gray, box)
        shifts = _edge_shifts(box, left, right)
        if shifts is None or max(abs(shifts[0]), abs(shifts[1]), abs(dy or 0)) > MAX_SHIFT_PX:
            continue
        if left is not None and right is not None:
            source = "line"
        else:
            source = "line_left" if left is not None else "line_right"
        fits[box.name] = (*shifts, dy or 0, source)
        edges[box.name] = (left, right)

    def fit_by_length(box: Box, search_dy: int, source: str) -> None:
        assert box.line is not None
        seg = _fit_line_len(gray, box, search_dy)
        if seg is not None:
            shifts = _edge_shifts(box, seg.x0, seg.x1)
            assert shifts is not None
            fits[box.name] = (*shifts, round(seg.y - box.line[2]), source)
            edges[box.name] = (seg.x0, seg.x1)

    for box in lined:
        if box.name not in fits:
            fit_by_length(box, SEARCH_DY_PX, "line_len")

    def row_mates(box: Box) -> list[Box]:
        assert box.line is not None
        return [
            other
            for other in lined
            if other.name != box.name
            and other.line is not None
            and abs(other.line[2] - box.line[2]) <= SAME_ROW_PX
        ]

    for box in lined:
        if box.name not in fits and not any(m.name in fits for m in row_mates(box)):
            fit_by_length(box, SEARCH_DY_WIDE_PX, "line_wide")

    placed: dict[str, tuple[int, int, int, str]] = {}
    for box in layout.boxes:
        found_mates = [m for m in row_mates(box) if m.name in fits] if box.line else []
        column = [
            m for m in lined if m.name in fits and m.part is not None and m.part == box.part
        ]
        if box.name in fits:
            placed[box.name] = fits[box.name]
        elif box.line is not None and column:
            # Та же часть в других строках: бланк набран в редакторе, табуляция у строк
            # общая. Высота — от ближайшего соседа по строке (строка бывает наклонена).
            d0 = round(statistics.median(fits[m.name][0] for m in column))
            d1 = round(statistics.median(fits[m.name][1] for m in column))
            if found_mates:
                dy = fits[_nearest_box(found_mates, box).name][2]
            else:
                dy = round(statistics.median(fits[m.name][2] for m in column))
            placed[box.name] = (d0, d1, dy, "column")
        elif found_mates:
            # Ближайший по горизонтали сосед: часы и минуты плывут вместе, день и месяц —
            # вместе, а две половины строки — независимо.
            d0, d1, dy, _ = fits[_nearest_box(found_mates, box).name]
            placed[box.name] = (d0, d1, dy, "row")
        else:
            placed[box.name] = (0, 0, 0, "static")

    out: dict[str, Placement] = {}
    for box in layout.boxes:
        d0, d1, dy, source = placed[box.name]
        x0, x1 = box.x0 + d0, box.x1 + d1
        if box.line is not None:
            # П. 7: не заходить за найденные концы подчёркиваний соседей по строке.
            min_width = MIN_WIDTH_FRACTION * box.width
            for mate in row_mates(box):
                mate_left, mate_right = edges.get(mate.name, (None, None))
                if _center(mate) < _center(box) and mate_right is not None:
                    if mate_right > x0 and x1 - mate_right >= min_width:
                        x0 = mate_right
                elif _center(mate) > _center(box) and mate_left is not None:
                    if mate_left < x1 and mate_left - x0 >= min_width:
                        x1 = mate_left
        d0, d1 = x0 - box.x0, x1 - box.x1
        out[box.name] = Placement(_shifted(box, d0, d1, dy), source, d0, d1, dy)
    return out


def crop_box(warped: np.ndarray, box: Box, *, pad: int = 0, fill: int = 255) -> np.ndarray:
    """Кроп одного бокса с отступом ``pad`` px с каждой стороны.

    Форма кропа всегда ``(box.height + 2·pad, box.width + 2·pad)`` (плюс каналы, если они
    есть): часть, выходящая за изображение, заливается ``fill`` (по умолчанию белым), так что
    положение цифры внутри кропа не зависит от близости к краю.
    """
    if pad < 0:
        raise ValueError(f"pad должен быть ≥ 0: {pad}")
    h, w = warped.shape[:2]
    x0, y0, x1, y1 = box.x0 - pad, box.y0 - pad, box.x1 + pad, box.y1 + pad
    out = np.full((y1 - y0, x1 - x0, *warped.shape[2:]), fill, dtype=warped.dtype)
    sx0, sy0, sx1, sy1 = max(x0, 0), max(y0, 0), min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0 : sy1 - y0, sx0 - x0 : sx1 - x0] = warped[sy0:sy1, sx0:sx1]
    return out


def crop_subfields(
    warped: np.ndarray,
    layout: Layout,
    pad: int = 0,
    *,
    names: Iterable[str] | None = None,
    fill: int = 255,
    refine: bool = True,
) -> dict[str, np.ndarray]:
    """Кропы всех (или только ``names``) подполей макета: ``{имя: кроп}``.

    ``warped`` — скан, выровненный по эталону варианта ``layout.variant``; его размер
    должен совпадать с ``layout.ref_size``, иначе ``ValueError`` (скан выровнен по
    другому эталону или не выровнен вовсе). Запас для рукописи уже заложен в боксы;
    ``pad`` добавляет поле сверх него (например, для аугментаций со сдвигом).
    ``refine`` — подгонять боксы по подчёркиваниям (:func:`locate_boxes`); ``False`` —
    резать неподвижные боксы макета.
    """
    h, w = warped.shape[:2]
    if (w, h) != tuple(layout.ref_size):
        raise ValueError(
            f"размер скана {w}×{h} не совпадает с эталоном {layout.variant} "
            f"{layout.ref_size[0]}×{layout.ref_size[1]}"
        )
    wanted = None if names is None else set(names)
    if wanted is not None:
        unknown = wanted.difference(layout.names)
        if unknown:
            raise KeyError(f"нет в макете {layout.variant}: {', '.join(sorted(unknown))}")
    if refine:
        boxes = [p.box for p in locate_boxes(warped, layout).values()]
    else:
        boxes = list(layout.boxes)
    return {
        box.name: crop_box(warped, box, pad=pad, fill=fill)
        for box in boxes
        if wanted is None or box.name in wanted
    }
