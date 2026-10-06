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
   Высота строки (T09). Подчёркивания одной строки набраны одной строкой текста и лежат
   на одной прямой (строка бывает наклонена). Ожидаемая высота бокса — прямая через
   найденные линии соседей по строке (нужно не меньше :data:`ROW_MIN_FITS` соседей, наклон
   не больше :data:`MAX_ROW_SLOPE`). Если линия бокса не найдена или отстоит от ожидаемой
   высоты дальше :data:`ROW_DY_TOL_PX`, поиск (п. 2, затем п. 4) повторяется только среди
   линий на этой высоте, и в п. 2 берётся только целая линия (оба конца). Так отсекаются
   ножки засечек печатного слова: у Courier на бланке Пионера низ слова «Time» — сплошная
   линия длиной с подчёркивание часов, на 10–12 px выше него. Если и так ничего нет,
   первая находка отбрасывается, и бокс идёт на запасные пути (п. 6).
4. Если краёв рядом нет (подчёркивание уехало дальше :data:`EDGE_TOLERANCE_PX` — так
   бывает с часами и минутами, они плывут отдельно от дня и месяца), ищется целая линия
   длиной как подчёркивание эталона (±:data:`LENGTH_TOL`), ближайшая к нему по центру,
   в окне шире по горизонтали (:data:`SEARCH_DX_WIDE_PX`; ``source="line_len"``). Линии,
   уже найденные соседями по строке, не берутся: иначе бокс месяца без своего
   подчёркивания (печатный месяц «_03_») уезжает на подчёркивание дня той же длины.
5. Если не нашлось ничего во всей строке (так бывает с шапкой: на единичных сканах
   она съезжает на 70 px по вертикали и 100 px по горизонтали), такой же поиск идёт и
   в окне выше по вертикали (:data:`SEARCH_DY_WIDE_PX`; ``source="line_wide"``). Если
   линии подходящей длины нет и там, в том же окне ищется целая линия по обоим концам,
   как в п. 2 (T09: у Пионера длина подчёркивания номера от скана к скану меняется
   вдвое, от 65 до 170 px).
6. Иначе бокс берёт медианный сдвиг той же части в других строках (``source="column"``:
   табуляция у строк общая), а высоту — от ближайшего найденного бокса своей строки.
   Если такой части нигде не нашлось — сдвиг ближайшего по горизонтали найденного бокса
   той же строки (``source="row"``): бокс целиком едет за ближним к нему краем соседа
   (за правым, если сосед слева). Текст между ними — печатный, и короткое подчёркивание
   соседа сдвигает всё, что правее. Если и в строке ничего — бокс остаётся на месте
   (``source="static"``).
7. Соседи по строке не залезают друг к другу: бокс не заходит левее найденного правого
   края подчёркивания левого соседа и правее найденного левого края правого соседа.
   Так в бокс минут не попадают часы, а в бокс месяца — день. Если от такой обрезки бокс
   стал бы уже :data:`MIN_WIDTH_FRACTION` эталонного, она не применяется. Если
   подчёркивания соседей слились в одну линию (у левого найден только левый край, у
   правого — только правый), стык считается по длине подчёркивания соседа в эталоне
   (T09: у Коммунара так в строке «Начало работ» на трети сканов, и в бокс месяца
   попадал хвост дня).

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
#: Высота строки (п. 3): сколько найденных соседей по строке нужно для оценки, допуск
#: от ожидаемой высоты (px) и предельный наклон строки (px на px).
ROW_MIN_FITS = 2
ROW_DY_TOL_PX = 7
MAX_ROW_SLOPE = 0.03
#: Штраф за наклон строки при поиске выбившейся линии: px отклонения на единицу наклона.
ROW_SLOPE_COST_PX = 300
#: Конец линии ближе этого к найденному концу линии соседа — это линия соседа.
CLAIMED_PX = 2


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
    gray: np.ndarray,
    box: Box,
    dx: int,
    dy: int,
    y_range: tuple[float, float] | None = None,
) -> tuple[list[LineSegment], int, int]:
    """Линии в окне вокруг подчёркивания бокса и левая/правая границы окна.

    ``y_range`` — оставить только линии с высотой в этих пределах (высота строки, п. 3).
    """
    assert box.line is not None
    lx0, lx1, ly = box.line
    h, w = gray.shape[:2]
    wx0, wx1 = max(lx0 - dx, 0), min(lx1 + dx, w)
    wy0, wy1 = max(ly - dy, 0), min(ly + dy + 1, h)
    if wx1 <= wx0 or wy1 <= wy0:
        return [], wx0, wx1
    segments = find_lines(gray[wy0:wy1, wx0:wx1], offset=(wx0, wy0))
    if y_range is not None:
        segments = [s for s in segments if y_range[0] <= s.y <= y_range[1]]
    return segments, wx0, wx1


def _fit_line_len(
    gray: np.ndarray,
    box: Box,
    search_dy: int,
    *,
    y_range: tuple[float, float] | None = None,
    claimed: Iterable[tuple[int | None, int | None]] = (),
    expected: tuple[float, list[float]] | None = None,
) -> LineSegment | None:
    """Целая линия длиной как подчёркивание эталона, ближайшая к нему, или ``None``.

    ``claimed`` — найденные края подчёркиваний соседей по строке: их линии не берутся.
    ``expected`` — ожидаемые центры по x подчёркиваний ``(своего, [соседей по строке])``:
    линия, чей центр ближе к ожидаемому месту соседа, чем к своему, — линия соседа.
    """
    assert box.line is not None
    lx0, lx1, ly = box.line
    segments, wx0, wx1 = _window(gray, box, SEARCH_DX_WIDE_PX, search_dy, y_range)
    length = lx1 - lx0
    taken = list(claimed)

    def is_claimed(s: LineSegment) -> bool:
        return any(
            (left is not None and abs(s.x0 - left) <= CLAIMED_PX)
            or (right is not None and abs(s.x1 - right) <= CLAIMED_PX)
            for left, right in taken
        )

    whole = [
        s
        for s in segments
        if s.x0 > wx0
        and s.x1 < wx1
        and abs((s.x1 - s.x0) - length) <= LENGTH_TOL * length
        and not is_claimed(s)
        and (
            expected is None
            or all(
                abs((s.x0 + s.x1) / 2 - expected[0]) <= abs((s.x0 + s.x1) / 2 - rival)
                for rival in expected[1]
            )
        )
    ]
    if not whole:
        return None
    cx, cy = (lx0 + lx1) / 2, ly
    return min(whole, key=lambda s: math.hypot((s.x0 + s.x1) / 2 - cx, s.y - cy))


def _fit_line(
    gray: np.ndarray,
    box: Box,
    y_range: tuple[float, float] | None = None,
    search_dy: int = SEARCH_DY_PX,
) -> tuple[int | None, int | None, int | None]:
    """Найденные края подчёркивания ``(левый x, правый x)`` и вертикальный сдвиг.

    Каждое значение — ``None``, если не найдено.
    """
    assert box.line is not None
    lx0, lx1, ly = box.line
    segments, wx0, wx1 = _window(gray, box, SEARCH_DX_PX, search_dy, y_range)
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


def _row_line(points: list[tuple[float, float]]) -> tuple[float, float, float]:
    """Прямая строки по точкам ``(x, dy)``: ``(средний x, средний dy, наклон)``.

    МНК; наклон ограничен :data:`MAX_ROW_SLOPE`, чтобы два близких соседа не дали дикой
    экстраполяции.
    """
    xs = np.array([p[0] for p in points], dtype=float)
    ys = np.array([p[1] for p in points], dtype=float)
    mx, my = float(xs.mean()), float(ys.mean())
    var = float(((xs - mx) ** 2).sum())
    slope = 0.0 if var == 0 else float(((xs - mx) * (ys - my)).sum()) / var
    return mx, my, max(-MAX_ROW_SLOPE, min(MAX_ROW_SLOPE, slope))


def _row_height(points: list[tuple[float, float]], x: float) -> float:
    """Ожидаемый вертикальный сдвиг подчёркивания в точке ``x`` по сдвигам ``points``."""
    mx, my, slope = _row_line(points)
    return my + slope * (x - mx)


def _row_spread(points: list[tuple[float, float]]) -> tuple[float, float]:
    """Наибольшее отклонение точек от прямой строки и её наклон."""
    mx, my, slope = _row_line(points)
    return max(abs(y - (my + slope * (x - mx))) for x, y in points), slope


def _row_outlier(points: dict[str, tuple[float, float]], candidates: set[str]) -> str | None:
    """Бокс из ``candidates``, чья линия выбивается из прямой строки, или ``None``.

    ``points`` — ``{имя: (x, dy)}`` найденных линий строки (не меньше трёх). Если все лежат
    на прямой с допуском :data:`ROW_DY_TOL_PX` — ``None``. Иначе выбивается та линия, без
    которой остальные ложатся на прямую лучше всего; к отклонению добавляется штраф за
    наклон (:data:`ROW_SLOPE_COST_PX`). Без штрафа при трёх линиях любые две лежат на
    прямой, а сильный наклон строки встречается реже чужой линии.
    """
    if len(points) < ROW_MIN_FITS + 1:
        return None
    # Каждая линия сверяется с прямой через остальные: прямая через все сразу «съедает»
    # выброс (при четырёх линиях отклонение засечек от неё меньше допуска).
    if all(
        abs(y - _row_height([q for m, q in points.items() if m != n], x)) <= ROW_DY_TOL_PX
        for n, (x, y) in points.items()
    ):
        return None
    best: tuple[float, str] | None = None
    for name in sorted(candidates & set(points)):
        spread, slope = _row_spread([p for n, p in points.items() if n != name])
        cost = spread + abs(slope) * ROW_SLOPE_COST_PX
        if best is None or cost < best[0]:
            best = (cost, name)
    return None if best is None else best[1]


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

    def fit_by_edges(
        box: Box,
        y_range: tuple[float, float] | None = None,
        *,
        search_dy: int = SEARCH_DY_PX,
        source: str | None = None,
    ) -> bool:
        left, right, dy = _fit_line(gray, box, y_range, search_dy)
        if (y_range is not None or source is not None) and (left is None or right is None):
            # Повторный поиск на высоте строки (п. 3) и широкий поиск (п. 5) берут только
            # целую линию: край чужой линии хуже первой находки или запасного пути.
            return False
        shifts = _edge_shifts(box, left, right)
        if shifts is None or max(abs(shifts[0]), abs(shifts[1]), abs(dy or 0)) > MAX_SHIFT_PX:
            return False
        if source is None:
            if left is not None and right is not None:
                source = "line"
            else:
                source = "line_left" if left is not None else "line_right"
        fits[box.name] = (*shifts, dy or 0, source)
        edges[box.name] = (left, right)
        return True

    def row_mates(box: Box) -> list[Box]:
        assert box.line is not None
        return [
            other
            for other in lined
            if other.name != box.name
            and other.line is not None
            and abs(other.line[2] - box.line[2]) <= SAME_ROW_PX
        ]

    def fit_by_length(
        box: Box, search_dy: int, source: str, y_range: tuple[float, float] | None = None
    ) -> bool:
        assert box.line is not None
        # Чужой считается только линия, найденная соседом целиком (оба конца): по одному
        # краю сосед мог зацепить общую длинную линию («___02 2026»), которая нужна и боксу.
        claimed = [
            edges[m.name]
            for m in row_mates(box)
            if m.name in edges and None not in edges[m.name]
        ]
        # Ожидаемые места подчёркиваний: найденные соседи — где нашлись, остальные — на
        # месте эталона со сдвигом строки (медиана сдвигов найденных соседей). У Пионера
        # подчёркивания дня и месяца почти одной длины: без этого месяц брал линию
        # ненайденного дня.
        mates = row_mates(box)
        found = [(fits[m.name][0] + fits[m.name][1]) / 2 for m in mates if m.name in fits]
        row_shift = statistics.median(found) if found else 0.0
        rivals = []
        for m in mates:
            assert m.line is not None
            shift = (fits[m.name][0] + fits[m.name][1]) / 2 if m.name in fits else row_shift
            rivals.append((m.line[0] + m.line[1]) / 2 + shift)
        own = (box.line[0] + box.line[1]) / 2 + row_shift
        seg = _fit_line_len(
            gray, box, search_dy, y_range=y_range, claimed=claimed, expected=(own, rivals)
        )
        if seg is None:
            return False
        shifts = _edge_shifts(box, seg.x0, seg.x1)
        assert shifts is not None
        fits[box.name] = (*shifts, round(seg.y - box.line[2]), source)
        edges[box.name] = (seg.x0, seg.x1)
        return True

    for box in lined:
        fit_by_edges(box)

    # П. 3, высота строки. В каждой строке по одному разбираются боксы, чья линия
    # выбивается из прямой через остальные (_row_outlier), затем не найденные боксы.
    # После каждой правки прямая пересчитывается.
    rows: list[list[Box]] = []
    for box in lined:
        for row in rows:
            if box in row_mates(row[0]):
                row.append(box)
                break
        else:
            rows.append([box])
    dropped: set[str] = set()

    def refit_at_row_height(box: Box) -> None:
        assert box.line is not None
        others = [(_center(m), float(fits[m.name][2])) for m in row_mates(box) if m.name in fits]
        y = box.line[2] + _row_height(others, _center(box))
        y_range = (y - ROW_DY_TOL_PX, y + ROW_DY_TOL_PX)
        if fit_by_edges(box, y_range) or fit_by_length(box, SEARCH_DY_PX, "line_len", y_range):
            return
        # Линии на высоте строки нет: первая находка — чужая линия (засечки печатного
        # слова), бокс уходит на запасные пути (п. 6).
        fits.pop(box.name, None)
        edges.pop(box.name, None)
        dropped.add(box.name)

    for row in rows:
        candidates = {box.name for box in row}
        while True:
            points = {b.name: (_center(b), float(fits[b.name][2])) for b in row if b.name in fits}
            name = _row_outlier(points, candidates)
            if name is None:
                break
            candidates.discard(name)
            refit_at_row_height(layout.box(name))
        for box in row:
            found = sum(m.name in fits for m in row_mates(box))
            if box.name not in fits and found >= ROW_MIN_FITS:
                refit_at_row_height(box)

    for box in lined:
        if box.name not in fits and box.name not in dropped:
            fit_by_length(box, SEARCH_DY_PX, "line_len")

    for box in lined:
        if box.name not in fits and not any(m.name in fits for m in row_mates(box)):
            if not fit_by_length(box, SEARCH_DY_WIDE_PX, "line_wide"):
                fit_by_edges(box, search_dy=SEARCH_DY_WIDE_PX, source="line_wide")

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
            # вместе, а две половины строки — независимо. Бокс едет целиком за ближним
            # к нему краем соседа.
            mate = _nearest_box(found_mates, box)
            d0, d1, dy, _ = fits[mate.name]
            shift = d1 if _center(mate) < _center(box) else d0
            placed[box.name] = (shift, shift, dy, "row")
        else:
            placed[box.name] = (0, 0, 0, "static")

    out: dict[str, Placement] = {}
    for box in layout.boxes:
        d0, d1, dy, source = placed[box.name]
        x0, x1 = box.x0 + d0, box.x1 + d1
        if box.line is not None:
            # П. 7: не заходить за найденные концы подчёркиваний соседей по строке.
            min_width = MIN_WIDTH_FRACTION * box.width
            own_left, own_right = edges.get(box.name, (None, None))
            for mate in row_mates(box):
                assert mate.line is not None
                mate_left, mate_right = edges.get(mate.name, (None, None))
                length = mate.line[1] - mate.line[0]
                # Слитые подчёркивания: у соседа найден только дальний конец, у бокса —
                # только свой дальний, общий стык не виден. Конец соседа — по длине его
                # подчёркивания в эталоне: раз линии сомкнулись, настоящий конец не ближе.
                if _center(mate) < _center(box) and mate_left is not None:
                    if mate_right is None and own_left is None and own_right is not None:
                        mate_right = mate_left + length
                elif _center(mate) > _center(box) and mate_right is not None:
                    if mate_left is None and own_right is None and own_left is not None:
                        mate_left = mate_right - length
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
