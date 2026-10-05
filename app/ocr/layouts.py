"""Макеты боксов подполей: где на выровненном бланке лежат цифры дат и номер ваучера.

Макет описывает один вариант бланка (см. :mod:`app.ocr.variants`) и задаётся в пикселях
его статичного эталона: скан сначала выравнивается по эталону, затем из ``warped``
вырезаются боксы (:func:`app.ocr.crops.crop_subfields`).

Файлы макетов — ``app/ocr/layouts/<variant>.json``. Координаты — не данные, их можно
коммитить. Формат::

    {
      "variant": "kommunar_v1",
      "tug_code": "k",
      "ref_size": [1654, 2340],
      "version": 1,
      "boxes": [
        {"name": "voucher_number", "x0": 796, "y0": 620, "x1": 1040, "y1": 712,
         "kind": "number", "printed_by_template": true, "line": [806, 1013, 696]},
        ...
      ]
    }

``line`` (необязательно) — подчёркивание «____» этого подполя в эталоне: ``[x0, x1, y]``.
Бланк заполняют в редакторе, поэтому печатный текст строки «плывёт» от скана к скану
(до 70 px по горизонтали, наклон строки до 15 px), а подчёркивание едет вместе с ним.
По найденному на скане подчёркиванию бокс сдвигается (:func:`app.ocr.crops.locate_boxes`).

Имена подполей: ``voucher_number`` и ``<строка>.<часть>``, где строка — одна из
:data:`DATE_ROWS`, часть — одна из :data:`DATE_PARTS`. Всего 17 боксов. Год напечатан
в бланке и не режется.

Зависимости — только стандартная библиотека.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

#: Каталог макетов, которые идут вместе с приложением.
DEFAULT_LAYOUTS_DIR = Path(__file__).resolve().parent / "layouts"

VOUCHER_NUMBER = "voucher_number"
#: Строки дат в хронологическом порядке: ``left_base ≤ started_work ≤ finished_work ≤
#: arrived_base``. Здесь — в порядке сверху вниз на бланке.
DATE_ROWS = ("left_base", "arrived_base", "started_work", "finished_work")
DATE_PARTS = ("day", "month", "hour", "minute")
#: Все подполя макета (17 штук) в каноническом порядке.
SUBFIELD_NAMES: tuple[str, ...] = (
    VOUCHER_NUMBER,
    *(f"{row}.{part}" for row in DATE_ROWS for part in DATE_PARTS),
)

#: ``two_digit`` — одна-две цифры (день, месяц, час, минуты); ``number`` — номер ваучера
#: (одна-четыре цифры, бывает с буквой: «22а»).
BOX_KINDS = ("two_digit", "number")
#: Ожидаемый вид бокса по имени подполя.
EXPECTED_KIND = {
    name: "number" if name == VOUCHER_NUMBER else "two_digit" for name in SUBFIELD_NAMES
}


class LayoutError(ValueError):
    """Макет не прошёл проверку или не читается."""


@dataclass(frozen=True)
class Box:
    """Бокс подполя в пикселях эталона: ``[x0, x1) × [y0, y1)``.

    Поля для рукописи, вылезающей за линию, уже заложены в координаты бокса.
    ``printed_by_template`` — значение в этом подполе бывает напечатано при заполнении
    бланка в редакторе (номер, день, месяц), а не только вписано от руки.
    ``line`` — подчёркивание подполя в эталоне ``(x0, x1, y)`` для подгонки бокса по скану;
    ``None`` — бокс неподвижен.
    """

    name: str
    x0: int
    y0: int
    x1: int
    y1: int
    kind: str
    printed_by_template: bool = False
    line: tuple[int, int, int] | None = None

    @property
    def width(self) -> int:
        return self.x1 - self.x0

    @property
    def height(self) -> int:
        return self.y1 - self.y0

    @property
    def row(self) -> str | None:
        """Строка дат (``left_base`` …) или ``None`` для номера ваучера."""
        return self.name.split(".", 1)[0] if "." in self.name else None

    @property
    def part(self) -> str | None:
        """Часть строки (``day`` …) или ``None`` для номера ваучера."""
        return self.name.split(".", 1)[1] if "." in self.name else None


@dataclass(frozen=True)
class Layout:
    """Макет боксов одного варианта бланка.

    ``ref_size`` — ``(ширина, высота)`` эталона варианта; ``version`` — номер ревизии
    разметки (растёт при правке координат).
    """

    variant: str
    tug_code: str
    ref_size: tuple[int, int]
    boxes: tuple[Box, ...]
    version: int = 1

    @property
    def names(self) -> tuple[str, ...]:
        return tuple(box.name for box in self.boxes)

    def box(self, name: str) -> Box:
        for box in self.boxes:
            if box.name == name:
                return box
        raise KeyError(name)


def validate_layout(layout: Layout, *, require_all: bool = True) -> None:
    """Проверяет макет; при ошибке — :class:`LayoutError` со списком всех проблем.

    Проверки: размер эталона положительный; имена из :data:`SUBFIELD_NAMES`, без дублей;
    вид бокса из :data:`BOX_KINDS` и соответствует подполю; бокс непустой и целиком внутри
    эталона; при ``require_all`` — есть все 17 подполей.
    """
    problems: list[str] = []
    w, h = layout.ref_size
    if w <= 0 or h <= 0:
        problems.append(f"ref_size должен быть положительным: {layout.ref_size}")
    if not layout.variant:
        problems.append("пустое имя варианта")
    if layout.version < 1:
        problems.append(f"version должна быть ≥ 1: {layout.version}")
    seen: set[str] = set()
    for box in layout.boxes:
        if box.name not in SUBFIELD_NAMES:
            problems.append(f"неизвестное подполе: {box.name!r}")
        elif box.name in seen:
            problems.append(f"дубль подполя: {box.name}")
        seen.add(box.name)
        if box.kind not in BOX_KINDS:
            problems.append(f"{box.name}: неизвестный вид {box.kind!r}")
        elif box.name in EXPECTED_KIND and box.kind != EXPECTED_KIND[box.name]:
            problems.append(f"{box.name}: вид {box.kind!r}, ожидался {EXPECTED_KIND[box.name]!r}")
        if box.x1 <= box.x0 or box.y1 <= box.y0:
            problems.append(f"{box.name}: пустой бокс ({box.x0}, {box.y0}, {box.x1}, {box.y1})")
        if box.x0 < 0 or box.y0 < 0 or box.x1 > w or box.y1 > h:
            problems.append(
                f"{box.name}: бокс ({box.x0}, {box.y0}, {box.x1}, {box.y1}) "
                f"выходит за эталон {w}×{h}"
            )
        if box.line is not None:
            lx0, lx1, ly = box.line
            if lx1 <= lx0 or lx0 < 0 or lx1 > w or not 0 <= ly < h:
                problems.append(f"{box.name}: подчёркивание {box.line} вне эталона или пустое")
    if require_all:
        missing = [name for name in SUBFIELD_NAMES if name not in seen]
        if missing:
            problems.append(f"нет подполей: {', '.join(missing)}")
    if problems:
        raise LayoutError(f"макет {layout.variant or '?'}: " + "; ".join(problems))


def _as_int(value: Any, what: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int | float) or value != int(value):
        raise LayoutError(f"{what}: ожидалось целое число, получено {value!r}")
    return int(value)


def layout_from_dict(data: Mapping[str, Any], *, require_all: bool = True) -> Layout:
    """Макет из словаря (содержимого JSON) с проверкой :func:`validate_layout`."""
    try:
        size = data["ref_size"]
        if not isinstance(size, list | tuple) or len(size) != 2:
            raise LayoutError(f"ref_size должен быть [ширина, высота]: {size!r}")
        boxes = []
        for item in data["boxes"]:
            name = str(item["name"])
            printed = item.get("printed_by_template", False)
            if not isinstance(printed, bool):
                raise LayoutError(f"{name}: printed_by_template должен быть bool")
            line = item.get("line")
            if line is not None and (not isinstance(line, list | tuple) or len(line) != 3):
                raise LayoutError(f"{name}: line должен быть [x0, x1, y]: {line!r}")
            boxes.append(
                Box(
                    name=name,
                    x0=_as_int(item["x0"], f"{name}.x0"),
                    y0=_as_int(item["y0"], f"{name}.y0"),
                    x1=_as_int(item["x1"], f"{name}.x1"),
                    y1=_as_int(item["y1"], f"{name}.y1"),
                    kind=str(item["kind"]),
                    printed_by_template=printed,
                    line=None
                    if line is None
                    else (
                        _as_int(line[0], f"{name}.line[0]"),
                        _as_int(line[1], f"{name}.line[1]"),
                        _as_int(line[2], f"{name}.line[2]"),
                    ),
                )
            )
        layout = Layout(
            variant=str(data["variant"]),
            tug_code=str(data.get("tug_code", "")),
            ref_size=(_as_int(size[0], "ref_size[0]"), _as_int(size[1], "ref_size[1]")),
            boxes=tuple(boxes),
            version=_as_int(data.get("version", 1), "version"),
        )
    except KeyError as exc:
        raise LayoutError(f"в макете нет ключа {exc.args[0]!r}") from None
    except TypeError as exc:
        raise LayoutError(f"неверная структура макета: {exc}") from None
    validate_layout(layout, require_all=require_all)
    return layout


def layout_to_dict(layout: Layout) -> dict[str, Any]:
    """Словарь для JSON; боксы — в каноническом порядке :data:`SUBFIELD_NAMES`."""
    order = {name: i for i, name in enumerate(SUBFIELD_NAMES)}
    boxes = sorted(layout.boxes, key=lambda b: order.get(b.name, len(order)))
    out_boxes: list[dict[str, Any]] = []
    for b in boxes:
        item: dict[str, Any] = {
            "name": b.name,
            "x0": b.x0,
            "y0": b.y0,
            "x1": b.x1,
            "y1": b.y1,
            "kind": b.kind,
            "printed_by_template": b.printed_by_template,
        }
        if b.line is not None:
            item["line"] = list(b.line)
        out_boxes.append(item)
    return {
        "variant": layout.variant,
        "tug_code": layout.tug_code,
        "ref_size": [layout.ref_size[0], layout.ref_size[1]],
        "version": layout.version,
        "boxes": out_boxes,
    }


def load_layout(path: Path, *, require_all: bool = True) -> Layout:
    """Читает и проверяет макет из JSON-файла."""
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise LayoutError(f"{path.name}: не JSON ({exc.msg}, строка {exc.lineno})") from None
    if not isinstance(data, dict):
        raise LayoutError(f"{path.name}: ожидался JSON-объект")
    try:
        return layout_from_dict(data, require_all=require_all)
    except LayoutError as exc:
        raise LayoutError(f"{path.name}: {exc}") from None


def load_layouts(directory: Path = DEFAULT_LAYOUTS_DIR) -> dict[str, Layout]:
    """Все макеты ``*.json`` каталога, по имени варианта. Дубль варианта — ошибка."""
    out: dict[str, Layout] = {}
    if not directory.is_dir():
        return out
    for path in sorted(directory.glob("*.json")):
        layout = load_layout(path)
        if layout.variant in out:
            raise LayoutError(f"{path.name}: вариант {layout.variant} уже загружен")
        out[layout.variant] = layout
    return out


def save_layout(layout: Layout, path: Path) -> None:
    """Пишет макет в JSON (после проверки), по боксу на строку — чтобы дифф был читаемым."""
    validate_layout(layout, require_all=False)
    data = layout_to_dict(layout)
    head = {k: v for k, v in data.items() if k != "boxes"}
    lines = [json.dumps(head, ensure_ascii=False)[:-1] + ', "boxes": [']
    rows = [json.dumps(b, ensure_ascii=False) for b in data["boxes"]]
    lines.extend(f"  {row}," for row in rows[:-1])
    if rows:
        lines.append(f"  {rows[-1]}")
    lines.append("]}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")
