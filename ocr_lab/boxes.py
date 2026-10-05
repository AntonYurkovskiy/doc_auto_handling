"""T08: разметка и проверка боксов подполей (номер ваучера, день/месяц/час/минуты).

CLI ``python -m ocr_lab.boxes <шаг> --variant <вариант>``, шаги:

- ``grid`` — эталон варианта с сеткой координат через 50 px (для разметки глазами);
- ``overlay`` — боксы макета ``app/ocr/layouts/<variant>.json`` поверх эталона →
  ``data/ocr/layouts/<variant>/boxes_overlay.png``;
- ``select`` — выборка сканов для проверки (разные годы; печатные и рукописные номер и
  день — по сверке T04) → ``data/ocr/layouts/<variant>/boxes_check_<tag>.csv``;
- ``align`` — выравнивание выборки по эталону варианта, ``H`` дописывается в тот же CSV;
- ``check`` — кропы с окрестностью и рамкой бокса, по листу на подполе →
  ``data/ocr/sheets/boxes/<variant>/<tag>/<подполе>.png``; боксы подгоняются по
  подчёркиваниям (:func:`app.ocr.crops.locate_boxes`), как они найдены — в
  ``boxes_placements_<tag>.csv``;
- ``scan`` — подгонка боксов (без листов) на всех выровненных сканах варианта по ``H`` из
  T07 (``_work/align_static.csv``) → ``boxes_placements_all.csv``: сколько боксов найдено
  по подчёркиванию, а сколько осталось на запасных путях (``row``, ``static``).

Листы содержат только кропы подполей (решение Q5), страницу целиком не показывают.
Выводятся только агрегаты и ``scan_id``.
"""

from __future__ import annotations

import argparse
import random
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from app.ocr.align import align
from app.ocr.crops import OPEN_PX, crop_box, locate_boxes
from app.ocr.layouts import DEFAULT_LAYOUTS_DIR, SUBFIELD_NAMES, Box, Layout, load_layout
from app.ocr.variants import load_layout as load_variant
from ocr_lab.layouts import ASSIGNMENT_CSV, STATIC_CSV, h_from_str, load_align, read_page
from ocr_lab.paths import LAYOUTS_DIR, REVIEW_DIR, SHEETS_DIR, ensure_dir
from ocr_lab.sheets import make_sheet

TRUTH_CHECK_CSV = REVIEW_DIR / "truth_check.csv"
N_CHECK = 40
#: Сколько сканов каждого способа заполнения (по сверке T04) взять в выборку.
N_PER_FILL = 3
#: Окрестность бокса на листах проверки, доля размера бокса с каждой стороны.
CONTEXT_X = 0.3
CONTEXT_Y = 0.45
#: Кропов на листе крупных (×2) кропов со штрихом на границе.
ZOOM_PER_SHEET = 12

#: Подсказка «штрих на границе бокса»: порог тёмного, полоса снаружи, мелкий мусор.
TOUCH_THRESHOLD = 150
TOUCH_MARGIN_PX = 4
TOUCH_MIN_PX = 10

BOX_COLORS = {
    "voucher_number": (0, 0, 255),
    "day": (0, 0, 255),
    "month": (0, 160, 0),
    "hour": (255, 0, 0),
    "minute": (200, 0, 200),
}


def layout_json(variant: str) -> Path:
    return DEFAULT_LAYOUTS_DIR / f"{variant}.json"


def variant_dir(variant: str) -> Path:
    return LAYOUTS_DIR / variant


def check_csv(variant: str, tag: str) -> Path:
    return variant_dir(variant) / f"boxes_check_{tag}.csv"


def placements_csv(variant: str, tag: str) -> Path:
    return variant_dir(variant) / f"boxes_placements_{tag}.csv"


def _write_png(path: Path, img: np.ndarray) -> None:
    ensure_dir(path.parent)
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"не кодируется PNG: {path.name}")
    path.write_bytes(buf.tobytes())


def _read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")


# --- разметка ------------------------------------------------------------------------------


def draw_grid(image: np.ndarray, step: int = 50) -> np.ndarray:
    """Копия (BGR) с линиями через ``step`` px; каждая вторая линия — красная, с подписью."""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    h, w = bgr.shape[:2]
    for x in range(step, w, step):
        major = x % (2 * step) == 0
        cv2.line(bgr, (x, 0), (x, h - 1), (0, 0, 255) if major else (255, 170, 0), 1)
        if major:
            for y in range(12, h, 400):
                cv2.putText(bgr, str(x), (x + 2, y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200))
    for y in range(step, h, step):
        major = y % (2 * step) == 0
        cv2.line(bgr, (0, y), (w - 1, y), (0, 0, 255) if major else (255, 170, 0), 1)
        if major:
            for x in range(2, w, 400):
                cv2.putText(bgr, str(y), (x, y - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (0, 0, 200))
    return bgr


def _color(box: Box) -> tuple[int, int, int]:
    return BOX_COLORS[box.part or box.name]


def draw_boxes(image: np.ndarray, layout: Layout, labels: bool = True) -> np.ndarray:
    """Копия (BGR) с рамками боксов. Рамка рисуется снаружи бокса, чтобы не закрывать
    его содержимое: внутренняя область бокса остаётся нетронутой."""
    bgr = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR) if image.ndim == 2 else image.copy()
    for box in layout.boxes:
        color = _color(box)
        cv2.rectangle(bgr, (box.x0 - 1, box.y0 - 1), (box.x1, box.y1), color, 1)
        if labels:
            text = box.part or "number"
            cv2.putText(
                bgr, text, (box.x0, box.y0 - 4), cv2.FONT_HERSHEY_SIMPLEX, 0.35, color
            )
    return bgr


def step_grid(variant: str) -> Path:
    ref = load_variant(variant_dir(variant)).reference.image
    out = variant_dir(variant) / "grid.png"
    _write_png(out, draw_grid(ref))
    return out


def step_overlay(variant: str) -> Path:
    ref = load_variant(variant_dir(variant)).reference.image
    layout = load_layout(layout_json(variant))
    if (ref.shape[1], ref.shape[0]) != layout.ref_size:
        raise ValueError(f"{variant}: размер эталона {ref.shape[::-1]} ≠ ref_size макета")
    out = variant_dir(variant) / "boxes_overlay.png"
    _write_png(out, draw_boxes(ref, layout))
    return out


# --- выборка и выравнивание ----------------------------------------------------------------


def fill_type(printed: dict[str, str]) -> str:
    """Способ заполнения ваучера по флагам T04 (``printed`` по номеру и строке дат)."""
    number = printed.get("voucher_number", "")
    row = printed.get("left_base", "")
    if number == "printed":
        return "number_printed"
    if "day=printed" in row:
        return "date_printed"
    if "month=printed" in row:
        return "month_printed"
    return "handwritten"


def t04_fill_types(path: Path = TRUTH_CHECK_CSV) -> dict[str, str]:
    """``scan_id → способ заполнения`` для сканов, просмотренных в T04."""
    if not path.exists():
        return {}
    df = _read_csv(path)
    out: dict[str, str] = {}
    for scan_id, group in df.groupby("scan_id"):
        out[str(scan_id)] = fill_type(dict(zip(group["field"], group["printed"], strict=True)))
    return out


def select_scans(
    variant: str, n: int, seed: int, exclude: set[str], per_fill: int = N_PER_FILL
) -> list[dict[str, str]]:
    """Выборка: по ``per_fill`` сканов каждого способа заполнения из T04, остальное —
    случайно, поровну по годам. Берутся только сканы варианта с успешным выравниванием."""
    df = _read_csv(ASSIGNMENT_CSV)
    pool = df[(df["variant"] == variant) & (df["ok"] == "True")]
    pool = pool[~pool["scan_id"].isin(exclude)]
    rng = random.Random(seed)
    fills = t04_fill_types()
    chosen: list[dict[str, str]] = []
    taken: set[str] = set()
    by_fill: dict[str, list[str]] = {}
    for sid in sorted(pool["scan_id"]):
        if sid in fills:
            by_fill.setdefault(fills[sid], []).append(sid)
    for fill in sorted(by_fill):
        for sid in rng.sample(by_fill[fill], min(per_fill, len(by_fill[fill]))):
            chosen.append({"scan_id": sid, "stratum": f"t04_{fill}"})
            taken.add(sid)
    years = dict(zip(pool["scan_id"], pool["year"], strict=True))
    year_list = sorted(set(years.values()))
    per_year = {y: n // len(year_list) + (i < n % len(year_list)) for i, y in enumerate(year_list)}
    for c in chosen:
        per_year[years[c["scan_id"]]] -= 1
    for year in year_list:
        rest = sorted(s for s, y in years.items() if y == year and s not in taken)
        for sid in rng.sample(rest, max(0, per_year[year])):
            chosen.append({"scan_id": sid, "stratum": "random"})
            taken.add(sid)
    for c in chosen:
        c["year"] = years[c["scan_id"]]
        c["fill"] = fills.get(c["scan_id"], "")
    return chosen


def step_select(variant: str, tag: str, n: int, seed: int, exclude_tags: list[str]) -> Path:
    exclude: set[str] = set()
    for other in exclude_tags:
        exclude.update(_read_csv(check_csv(variant, other))["scan_id"])
    rows = select_scans(variant, n, seed, exclude)
    out = check_csv(variant, tag)
    pd.DataFrame(rows).to_csv(out, index=False, encoding="utf-8")
    return out


def _h_to_str(H: np.ndarray) -> str:
    return " ".join(f"{v:.9g}" for v in np.asarray(H).ravel())


def step_align(variant: str, tag: str) -> None:
    path = check_csv(variant, tag)
    df = _read_csv(path)
    ref = load_variant(variant_dir(variant)).reference
    for col in ("ok", "score", "H"):
        if col not in df.columns:
            df[col] = ""
    for i, sid in enumerate(df["scan_id"]):
        if df.at[i, "H"]:
            continue
        res = align(read_page(sid), ref)
        df.at[i, "ok"] = str(res.ok)
        df.at[i, "score"] = f"{res.score:.4f}"
        df.at[i, "H"] = _h_to_str(res.H) if res.H is not None else ""
    df.to_csv(path, index=False, encoding="utf-8")
    print(f"выровнено: ok {(df['ok'] == 'True').sum()} из {len(df)}")


def warp_page(scan_id: str, H: np.ndarray, size: tuple[int, int]) -> np.ndarray:
    """Страница кэша в кадре эталона (как ``AlignResult.warped``)."""
    return cv2.warpPerspective(
        read_page(scan_id), H, size, flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=255,
    )


# --- проверка ------------------------------------------------------------------------------


def context_box(box: Box) -> Box:
    """Бокс с окрестностью для листа проверки."""
    dx = round(box.width * CONTEXT_X)
    dy = round(box.height * CONTEXT_Y)
    return Box(box.name, box.x0 - dx, box.y0 - dy, box.x1 + dx, box.y1 + dy, box.kind)


def context_crop(warped: np.ndarray, box: Box) -> np.ndarray:
    """Кроп с окрестностью (BGR), граница бокса — цветная рамка снаружи бокса."""
    ctx = context_box(box)
    crop = crop_box(warped, ctx)
    bgr = cv2.cvtColor(crop, cv2.COLOR_GRAY2BGR)
    x0, y0 = box.x0 - ctx.x0, box.y0 - ctx.y0
    cv2.rectangle(bgr, (x0 - 1, y0 - 1), (x0 + box.width, y0 + box.height), _color(box), 1)
    return bgr


def touched_sides(warped: np.ndarray, box: Box) -> str:
    """Стороны бокса (``L``, ``R``, ``T``, ``B``), которые пересекает тёмный штрих.

    Подчёркивания (горизонтальные линии) не считаются. Это подсказка, какие кропы смотреть
    крупно: штрих на границе — либо значение не влезло, либо внутрь заходит печать или
    соседнее подполе.
    """
    m = TOUCH_MARGIN_PX
    crop = crop_box(warped, box, pad=m)
    dark = (crop < TOUCH_THRESHOLD).astype(np.uint8)
    lines = cv2.morphologyEx(
        dark, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (OPEN_PX, 1))
    )
    ink = dark & ~cv2.dilate(lines, np.ones((5, 1), np.uint8))
    n, labels, stats, _ = cv2.connectedComponentsWithStats(ink, connectivity=8)
    h, w = ink.shape
    inside = np.zeros((h, w), bool)
    inside[m : h - m, m : w - m] = True
    sides = set()
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] < TOUCH_MIN_PX:
            continue
        comp = labels == i
        if not (comp & inside).any() or not (comp & ~inside).any():
            continue
        ys, xs = np.nonzero(comp)
        if xs.min() < m:
            sides.add("L")
        if xs.max() >= w - m:
            sides.add("R")
        if ys.min() < m:
            sides.add("T")
        if ys.max() >= h - m:
            sides.add("B")
    return "".join(s for s in "LRTB" if s in sides)


def step_check(variant: str, tag: str, names: list[str] | None = None) -> None:
    """Листы проверки по подполям и сводка, как найдены боксы (``locate_boxes``)."""
    df = _read_csv(check_csv(variant, tag))
    layout = load_layout(layout_json(variant))
    wanted = names or list(SUBFIELD_NAMES)
    crops: dict[str, list[np.ndarray]] = {name: [] for name in wanted}
    captions: list[str] = []
    rows: list[dict[str, Any]] = []
    zoom: list[np.ndarray] = []
    zoom_captions: list[str] = []
    for sid, h_text in zip(df["scan_id"], df["H"], strict=True):
        H = h_from_str(h_text)
        if H is None:
            continue
        warped = warp_page(sid, H, layout.ref_size)
        placed = locate_boxes(warped, layout)
        captions.append(sid)
        for name in wanted:
            pl = placed[name]
            crops[name].append(context_crop(warped, pl.box))
            touch = touched_sides(warped, pl.box)
            rows.append(
                {"scan_id": sid, "subfield": name, "source": pl.source,
                 "dx0": pl.dx0, "dx1": pl.dx1, "dy": pl.dy, "touch": touch}
            )
            if touch:
                zoom.append(cv2.resize(context_crop(warped, pl.box), None, fx=2, fy=2,
                                       interpolation=cv2.INTER_CUBIC))
                zoom_captions.append(f"{name} {sid} {touch}")
    sheet_dir = ensure_dir(SHEETS_DIR / "boxes" / variant / tag)
    for name in wanted:
        ctx_w = context_box(layout.box(name)).width
        cols = max(1, min(5, 1900 // (ctx_w + 6)))
        make_sheet(crops[name], captions, cols, ctx_w, sheet_dir / f"{name}.png")
    for i in range(0, len(zoom), ZOOM_PER_SHEET):
        part = zoom[i : i + ZOOM_PER_SHEET]
        cell_w = max(img.shape[1] for img in part)
        make_sheet(part, zoom_captions[i : i + ZOOM_PER_SHEET], max(1, 1900 // (cell_w + 6)),
                   cell_w, sheet_dir / f"_touch_{i // ZOOM_PER_SHEET:02d}.png")
    pdf = pd.DataFrame(rows)
    pdf.to_csv(placements_csv(variant, tag), index=False, encoding="utf-8")
    print(f"листы: {sheet_dir}; сканов: {len(captions)}")
    table = pdf.pivot_table(index="subfield", columns="source", values="scan_id",
                            aggfunc="count", fill_value=0)
    table["touch"] = pdf[pdf["touch"] != ""].groupby("subfield").size()
    print(table.reindex(wanted).fillna(0).astype(int).to_string())
    print(f"|dx| p95 {pdf[['dx0', 'dx1']].abs().max(axis=1).quantile(0.95):.0f} px, "
          f"max {pdf[['dx0', 'dx1']].abs().max(axis=1).max()} px; "
          f"|dy| p95 {pdf['dy'].abs().quantile(0.95):.0f} px, max {pdf['dy'].abs().max()} px")


def step_scan(variant: str) -> None:
    """Подгонка боксов на всех выровненных сканах варианта; сводка по способам."""
    static = load_align(STATIC_CSV)
    rows_in = static[
        (static["variant"] == variant)
        & (static["is_best"].astype(str) == "True")
        & (static["ok"].astype(str) == "True")
    ]
    layout = load_layout(layout_json(variant))
    rows: list[dict[str, Any]] = []
    for sid, h_text in zip(rows_in["scan_id"], rows_in["H"], strict=True):
        H = h_from_str(str(h_text))
        if H is None:
            continue
        placed = locate_boxes(warp_page(str(sid), H, layout.ref_size), layout)
        for name, pl in placed.items():
            rows.append({"scan_id": sid, "subfield": name, "source": pl.source,
                         "dx0": pl.dx0, "dx1": pl.dx1, "dy": pl.dy})
    pdf = pd.DataFrame(rows)
    pdf.to_csv(placements_csv(variant, "all"), index=False, encoding="utf-8")
    print(f"сканов: {pdf['scan_id'].nunique()}")
    table = pdf.pivot_table(index="subfield", columns="source", values="scan_id",
                            aggfunc="count", fill_value=0)
    print(table.reindex(list(SUBFIELD_NAMES)).fillna(0).astype(int).to_string())
    weak = pdf[pdf["source"].isin(["row", "static"])]
    for name, group in weak.groupby("subfield"):
        print(f"{name}: {group['source'].iloc[0]} — {', '.join(sorted(group['scan_id']))}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T08: боксы подполей")
    parser.add_argument("step", choices=["grid", "overlay", "select", "align", "check", "scan"])
    parser.add_argument("--variant", required=True)
    parser.add_argument("--tag", default="dev", help="имя выборки (dev, holdout, …)")
    parser.add_argument("--n", type=int, default=N_CHECK)
    parser.add_argument("--seed", type=int, default=8)
    parser.add_argument("--exclude", nargs="*", default=[], help="теги выборок-исключений")
    parser.add_argument("--names", nargs="*", default=None, help="подполя для check")
    args = parser.parse_args(argv)
    started = time.perf_counter()
    if args.step == "grid":
        print(step_grid(args.variant))
    elif args.step == "overlay":
        print(step_overlay(args.variant))
    elif args.step == "select":
        print(step_select(args.variant, args.tag, args.n, args.seed, args.exclude))
    elif args.step == "align":
        step_align(args.variant, args.tag)
    elif args.step == "scan":
        step_scan(args.variant)
    else:
        step_check(args.variant, args.tag, args.names)
    print(f"готово за {time.perf_counter() - started:.1f} с")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
