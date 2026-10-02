"""Варианты бланка, статичные эталоны и калибровка порогов выравнивания (T07).

Шаги (каждый — подкоманда CLI, все идемпотентны и возобновляемы):

1. ``metrics`` — метрики страниц кэша: контраст, наклон, резкость, типичность положения.
2. ``candidates`` — по 12 кандидатов в начальные эталоны на буксир и обезличенные листы.
3. ``initial`` — выравнивание всех страниц по начальным эталонам (одиночным сканам).
4. ``worst`` — обезличенные листы худших по ``score`` страниц (поиск вариантов и типов
   отказов).
5. ``static`` — статичный эталон варианта: медиана 30 лучших выровненных сканов, маска
   статичных зон, ``meta.json``.
6. ``realign`` — ``detect_variant`` на всех страницах по статичным эталонам плюс
   независимая оценка остаточного сдвига по якорям.
   Перед ним откалиброванные пороги пишутся шагом ``params``: ``--set min_score=0.70
   --set max_reproj_err=4.0 --set use_ecc=false``.
7. ``blockfit`` — посадка шапки и блока строк дат после выравнивания: остаточный сдвиг
   печати внутри каждой полосы (независимая от ``score`` мера «правильности»).
8. ``overlays`` — наложения краёв эталона на выровненный скан по диапазонам ``score``.
9. ``report`` — ``layout_assignment.csv`` и ``reports/layouts_report.md``.

Обезличивание (решение Q5): страницу целиком модели показывать нельзя. На листах зоны
буксира, судна, агента, вида работ, примечаний, «совместно с», подписей и печатей
закрашены белым (:data:`SENSITIVE_ZONES`), а если положение бланка на странице неизвестно,
страница заменяется грубой мозаикой (:func:`pixelate`), на которой текст нечитаем.

Запуск: ``python -m ocr_lab.layouts <шаг> [--workers 4] [--max-minutes 9]``.
"""

from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import sys
import time
from collections import Counter, defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from app.ocr.align import (
    AlignParams,
    AlignResult,
    Reference,
    align,
    build_reference,
    map_points_to_scan,
)
from app.ocr.variants import Layout, detect_variant, load_layouts, save_layout
from app.services.voucher_number import parse_voucher_name
from ocr_lab.paths import (
    LAYOUTS_DIR,
    MANIFEST,
    PAGES_DIR,
    REPORTS_DIR,
    SHEETS_DIR,
    WORK_DIR,
    ensure_dir,
)
from ocr_lab.sheets import make_sheet

Rect = tuple[float, float, float, float]


@dataclass(frozen=True)
class VariantSpec:
    """Вариант бланка: имя, буксир и скан начального эталона."""

    name: str
    tug_code: str
    seed: str


#: Варианты бланка. ``seed`` — начальный эталон, выбран в T07 глазами из 12 кандидатов
#: по обезличенным миниатюрам (шаг ``candidates``).
#: У Пионера две вёрстки: ``pioneer_v1`` — шапка с отступом, имя буксира вписывается;
#: ``pioneer_v2`` — компактная шапка, имя буксира напечатано, строки дат стоят выше.
VARIANTS: tuple[VariantSpec, ...] = (
    VariantSpec("kommunar_v1", "k", "2025_180k"),
    VariantSpec("pioneer_v1", "p", "2025_16p"),
    VariantSpec("pioneer_v2", "p", "2026_199p"),
)

#: Зоны переменного текста в долях эталона (x0, y0, x1, y1), по одному набору на вариант.
#: Заданы в кадре начального эталона варианта, подобраны по статичным эталонам:
#: значения буксира, судна, агента и вида работ,
#: примечания, «совместно с», подписи и печати, а также шапка факса в правом верхнем углу.
SENSITIVE_ZONES: dict[str, tuple[Rect, ...]] = {
    "kommunar_v1": (
        (0.48, 0.0, 1.0, 0.105),
        (0.245, 0.325, 1.0, 0.415),
        (0.125, 0.415, 1.0, 0.472),
        (0.175, 0.472, 1.0, 0.497),
        (0.205, 0.662, 1.0, 0.705),
        (0.430, 0.705, 1.0, 0.750),
        (0.0, 0.750, 1.0, 1.0),
    ),
    "pioneer_v1": (
        (0.48, 0.0, 1.0, 0.090),
        (0.340, 0.310, 1.0, 0.365),
        (0.330, 0.365, 1.0, 0.410),
        (0.210, 0.410, 1.0, 0.455),
        (0.260, 0.455, 1.0, 0.505),
        (0.245, 0.725, 1.0, 0.772),
        (0.535, 0.772, 1.0, 0.815),
        (0.0, 0.808, 1.0, 1.0),
    ),
    "pioneer_v2": (
        (0.48, 0.0, 1.0, 0.075),
        (0.330, 0.265, 1.0, 0.320),
        (0.310, 0.320, 1.0, 0.365),
        (0.200, 0.365, 1.0, 0.410),
        (0.250, 0.410, 1.0, 0.465),
        (0.235, 0.675, 1.0, 0.720),
        (0.520, 0.720, 1.0, 0.765),
        (0.0, 0.757, 1.0, 1.0),
    ),
}

#: Полосы для наложений (доли эталона): шапка с номером и блок строк дат.
OVERLAY_BANDS: dict[str, tuple[Rect, ...]] = {
    "kommunar_v1": ((0.0, 0.10, 1.0, 0.31), (0.0, 0.497, 1.0, 0.662)),
    "pioneer_v1": ((0.0, 0.09, 1.0, 0.31), (0.0, 0.505, 1.0, 0.725)),
    "pioneer_v2": ((0.0, 0.06, 1.0, 0.265), (0.0, 0.465, 1.0, 0.675)),
}

N_CANDIDATES = 12
N_STATIC_SOURCES = 30
N_WORST = 48
DARK_FRAC = 0.7  # доля сканов, в которых пиксель тёмный, чтобы считаться статичным
MAX_SOURCES_PER_MONTH = 3  # не больше стольких источников эталона из одного месяца
MASK_DILATE_PX = 4
UNALIGNED_MARGIN = 0.012  # запас зон, когда страница не выровнена
ALIGNED_MARGIN = 0.006
MIN_SCORE_FOR_ZONES = 0.35  # ниже — зонам на скане не верим, показываем мозаику

ANCHOR_SIZE = 128
ANCHOR_GRID = (6, 8)  # ячеек по x и по y
ANCHOR_MIN_GRADIENT = 3.0  # средний |градиент| в окне по слабой из двух осей
ANCHOR_SEARCH_PX = 14
ANCHOR_MIN_PEAK = 0.35
# Посадка блоков (шапка, строки дат): якоря гуще и поиск шире — сдвиги блоков до ~20 px.
BLOCK_GRID = (8, 3)
BLOCK_SEARCH_PX = 24
BLOCK_TOLERANCE_PX = 6.0  # сдвиг блока дат, который поглощают поля кропов (T08: 15–25 %)

WORK = LAYOUTS_DIR / "_work"
METRICS_CSV = WORK / "page_metrics.csv"
CANDIDATES_CSV = WORK / "candidates.csv"
INITIAL_CSV = WORK / "align_initial.csv"
STATIC_CSV = WORK / "align_static.csv"
BLOCK_FIT_CSV = WORK / "block_fit.csv"
FAIL_TYPES_JSON = WORK / "fail_types.json"
ASSIGNMENT_CSV = WORK_DIR / "layout_assignment.csv"
LAYOUT_SHEETS_DIR = SHEETS_DIR / "layouts"
REPORT_MD = REPORTS_DIR / "layouts_report.md"

ALIGN_COLUMNS = (
    "scan_id", "variant", "ok", "has_warp", "score", "inliers", "inlier_ratio", "reproj_err",
    "rotated180", "method", "reason", "H", "is_best", "margin", "resid_med", "resid_p75",
    "resid_p90", "resid_max",
    "anchors_ok", "anchors", "seconds", "seconds_fast",
)


# --- ввод-вывод ---------------------------------------------------------------------------


def read_page(scan_id: str, pages_dir: Path = PAGES_DIR) -> np.ndarray:
    """Страница кэша T03 (серое). Путь может содержать кириллицу."""
    data = np.fromfile(str(pages_dir / f"{scan_id}.png"), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"не читается страница {scan_id}")
    return img


def load_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def variants_for_tug(tug_code: str) -> list[str]:
    """Имена вариантов бланка этого буксира."""
    return [v.name for v in VARIANTS if v.tug_code == tug_code]


def _read_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def _append_rows(path: Path, columns: tuple[str, ...], rows: list[dict[str, Any]]) -> None:
    new = not path.exists()
    ensure_dir(path.parent)
    with path.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=columns, extrasaction="ignore")
        if new:
            writer.writeheader()
        writer.writerows(rows)


# --- метрики страниц -----------------------------------------------------------------------


def estimate_skew(dark: np.ndarray, max_deg: float = 3.0) -> float:
    """Угол (градусы), поворот на который делает строки горизонтальными.

    Критерий — резкость профиля строк: сумма квадратов разностей соседних сумм по строкам.
    """
    f = dark.astype(np.float32)
    h, w = f.shape
    center = (w / 2.0, h / 2.0)

    def crit(angle: float) -> float:
        M = cv2.getRotationMatrix2D(center, angle, 1.0)
        prof = cv2.warpAffine(f, M, (w, h)).sum(axis=1)
        return float(((prof[1:] - prof[:-1]) ** 2).sum())

    coarse = np.arange(-max_deg, max_deg + 1e-9, 0.5)
    best = float(coarse[int(np.argmax([crit(a) for a in coarse]))])
    fine = np.arange(best - 0.4, best + 0.4 + 1e-9, 0.1)
    return float(fine[int(np.argmax([crit(a) for a in fine]))])


def page_metrics(img: np.ndarray) -> tuple[dict[str, float], np.ndarray, np.ndarray]:
    """Метрики страницы и профили тёмных пикселей по строкам и столбцам (копия 1/4)."""
    small = cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    thr, _ = cv2.threshold(small, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    dark = small < thr
    if not dark.any() or dark.all():
        zeros = {"contrast": 0.0, "ink": float(dark.mean()), "skew_deg": 0.0, "sharpness": 0.0}
        return zeros, dark.mean(axis=1), dark.mean(axis=0)
    metrics = {
        "contrast": float(np.median(small[~dark])) - float(np.median(small[dark])),
        "ink": float(dark.mean()),
        "skew_deg": estimate_skew(dark),
        "sharpness": float(cv2.Laplacian(img, cv2.CV_64F).var()),
    }
    return metrics, dark.mean(axis=1), dark.mean(axis=0)


def profile_shift(profile: np.ndarray, median: np.ndarray, max_shift: int) -> tuple[int, float]:
    """Сдвиг профиля относительно медианного (в отсчётах) и корреляция при этом сдвиге."""
    a = median - median.mean()
    best = (0, -1.0)
    for shift in range(-max_shift, max_shift + 1):
        b = np.roll(profile, -shift)
        b = b - b.mean()
        denom = float(np.linalg.norm(a) * np.linalg.norm(b))
        corr = float(a @ b) / denom if denom > 0 else 0.0
        if corr > best[1]:
            best = (shift, corr)
    return best


def _metrics_worker(scan_id: str) -> tuple[str, dict[str, float], np.ndarray, np.ndarray]:
    metrics, rows, cols = page_metrics(read_page(scan_id))
    return scan_id, metrics, rows, cols


def step_metrics(manifest: pd.DataFrame, workers: int) -> pd.DataFrame:
    """Метрики всех страниц; типичность считается относительно медианного профиля буксира."""
    ids = list(manifest["scan_id"])
    tug = dict(zip(manifest["scan_id"], manifest["tug_code"], strict=True))
    raw: dict[str, tuple[dict[str, float], np.ndarray, np.ndarray]] = {}
    with ProcessPoolExecutor(max_workers=workers) as pool:
        for scan_id, metrics, rows, cols in pool.map(_metrics_worker, ids, chunksize=16):
            raw[scan_id] = (metrics, rows, cols)
    records: list[dict[str, Any]] = []
    for code in sorted(set(tug.values())):
        group = [s for s in ids if tug[s] == code]
        med_rows = np.median(np.stack([raw[s][1] for s in group]), axis=0)
        med_cols = np.median(np.stack([raw[s][2] for s in group]), axis=0)
        for scan_id in group:
            metrics, rows, cols = raw[scan_id]
            sy, cy = profile_shift(rows, med_rows, max_shift=30)
            sx, cx = profile_shift(cols, med_cols, max_shift=30)
            records.append(
                {
                    "scan_id": scan_id,
                    "tug_code": code,
                    **metrics,
                    "shift_y": sy / len(rows),
                    "corr_y": cy,
                    "shift_x": sx / len(cols),
                    "corr_x": cx,
                }
            )
    df = pd.DataFrame.from_records(records)
    ensure_dir(WORK)
    df.to_csv(METRICS_CSV, index=False, encoding="utf-8")
    return df


def pick_candidates(metrics: pd.DataFrame, n: int = N_CANDIDATES) -> pd.DataFrame:
    """По ``n`` кандидатов на буксир: малый наклон, типичное положение, высокий контраст.

    Сначала фильтр «типичности» (наклон, сдвиг и корреляция профилей), затем ранжирование
    по сумме рангов контраста и резкости. Если фильтр оставил меньше ``n``, он ослабляется.
    """
    picked: list[pd.DataFrame] = []
    for _, group in metrics.groupby("tug_code"):
        chosen = group.iloc[0:0]
        for k in (1.0, 2.0, 4.0, 1e9):
            mask = (
                (group["skew_deg"].abs() <= 0.3 * k)
                & (group["shift_y"].abs() <= 0.01 * k)
                & (group["shift_x"].abs() <= 0.01 * k)
                & (group["corr_y"] >= group["corr_y"].quantile(0.5 / k))
            )
            chosen = group[mask]
            if len(chosen) >= n:
                break
        rank = chosen["contrast"].rank(ascending=False) + chosen["sharpness"].rank(ascending=False)
        picked.append(chosen.assign(rank=rank).sort_values(["rank", "scan_id"]).head(n))
    return pd.concat(picked, ignore_index=True)


# --- обезличивание -------------------------------------------------------------------------


def zone_rects(variants: str | list[str], margin: float = 0.0) -> list[Rect]:
    """Зоны переменного текста варианта (или объединение нескольких) с запасом ``margin``."""
    names = [variants] if isinstance(variants, str) else variants
    return [
        (x0 - margin, y0 - margin, x1 + margin, y1 + margin)
        for name in names
        for x0, y0, x1, y1 in SENSITIVE_ZONES[name]
    ]


def _extend(rect: Rect) -> Rect:
    """Края зоны, упирающиеся в границу эталона, уводятся далеко за неё: поля скана за
    пределами эталона тоже должны быть закрыты."""
    x0, y0, x1, y1 = rect
    return (
        x0 - 0.5 if x0 <= 0.02 else x0,
        y0 - 0.5 if y0 <= 0.02 else y0,
        x1 + 0.5 if x1 >= 0.98 else x1,
        y1 + 0.5 if y1 >= 0.98 else y1,
    )


def anonymize(
    page: np.ndarray,
    variants: str | list[str],
    H: np.ndarray | None = None,
    ref_size: tuple[int, int] | None = None,
    margin: float = UNALIGNED_MARGIN,
) -> np.ndarray:
    """Копия страницы с закрашенными белым зонами переменного текста.

    Без ``H`` зоны берутся в координатах самой страницы (годится только для страниц
    с типичным положением бланка; если вариант неизвестен, передаётся список вариантов
    буксира и закрашивается объединение их зон). С ``H`` (скан → эталон) и ``ref_size``
    зоны переводятся из координат эталона в координаты скана.
    """
    out = page.copy()
    h, w = page.shape[:2]
    white = 255 if page.ndim == 2 else (255, 255, 255)
    for rect in zone_rects(variants, margin):
        x0, y0, x1, y1 = _extend(rect)
        if H is None:
            pts = np.array([[x0 * w, y0 * h], [x1 * w, y0 * h], [x1 * w, y1 * h], [x0 * w, y1 * h]])
        else:
            if ref_size is None:
                raise ValueError("anonymize: с H нужен ref_size")
            rw, rh = ref_size
            ref_pts = np.array(
                [[x0 * rw, y0 * rh], [x1 * rw, y0 * rh], [x1 * rw, y1 * rh], [x0 * rw, y1 * rh]]
            )
            pts = map_points_to_scan(ref_pts, H)
        cv2.fillPoly(out, [np.round(pts).astype(np.int32)], white)
    return out


def pixelate(page: np.ndarray, blocks: int = 40) -> np.ndarray:
    """Грубая мозаика страницы (``blocks`` ячеек по ширине): видна компоновка, текст — нет."""
    h, w = page.shape[:2]
    small = cv2.resize(
        page, (blocks, max(1, round(blocks * h / w))), interpolation=cv2.INTER_AREA
    )
    return cv2.resize(small, (w, h), interpolation=cv2.INTER_NEAREST)


def thumbnail(img: np.ndarray, width: int) -> np.ndarray:
    h, w = img.shape[:2]
    return cv2.resize(img, (width, max(1, round(h * width / w))), interpolation=cv2.INTER_AREA)


def step_candidates(metrics: pd.DataFrame) -> pd.DataFrame:
    """Листы кандидатов в начальные эталоны: по 6 обезличенных миниатюр на лист."""
    cands = pick_candidates(metrics)
    ensure_dir(WORK)
    cands.to_csv(CANDIDATES_CSV, index=False, encoding="utf-8")
    for code, group in cands.groupby("tug_code"):
        ids = list(group["scan_id"])
        for part in range(0, len(ids), 6):
            chunk = ids[part : part + 6]
            zones = variants_for_tug(str(code))
            thumbs = [thumbnail(anonymize(read_page(s), zones), 640) for s in chunk]
            make_sheet(
                thumbs, chunk, cols=3, cell_w=640,
                out_path=LAYOUT_SHEETS_DIR / f"candidates_{code}_{part // 6 + 1}.png",
            )
    return cands


# --- выравнивание всех страниц --------------------------------------------------------------

_REFS: dict[str, Reference] = {}
_LAYOUTS: dict[str, Layout] = {}
_ANCHORS: dict[str, list[tuple[int, int]]] = {}


def _h_to_str(H: np.ndarray | None) -> str:
    return "" if H is None else " ".join(f"{v:.9g}" for v in np.asarray(H).ravel())


def h_from_str(text: str) -> np.ndarray | None:
    if not text:
        return None
    return np.array([float(v) for v in text.split()], dtype=np.float64).reshape(3, 3)


def result_row(scan_id: str, variant: str, res: AlignResult) -> dict[str, Any]:
    return {
        "scan_id": scan_id,
        "variant": variant,
        "ok": res.ok,
        "has_warp": res.warped is not None,
        "score": f"{res.score:.4f}",
        "inliers": res.inliers,
        "inlier_ratio": f"{res.inlier_ratio:.3f}",
        "reproj_err": f"{res.reproj_err:.3f}",
        "rotated180": res.rotated180,
        "method": res.method,
        "reason": res.reason,
        "H": _h_to_str(res.H),
    }


def _init_initial() -> None:
    cv2.setNumThreads(1)  # процессов несколько: внутренние потоки opencv только мешают
    for spec in VARIANTS:
        _REFS[spec.name] = build_reference(read_page(spec.seed), name=spec.name)


def _initial_worker(scan_id: str) -> list[dict[str, Any]]:
    page = read_page(scan_id)
    rows = []
    for name, ref in _REFS.items():
        started = time.perf_counter()
        row = result_row(scan_id, name, align(page, ref))
        row["seconds"] = f"{time.perf_counter() - started:.3f}"
        rows.append(row)
    return rows


def find_anchors(
    ref_image: np.ndarray,
    mask: np.ndarray,
    region: Rect | None = None,
    grid: tuple[int, int] = ANCHOR_GRID,
) -> list[tuple[int, int]]:
    """Левые верхние углы окон-якорей: в каждой ячейке сетки — окно с самой «двумерной»
    статичной печатью.

    Качество окна — меньшая из средних горизонтальной и вертикальной составляющих градиента
    внутри маски: одиночная линия (сдвиг вдоль неё не определить) якорем не становится.
    ``region`` (доли эталона) — сетка строится только внутри этой полосы, окна целиком в ней.
    """
    h, w = mask.shape
    rx0, ry0, rx1, ry1 = (0, 0, w, h)
    if region is not None:
        rx0, ry0 = round(region[0] * w), round(region[1] * h)
        rx1, ry1 = round(region[2] * w), round(region[3] * h)
    img = ref_image.astype(np.float32)
    inside = (mask > 0).astype(np.float32)
    win = (ANCHOR_SIZE, ANCHOR_SIZE)
    energy = []
    for dx, dy in ((1, 0), (0, 1)):
        grad = np.abs(cv2.Sobel(img, cv2.CV_32F, dx, dy, ksize=3)) / 8.0 * inside
        energy.append(
            cv2.boxFilter(grad, -1, win, anchor=(0, 0), borderType=cv2.BORDER_CONSTANT)
        )
    quality = np.minimum(energy[0], energy[1])
    pad = BLOCK_SEARCH_PX + 2
    quality[: max(pad, ry0), :] = 0
    quality[:, : max(pad, rx0)] = 0
    quality[min(h - pad, ry1) - ANCHOR_SIZE :, :] = 0
    quality[:, min(w - pad, rx1) - ANCHOR_SIZE :] = 0
    nx, ny = grid
    rw, rh = rx1 - rx0, ry1 - ry0
    anchors: list[tuple[int, int]] = []
    for iy in range(ny):
        for ix in range(nx):
            y0, y1 = ry0 + iy * rh // ny, ry0 + (iy + 1) * rh // ny
            x0, x1 = rx0 + ix * rw // nx, rx0 + (ix + 1) * rw // nx
            cell = quality[y0:y1, x0:x1]
            pos = np.unravel_index(int(np.argmax(cell)), cell.shape)
            if cell[pos] >= ANCHOR_MIN_GRADIENT:
                anchors.append((x0 + int(pos[1]), y0 + int(pos[0])))
    return anchors


def anchor_residuals(
    warped: np.ndarray,
    ref_image: np.ndarray,
    anchors: list[tuple[int, int]],
    search_px: int = ANCHOR_SEARCH_PX,
) -> list[float]:
    """Остаточный сдвиг (px) в каждом якоре, где совпадение найдено уверенно.

    Независимая от ``score`` проверка: окно эталона ищется в выровненном скане в пределах
    ±``search_px``. Сдвиг пика от нуля — локальная ошибка выравнивания. Среди
    равных пиков берётся ближайший к нулевому сдвигу.
    """
    out: list[float] = []
    r = search_px
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    dist = np.hypot(xx, yy)
    for x, y in anchors:
        tmpl = ref_image[y : y + ANCHOR_SIZE, x : x + ANCHOR_SIZE]
        area = warped[y - r : y + ANCHOR_SIZE + r, x - r : x + ANCHOR_SIZE + r]
        if float(area.std()) < 1.0 or float(tmpl.std()) < 1.0:
            continue
        resp = cv2.matchTemplate(area, tmpl, cv2.TM_CCOEFF_NORMED)
        peak = float(resp.max())
        if peak >= ANCHOR_MIN_PEAK:
            out.append(float(dist[resp >= peak - 0.01].min()))
    return out


def _init_static() -> None:
    cv2.setNumThreads(1)
    _LAYOUTS.update(load_layouts(LAYOUTS_DIR))
    for name, layout in _LAYOUTS.items():
        _ANCHORS[name] = find_anchors(layout.reference.image, layout.reference.mask)


def _static_worker(scan_id: str) -> list[dict[str, Any]]:
    page = read_page(scan_id)
    started = time.perf_counter()
    match = detect_variant(page, _LAYOUTS, confident_score=None)
    seconds = time.perf_counter() - started
    # Время рабочего режима: с ранним выходом и подсказкой буксира из имени файла.
    started = time.perf_counter()
    detect_variant(page, _LAYOUTS, prefer=name_tug_code(scan_id) or None)
    seconds_fast = time.perf_counter() - started
    rows = []
    for name, res in match.results.items():
        row = result_row(scan_id, name, res)
        row["is_best"] = name == match.variant
        row["seconds"] = f"{seconds:.3f}"
        row["seconds_fast"] = f"{seconds_fast:.3f}"
        if name == match.variant:
            row["margin"] = f"{match.margin:.4f}"
            anchors = _ANCHORS[name]
            row["anchors"] = len(anchors)
            if res.warped is not None:
                resid = anchor_residuals(res.warped, _LAYOUTS[name].reference.image, anchors)
                row["anchors_ok"] = len(resid)
                if resid:
                    row["resid_med"] = f"{np.median(resid):.2f}"
                    row["resid_p75"] = f"{np.percentile(resid, 75):.2f}"
                    row["resid_p90"] = f"{np.percentile(resid, 90):.2f}"
                    row["resid_max"] = f"{max(resid):.2f}"
        rows.append(row)
    return rows


BLOCK_COLUMNS = (
    "scan_id", "variant", "head_n", "head_med", "head_max", "date_n", "date_med", "date_p75",
    "date_max",
)
_BLOCK_ANCHORS: dict[str, dict[str, list[tuple[int, int]]]] = {}


def _init_blocks() -> None:
    cv2.setNumThreads(1)
    _LAYOUTS.update(load_layouts(LAYOUTS_DIR))
    for name, layout in _LAYOUTS.items():
        ref = layout.reference
        head, date = OVERLAY_BANDS[name]
        _BLOCK_ANCHORS[name] = {
            "head": find_anchors(ref.image, ref.mask, head, BLOCK_GRID),
            "date": find_anchors(ref.image, ref.mask, date, BLOCK_GRID),
        }


def block_fit_row(
    scan_id: str, variant: str, warped: np.ndarray, layout: Layout,
    anchors: dict[str, list[tuple[int, int]]],
) -> dict[str, Any]:
    """Посадка шапки и блока строк дат: остаточные сдвиги якорей внутри каждой полосы."""
    row: dict[str, Any] = {"scan_id": scan_id, "variant": variant}
    for key in ("head", "date"):
        resid = anchor_residuals(
            warped, layout.reference.image, anchors[key], search_px=BLOCK_SEARCH_PX
        )
        row[f"{key}_n"] = len(resid)
        if resid:
            row[f"{key}_med"] = f"{np.median(resid):.2f}"
            row[f"{key}_max"] = f"{max(resid):.2f}"
            if key == "date":
                row["date_p75"] = f"{np.percentile(resid, 75):.2f}"
    return row


def _block_worker(item: tuple[str, str]) -> list[dict[str, Any]]:
    scan_id, variant = item
    layout = _LAYOUTS[variant]
    res = align(read_page(scan_id), layout.reference)
    if res.warped is None:
        return [{"scan_id": scan_id, "variant": variant, "head_n": 0, "date_n": 0}]
    return [block_fit_row(scan_id, variant, res.warped, layout, _BLOCK_ANCHORS[variant])]


def step_blockfit(workers: int) -> None:
    """Посадка блоков для лучшего варианта каждой страницы (``_work/block_fit.csv``)."""
    best = load_align(STATIC_CSV)
    best = best[best["is_best"] & best["has_warp"]].sort_values("scan_id")
    done = {row["scan_id"] for row in _read_rows(BLOCK_FIT_CSV)}
    todo = [(s, v) for s, v in zip(best["scan_id"], best["variant"], strict=True)
            if s not in done]
    with ProcessPoolExecutor(max_workers=workers, initializer=_init_blocks) as pool:
        for n, rows in enumerate(pool.map(_block_worker, todo, chunksize=8), start=1):
            _append_rows(BLOCK_FIT_CSV, BLOCK_COLUMNS, rows)
            if n % 100 == 0:
                print(f"blockfit: {n}/{len(todo)}", flush=True)
    print(f"blockfit: записано {len(todo)}")


def _run_align(
    kind: str, ids: list[str], out_csv: Path, workers: int, max_minutes: float | None
) -> int:
    """Общий цикл выравнивания: пропуск готовых, запись по мере готовности, лимит времени."""
    done = {row["scan_id"] for row in _read_rows(out_csv)}
    todo = [s for s in ids if s not in done]
    if not todo:
        print(f"{kind}: всё готово ({len(done)} страниц)")
        return 0
    init, worker = (
        (_init_initial, _initial_worker) if kind == "initial" else (_init_static, _static_worker)
    )
    started = time.monotonic()
    n = 0
    pool = ProcessPoolExecutor(max_workers=workers, initializer=init)
    try:
        futures = [pool.submit(worker, s) for s in todo]
        for fut in as_completed(futures):
            _append_rows(out_csv, ALIGN_COLUMNS, fut.result())
            n += 1
            if n % 100 == 0:
                print(f"{kind}: {n}/{len(todo)}", flush=True)
            if max_minutes is not None and time.monotonic() - started > max_minutes * 60:
                print(f"{kind}: остановка по --max-minutes, готово {n}; перезапустите")
                break
    finally:
        pool.shutdown(wait=True, cancel_futures=True)
    print(f"{kind}: записано {n}, осталось {len(todo) - n}")
    return len(todo) - n


def load_align(path: Path) -> pd.DataFrame:
    """CSV выравнивания с приведёнными типами."""
    df = pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8")
    for col in ("ok", "has_warp", "rotated180", "is_best"):
        if col in df:
            df[col] = df[col] == "True"
    for col in ("score", "inlier_ratio", "reproj_err", "margin", "resid_med", "resid_p75",
                "resid_p90", "resid_max",
                "seconds", "seconds_fast", "inliers", "anchors_ok", "anchors"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


_BEST_ORDER = ["scan_id", "has_warp", "score", "inliers"]


def own_rows(align_df: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """По одной строке на скан: лучший по ``score`` вариант «своего» буксира из манифеста."""
    tug = dict(zip(manifest["scan_id"], manifest["tug_code"], strict=True))
    variant_tug = {v.name: v.tug_code for v in VARIANTS}
    keep = align_df["variant"].map(variant_tug) == align_df["scan_id"].map(tug)
    own = align_df[keep].sort_values(_BEST_ORDER, ascending=[True, False, False, False])
    return own.drop_duplicates("scan_id").set_index("scan_id", drop=False).rename_axis(None)


def best_rows(align_df: pd.DataFrame) -> pd.DataFrame:
    """По одной строке на скан: лучший по ``score`` вариант среди всех."""
    best = align_df.sort_values(_BEST_ORDER, ascending=[True, False, False, False])
    return best.drop_duplicates("scan_id").set_index("scan_id", drop=False).rename_axis(None)


# --- листы худших --------------------------------------------------------------------------


def anonymized_page(scan_id: str, rows: pd.DataFrame, ref_size: tuple[int, int]) -> np.ndarray:
    """Страница для листа: зоны закрашены по лучшей доступной H, иначе — мозаика.

    ``rows`` — строки выравнивания этого скана по всем вариантам. H годится, только если
    выравнивание дошло до пост-проверки (``has_warp``): инлайеров хватило и гомография
    правдоподобна — и набрало ``score`` не ниже ``MIN_SCORE_FOR_ZONES``: при худшем
    совпадении положению зон на скане верить нельзя.
    """
    page = read_page(scan_id)
    usable = rows[rows["has_warp"] & (rows["score"] >= MIN_SCORE_FOR_ZONES)]
    usable = usable.sort_values("score", ascending=False)
    if usable.empty:
        return pixelate(page, blocks=64)
    best = usable.iloc[0]
    H = h_from_str(str(best["H"]))
    return anonymize(page, str(best["variant"]), H, ref_size, margin=UNALIGNED_MARGIN)


def step_worst(manifest: pd.DataFrame, n: int = N_WORST, source: Path = INITIAL_CSV) -> list[str]:
    """Обезличенные листы ``n`` худших по ``score`` страниц (по эталону своего буксира)."""
    df = load_align(source)
    own = own_rows(df, manifest).sort_values(["score", "scan_id"])
    worst = list(own["scan_id"].head(n))
    ref_size = _page_size()
    tag = source.stem.replace("align_", "")
    for part in range(0, len(worst), 24):
        chunk = worst[part : part + 24]
        thumbs, captions = [], []
        for scan_id in chunk:
            rows = df[df["scan_id"] == scan_id]
            thumbs.append(thumbnail(anonymized_page(scan_id, rows, ref_size), 320))
            captions.append(f"{scan_id} s={own.loc[scan_id, 'score']:.2f}")
        make_sheet(
            thumbs, captions, cols=6, cell_w=320,
            out_path=LAYOUT_SHEETS_DIR / f"worst_{tag}_{part // 24 + 1}.png",
        )
    return worst


def _page_size() -> tuple[int, int]:
    img = read_page(VARIANTS[0].seed)
    return int(img.shape[1]), int(img.shape[0])


# --- статичные эталоны ----------------------------------------------------------------------


def build_static(
    warped: list[np.ndarray],
    zones: list[Rect],
    dark_frac: float = DARK_FRAC,
    dilate_px: int = MASK_DILATE_PX,
) -> tuple[np.ndarray, np.ndarray]:
    """Статичный эталон и маска статичных зон по стопке выровненных сканов.

    Эталон — попиксельная медиана. Статичный пиксель — тёмный (ниже порога Оцу своей
    страницы) не менее чем в ``dark_frac`` сканов. Зоны переменного текста ``zones``
    (доли кадра) в эталоне закрашены белым и в маску не входят: там остаются следы частых
    значений (самый частый агент, вид работ), а это уже не форма бланка.
    Маска — статичные пиксели, расширенные на ``dilate_px``.
    """
    stack = np.stack(warped)
    median = np.median(stack, axis=0).astype(np.uint8)
    frac = np.zeros(median.shape, np.float32)
    for page in warped:
        thr, _ = cv2.threshold(page, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        frac += page < thr
    frac /= len(warped)

    h, w = median.shape
    in_zone = np.zeros((h, w), bool)
    for x0, y0, x1, y1 in zones:
        xa, xb = max(0, round(x0 * w)), min(w, round(x1 * w))
        ya, yb = max(0, round(y0 * h)), min(h, round(y1 * h))
        in_zone[ya:yb, xa:xb] = True
    static = (frac >= dark_frac) & ~in_zone

    reference = median.copy()
    reference[in_zone] = 255
    size = 2 * dilate_px + 1
    mask = cv2.dilate(
        static.astype(np.uint8) * 255, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    )
    mask[in_zone] = 0
    return reference, mask


def pick_sources(ranked: pd.DataFrame, month: dict[str, str], n: int) -> list[str]:
    """Кандидаты в источники эталона: лучшие по ``score``, но не больше
    ``MAX_SOURCES_PER_MONTH`` из одного месяца.

    Без ограничения 30 лучших — почти одинаковые бланки одной пачки (с одним и тем же
    напечатанным месяцем), и эти цифры попадают в «статичную» печать.
    """
    used: Counter[str] = Counter()
    out: list[str] = []
    for scan_id in ranked["scan_id"]:
        key = month.get(scan_id, "")
        if used[key] >= MAX_SOURCES_PER_MONTH:
            continue
        used[key] += 1
        out.append(scan_id)
        if len(out) >= 2 * n:  # запас на случай неудачного повторного выравнивания
            break
    return out


def step_static(manifest: pd.DataFrame, n: int = N_STATIC_SOURCES) -> None:
    """Статичные эталоны всех вариантов: ``data/ocr/layouts/<variant>/``."""
    best = best_rows(load_align(INITIAL_CSV))
    tug = dict(zip(manifest["scan_id"], manifest["tug_code"], strict=True))
    month = {
        row.scan_id: f"{row.left_base_year}-{row.left_base_month}" for row in manifest.itertuples()
    }
    for spec in VARIANTS:
        ref = build_reference(read_page(spec.seed), name=spec.name)
        # Источники — сканы своего буксира, для которых этот вариант лучший среди всех.
        mine = (best["variant"] == spec.name) & (best["scan_id"].map(tug) == spec.tug_code)
        ranked = best[mine & best["has_warp"]].sort_values(
            ["score", "scan_id"], ascending=[False, True]
        )
        sources: list[str] = []
        warped: list[np.ndarray] = []
        for scan_id in pick_sources(ranked, month, n):
            if len(warped) >= n:
                break
            res = align(read_page(scan_id), ref)
            if res.warped is not None:
                sources.append(scan_id)
                warped.append(res.warped)
        reference, mask = build_static(warped, zone_rects(spec.name))
        meta = {
            "name": spec.name,
            "tug_code": spec.tug_code,
            "seed_scan_id": spec.seed,
            "source_scan_ids": sources,
            "params": {
                "n_sources": len(sources),
                "dark_frac": DARK_FRAC,
                "max_sources_per_month": MAX_SOURCES_PER_MONTH,
                "mask_dilate_px": MASK_DILATE_PX,
                "page_width": int(reference.shape[1]),
            },
            "sensitive_zones": [list(z) for z in SENSITIVE_ZONES[spec.name]],
            "align_params": _existing_align_params(spec.name),
        }
        save_layout(LAYOUTS_DIR / spec.name, reference, mask, meta)
        print(f"{spec.name}: эталон по {len(sources)} сканам, статичных пикселей "
              f"{float((mask > 0).mean()):.3f}")


def _existing_align_params(name: str) -> dict[str, Any]:
    """Откалиброванные пороги из прежнего ``meta.json`` варианта (пересборка их не теряет)."""
    path = LAYOUTS_DIR / name / "meta.json"
    if not path.exists():
        return {}
    return dict(json.loads(path.read_text(encoding="utf-8")).get("align_params") or {})


def set_align_params(overrides: dict[str, Any]) -> None:
    """Записывает откалиброванные пороги в ``meta.json`` всех вариантов."""
    valid = {f.name for f in dataclasses.fields(AlignParams)}
    unknown = set(overrides) - valid
    if unknown:
        raise ValueError(f"неизвестные поля AlignParams: {sorted(unknown)}")
    for spec in VARIANTS:
        path = LAYOUTS_DIR / spec.name / "meta.json"
        meta = json.loads(path.read_text(encoding="utf-8"))
        meta["align_params"] = {**dict(meta.get("align_params") or {}), **overrides}
        path.write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


# --- наложения -----------------------------------------------------------------------------


def overlay_bands(warped: np.ndarray, layout: Layout) -> np.ndarray:
    """Наложение для просмотра: полосы шапки и строк дат, края эталона — красным.

    От скана остаётся только окрестность статичной печати («трафарет»): переменный текст
    вне её не виден, а зоны переменного текста закрашены белым.
    """
    ref = layout.reference
    near = cv2.dilate(ref.mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (25, 25))) > 0
    gray = np.where(near, warped, 255).astype(np.uint8)
    gray = anonymize(gray, layout.name, margin=ALIGNED_MARGIN)
    bgr = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
    edges = cv2.Canny(cv2.GaussianBlur(ref.image, (3, 3), 0), 50, 150)
    bgr[(edges > 0) & (ref.mask > 0)] = (0, 0, 255)
    h, w = gray.shape
    parts = []
    for x0, y0, x1, y1 in OVERLAY_BANDS[layout.name]:
        parts.append(bgr[round(y0 * h) : round(y1 * h), round(x0 * w) : round(x1 * w)])
        parts.append(np.full((8, parts[-1].shape[1], 3), 160, np.uint8))
    return np.vstack(parts[:-1])


def step_overlays(
    manifest: pd.DataFrame, bins: list[float], per_bin: int = 8, seed: int = 7
) -> None:
    """Листы наложений: по ``per_bin`` сканов из каждого диапазона ``score`` лучшего варианта."""
    df = load_align(STATIC_CSV)
    best = df[df["is_best"] & df["has_warp"]].sort_values("scan_id")
    layouts = load_layouts(LAYOUTS_DIR)
    edges = [-1.0, *bins, 2.0]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        group = best[(best["score"] >= lo) & (best["score"] < hi)]
        if group.empty:
            continue
        sample = group.sample(min(per_bin, len(group)), random_state=seed).sort_values("score")
        tag = f"{max(lo, 0):.2f}-{min(hi, 1):.2f}"
        items = list(sample.itertuples())
        for part in range(0, len(items), 4):
            images, captions = [], []
            for row in items[part : part + 4]:
                layout = layouts[str(row.variant)]
                res = align(read_page(str(row.scan_id)), layout.reference)
                if res.warped is None:
                    continue
                images.append(overlay_bands(res.warped, layout))
                captions.append(
                    f"{row.scan_id} s={row.score:.2f} in={row.inliers} resid={row.resid_p90}"
                )
            if images:
                make_sheet(
                    images, captions, cols=2, cell_w=990,
                    out_path=LAYOUT_SHEETS_DIR / f"overlay_{tag}_{part // 4 + 1}.png",
                )
        print(f"наложения {tag}: {len(items)} из {len(group)}")


# --- сводка и отчёт -------------------------------------------------------------------------


def ok_mask(df: pd.DataFrame, params: AlignParams) -> pd.Series:
    """``ok`` по сохранённым метрикам для заданных порогов (без повторного выравнивания)."""
    return (
        df["has_warp"]
        & (df["inliers"] >= params.min_inliers)
        & (df["inlier_ratio"] >= params.min_inlier_ratio)
        & (df["reproj_err"] <= params.max_reproj_err)
        & (df["score"] >= params.min_score)
    )


def name_tug_code(scan_id: str) -> str:
    """Код буксира из имени файла (``scan_id`` = ``<год>_<имя>``); пусто, если кода нет."""
    name = scan_id.split("_", 1)[1] if "_" in scan_id else scan_id
    return parse_voucher_name(name).tug_code or ""


def build_assignment(manifest: pd.DataFrame) -> pd.DataFrame:
    """``layout_assignment.csv``: метрики шага 2 (начальный эталон) и итог по статичным."""
    base = manifest[["scan_id", "tug_code", "year"]].copy()
    base["name_code"] = [name_tug_code(s) for s in base["scan_id"]]
    init = own_rows(load_align(INITIAL_CSV), manifest)
    cols = ["ok", "score", "inliers", "reproj_err", "rotated180"]
    out = base.join(init[cols].add_prefix("init_"), on="scan_id")
    if STATIC_CSV.exists():
        static = load_align(STATIC_CSV)
        own = own_rows(static, manifest)
        out = out.join(own[["variant", *cols]].add_prefix("own_"), on="scan_id")
        best = static[static["is_best"]].set_index("scan_id")
        best_cols = ["variant", "ok", "score", "inliers", "inlier_ratio", "reproj_err",
                     "rotated180", "method", "margin", "resid_med", "resid_p75", "resid_p90",
                     "resid_max", "anchors_ok",
                     "anchors", "seconds", "seconds_fast"]
        out = out.join(best[best_cols], on="scan_id")
        variant_tug = {v.name: v.tug_code for v in VARIANTS}
        out["variant_matches_tug"] = out["variant"].map(variant_tug) == out["tug_code"]
        out["variant_matches_name"] = out["variant"].map(variant_tug) == out["name_code"]
    out.to_csv(ASSIGNMENT_CSV, index=False, encoding="utf-8")
    return out


def _pct(num: float, den: float) -> str:
    return "—" if den == 0 else f"{100.0 * num / den:.1f} %"


def _share_table(df: pd.DataFrame) -> list[str]:
    """Доля ``ok`` по найденному варианту и году: начальный эталон против статичного."""
    lines = ["| Вариант | Год | Сканов | до, итоговые пороги | до, `min_score=0.3` | "
             "после (статичный) | медиана `score` до → после |",
             "|---|---|---:|---:|---:|---:|---:|"]

    def row(label: str, year: str, g: pd.DataFrame) -> str:
        n = len(g)
        return (
            f"| {label} | {year} | {n} | {_pct(g['init_ok'].sum(), n)} | "
            f"{_pct(g['init_ok_soft'].sum(), n)} | {_pct(g['ok'].sum(), n)} | "
            f"{g['init_score'].median():.2f} → {g['score'].median():.2f} |"
        )

    for (variant, year), g in df.groupby(["variant", "year"]):
        lines.append(row(str(variant), str(year), g))
    lines.append(row("все", "все", df))
    return lines


def render_report(assign: pd.DataFrame) -> str:
    """Отчёт только с агрегатами: без построчных значений и текстов бланков."""
    layouts = load_layouts(LAYOUTS_DIR)
    df = assign.copy()
    for col in ("init_ok", "init_ok_soft", "own_ok", "ok", "variant_matches_tug",
                "variant_matches_name"):
        df[col] = df[col].astype(bool)
    lines = ["# Варианты бланка и выравнивание (T07)", ""]
    lines += ["## Варианты", "",
              "| Вариант | Буксир | Определено `detect_variant` | из них 2025 | 2026 |",
              "|---|---|---:|---:|---:|"]
    for spec in VARIANTS:
        g = df[df["variant"] == spec.name]
        lines.append(
            f"| {spec.name} | {spec.tug_code} | {len(g)} | {(g['year'] == '2025').sum()} | "
            f"{(g['year'] == '2026').sum()} |"
        )
    by_tug = df.groupby("tug_code").size().to_dict()
    lines += ["", "Сканов по манифесту: "
              + ", ".join(f"буксир `{k}` — {v}" for k, v in sorted(by_tug.items())) + "."]
    lines += ["", "Пороги `ok` (из `meta.json` вариантов):", ""]
    for name, layout in layouts.items():
        p = layout.reference.params
        lines.append(
            f"- `{name}`: `min_score={p.min_score}`, `min_inliers={p.min_inliers}`, "
            f"`min_inlier_ratio={p.min_inlier_ratio}`, `max_reproj_err={p.max_reproj_err}`, "
            f"`use_ecc={p.use_ecc}`; "
            f"эталон по {len(layout.meta.get('source_scan_ids', []))} сканам"
        )

    lines += ["", "## Доля `ok` по вариантам и годам", "",
              "Вариант — найденный `detect_variant`. «До» — выравнивание по начальному эталону "
              "этого варианта (один скан, без маски статичных зон), «после» — по статичному "
              "эталону. Без маски `score` считается по всем краям, включая рукопись, и "
              "структурно ниже, поэтому «до» дано и с итоговыми порогами, и с `min_score=0.3` "
              "(порог T04 для эталона без маски).", ""]
    lines += _share_table(df)

    aligned = df[df["ok"]]
    failed = df[~df["ok"]]
    tug_margin = _tug_margins(df)
    lines += ["", "## `detect_variant`", "",
              f"- выровнено (лучший вариант прошёл пороги): {len(aligned)} из {len(df)} "
              f"({_pct(len(aligned), len(df))});",
              f"- неудачных выравниваний: {len(failed)} ({_pct(len(failed), len(df))});",
              f"- вариант совпал с буксиром из манифеста среди выровненных: "
              f"{int(aligned['variant_matches_tug'].sum())} из {len(aligned)} "
              f"({_pct(aligned['variant_matches_tug'].sum(), len(aligned))});",
              f"- то же среди всех страниц: {_pct(df['variant_matches_tug'].sum(), len(df))};",
              f"- повёрнутых на 180° среди выровненных: "
              f"{int(aligned['rotated180'].astype(bool).sum())};",
              f"- отрыв лучшего `score` от лучшего варианта другого буксира среди выровненных: "
              f"минимум {tug_margin.min():.2f}, 1-й перцентиль {tug_margin.quantile(0.01):.2f};",
              f"- отрыв от второго варианта того же буксира (`pioneer_v1`/`pioneer_v2`) бывает "
              f"почти нулевым (минимум {aligned['margin'].min():.2f}): вёрстки близки, выбор "
              "между ними влияет только на эталон для кропов."]
    mismatch = aligned[~aligned["variant_matches_tug"]]
    if len(mismatch):
        by_name = int(mismatch["variant_matches_name"].sum())
        lines += [
            f"- не совпали с манифестом (`scan_id`): {', '.join(sorted(mismatch['scan_id']))};",
            f"- из них в {by_name} бланк совпадает с кодом буксира в имени файла, то есть "
            "расходится сама колонка `tug` выгрузки, а не определение варианта;",
            f"- вариант совпал с кодом буксира из имени файла среди выровненных: "
            f"{int(aligned['variant_matches_name'].sum())} из {len(aligned)} "
            f"({_pct(aligned['variant_matches_name'].sum(), len(aligned))}).",
        ]
    lines += [f"- {note}" for note in _notes("variant_notes")]
    calibration = _notes("calibration_notes")
    if calibration:
        lines += ["", "## Калибровка порогов", ""]
        lines += [f"- {note}" for note in calibration]

    lines += ["", "## `score` и остаточный сдвиг по якорям", "",
              "Остаточный сдвиг — локальное смещение окон статичной печати после выравнивания "
              "(поиск ±14 px), проверка, независимая от `score`. Берётся медиана и 75-й "
              "перцентиль по якорям скана: отдельные якоря стоят на переменной печати "
              "(месяц, год, номер) и дают выбросы.", "",
              "| `score` | Сканов | медиана сдвига ≤ 2 px | p75 ≤ 4 px | медиана медиан, px |",
              "|---|---:|---:|---:|---:|"]
    edges = [0.0, 0.3, 0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        g = df[(df["score"] >= lo) & (df["score"] < hi)]
        if g.empty:
            continue
        lines.append(
            f"| {lo:.2f}–{min(hi, 1.0):.2f} | {len(g)} | "
            f"{_pct((g['resid_med'] <= 2).sum(), len(g))} | "
            f"{_pct((g['resid_p75'] <= 4).sum(), len(g))} | "
            + ("—" if g["resid_med"].isna().all() else f"{g['resid_med'].median():.1f}")
            + " |"
        )
    cross = _cross_tug_scores(df)
    if len(cross):
        lines += ["", f"Чужой бланк (эталон другого буксира), {len(cross)} выравниваний: "
                  f"до пост-проверки дошли {int((cross > 0).sum())}, их `score` — "
                  f"медиана {cross[cross > 0].median():.2f}, максимум {cross.max():.2f}. "
                  f"Свой бланк: минимум `score` среди выровненных {aligned['score'].min():.2f}."]

    lines += _block_fit_section(df)

    lines += ["", "## Типы отказов", ""]
    if FAIL_TYPES_JSON.exists():
        types = json.loads(FAIL_TYPES_JSON.read_text(encoding="utf-8"))
        counts = Counter(types.get("by_scan", {}).values())
        for kind, count in counts.most_common():
            note = types.get("descriptions", {}).get(kind, "")
            lines.append(f"- **{kind}** — {count}" + (f": {note}" if note else ""))
        for note in types.get("notes", []):
            lines.append(f"- {note}")
    else:
        lines.append("- типы не размечены.")
    reasons: dict[str, int] = defaultdict(int)
    for scan_id in failed["scan_id"]:
        reasons[_reason_kind(scan_id)] += 1
    if reasons:
        lines += ["", "Причины отказа по данным `align` (лучший вариант):", ""]
        lines += [f"- {kind}: {count}" for kind, count in sorted(reasons.items())]

    sec = df["seconds"].dropna()
    fast = df["seconds_fast"].dropna()
    lines += ["", "## Время", "",
              f"- `detect_variant` (выравнивание по {len(layouts)} эталонам), на страницу: "
              f"среднее {sec.mean():.2f} с, p50 {sec.median():.2f} с, "
              f"p95 {sec.quantile(0.95):.2f} с;",
              f"- рабочий режим (подсказка буксира из имени файла и ранний выход): "
              f"среднее {fast.mean():.2f} с, p50 {fast.median():.2f} с, "
              f"p95 {fast.quantile(0.95):.2f} с;",
              "- измерено в 4 параллельных процессах на CPU, страница 1654×2340."]
    return "\n".join(lines) + "\n"


def _notes(key: str) -> list[str]:
    """Заметки ручной проверки из ``fail_types.json`` (список строк по ключу)."""
    if not FAIL_TYPES_JSON.exists():
        return []
    return list(json.loads(FAIL_TYPES_JSON.read_text(encoding="utf-8")).get(key, []))


def _tug_margins(assign: pd.DataFrame) -> pd.Series:
    """Отрыв ``score`` найденного варианта от лучшего варианта другого буксира (выровненные)."""
    variant_tug = {v.name: v.tug_code for v in VARIANTS}
    static = load_align(STATIC_CSV)
    static["tug"] = static["variant"].map(variant_tug)
    by_tug = static.groupby(["scan_id", "tug"])["score"].max().unstack()
    aligned = assign[assign["ok"]].set_index("scan_id")
    found = aligned["variant"].map(variant_tug)
    own = pd.Series([by_tug.at[s, t] for s, t in found.items()], index=found.index)
    other = by_tug.loc[found.index].where(
        by_tug.loc[found.index].columns.to_numpy() != found.to_numpy()[:, None]
    ).max(axis=1)
    return (own - other.fillna(0.0)).astype(float)


def _block_fit_section(df: pd.DataFrame) -> list[str]:
    """Раздел отчёта о посадке шапки и блока строк дат (если шаг ``blockfit`` выполнен)."""
    if not BLOCK_FIT_CSV.exists():
        return []
    fit = pd.read_csv(BLOCK_FIT_CSV, encoding="utf-8").drop(columns="variant")
    df = df.merge(fit, on="scan_id", how="left")
    tol = BLOCK_TOLERANCE_PX
    date_bad = df["date_med"] > tol
    head_bad = (df["head_n"].fillna(0) < 3) | (df["head_med"] > tol)
    lines = ["", "## Посадка блоков (шапка и строки дат)", "",
             "Медиана остаточного сдвига печати внутри полосы после выравнивания (окна по сетке "
             f"8×3, поиск ±{BLOCK_SEARCH_PX} px). Медиана, а не перцентиль: часть окон блока дат "
             "попадает на печатные день и месяц предзаполненных бланков. Допуск — "
             f"{tol:.0f} px: столько поглощают поля кропов.", "",
             f"| `score` | Сканов | дат ≤ 3 px | дат ≤ {tol:.0f} px | дат > {tol:.0f} px | "
             f"шапка > {tol:.0f} px |",
             "|---|---:|---:|---:|---:|---:|"]
    edges = [0.0, 0.62, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.01]
    for lo, hi in zip(edges[:-1], edges[1:], strict=True):
        sel = (df["score"] >= lo) & (df["score"] < hi)
        g = df[sel]
        if g.empty:
            continue
        lines.append(
            f"| {lo:.2f}–{min(hi, 1.0):.2f} | {len(g)} | {int((g['date_med'] <= 3).sum())} | "
            f"{int((g['date_med'] <= tol).sum())} | {int(date_bad[sel].sum())} | "
            f"{int(head_bad[sel].sum())} |"
        )
    ok = df["ok"]
    lines += ["",
              f"- среди выровненных ({int(ok.sum())}) блок дат сдвинут больше допуска у "
              f"{int((date_bad & ok).sum())} ({_pct((date_bad & ok).sum(), ok.sum())}), "
              f"шапка (номер ваучера) — у {int((head_bad & ok).sum())} "
              f"({_pct((head_bad & ok).sum(), ok.sum())});",
              "- сдвиги непрерывны (не кластеры), то есть это не отдельные варианты, а плавающая "
              "вёрстка: бланк заполняют в редакторе, и многострочные поля сдвигают таблицу. "
              "Нужно локальное уточнение блока после гомографии (T08–T10, путь А T11)."]
    return lines


def _cross_tug_scores(assign: pd.DataFrame) -> pd.Series:
    """``score`` выравниваний по эталонам чужого буксира (буксир — по найденному варианту)."""
    variant_tug = {v.name: v.tug_code for v in VARIANTS}
    static = load_align(STATIC_CSV)
    found = assign.set_index("scan_id")["variant"].map(variant_tug)
    cross = static["variant"].map(variant_tug) != static["scan_id"].map(found)
    known = static["scan_id"].map(assign.set_index("scan_id")["ok"]).astype(bool)
    return static.loc[cross & known, "score"]


_REASONS: dict[str, str] = {}


def _reason_kind(scan_id: str) -> str:
    if not _REASONS:
        static = load_align(STATIC_CSV)
        for row in static[static["is_best"]].itertuples():
            reason = str(row.reason)
            if "score" in reason:
                kind = "низкий score"
            elif "перепроецирования" in reason:
                kind = "ошибка перепроецирования"
            elif "инлайеров" in reason or "признаков" in reason or "пар" in reason:
                kind = "мало совпавших признаков"
            elif reason:
                kind = "неправдоподобная гомография"
            else:
                kind = "прочее"
            _REASONS[str(row.scan_id)] = kind
    return _REASONS.get(scan_id, "прочее")


def step_report(manifest: pd.DataFrame) -> pd.DataFrame:
    """Пересчитывает ``ok`` «до» и «после» по итоговым порогам, пишет CSV и отчёт."""
    assign = build_assignment(manifest)
    layouts = load_layouts(LAYOUTS_DIR)
    initial = load_align(INITIAL_CSV)
    # «До» — начальный эталон того же варианта, который нашёл detect_variant.
    found = dict(zip(assign["scan_id"], assign["variant"], strict=True))
    same = initial[initial["variant"] == initial["scan_id"].map(found)].set_index("scan_id")
    own = own_rows(load_align(STATIC_CSV), manifest)
    soft = {name: dataclasses.replace(layout.reference.params, min_score=0.3)
            for name, layout in layouts.items()}
    for column, frame, params in (
        ("init_ok", same, {n: lay.reference.params for n, lay in layouts.items()}),
        ("init_ok_soft", same, soft),
        ("own_ok", own, {n: lay.reference.params for n, lay in layouts.items()}),
    ):
        flags = pd.Series(False, index=frame.index, dtype=bool)
        for name, p in params.items():
            sel = frame["variant"] == name
            # Массив, а не Series: иначе pandas выравнивает значение по всему индексу (NaN).
            flags[sel] = ok_mask(frame[sel], p).to_numpy(dtype=bool)
        assign[column] = assign["scan_id"].isin(flags.index[flags])
    assign["init_score"] = assign["scan_id"].map(same["score"])
    assign.to_csv(ASSIGNMENT_CSV, index=False, encoding="utf-8")
    ensure_dir(REPORT_MD.parent)
    REPORT_MD.write_text(render_report(assign), encoding="utf-8")
    return assign


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T07: варианты бланка и эталоны")
    parser.add_argument(
        "step",
        choices=["metrics", "candidates", "initial", "worst", "static", "params", "realign",
                 "blockfit", "overlays", "report"],
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--bins", type=str, default="0.3,0.4,0.5,0.6,0.7",
                        help="границы диапазонов score для наложений")
    parser.add_argument("--set", action="append", default=[], metavar="ПОЛЕ=ЗНАЧЕНИЕ",
                        help="для шага params: порог AlignParams в meta.json вариантов (JSON)")
    parser.add_argument("--source", choices=["initial", "static"], default="initial",
                        help="для шага worst: по какому выравниванию брать худших")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    ids = sorted(manifest["scan_id"])
    if args.step == "metrics":
        df = step_metrics(manifest, args.workers)
        print(f"метрики: {len(df)} страниц")
    elif args.step == "candidates":
        cands = step_candidates(pd.read_csv(METRICS_CSV, encoding="utf-8"))
        print(f"кандидатов: {len(cands)}; листы — {LAYOUT_SHEETS_DIR}")
    elif args.step == "initial":
        return 1 if _run_align("initial", ids, INITIAL_CSV, args.workers, args.max_minutes) else 0
    elif args.step == "worst":
        worst = step_worst(manifest, source=INITIAL_CSV if args.source == "initial" else STATIC_CSV)
        print(f"худших: {len(worst)}; листы — {LAYOUT_SHEETS_DIR}")
    elif args.step == "static":
        step_static(manifest)
    elif args.step == "params":
        overrides = {k: json.loads(v) for k, v in (item.split("=", 1) for item in args.set)}
        set_align_params(overrides)
        print(f"пороги записаны в meta.json вариантов: {overrides}")
    elif args.step == "realign":
        return 1 if _run_align("static", ids, STATIC_CSV, args.workers, args.max_minutes) else 0
    elif args.step == "blockfit":
        step_blockfit(args.workers)
    elif args.step == "overlays":
        step_overlays(manifest, [float(v) for v in args.bins.split(",")])
    elif args.step == "report":
        assign = step_report(manifest)
        print(f"отчёт: {REPORT_MD}; страниц {len(assign)}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
