"""T15: аугментации кропов двузначных подполей — только cv2/numpy, без albumentations.

Все искажения детерминированы по `seed` (:func:`augment`): одинаковые `image` и `seed`
всегда дают одинаковый результат, что важно и для тестов, и для воспроизводимости
(``cv2.setRNGSeed`` сюда не относится — здесь только `numpy.random.Generator`, cv2
используется исключительно для геометрии/блюра/JPEG без собственного ГСЧ).

Набор техник (п. 2 промпта T15, план «Этап 2»):

- `geometric` — поворот ±5°, сдвиг и масштаб кропа ±10 % (имитирует ошибку
  выравнивания), лёгкая перспектива;
- `morph` — эрозия/дилатация 1–2 px (толщина штриха);
- `photometric` — яркость, контраст, гамма, шум;
- `blur` — Gaussian blur;
- `underline` — синтетическая линия подчёркивания;
- `neighbor` — обрывок печатного символа соседнего поля у края кропа;
- `erasing` — маленький стёртый (белый) прямоугольник;
- `jpeg` — пережатие JPEG качеством 30–90.

:func:`augment` без `ops` применяет случайное подмножество техник (у каждой — своя
вероятность, :data:`OP_PROB`), чтобы не ломать цифру комбинацией всех искажений разом.
Превью (п. 5 промпта) — :func:`build_preview`, CLI `python -m ocr_lab.augment preview`.
"""

from __future__ import annotations

import argparse
import csv
import random
from collections.abc import Callable, Sequence
from pathlib import Path

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

from ocr_lab.cut_crops import read_csv_rows, read_gray
from ocr_lab.paths import SHEETS_DIR, WORK_DIR, ensure_dir
from ocr_lab.predictions import PARTS, ROWS

#: Двузначные подполя (день/месяц/час/минута всех строк) — без номера ваучера, у него
#: другой вид бокса («number», не «two_digit», см. app.ocr.layouts).
SUBFIELDS: tuple[str, ...] = tuple(f"{r}.{p}" for r in ROWS for p in PARTS)

INDEX_CSV = WORK_DIR / "crops_index.csv"

# --- геометрия -------------------------------------------------------------------------

MAX_ROTATE_DEG = 5.0
MAX_SHIFT_FRAC = 0.10
SCALE_RANGE = (0.90, 1.10)
PERSPECTIVE_JITTER_FRAC = 0.02
#: Цвет фона кропа (белый бланк) — заливка границ при геометрических искажениях.
BORDER_VALUE = 255


def geometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Поворот ±5°, сдвиг и масштаб ±10 % (ошибка выравнивания), лёгкая перспектива."""
    h, w = image.shape[:2]
    angle = rng.uniform(-MAX_ROTATE_DEG, MAX_ROTATE_DEG)
    scale = rng.uniform(*SCALE_RANGE)
    dx = rng.uniform(-MAX_SHIFT_FRAC, MAX_SHIFT_FRAC) * w
    dy = rng.uniform(-MAX_SHIFT_FRAC, MAX_SHIFT_FRAC) * h
    matrix = cv2.getRotationMatrix2D((w / 2, h / 2), angle, scale)
    matrix[0, 2] += dx
    matrix[1, 2] += dy
    out = cv2.warpAffine(
        image, matrix, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_VALUE,
    )
    jitter = PERSPECTIVE_JITTER_FRAC * min(h, w)
    src = np.array([[0, 0], [w, 0], [w, h], [0, h]], dtype=np.float32)
    dst = src + rng.uniform(-jitter, jitter, size=src.shape).astype(np.float32)
    persp = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(
        out, persp, (w, h), flags=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_CONSTANT, borderValue=BORDER_VALUE,
    )


# --- толщина штриха ----------------------------------------------------------------------


def stroke_thickness(image: np.ndarray, delta_px: int) -> np.ndarray:
    """Изменить толщину штриха на ``|delta_px|`` пикселей (обычно 1 или 2).

    Фон светлый (255), чернила — тёмные, поэтому направления морфологии обратные
    интуиции: чтобы утолщить штрих, нужен ``cv2.erode`` (он расширяет тёмные области —
    берёт минимум по окну), а чтобы утончить — ``cv2.dilate`` (расширяет светлые, берёт
    максимум). ``delta_px > 0`` — толще, ``delta_px < 0`` — тоньше, ``0`` — без изменений.
    """
    if delta_px == 0:
        return image
    size = 2 * abs(delta_px) + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (size, size))
    return cv2.erode(image, kernel) if delta_px > 0 else cv2.dilate(image, kernel)


def morph(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    delta = int(rng.choice([-2, -1, 1, 2]))
    return stroke_thickness(image, delta)


# --- blur / jpeg -------------------------------------------------------------------------

BLUR_SIGMA_RANGE = (0.3, 1.2)
JPEG_QUALITY_RANGE = (30, 90)


def blur(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    sigma = float(rng.uniform(*BLUR_SIGMA_RANGE))
    k = max(3, int(2 * round(3 * sigma) + 1))
    return cv2.GaussianBlur(image, (k, k), sigma)


def jpeg(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Пережать кроп в JPEG заданного качества и раскодировать обратно."""
    quality = int(rng.integers(JPEG_QUALITY_RANGE[0], JPEG_QUALITY_RANGE[1] + 1))
    ok, buf = cv2.imencode(".jpg", image, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return image
    decoded = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    return image if decoded is None else decoded


# --- фотометрия ----------------------------------------------------------------------------

BRIGHTNESS_RANGE = (-25.0, 25.0)
CONTRAST_RANGE = (0.8, 1.2)
GAMMA_RANGE = (0.7, 1.4)
NOISE_SIGMA_RANGE = (2.0, 10.0)


def photometric(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Яркость, контраст, гамма и шум — одно случайное сочетание параметров."""
    brightness = rng.uniform(*BRIGHTNESS_RANGE)
    contrast = rng.uniform(*CONTRAST_RANGE)
    gamma = rng.uniform(*GAMMA_RANGE)
    out = np.clip(image.astype(np.float32) * contrast + brightness, 0, 255).astype(np.uint8)
    table = (np.linspace(0, 1, 256) ** (1.0 / gamma) * 255.0).astype(np.float32)
    out = table[out]
    sigma = float(rng.uniform(*NOISE_SIGMA_RANGE))
    out = out + rng.normal(0.0, sigma, size=out.shape)
    return np.clip(out, 0, 255).astype(np.uint8)


# --- синтетический «мусор» бланка ----------------------------------------------------------

UNDERLINE_THICKNESS = (1, 2)


def underline(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Синтетическая линия подчёркивания у низа кропа (часть бланка под полем)."""
    h, w = image.shape[:2]
    out = image.copy()
    y = int(rng.integers(max(1, int(h * 0.82)), h))
    x0 = int(rng.integers(0, max(1, w // 4)))
    x1 = int(rng.integers(max(x0 + 1, 3 * w // 4), w))
    thickness = int(rng.integers(UNDERLINE_THICKNESS[0], UNDERLINE_THICKNESS[1] + 1))
    gray = int(rng.integers(0, 60))
    cv2.line(out, (x0, y), (x1, y), gray, thickness, cv2.LINE_AA)
    return out


def neighbor_fragment(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Обрывок печатного символа соседнего поля у левого или правого края кропа."""
    h, w = image.shape[:2]
    out = image.copy()
    frag_w = max(2, int(w * rng.uniform(0.03, 0.08)))
    frag_h = max(3, int(h * rng.uniform(0.3, 0.7)))
    y0 = int(rng.integers(0, max(1, h - frag_h)))
    x0 = 0 if bool(rng.integers(0, 2)) else max(0, w - frag_w)
    for _ in range(int(rng.integers(1, 3))):
        p1 = (x0 + int(rng.integers(0, frag_w)), y0 + int(rng.integers(0, frag_h)))
        p2 = (x0 + int(rng.integers(0, frag_w)), y0 + int(rng.integers(0, frag_h)))
        cv2.line(out, p1, p2, int(rng.integers(0, 90)), 1, cv2.LINE_AA)
    return out


ERASE_FRAC_RANGE = (0.05, 0.12)


def random_erasing(image: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """Маленький стёртый (белый) прямоугольник — пропуск чернил или блик на скане."""
    h, w = image.shape[:2]
    out = image.copy()
    eh = max(1, int(h * rng.uniform(*ERASE_FRAC_RANGE)))
    ew = max(1, int(w * rng.uniform(*ERASE_FRAC_RANGE)))
    y0 = int(rng.integers(0, max(1, h - eh)))
    x0 = int(rng.integers(0, max(1, w - ew)))
    out[y0 : y0 + eh, x0 : x0 + ew] = BORDER_VALUE
    return out


# --- композиция ------------------------------------------------------------------------

OPS: dict[str, Callable[[np.ndarray, np.random.Generator], np.ndarray]] = {
    "geometric": geometric,
    "morph": morph,
    "photometric": photometric,
    "underline": underline,
    "neighbor": neighbor_fragment,
    "erasing": random_erasing,
    "blur": blur,
    "jpeg": jpeg,
}
#: Порядок применения техник при случайной композиции: сначала свойства «листа» и
#: почерка (геометрия, толщина штриха), затем освещение/шум, затем мусор страницы и
#: блюр, JPEG — последним шагом (имитирует сохранение файла после сканирования).
OP_ORDER: tuple[str, ...] = (
    "geometric", "morph", "photometric", "underline", "neighbor", "erasing", "blur", "jpeg",
)
#: Вероятность применить технику при случайной композиции — не все сразу, иначе
#: комбинация всех искажений может уничтожить цифру.
OP_PROB: dict[str, float] = {
    "geometric": 0.9, "morph": 0.5, "photometric": 0.8, "underline": 0.25,
    "neighbor": 0.2, "erasing": 0.15, "blur": 0.5, "jpeg": 0.6,
}


def augment(image: np.ndarray, seed: int, *, ops: Sequence[str] | None = None) -> np.ndarray:
    """Детерминированная аугментация кропа по `seed`.

    `ops=None` — случайный набор техник (вероятности :data:`OP_PROB`, не меньше одной).
    `ops=[...]` — применить именно эти техники, в каноническом порядке
    :data:`OP_ORDER` (для превью и тестов отдельных эффектов).
    Одинаковые `image` и `seed` всегда дают одинаковый результат (детерминизм — п. 2
    промпта T15); разные `seed` почти всегда дают разные результаты.
    """
    rng = np.random.default_rng(seed)
    if ops is not None:
        unknown = set(ops) - set(OPS)
        if unknown:
            raise ValueError(f"неизвестные техники аугментации: {sorted(unknown)}")
        chosen = [op for op in OP_ORDER if op in ops]
    else:
        chosen = [op for op in OP_ORDER if rng.random() < OP_PROB[op]]
        if not chosen:
            chosen = [str(rng.choice(OP_ORDER))]
    out = image
    for op in chosen:
        out = OPS[op](out, rng)
    return out


# --- превью (п. 5 промпта) ---------------------------------------------------------------

PREVIEW_N = 12
PREVIEW_SEED = 0
PREVIEW_PATH = SHEETS_DIR / "augment_preview.png"
PREVIEW_CELL_W = 140
PREVIEW_LABEL_W = 170
PREVIEW_FONT_SIZE = 12

_FONT_CANDIDATES = (
    "DejaVuSans.ttf", "arial.ttf", "C:/Windows/Fonts/arial.ttf", "C:/Windows/Fonts/segoeui.ttf",
    "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf", "/Library/Fonts/Arial.ttf",
)


def _font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    for name in _FONT_CANDIDATES:
        try:
            return ImageFont.truetype(name, size)
        except OSError:
            continue
    return ImageFont.load_default(size=size)


def _resize_to_width(image: np.ndarray, width: int) -> Image.Image:
    img = Image.fromarray(image, mode="L")
    if img.width == width:
        return img
    height = max(1, round(img.height * width / img.width))
    return img.resize((width, height), Image.Resampling.LANCZOS)


def _sample_crops(
    index_csv: Path, *, n: int = PREVIEW_N, seed: int = PREVIEW_SEED
) -> list[tuple[str, str, Path]]:
    """`n` случайных (scan_id, subfield, путь) двузначных подполей — вход превью."""
    rows = [
        r for r in read_csv_rows(index_csv)
        if r["subfield"] in SUBFIELDS and (r.get("align_ok") or "True") == "True"
    ]
    rng = random.Random(seed)
    picked = rng.sample(rows, min(n, len(rows)))
    return [(r["scan_id"], r["subfield"], WORK_DIR / r["path"]) for r in picked]


def build_preview(
    *, index_csv: Path = INDEX_CSV, out_path: Path = PREVIEW_PATH,
    n: int = PREVIEW_N, seed: int = PREVIEW_SEED,
) -> Path:
    """Лист превью: `n` кропов × (оригинал + 8 аугментаций), п. 5 промпта T15.

    Не через `ocr_lab.sheets.make_sheet` — лимит 100 кропов на лист не подходит для
    ``n × 9`` ячеек (12 × 9 = 108). Раскладка — «строка на кроп, колонка на технику»,
    длинная сторона листа укладывается в лимит 2000 px при n=12 без дополнительного
    масштабирования (посчитано по размеру ячеек).
    """
    picks = _sample_crops(index_csv, n=n, seed=seed)
    if not picks:
        raise ValueError("build_preview: в индексе нет подходящих кропов")
    columns = ("оригинал", *OP_ORDER)
    font = _font(PREVIEW_FONT_SIZE)
    header_h = PREVIEW_FONT_SIZE + 8

    rows_imgs: list[list[Image.Image]] = []
    for i, (_scan_id, _subfield, path) in enumerate(picks):
        crop = read_gray(path)
        variants = [crop] + [augment(crop, seed=seed * 1000 + i, ops=(op,)) for op in OP_ORDER]
        rows_imgs.append([_resize_to_width(v, PREVIEW_CELL_W) for v in variants])

    row_h = max(img.height for row in rows_imgs for img in row)
    sheet_w = PREVIEW_LABEL_W + len(columns) * PREVIEW_CELL_W
    sheet_h = header_h + len(rows_imgs) * row_h
    sheet = Image.new("L", (sheet_w, sheet_h), 255)
    draw = ImageDraw.Draw(sheet)
    for c, name in enumerate(columns):
        draw.text((PREVIEW_LABEL_W + c * PREVIEW_CELL_W + 4, 2), name, fill=0, font=font)
    for r, (scan_id, subfield, _path) in enumerate(picks):
        y = header_h + r * row_h
        draw.text((4, y + row_h // 2 - PREVIEW_FONT_SIZE - 1), scan_id, fill=0, font=font)
        draw.text((4, y + row_h // 2 + 1), subfield, fill=0, font=font)
        for c, img in enumerate(rows_imgs[r]):
            x = PREVIEW_LABEL_W + c * PREVIEW_CELL_W
            sheet.paste(img, (x, y + (row_h - img.height) // 2))
    ensure_dir(out_path.parent)
    sheet.save(out_path)
    with out_path.with_suffix(".csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["row", "scan_id", "subfield"])
        writer.writerows((r, scan_id, subfield) for r, (scan_id, subfield, _p) in enumerate(picks))
    return out_path


# --- CLI ----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ocr_lab.augment", description=__doc__)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_preview = sub.add_parser("preview", help="лист превью аугментаций (п. 5 промпта T15)")
    p_preview.add_argument("--n", type=int, default=PREVIEW_N)
    p_preview.add_argument("--seed", type=int, default=PREVIEW_SEED)
    p_preview.add_argument("--out", type=Path, default=PREVIEW_PATH)

    args = parser.parse_args(argv)
    out = build_preview(n=args.n, seed=args.seed, out_path=args.out)
    print(f"записано: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
