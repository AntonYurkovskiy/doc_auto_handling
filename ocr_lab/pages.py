"""Кэш страниц сканов: `data/ocr/pages/<scan_id>.png` — серое, ширина 1654 px.

Для каждой строки `manifest.csv` рендерится страница 0 скана при 200 dpi и
приводится к ширине A4 (`app.ocr.io.load_scan` + `normalize_width`). Кэш нужен
выравниванию (T06/T07), нарезке подполей и рантайму: дальше все работают с
одинаковым масштабом.

Запуск: `python -m ocr_lab.pages build [--workers 4] [--limit N]`.
Побочные артефакты: `data/ocr/pages_index.csv` (метаданные всех прогонов —
число страниц, исходный размер, dpi; чтобы отчёт был полным после повторных
запусков), `data/ocr/reports/pages_report.md` и лист-превью
`data/ocr/sheets/pages_sample.png` (48 случайных страниц, для T07).
"""

from __future__ import annotations

import argparse
import csv
import math
import random
import sys
import time
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import pandas as pd
from PIL import Image, ImageDraw, ImageFont

from app.ocr.io import (
    DEFAULT_DPI,
    PAGE_WIDTH_200DPI,
    ScanLoadError,
    load_scan,
    normalize_width,
    page_count,
    page_size_px,
)
from ocr_lab.paths import MANIFEST, PAGES_DIR, REPORTS_DIR, SHEETS_DIR, ensure_dir

#: Сколько случайных страниц собирать в лист-превью `pages_sample.png`.
SAMPLE_SHEET_SIZE = 48
SAMPLE_SHEET_SEED = 42

#: Колонки индекса страниц `data/ocr/pages_index.csv`: метаданные всех когда-либо
#: отрендеренных сканов, чтобы отчёт оставался полным после идемпотентных прогонов.
INDEX_COLUMNS = ("scan_id", "n_pages", "src_width", "src_height", "dpi")


@dataclass
class PageStat:
    """Результат обработки одного скана — только агрегаты и scan_id."""

    scan_id: str
    status: str  # "ok" | "skipped" | "error"
    error: str = ""
    n_pages: int = 0
    src_width: int = 0  # размер рендера страницы 0 до нормализации, px
    src_height: int = 0
    dpi: int = 0  # dpi рендера (для картинок — 0, у них нет dpi рендера)
    elapsed: float = 0.0  # секунды на загрузку+нормализацию+запись


def _process_one(scan_id: str, scan_path: str, out_dir: str) -> PageStat:
    """Обработать один скан. Функция верхнего уровня — нужна для spawn-процессов."""
    out_path = Path(out_dir) / f"{scan_id}.png"
    path = Path(scan_path)
    is_pdf = path.suffix.lower() == ".pdf"
    if out_path.exists():
        # Идемпотентность: файл уже есть — рендер пропускаем, но метаданные
        # (число страниц, исходный размер) считаем — это дёшево, без рендера.
        stat = PageStat(scan_id=scan_id, status="skipped")
        try:
            stat.n_pages = page_count(path)
            stat.src_width, stat.src_height = page_size_px(path, dpi=DEFAULT_DPI)
            stat.dpi = DEFAULT_DPI if is_pdf else 0
        except Exception:
            pass
        return stat

    t0 = time.perf_counter()
    try:
        n_pages = page_count(path)
        img = load_scan(path, dpi=DEFAULT_DPI, page=0)
        src_h, src_w = img.shape
        img = normalize_width(img, PAGE_WIDTH_200DPI)
        # Не cv2.imwrite: он не умеет не-ASCII пути на Windows (scan_id бывает с
        # кириллицей — «67к») и пишет файл под искажённым именем. imencode +
        # write_bytes идёт через Python-файл, где UTF-8 имя работает.
        ok, encoded = cv2.imencode(".png", img)
        if not ok:
            raise ScanLoadError(f"cv2.imencode не смог закодировать {out_path.name}")
        out_path.write_bytes(encoded.tobytes())
        return PageStat(
            scan_id=scan_id,
            status="ok",
            n_pages=n_pages,
            src_width=src_w,
            src_height=src_h,
            dpi=DEFAULT_DPI if is_pdf else 0,
            elapsed=time.perf_counter() - t0,
        )
    except Exception as exc:
        # Ошибка одного скана не должна ронять прогон; в отчёт — только scan_id
        # и тип ошибки, без содержимого файла.
        message = f"{type(exc).__name__}: {exc}"
        return PageStat(
            scan_id=scan_id,
            status="error",
            error=message[:200],
            elapsed=time.perf_counter() - t0,
        )


def build(
    workers: int = 4,
    limit: int | None = None,
    manifest_path: Path = MANIFEST,
    out_dir: Path | None = None,
) -> list[PageStat]:
    """Построить кэш страниц по манифесту. Готовые файлы пропускаются."""
    out = ensure_dir(out_dir or PAGES_DIR)
    df = pd.read_csv(manifest_path, dtype=str)
    jobs = [(str(row.scan_id), str(row.scan_path)) for row in df.itertuples()]
    if limit is not None:
        jobs = jobs[:limit]

    if workers > 1:
        with ProcessPoolExecutor(max_workers=workers) as executor:
            futures = [
                executor.submit(_process_one, scan_id, scan_path, str(out))
                for scan_id, scan_path in jobs
            ]
            stats = [f.result() for f in futures]
    else:
        stats = [_process_one(scan_id, scan_path, str(out)) for scan_id, scan_path in jobs]
    return stats


def _index_path(out_dir: Path) -> Path:
    """Индекс страниц лежит рядом с каталогом кэша: `data/ocr/pages_index.csv`."""
    return out_dir.parent / "pages_index.csv"


def _load_index(path: Path) -> dict[str, dict[str, str]]:
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return {row["scan_id"]: row for row in csv.DictReader(fh)}


def update_index(stats: list[PageStat], out_dir: Path) -> dict[str, dict[str, str]]:
    """Слить результаты прогона в `pages_index.csv` и вернуть весь индекс.

    `ok` перезаписывает строку; `skipped` заполняет только те поля, которых
    в индексе ещё нет (нулевые), данные прошлых прогонов не затирает;
    `error` индекс не трогает.
    """
    index_path = _index_path(out_dir)
    rows = _load_index(index_path)
    for s in stats:
        if s.status == "ok":
            rows[s.scan_id] = {
                "scan_id": s.scan_id,
                "n_pages": str(s.n_pages),
                "src_width": str(s.src_width),
                "src_height": str(s.src_height),
                "dpi": str(s.dpi),
            }
        elif s.status == "skipped":
            row = rows.setdefault(
                s.scan_id,
                {"scan_id": s.scan_id, "n_pages": "0", "src_width": "0",
                 "src_height": "0", "dpi": "0"},
            )
            for key, val in (
                ("n_pages", s.n_pages),
                ("src_width", s.src_width),
                ("src_height", s.src_height),
                ("dpi", s.dpi),
            ):
                if val and int(row.get(key) or 0) == 0:
                    row[key] = str(val)
    ensure_dir(index_path.parent)
    with index_path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=INDEX_COLUMNS)
        writer.writeheader()
        for scan_id in sorted(rows):
            writer.writerow(rows[scan_id])
    return rows


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def render_report(
    stats: list[PageStat],
    index_rows: dict[str, dict[str, str]],
    total_rows: int,
) -> str:
    """Отчёт только из агрегатов: счётчики и scan_id ошибок.

    Число страниц, размеры и ориентация — из `index_rows` (метаданные всех
    прогонов), поэтому отчёт полон и после идемпотентного запуска. Время —
    только по страницам, записанным в этом прогоне.
    """
    done = [s for s in stats if s.status == "ok"]
    skipped = [s for s in stats if s.status == "skipped"]
    errors = [s for s in stats if s.status == "error"]

    lines: list[str] = ["# Отчёт по кэшу страниц (`data/ocr/pages`)", ""]
    lines.append("## Итог")
    lines.append(f"- строк манифеста: {total_rows}")
    lines.append(f"- записано страниц: {len(done)}")
    lines.append(f"- пропущено (файл уже есть): {len(skipped)}")
    lines.append(f"- ошибок: {len(errors)}")
    for s in errors:
        lines.append(f"  - {s.scan_id}: {s.error}")
    lines.append("")

    # Метаданные по сканам этого прогона — из индекса (живёт между прогонами).
    indexed = [index_rows[s.scan_id] for s in stats if s.scan_id in index_rows]

    lines.append("## Число страниц в исходном скане")
    pages_counter = Counter(int(r["n_pages"]) for r in indexed if int(r["n_pages"]) > 0)
    unknown = len(stats) - sum(pages_counter.values())
    for n, cnt in sorted(pages_counter.items()):
        lines.append(f"- {n} стр.: {cnt}")
    if unknown:
        lines.append(f"- неизвестно (ошибка чтения): {unknown}")
    lines.append("")

    sized = [r for r in indexed if int(r["src_width"]) > 0]
    lines.append(f"## Исходные размеры рендера (dpi={DEFAULT_DPI} для PDF)")
    if sized:
        widths = sorted(int(r["src_width"]) for r in sized)
        heights = sorted(int(r["src_height"]) for r in sized)
        lines.append(
            f"- ширина, px: min={widths[0]}, p50={_percentile(widths, 0.5):.0f}, "
            f"max={widths[-1]}"
        )
        lines.append(
            f"- высота, px: min={heights[0]}, p50={_percentile(heights, 0.5):.0f}, "
            f"max={heights[-1]}"
        )
        dpi_counter = Counter(int(r["dpi"]) for r in sized)
        lines.append("- dpi рендера: " + ", ".join(
            f"{d}: {c}" for d, c in sorted(dpi_counter.items())
        ))
        size_counter = Counter((r["src_width"], r["src_height"]) for r in sized)
        lines.append("- частые размеры (ШxВ):")
        for (w, h), cnt in size_counter.most_common(10):
            lines.append(f"  - {w}x{h}: {cnt}")
    else:
        lines.append("- нет данных")
    lines.append("")

    lines.append("## Ориентация")
    landscape = sum(1 for r in sized if int(r["src_width"]) > int(r["src_height"]))
    if sized:
        lines.append(
            f"- альбомных страниц: {landscape} из {len(sized)} "
            f"({100.0 * landscape / len(sized):.1f} %)"
        )
    else:
        lines.append("- нет данных")
    lines.append("")

    lines.append("## Время на страницу (этот прогон)")
    times = [s.elapsed for s in done]
    if times:
        lines.append(
            f"- среднее={sum(times) / len(times):.2f} с, "
            f"p50={_percentile(times, 0.5):.2f} с, "
            f"p95={_percentile(times, 0.95):.2f} с, "
            f"всего={sum(times):.1f} с"
        )
    else:
        lines.append("- в этом прогоне страницы не записывались")
    lines.append("")
    return "\n".join(lines)


def write_sample_sheet(
    stats: list[PageStat],
    pages_dir: Path,
    sheet_path: Path,
    n: int = SAMPLE_SHEET_SIZE,
    seed: int = SAMPLE_SHEET_SEED,
    thumb_w: int = 200,
    thumb_h: int = 280,
    cols: int = 8,
) -> int:
    """Лист-превью: `n` случайных страниц кэша, миниатюры с подписью scan_id.

    Длинная сторона листа остаётся ≤ 2000 px (правило `_common.md`).
    Возвращает число размещённых миниатюр.
    """
    produced = [
        (s.scan_id, pages_dir / f"{s.scan_id}.png")
        for s in stats
        if s.status in ("ok", "skipped") and (pages_dir / f"{s.scan_id}.png").exists()
    ]
    rng = random.Random(seed)
    if len(produced) > n:
        produced = rng.sample(produced, n)
    else:
        produced = sorted(produced)
    if not produced:
        return 0

    label_h = 22
    pad = 8
    cell_w, cell_h = thumb_w + pad, thumb_h + label_h + pad
    rows = math.ceil(len(produced) / cols)
    sheet = Image.new("L", (cols * cell_w + pad, rows * cell_h + pad), 255)
    draw = ImageDraw.Draw(sheet)
    font = ImageFont.load_default(size=13)
    for i, (scan_id, png_path) in enumerate(produced):
        with Image.open(png_path) as img:
            thumb = img.convert("L")
            thumb.thumbnail((thumb_w, thumb_h))
        x = pad + (i % cols) * cell_w
        y = pad + (i // cols) * cell_h
        sheet.paste(thumb, (x, y))
        draw.text((x, y + thumb_h + 2), scan_id, fill=0, font=font)
    ensure_dir(sheet_path.parent)
    sheet.save(sheet_path)
    return len(produced)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Кэш страниц сканов ваучеров (серое, 1654 px)")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="отрендерить страницы манифеста в data/ocr/pages")
    p_build.add_argument("--workers", type=int, default=4, help="число процессов (1 — без пула)")
    p_build.add_argument("--limit", type=int, default=None, help="обработать только первые N строк")
    args = parser.parse_args(argv)

    if args.command == "build":
        df = pd.read_csv(MANIFEST, dtype=str)
        out_dir = ensure_dir(PAGES_DIR)
        stats = build(workers=args.workers, limit=args.limit, out_dir=out_dir)
        index_rows = update_index(stats, out_dir)
        ensure_dir(REPORTS_DIR)
        report_path = REPORTS_DIR / "pages_report.md"
        report_path.write_text(
            render_report(stats, index_rows, total_rows=len(df)), encoding="utf-8"
        )
        n_sheet = write_sample_sheet(stats, out_dir, SHEETS_DIR / "pages_sample.png")
        ok = sum(1 for s in stats if s.status == "ok")
        skipped = sum(1 for s in stats if s.status == "skipped")
        errors = sum(1 for s in stats if s.status == "error")
        print(f"pages: записано {ok}, пропущено {skipped}, ошибок {errors} -> {PAGES_DIR}")
        print(f"отчёт -> {report_path}")
        print(f"лист-превью ({n_sheet} страниц) -> {SHEETS_DIR / 'pages_sample.png'}")
        return 0 if errors == 0 else 1
    return 1


if __name__ == "__main__":
    sys.exit(main())
