"""Массовая нарезка кропов подполей (T10) и контроль качества.

Для каждого скана манифеста: страница из кэша T03 → вариант бланка (из
``layout_assignment.csv`` T07, для новых сканов — ``detect_variant``) → ``align`` по
статичному эталону варианта → ``crop_subfields`` по макету боксов T08/T09.

Шаги CLI (``python -m ocr_lab.cut_crops <шаг>``):

- ``build [--workers 4] [--force] [--variant V] [--subfields a,b] [--limit N]`` —
  кропы ``data/ocr/crops/<subfield>/<scan_id>.png`` (серые, в разрешении эталона, без
  ресайза), индекс ``data/ocr/crops_index.csv`` и отказы выравнивания
  ``data/ocr/reports/align_failures.csv``. Готовые сканы (все кропы на месте и строка в
  индексе) пропускаются; ``--force`` режет заново. Параллельно — процессы (spawn).
- ``sheets [--seed 0] [--n 200]`` — листы контроля: по каждому подполю ``n`` случайных
  кропов, листы по 100 в ``data/ocr/sheets/crops/`` с CSV «индекс → scan_id».
- ``report`` — ``data/ocr/reports/crops_report.md`` по индексу, отказам и ручной оценке
  ``data/ocr/review/crops_qc.csv`` (``subfield, scan_id, problem``) и заметки
  ``data/ocr/review/crops_qc_notes.md`` (если есть).

В консоль и отчёты идут только агрегаты и ``scan_id``.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import Counter
from collections.abc import Iterable, Sequence
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from app.ocr.align import AlignResult, align
from app.ocr.crops import crop_box, locate_boxes
from app.ocr.layouts import SUBFIELD_NAMES, load_layouts
from app.ocr.layouts import Layout as BoxLayout
from app.ocr.variants import Layout as VariantLayout
from app.ocr.variants import detect_variant
from app.ocr.variants import load_layouts as load_variants
from ocr_lab.paths import (
    CROPS_DIR,
    LAYOUTS_DIR,
    MANIFEST,
    PAGES_DIR,
    REPORTS_DIR,
    REVIEW_DIR,
    SHEETS_DIR,
    WORK_DIR,
    ensure_dir,
)
from ocr_lab.sheets import MAX_CROPS, make_sheet

INDEX_CSV = WORK_DIR / "crops_index.csv"
ASSIGNMENT_CSV = WORK_DIR / "layout_assignment.csv"
FAILURES_CSV = REPORTS_DIR / "align_failures.csv"
QC_CSV = REVIEW_DIR / "crops_qc.csv"
#: Заметки ручной проверки листов (markdown); если файл есть, он дописывается в отчёт.
QC_NOTES_MD = REVIEW_DIR / "crops_qc_notes.md"
CROP_SHEETS_DIR = SHEETS_DIR / "crops"
REPORT_MD = REPORTS_DIR / "crops_report.md"

#: Колонки индекса. ``source`` — как найден бокс (``Placement.source``), сверх промпта:
#: признак качества для T11/T19.
INDEX_COLUMNS = (
    "scan_id", "variant", "subfield", "path", "w", "h", "align_ok", "score", "inliers",
    "rotated180", "source",
)
FAILURE_COLUMNS = (
    "scan_id", "year", "variant", "score", "inliers", "reproj_err", "rotated180", "reason",
)
QC_COLUMNS = ("subfield", "scan_id", "problem")
#: Типы проблем ручной оценки листов.
PROBLEMS = {
    "cut_digit": "цифра обрезана краем кропа",
    "empty_misaligned": "пусто из-за промаха выравнивания или подгонки",
    "neighbour_digits": "попали цифры соседнего подполя",
    "rotated": "перевёрнуто",
    "other": "другое",
}

#: Цель этапа 1: значение видно целиком больше чем в 98 % кропов.
VISIBLE_TARGET = 0.98
#: Порог доли отказов выравнивания, выше которого нужна T11.
T11_FAIL_SHARE = 0.05
SAMPLE_PER_SUBFIELD = 200
SAMPLE_SEED = 0
SHEET_COLS = 10
SHEET_CELL_W = 180

#: Сид ГСЧ OpenCV перед каждым выравниванием: RANSAC иначе слегка недетерминирован
#: (T07: разброс ``score`` на ~0,6 % пар), а прогон должен быть воспроизводимым.
CV_SEED = 0


@dataclass
class ScanResult:
    """Итог обработки одного скана (только агрегаты и scan_id)."""

    scan_id: str
    year: str
    status: str  # "ok" | "skipped" | "failed" | "error"
    variant: str = ""
    score: float = 0.0
    inliers: int = 0
    reproj_err: float = 0.0
    rotated180: bool = False
    reason: str = ""
    seconds: float = 0.0
    rows: tuple[tuple[str, ...], ...] = ()  # строки индекса (по INDEX_COLUMNS)


# --- ввод-вывод ----------------------------------------------------------------------------


def read_gray(path: Path) -> np.ndarray:
    """Серое PNG; путь может содержать кириллицу (cv2.imread на Windows не читает)."""
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"не читается {path.name}")
    return img


def write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"не кодируется PNG: {path.name}")
    path.write_bytes(buf.tobytes())


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv_rows(path: Path, columns: Sequence[str], rows: Iterable[Sequence[object]]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(columns)
        writer.writerows(rows)


def crop_path(crops_dir: Path, subfield: str, scan_id: str) -> Path:
    return crops_dir / subfield / f"{scan_id}.png"


def rel_path(path: Path) -> str:
    """Путь кропа в индексе — относительно ``WORK_DIR``, с прямыми слешами."""
    try:
        return path.relative_to(WORK_DIR).as_posix()
    except ValueError:
        return path.as_posix()


def tug_hint(scan_id: str) -> str:
    """Код буксира из имени файла (``2025_12k`` → ``k``) — подсказка порядка вариантов."""
    from ocr_lab.layouts import name_tug_code

    return name_tug_code(scan_id)


# --- обработка одного скана ----------------------------------------------------------------

#: Кэш эталонов и макетов внутри процесса-воркера (грузятся один раз на процесс).
_WORKER: dict[str, object] = {}


def _resources(
    layouts_dir: str, boxes_dir: str | None
) -> tuple[dict[str, VariantLayout], dict[str, BoxLayout]]:
    key = f"{layouts_dir}|{boxes_dir}"
    if _WORKER.get("key") != key:
        _WORKER["variants"] = load_variants(Path(layouts_dir))
        _WORKER["boxes"] = load_layouts(Path(boxes_dir)) if boxes_dir else load_layouts()
        _WORKER["key"] = key
    return _WORKER["variants"], _WORKER["boxes"]  # type: ignore[return-value]


def _align_scan(
    page: np.ndarray, scan_id: str, variant: str, variants: dict[str, VariantLayout]
) -> tuple[str, AlignResult]:
    """Выравнивание по назначенному варианту; если его нет или он не прошёл — перебор."""
    cv2.setRNGSeed(CV_SEED)
    if variant in variants:
        res = align(page, variants[variant].reference)
        if res.ok:
            return variant, res
    cv2.setRNGSeed(CV_SEED)
    match = detect_variant(page, variants, prefer=tug_hint(scan_id) or None)
    if match.ok or variant not in variants:
        return match.variant, match.result
    # Ни один вариант не прошёл: в отказ пишем назначенный T07 вариант.
    return variant, match.results.get(variant, match.result)


def process_scan(
    scan_id: str,
    year: str,
    variant: str,
    pages_dir: str,
    crops_dir: str,
    layouts_dir: str,
    boxes_dir: str | None = None,
    subfields: tuple[str, ...] | None = None,
) -> ScanResult:
    """Выровнять скан и нарезать кропы. Функция верхнего уровня — нужна для spawn."""
    t0 = time.perf_counter()
    try:
        variants, box_layouts = _resources(layouts_dir, boxes_dir)
        page = read_gray(Path(pages_dir) / f"{scan_id}.png")
        variant, res = _align_scan(page, scan_id, variant, variants)
        base = ScanResult(
            scan_id=scan_id, year=year, status="ok", variant=variant, score=float(res.score),
            inliers=int(res.inliers), reproj_err=float(res.reproj_err),
            rotated180=bool(res.rotated180),
        )
        if not res.ok or res.warped is None:
            base.status = "failed"
            base.reason = res.reason or "нет выровненного изображения"
            base.seconds = time.perf_counter() - t0
            return base
        if variant not in box_layouts:
            raise KeyError(f"нет макета боксов для варианта {variant}")
        layout = box_layouts[variant]
        placements = locate_boxes(res.warped, layout)
        rows = []
        for name in SUBFIELD_NAMES:
            pl = placements[name]
            out = crop_path(Path(crops_dir), name, scan_id)
            if subfields is None or name in subfields:
                crop = crop_box(res.warped, pl.box)
                ensure_dir(out.parent)
                write_png(out, crop)
            rows.append((
                scan_id, variant, name, rel_path(out), str(pl.box.width), str(pl.box.height),
                "True", f"{res.score:.4f}", str(res.inliers), str(res.rotated180), pl.source,
            ))
        base.rows = tuple(rows)
        base.seconds = time.perf_counter() - t0
        return base
    except Exception as exc:
        # Сбой одного скана не роняет прогон; в сообщение — только тип и scan_id.
        return ScanResult(
            scan_id=scan_id, year=year, status="error", variant=variant,
            reason=f"{type(exc).__name__}: {str(exc)[:160]}",
            seconds=time.perf_counter() - t0,
        )


# --- build ---------------------------------------------------------------------------------


def load_jobs(
    manifest: Path = MANIFEST, assignment: Path = ASSIGNMENT_CSV
) -> list[tuple[str, str, str]]:
    """``(scan_id, year, variant)`` для всех сканов манифеста; вариант — из T07 или пусто."""
    assigned = {r["scan_id"]: r.get("variant", "") for r in read_csv_rows(assignment)}
    return [
        (r["scan_id"], r.get("year", ""), assigned.get(r["scan_id"], ""))
        for r in read_csv_rows(manifest)
    ]


def is_done(scan_id: str, index: dict[str, list[dict[str, str]]], crops_dir: Path) -> bool:
    """Скан готов: в индексе все 17 подполей и все файлы кропов на месте."""
    rows = index.get(scan_id)
    if not rows or {r["subfield"] for r in rows} != set(SUBFIELD_NAMES):
        return False
    return all(crop_path(crops_dir, name, scan_id).exists() for name in SUBFIELD_NAMES)


def group_index(rows: Iterable[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    out: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        out.setdefault(row["scan_id"], []).append(row)
    return out


def merge_results(
    index: dict[str, list[dict[str, str]]],
    failures: dict[str, dict[str, str]],
    results: Iterable[ScanResult],
) -> None:
    """Слить итоги прогона в индекс и отказы (на месте). Пропуски и ошибки не трогают их."""
    for r in results:
        if r.status == "ok":
            index[r.scan_id] = [dict(zip(INDEX_COLUMNS, row, strict=True)) for row in r.rows]
            failures.pop(r.scan_id, None)
        elif r.status == "failed":
            index.pop(r.scan_id, None)
            failures[r.scan_id] = {
                "scan_id": r.scan_id, "year": r.year, "variant": r.variant,
                "score": f"{r.score:.4f}", "inliers": str(r.inliers),
                "reproj_err": f"{r.reproj_err:.3f}", "rotated180": str(r.rotated180),
                "reason": r.reason,
            }


def remove_stale_crops(scan_ids: Iterable[str], crops_dir: Path) -> int:
    """Удалить кропы сканов, которые теперь в отказах (остались от прошлых прогонов)."""
    removed = 0
    for sid in scan_ids:
        for name in SUBFIELD_NAMES:
            path = crop_path(crops_dir, name, sid)
            if path.exists():
                path.unlink()
                removed += 1
    return removed


def build(
    *,
    workers: int = 4,
    force: bool = False,
    variant: str | None = None,
    subfields: tuple[str, ...] | None = None,
    limit: int | None = None,
    manifest: Path = MANIFEST,
    assignment: Path = ASSIGNMENT_CSV,
    pages_dir: Path = PAGES_DIR,
    crops_dir: Path = CROPS_DIR,
    layouts_dir: Path = LAYOUTS_DIR,
    boxes_dir: Path | None = None,
    index_csv: Path = INDEX_CSV,
    failures_csv: Path = FAILURES_CSV,
) -> list[ScanResult]:
    """Нарезать кропы по манифесту; индекс и отказы обновляются, готовое пропускается.

    ``variant`` — обработать только сканы, назначенные этому варианту; ``subfields`` —
    переписать только эти кропы (остальные файлы не трогаются, индекс обновляется целиком).
    """
    jobs = load_jobs(manifest, assignment)
    if variant:
        jobs = [j for j in jobs if j[2] == variant]
    if limit is not None:
        jobs = jobs[:limit]
    index = group_index(read_csv_rows(index_csv))
    failures = {r["scan_id"]: r for r in read_csv_rows(failures_csv)}

    results: list[ScanResult] = []
    todo = []
    for sid, year, var in jobs:
        if not force and (is_done(sid, index, crops_dir) or sid in failures):
            results.append(ScanResult(scan_id=sid, year=year, status="skipped", variant=var))
        else:
            todo.append((sid, year, var))

    args = (str(pages_dir), str(crops_dir), str(layouts_dir),
            str(boxes_dir) if boxes_dir else None, subfields)
    if workers > 1 and len(todo) > 1:
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(process_scan, sid, year, var, *args) for sid, year, var in todo]
            done = 0
            for fut in futures:
                results.append(fut.result())
                done += 1
                if done % 100 == 0:
                    print(f"  обработано {done} из {len(todo)}", flush=True)
    else:
        results.extend(process_scan(sid, year, var, *args) for sid, year, var in todo)

    merge_results(index, failures, results)
    remove_stale_crops(failures, crops_dir)
    write_csv_rows(
        index_csv, INDEX_COLUMNS,
        ([row[c] for c in INDEX_COLUMNS] for sid in sorted(index) for row in index[sid]),
    )
    write_csv_rows(
        failures_csv, FAILURE_COLUMNS,
        ([failures[sid].get(c, "") for c in FAILURE_COLUMNS] for sid in sorted(failures)),
    )
    return results


# --- листы контроля ------------------------------------------------------------------------


def sample_for_sheets(
    index_rows: Sequence[dict[str, str]],
    *,
    n: int = SAMPLE_PER_SUBFIELD,
    seed: int = SAMPLE_SEED,
    subfields: Sequence[str] = SUBFIELD_NAMES,
) -> dict[str, list[str]]:
    """Случайные ``scan_id`` по подполям: один ГСЧ на все подполя в каноническом порядке.

    Выборки разных подполей различаются (больше сканов под контролем), но всё
    воспроизводимо от ``seed``. Сканы перед выборкой сортируются.
    """
    rng = random.Random(seed)
    by_field: dict[str, list[str]] = {name: [] for name in subfields}
    for row in index_rows:
        if row["subfield"] in by_field and row.get("align_ok", "True") == "True":
            by_field[row["subfield"]].append(row["scan_id"])
    out: dict[str, list[str]] = {}
    for name in subfields:
        pool = sorted(set(by_field[name]))
        out[name] = rng.sample(pool, min(n, len(pool)))
    return out


def sheet_paths(subfield: str, n_items: int, out_dir: Path = CROP_SHEETS_DIR) -> list[Path]:
    count = max(1, -(-n_items // MAX_CROPS))
    return [out_dir / f"{subfield}_{k + 1}.png" for k in range(count)]


def build_sheets(
    *,
    n: int = SAMPLE_PER_SUBFIELD,
    seed: int = SAMPLE_SEED,
    subfields: Sequence[str] = SUBFIELD_NAMES,
    index_csv: Path = INDEX_CSV,
    crops_dir: Path = CROPS_DIR,
    out_dir: Path = CROP_SHEETS_DIR,
) -> list[Path]:
    """Листы по 100 кропов на подполе; подпись — ``scan_id``, рядом CSV индексов."""
    samples = sample_for_sheets(read_csv_rows(index_csv), n=n, seed=seed, subfields=subfields)
    written: list[Path] = []
    for name in subfields:
        ids = samples[name]
        for k, path in enumerate(sheet_paths(name, len(ids), out_dir)):
            chunk = ids[k * MAX_CROPS : (k + 1) * MAX_CROPS]
            if not chunk:
                continue
            images = [read_gray(crop_path(crops_dir, name, sid)) for sid in chunk]
            make_sheet(images, chunk, SHEET_COLS, SHEET_CELL_W, path, font_size=13)
            written.append(path)
    return written


# --- отчёт ---------------------------------------------------------------------------------


def _pct(part: int, total: int) -> str:
    return f"{100.0 * part / total:.1f} %" if total else "—"


def render_report(
    jobs: Sequence[tuple[str, str, str]],
    index_rows: Sequence[dict[str, str]],
    failures: Sequence[dict[str, str]],
    qc_rows: Sequence[dict[str, str]],
    *,
    sample_n: int = SAMPLE_PER_SUBFIELD,
) -> str:
    """Отчёт из агрегатов: отказы выравнивания, видимость по подполям, типы проблем."""
    total = len(jobs)
    fail_ids = {r["scan_id"] for r in failures}
    indexed = {r["scan_id"] for r in index_rows}
    missing = [sid for sid, _, _ in jobs if sid not in indexed and sid not in fail_ids]
    variant_of = {r["scan_id"]: r["variant"] for r in index_rows}
    variant_of.update({r["scan_id"]: r["variant"] for r in failures})
    year_of = {sid: year for sid, year, _ in jobs}

    lines = ["# Нарезка кропов и контроль качества (T10)", ""]
    lines += ["## Выравнивание", ""]
    lines.append(f"- сканов в манифесте: {total}")
    lines.append(f"- выровнено и нарезано: {len(indexed)} ({_pct(len(indexed), total)})")
    lines.append(f"- неудачных выравниваний: {len(fail_ids)} ({_pct(len(fail_ids), total)})")
    lines.append(f"- не обработано (ошибка чтения или кода): {len(missing)}")
    for sid in missing:
        lines.append(f"  - {sid}")
    lines.append(f"- кропов: {len(index_rows)}")
    lines.append("")

    for title, key in (("варианту", "variant"), ("году", "year")):
        lines += [f"### Отказы по {title}", "", "| | сканов | отказов | доля |",
                  "|---|---|---|---|"]
        counts: Counter[str] = Counter()
        fails: Counter[str] = Counter()
        for sid, _, _ in jobs:
            value = variant_of.get(sid, "—") if key == "variant" else year_of.get(sid, "—")
            counts[value] += 1
            if sid in fail_ids:
                fails[value] += 1
        for value in sorted(counts):
            lines.append(
                f"| {value} | {counts[value]} | {fails[value]} | "
                f"{_pct(fails[value], counts[value])} |"
            )
        lines.append("")
    if failures:
        lines += ["### Отказы поимённо", ""]
        for r in failures:
            lines.append(
                f"- {r['scan_id']} ({r['variant']}): score {r['score']}, "
                f"инлайеров {r['inliers']} — {r['reason']}"
            )
        lines.append("")

    sources: Counter[tuple[str, str]] = Counter((r["subfield"], r["source"]) for r in index_rows)
    all_sources = sorted({s for _, s in sources})
    lines += ["## Способ подгонки боксов (все кропы)", ""]
    lines.append("| подполе | " + " | ".join(all_sources) + " |")
    lines.append("|---|" + "---|" * len(all_sources))
    for name in SUBFIELD_NAMES:
        cells = " | ".join(str(sources[(name, s)]) for s in all_sources)
        lines.append(f"| {name} | {cells} |")
    lines.append("")

    need = int(np.ceil(VISIBLE_TARGET * sample_n))
    bad: dict[str, set[str]] = {name: set() for name in SUBFIELD_NAMES}
    problems: Counter[str] = Counter()
    by_field_problem: Counter[tuple[str, str]] = Counter()
    for r in qc_rows:
        if r["subfield"] in bad:
            bad[r["subfield"]].add(r["scan_id"])
            problems[r["problem"]] += 1
            by_field_problem[(r["subfield"], r["problem"])] += 1
    lines += [
        f"## Видимость значения (листы, {sample_n} случайных кропов на подполе, seed "
        f"{SAMPLE_SEED})", "",
        f"Цель — не меньше {need} из {sample_n} ({VISIBLE_TARGET:.0%}).", "",
        "| подполе | видно целиком | плохих | цель |", "|---|---|---|---|",
    ]
    failed_fields = []
    for name in SUBFIELD_NAMES:
        visible = sample_n - len(bad[name])
        ok = visible >= need
        if not ok:
            failed_fields.append(name)
        lines.append(f"| {name} | {visible} | {len(bad[name])} | {'да' if ok else 'нет'} |")
    lines.append("")
    lines += ["## Типы проблем", ""]
    if problems:
        for code, cnt in problems.most_common():
            lines.append(f"- {code} ({PROBLEMS.get(code, code)}): {cnt}")
    else:
        lines.append("- проблем не найдено")
    lines.append("")

    share = len(fail_ids) / total if total else 0.0
    lines += ["## Решение по T11", ""]
    verdict = "нужна" if share > T11_FAIL_SHARE else "не нужна"
    lines.append(
        f"- доля неудачных выравниваний {share:.1%} при пороге {T11_FAIL_SHARE:.0%} — "
        f"T11 по этому критерию {verdict}."
    )
    if failed_fields:
        lines.append(f"- подполя ниже цели видимости: {', '.join(failed_fields)}.")
    else:
        lines.append("- цель видимости выполнена по всем 17 подполям.")
    lines.append("")
    return "\n".join(lines)


def write_report(
    *,
    index_csv: Path = INDEX_CSV,
    failures_csv: Path = FAILURES_CSV,
    qc_csv: Path = QC_CSV,
    notes_md: Path = QC_NOTES_MD,
    out_path: Path = REPORT_MD,
) -> Path:
    text = render_report(
        load_jobs(), read_csv_rows(index_csv), read_csv_rows(failures_csv),
        read_csv_rows(qc_csv),
    )
    if notes_md.exists():
        text += "\n" + notes_md.read_text(encoding="utf-8").strip() + "\n"
    ensure_dir(out_path.parent)
    out_path.write_text(text, encoding="utf-8")
    return out_path


# --- CLI -----------------------------------------------------------------------------------


def _summary(results: Sequence[ScanResult]) -> str:
    counts = Counter(r.status for r in results)
    times = sorted(r.seconds for r in results if r.status in ("ok", "failed"))
    parts = [f"{k} {counts[k]}" for k in ("ok", "skipped", "failed", "error")]
    if times:
        parts.append(f"с/скан: среднее {sum(times) / len(times):.2f}, "
                     f"p95 {times[int(0.95 * (len(times) - 1))]:.2f}")
    return ", ".join(parts)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T10: нарезка кропов подполей и контроль")
    sub = parser.add_subparsers(dest="command", required=True)
    p_build = sub.add_parser("build", help="выровнять сканы и нарезать кропы")
    p_build.add_argument("--workers", type=int, default=4)
    p_build.add_argument("--force", action="store_true", help="резать заново готовые сканы")
    p_build.add_argument("--variant", default=None, help="только сканы этого варианта")
    p_build.add_argument("--subfields", default=None, help="переписать только эти кропы")
    p_build.add_argument("--limit", type=int, default=None)
    p_sheets = sub.add_parser("sheets", help="листы контроля по подполям")
    p_sheets.add_argument("--seed", type=int, default=SAMPLE_SEED)
    p_sheets.add_argument("--n", type=int, default=SAMPLE_PER_SUBFIELD)
    sub.add_parser("report", help="отчёт crops_report.md")
    args = parser.parse_args(argv)

    if args.command == "build":
        subfields = tuple(args.subfields.split(",")) if args.subfields else None
        if subfields:
            unknown = set(subfields).difference(SUBFIELD_NAMES)
            if unknown:
                parser.error(f"неизвестные подполя: {', '.join(sorted(unknown))}")
        results = build(workers=args.workers, force=args.force, variant=args.variant,
                        subfields=subfields, limit=args.limit)
        print(f"build: {_summary(results)}")
        for r in results:
            if r.status in ("failed", "error"):
                print(f"  {r.status}: {r.scan_id} ({r.variant}) — {r.reason}")
        print(f"индекс -> {INDEX_CSV}; отказы -> {FAILURES_CSV}")
        return 0
    if args.command == "sheets":
        written = build_sheets(n=args.n, seed=args.seed)
        print(f"листов: {len(written)} -> {CROP_SHEETS_DIR}")
        return 0
    if args.command == "report":
        print(f"отчёт -> {write_report()}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
