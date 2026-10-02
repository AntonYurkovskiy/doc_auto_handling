"""Визуальная сверка истины (T04): выборка ваучеров и склейки «номер + 4 строки дат».

Шаги:
1. `refs` — для каждого буксира эталон бланка: страницы кэша выравниваются по одной
   опорной странице, затем берётся попиксельная медиана (рукопись уходит, остаётся
   форма). Эталоны пишутся в `data/ocr/review/truth_check_refs/<tug>.png`.
2. `select` — выборка: все `chain_ok=False`, все ваучеры с поправками истины
   (перепроверка), все с `24:00`/`00:00`, все с минутами не кратными 10 (вопрос Q1),
   50 случайных (25 Коммунар + 25 Пионер, расслоение по году-месяцу, seed 42),
   10 случайных с переходом через полночь.
   Пишется `data/ocr/review/truth_check_selection.csv`.
3. `strips` — каждый скан выравнивается по эталону своего буксира, из выровненной
   страницы вырезаются только номер ваучера и четыре строки дат (решение Q5) и
   склеиваются в `data/ocr/review/truth_check/<scan_id>.png` с подписью-истиной сверху.
   Заготовка CSV сверки — `data/ocr/review/truth_check_template.csv`.

Запуск: `python -m ocr_lab.truth_check {refs,select,strips,all} [--max-minutes 9]`.
"""

from __future__ import annotations

import argparse
import csv
import random
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from app.ocr.align import AlignParams, Reference, align, build_reference
from ocr_lab.paths import MANIFEST, PAGES_DIR, REVIEW_DIR, ensure_dir
from ocr_lab.sheets import make_strip

ROWS = ("left_base", "arrived_base", "started_work", "finished_work")
FIELDS = ("voucher_number", *ROWS)
ROW_LABELS = {
    "left_base": "Выход",
    "arrived_base": "Приход",
    "started_work": "Начало",
    "finished_work": "Окончание",
}

SEED = 42
RANDOM_PER_TUG = 25
MIDNIGHT_N = 10

#: Опорные страницы, по которым строится эталон (подобраны в T04 по медиане выравнивания).
REF_SEED_SCANS = {"k": "2025_283k", "p": "2025_3p"}
REF_N_PAGES = 40

#: Боксы в долях эталона (x0, y0, x1, y1). Подобраны по медианным эталонам T04 с запасом:
#: номер — вместе с печатным «VOUCHER №», строки — от «Date» до «Min.».
BOXES: dict[str, dict[str, tuple[float, float, float, float]]] = {
    "k": {
        "voucher_number": (0.30, 0.255, 0.66, 0.300),
        "left_base": (0.02, 0.478, 0.98, 0.516),
        "arrived_base": (0.02, 0.517, 0.98, 0.555),
        "started_work": (0.02, 0.555, 0.98, 0.593),
        "finished_work": (0.02, 0.593, 0.98, 0.632),
    },
    "p": {
        "voucher_number": (0.30, 0.255, 0.70, 0.300),
        "left_base": (0.06, 0.510, 0.98, 0.556),
        "arrived_base": (0.06, 0.563, 0.98, 0.609),
        "started_work": (0.06, 0.615, 0.98, 0.661),
        "finished_work": (0.06, 0.668, 0.98, 0.714),
    },
}

STRIP_WIDTH = 1600
ALIGN_PARAMS = AlignParams(try_rotate180=False, min_score=0.3)

REFS_DIR = REVIEW_DIR / "truth_check_refs"
STRIPS_DIR = REVIEW_DIR / "truth_check"
SELECTION_CSV = REVIEW_DIR / "truth_check_selection.csv"
TEMPLATE_CSV = REVIEW_DIR / "truth_check_template.csv"
ALIGN_LOG_CSV = REVIEW_DIR / "truth_check_align.csv"


def _read_gray(path: Path) -> np.ndarray:
    """Чтение PNG по пути с кириллицей (cv2.imread на Windows такие пути не читает)."""
    data = np.fromfile(str(path), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"не читается страница {path.stem}")
    return img


def _write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"не кодируется PNG {path.stem}")
    ensure_dir(path.parent)
    path.write_bytes(buf.tobytes())


def load_manifest(path: Path = MANIFEST) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


# --- Эталоны --------------------------------------------------------------------------------


def build_refs(manifest: pd.DataFrame, n_pages: int = REF_N_PAGES) -> dict[str, Path]:
    """Медианный эталон каждого буксира по `n_pages` страницам, выровненным по опорной."""
    out: dict[str, Path] = {}
    for tug, seed_scan in REF_SEED_SCANS.items():
        path = REFS_DIR / f"{tug}.png"
        out[tug] = path
        if path.exists():
            continue
        seed_img = _read_gray(PAGES_DIR / f"{seed_scan}.png")
        ref = build_reference(seed_img, name=seed_scan, params=ALIGN_PARAMS)
        ids = sorted(manifest.loc[manifest["tug_code"] == tug, "scan_id"])
        random.Random(1).shuffle(ids)
        warped = [seed_img]
        for scan_id in ids:
            if len(warped) >= n_pages:
                break
            if scan_id == seed_scan:
                continue
            res = align(_read_gray(PAGES_DIR / f"{scan_id}.png"), ref)
            if res.ok and res.warped is not None:
                warped.append(res.warped)
        median = np.median(np.stack(warped), axis=0).astype(np.uint8)
        _write_png(path, median)
        print(f"эталон {tug}: {len(warped)} страниц")
    return out


def load_refs() -> dict[str, Reference]:
    return {
        tug: build_reference(_read_gray(REFS_DIR / f"{tug}.png"), name=tug, params=ALIGN_PARAMS)
        for tug in REF_SEED_SCANS
    }


# --- Выборка --------------------------------------------------------------------------------


def _stratified(df: pd.DataFrame, n: int, rng: random.Random) -> list[str]:
    """`n` случайных scan_id с расслоением по (год, месяц) `left_base`: по кругу по слоям."""
    strata: dict[str, list[str]] = defaultdict(list)
    for row in df.itertuples():
        key = f"{row.left_base_year}-{int(row.left_base_month or 0):02d}"
        strata[key].append(str(row.scan_id))
    keys = sorted(strata)
    for key in keys:
        strata[key].sort()
        rng.shuffle(strata[key])
    rng.shuffle(keys)
    picked: list[str] = []
    while len(picked) < n and any(strata[k] for k in keys):
        for key in keys:
            if strata[key] and len(picked) < n:
                picked.append(strata[key].pop())
    return picked


def select(manifest: pd.DataFrame, seed: int = SEED) -> dict[str, list[str]]:
    """Выборка T04: `scan_id -> [причины]`, порядок — порядок добавления."""
    reasons: dict[str, list[str]] = {}

    def add(scan_id: str, reason: str) -> None:
        reasons.setdefault(scan_id, []).append(reason)

    for scan_id in manifest.loc[manifest["chain_ok"] != "True", "scan_id"]:
        add(scan_id, "chain_violation")
    for scan_id in manifest.loc[manifest["truth_corrected"] != "", "scan_id"]:
        add(scan_id, "truth_corrected")
    midnight_flags = manifest["has_hour24"] == "True"
    for row in ROWS:
        hour0 = manifest[f"{row}_hour"].isin(["0", "00"])
        midnight_flags |= hour0 & manifest[f"{row}_minute"].isin(["0", "00"])
    for scan_id in manifest.loc[midnight_flags, "scan_id"]:
        add(scan_id, "hour24_or_0000")

    rng = random.Random(seed)
    pool = manifest[~manifest["scan_id"].isin(list(reasons))]
    for tug in ("k", "p"):
        for scan_id in _stratified(pool[pool["tug_code"] == tug], RANDOM_PER_TUG, rng):
            add(scan_id, f"random_{tug}")

    cross = manifest[
        (manifest["crosses_midnight"] == "True") & ~manifest["scan_id"].isin(list(reasons))
    ]
    cross_ids = sorted(cross["scan_id"])
    for scan_id in rng.sample(cross_ids, min(MIDNIGHT_N, len(cross_ids))):
        add(scan_id, "crosses_midnight")
    # Минуты не кратны 10 в истине — проверка вопроса «пишут кратно 10 или округляют».
    # Добавляются после случайных выборок, чтобы не менять их состав.
    for scan_id in manifest.loc[manifest["minutes_mult10"] == "False", "scan_id"]:
        add(scan_id, "minutes_not_mult10")
    return reasons


def write_selection(reasons: dict[str, list[str]], path: Path = SELECTION_CSV) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scan_id", "reasons"])
        for scan_id, rs in reasons.items():
            writer.writerow([scan_id, ";".join(rs)])


def read_selection(path: Path = SELECTION_CSV) -> dict[str, list[str]]:
    with path.open("r", encoding="utf-8", newline="") as fh:
        return {row["scan_id"]: row["reasons"].split(";") for row in csv.DictReader(fh)}


# --- Истина в подписи ----------------------------------------------------------------------


def truth_value(rec: pd.Series, field: str) -> str:
    """Истина подполя в формате сверки: номер или `ДД.ММ ЧЧ:ММ` (как на бланке, 24:00)."""
    if field == "voucher_number":
        return f"{rec['voucher_number']}{rec['tug_code']}"
    day, month = rec[f"{field}_day"], rec[f"{field}_month"]
    hour, minute = rec[f"{field}_hour"], rec[f"{field}_minute"]
    if not day:
        return ""
    text = f"{int(day):02d}.{int(month):02d}"
    if hour != "" and minute != "":
        text += f" {int(hour):02d}:{int(minute):02d}"
    return text


def caption(rec: pd.Series, reasons: list[str], corrections: dict[str, str]) -> str:
    parts = [f"{rec['scan_id']} | № {truth_value(rec, 'voucher_number')}"]
    for row in ROWS:
        parts.append(f"{ROW_LABELS[row]} {truth_value(rec, row) or '—'}")
    text = " | ".join(parts)
    extra = [f"[{', '.join(reasons)}]"]
    if rec["chain_violation"]:
        extra.append(f"цепочка: {rec['chain_violation']}")
    for field, export_value in corrections.items():
        extra.append(f"поправка {ROW_LABELS.get(field, field)}: в выгрузке {export_value}")
    return text + "\n" + "; ".join(extra)


def _corrections_by_scan() -> dict[str, dict[str, str]]:
    from ocr_lab.paths import CORRECTIONS_CSV

    out: dict[str, dict[str, str]] = defaultdict(dict)
    if CORRECTIONS_CSV.exists():
        with CORRECTIONS_CSV.open("r", encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                if row.get("status") == "confirmed":
                    out[row["scan_id"]][row["field"]] = row["export_value"][:16]
    return out


# --- Склейки -------------------------------------------------------------------------------


def crop_box(img: np.ndarray, box: tuple[float, float, float, float]) -> np.ndarray:
    h, w = img.shape[:2]
    x0, y0, x1, y1 = box
    return img[int(y0 * h) : int(y1 * h), int(x0 * w) : int(x1 * w)]


@dataclass
class StripStat:
    scan_id: str
    ok: bool
    inliers: int
    score: float
    reason: str
    form: str = ""  # буксир, по эталону которого выровнен скан


def build_strip(
    rec: pd.Series,
    reasons: list[str],
    refs: dict[str, Reference],
    corrections: dict[str, str],
) -> StripStat:
    """Склейка одного скана. Сначала эталон своего буксира; если выравнивание не прошло,
    пробуются эталоны других буксиров (бланк мог быть взят у соседнего буксира)."""
    scan_id = str(rec["scan_id"])
    page = _read_gray(PAGES_DIR / f"{scan_id}.png")
    own = str(rec["tug_code"])
    order = [own, *(tug for tug in refs if tug != own)]
    tried = []
    for tug in order:
        res = align(page, refs[tug])
        tried.append((tug, res))
        if res.ok:
            break
    tug, res = next(((t, r) for t, r in tried if r.ok), tried[0])
    if res.warped is None:
        return StripStat(scan_id, False, res.inliers, res.score, res.reason or "нет H", tug)
    crops = [crop_box(res.warped, BOXES[tug][field]) for field in FIELDS]
    text = caption(rec, reasons, corrections)
    if tug != own:
        text += f"; бланк буксира {tug}"
    strip = make_strip(crops, text, width=STRIP_WIDTH)
    ensure_dir(STRIPS_DIR)
    strip.save(STRIPS_DIR / f"{scan_id}.png")
    return StripStat(scan_id, res.ok, res.inliers, res.score, res.reason, tug)


def build_strips(
    manifest: pd.DataFrame, reasons: dict[str, list[str]], max_minutes: float | None = None
) -> list[StripStat]:
    refs = load_refs()
    corrections = _corrections_by_scan()
    by_id = manifest.set_index("scan_id", drop=False)
    done = _read_align_log()
    started = time.monotonic()
    stats: list[StripStat] = []
    for scan_id, rs in reasons.items():
        if scan_id in done and (STRIPS_DIR / f"{scan_id}.png").exists():
            continue
        if max_minutes is not None and time.monotonic() - started > max_minutes * 60:
            print("остановка по --max-minutes; перезапустите для продолжения")
            break
        rec = by_id.loc[scan_id]
        stat = build_strip(rec, rs, refs, corrections[scan_id])
        stats.append(stat)
        _append_align_log(stat)
        print(f"{scan_id}: ok={stat.ok} inliers={stat.inliers} score={stat.score:.2f}")
    return stats


def _read_align_log() -> set[str]:
    if not ALIGN_LOG_CSV.exists():
        return set()
    with ALIGN_LOG_CSV.open("r", encoding="utf-8", newline="") as fh:
        return {row["scan_id"] for row in csv.DictReader(fh)}


def _append_align_log(stat: StripStat) -> None:
    new = not ALIGN_LOG_CSV.exists()
    ensure_dir(ALIGN_LOG_CSV.parent)
    with ALIGN_LOG_CSV.open("a", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        if new:
            writer.writerow(["scan_id", "ok", "inliers", "score", "reason", "form"])
        writer.writerow(
            [stat.scan_id, stat.ok, stat.inliers, f"{stat.score:.3f}", stat.reason, stat.form]
        )


def write_template(manifest: pd.DataFrame, reasons: dict[str, list[str]]) -> None:
    """Заготовка `truth_check.csv`: истина заполнена, остальные колонки пустые."""
    by_id = manifest.set_index("scan_id", drop=False)
    ensure_dir(TEMPLATE_CSV.parent)
    with TEMPLATE_CSV.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scan_id", "field", "truth", "seen", "verdict", "printed", "note"])
        for scan_id in reasons:
            rec = by_id.loc[scan_id]
            for field in FIELDS:
                writer.writerow([scan_id, field, truth_value(rec, field), "", "", "", ""])


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T04: склейки для визуальной сверки истины")
    parser.add_argument("step", choices=["refs", "select", "strips", "all"])
    parser.add_argument("--max-minutes", type=float, default=None)
    parser.add_argument("--limit", type=int, default=None, help="только первые N из выборки")
    args = parser.parse_args(argv)

    manifest = load_manifest()
    if args.step in ("refs", "all"):
        build_refs(manifest)
    if args.step in ("select", "all"):
        reasons = select(manifest)
        write_selection(reasons)
        write_template(manifest, reasons)
        counts: dict[str, int] = defaultdict(int)
        for rs in reasons.values():
            for r in rs:
                counts[r] += 1
        print(f"выбрано {len(reasons)} ваучеров: {dict(counts)}")
    if args.step in ("strips", "all"):
        reasons = read_selection()
        if args.limit is not None:
            reasons = dict(list(reasons.items())[: args.limit])
        build_strips(manifest, reasons, args.max_minutes)
    return 0


if __name__ == "__main__":
    sys.exit(main())
