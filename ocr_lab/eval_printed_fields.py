"""T30: оценка чтения печатных строк (судно, агент, вид работ) на сканах с известной истиной.

Для каждого скана: страница из кэша `data/ocr/pages/` -> выравнивание по эталону -> кропы
строк по `app.ocr.printed_rows` -> Tesseract -> сверка со списками известных значений
(`app.services.printed_match`) -> сравнение с истиной из выгрузки. Для сравнения тот же путь
можно прогнать по прежнему «трафарету» из долей страницы (`--legacy`).

Списки для сверки строятся по выгрузке **без строки самого скана**: судно, которого в
остальной истории нет, не должно находиться (проверяет, что пустое поле тоже верный ответ).

Запуск: `.venv/Scripts/python -m ocr_lab.eval_printed_fields --split test --n 40`.

В консоль и отчёт идут только агрегаты и `scan_id`, без названий судов и агентов
(`_common.md`, п. 3).
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import sys
import time
from collections import Counter, defaultdict
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.config import GROUP_A_AGENTS, GROUP_B_AGENTS, settings
from app.ocr import pipeline as ocr_pipeline
from app.ocr.io import normalize_width
from app.ocr.printed_rows import crop_printed_rows, has_printed_layout
from app.ocr.variants import detect_variant
from app.services import printed_match as pm
from app.services.calculation import normalize_work_type
from app.services.voucher_ocr import PRINTED_UPSCALE, read_printed_crop
from app.services.voucher_template import DEFAULT_VOUCHER_REGIONS
from ocr_lab.paths import DATASET_CSV, MANIFEST, PAGES_DIR, REPORTS_DIR, ensure_dir

FIELDS = ("vessel", "agent", "work_type")
#: Пары (минимальная схожесть, минимальный отрыв от следующего) для `--sweep`.
SWEEP = (
    (0.65, 0.05),
    (0.7, 0.05),
    (0.7, 0.1),
    (0.75, 0.05),
    (0.75, 0.1),
    (0.8, 0.05),
    (0.8, 0.1),
    (0.85, 0.1),
)
KNOWN_AGENTS = frozenset({*GROUP_A_AGENTS, *GROUP_B_AGENTS})


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _services(value: str | None) -> frozenset[str]:
    """Вид работ -> множество нормализованных услуг («Отшвартовка + Сопровождение»)."""
    parts = [p.strip() for p in (value or "").split("+") if p.strip()]
    return frozenset(normalize_work_type(p) or p.lower() for p in parts)


def same_value(field: str, predicted: str | None, truth: str | None) -> bool:
    """Совпадение прочитанного значения с истиной с учётом написаний."""
    if not predicted or not truth:
        return False
    if field == "vessel":
        return pm._fold_latin(predicted) == pm._fold_latin(truth)
    if field == "agent":
        return pm._fold_cyrillic(predicted) == pm._fold_cyrillic(truth)
    return _services(predicted) == _services(truth)


def build_samples(
    truth_csv: Path, split: str, n: int, seed: int
) -> list[tuple[str, str, dict[str, str]]]:
    """Сканы выбранной выборки: (`scan_id`, код буксира, строка истины)."""
    manifest = _read_csv(MANIFEST)
    by_path: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in _read_csv(truth_csv):
        by_path[row["voucher_scan_path"]].append(row)
    samples = []
    for row in manifest:
        if row["scan_exists"] != "True" or row["tug_code"] not in ("k", "p"):
            continue
        if split == "valtest":
            if row["split"] not in ("val", "test"):
                continue
        elif split != "all" and row["split"] != split:
            continue
        rows = by_path.get(row["scan_path"], [])
        if len(rows) != 1:  # дубли в выгрузке: истина неоднозначна
            continue
        samples.append((row["scan_id"].strip(), row["tug_code"], rows[0]))
    random.Random(seed).shuffle(samples)
    return samples[:n] if n > 0 else samples


def _row_date(row: dict[str, str]) -> datetime | None:
    for key in ("work_start", "base_departure"):
        try:
            return datetime.strptime(row[key].strip(), "%d.%m.%Y %H:%M")
        except (KeyError, ValueError):
            continue
    return None


def known_counts(
    truth_csv: Path,
) -> tuple[Counter[str], Counter[str], list[tuple[datetime | None, str, str]]]:
    """Сколько строк выгрузки приходится на каждое судно и агента (для «без себя»).

    Третье значение — все строки как `(дата, судно, ключ строки)`: из них строится окно судов
    «заявки около даты ваучера», как в приложении.
    """
    vessels: Counter[str] = Counter()
    agents: Counter[str] = Counter()
    dated: list[tuple[datetime | None, str, str]] = []
    for row in _read_csv(truth_csv):
        vessel = row["vessel"].strip()
        if vessel:
            vessels[vessel] += 1
            dated.append((_row_date(row), vessel, row["row_number"]))
        if row["agent"].strip():
            agents[row["agent"].strip()] += 1
    return vessels, agents, dated


def _legacy_crops(page: np.ndarray) -> dict[str, np.ndarray]:
    """Прежний способ: прямоугольники из долей страницы на невыровненной странице."""
    height, width = page.shape
    crops: dict[str, np.ndarray] = {}
    for region in DEFAULT_VOUCHER_REGIONS:
        if region.name not in FIELDS:
            continue
        left = max(0, int((region.center_x - region.width / 2) * width))
        right = min(width, int((region.center_x + region.width / 2) * width))
        top = max(0, int((region.center_y - region.height / 2) * height))
        bottom = min(height, int((region.center_y + region.height / 2) * height))
        crops[region.name] = page[top:bottom, left:right]
    return crops


def _outcome(found: pm.PrintedMatch | None, field: str, truth_value: str, in_list: bool) -> str:
    if found is None:
        return "empty_ok" if not in_list else "miss"
    return "correct" if same_value(field, found.value, truth_value) else "wrong"


def evaluate(
    samples: list[tuple[str, str, dict[str, str]]],
    vessels: Counter[str],
    agents: Counter[str],
    dated: list[tuple[datetime | None, str, str]],
    *,
    window_days: int,
    sweep: bool = False,
    legacy: bool,
    psm: int | None,
    upscale: float,
    resources: ocr_pipeline.PipelineResources,
) -> dict[str, Any]:
    stats: dict[str, dict[str, Counter[str]]] = {}
    per_scan: list[dict[str, Any]] = []
    timing: dict[str, float] = defaultdict(float)
    for scan_id, tug_code, truth in samples:
        page_path = PAGES_DIR / f"{scan_id}.png"
        if not page_path.is_file():
            continue
        # imdecode(fromfile): cv2.imread не открывает пути с кириллицей (`2025_67к`).
        gray = cv2.imdecode(np.fromfile(page_path, dtype=np.uint8), cv2.IMREAD_GRAYSCALE)
        if gray is None:
            continue
        page = normalize_width(gray)
        t0 = time.perf_counter()
        variant: str | None = None
        if legacy:
            crops = _legacy_crops(page)
        else:
            match = detect_variant(page, resources.variants, prefer=tug_code)
            variant = match.variant
            if not match.ok or match.result.warped is None or not has_printed_layout(variant):
                per_scan.append({"scan_id": scan_id, "variant": variant, "status": "align_failed"})
                stats.setdefault(variant or "unknown", {}).setdefault("align_failed", Counter())[
                    "n"
                ] += 1
                continue
            crops = crop_printed_rows(match.result.warped, variant, FIELDS)
        timing["align_crop"] += time.perf_counter() - t0

        own_vessel = truth["vessel"].strip()
        own_agent = truth["agent"].strip()
        vessel_list = [v for v, c in vessels.items() if c - (v == own_vessel) > 0]
        if window_days:
            own_date = _row_date(truth)
            if own_date is not None:
                span = timedelta(days=window_days)
                near = {
                    v
                    for when, v, key in dated
                    if key != truth["row_number"]
                    and when is not None
                    and abs(when - own_date) <= span
                }
                vessel_list = sorted(near)
        agent_list = sorted(
            KNOWN_AGENTS | {a for a, c in agents.items() if c - (a == own_agent) > 0}
        )
        group = stats.setdefault(variant or tug_code, {})
        record: dict[str, Any] = {"scan_id": scan_id, "variant": variant, "tug": tug_code}
        for field in FIELDS:
            crop = crops.get(field)
            t0 = time.perf_counter()
            text = (
                read_printed_crop(crop, psm=psm, upscale=upscale)[0] if crop is not None else None
            )
            timing["tesseract"] += time.perf_counter() - t0
            truth_value = truth[field].strip()
            if field == "vessel":
                found = pm.match_vessel(text, vessel_list)
                in_list = truth_value in vessel_list
                if sweep:
                    for score, margin in SWEEP:
                        alt = pm.match_vessel(text, vessel_list, min_score=score, min_margin=margin)
                        sw = stats.setdefault(f"sweep {score}/{margin}", {}).setdefault(
                            "vessel", Counter()
                        )
                        sw["n"] += 1
                        sw[_outcome(alt, "vessel", truth_value, in_list)] += 1
                        sw["matched"] += alt is not None
            elif field == "agent":
                found = pm.match_agent(text, agent_list)
                in_list = truth_value in agent_list
                if sweep:
                    for score, margin in SWEEP:
                        alt = pm.match_agent(text, agent_list, min_score=score, min_margin=margin)
                        sw = stats.setdefault(f"sweep {score}/{margin}", {}).setdefault(
                            "agent", Counter()
                        )
                        sw["n"] += 1
                        sw[_outcome(alt, "agent", truth_value, in_list)] += 1
                        sw["matched"] += alt is not None
            else:
                found = pm.match_work_type(text)
                in_list = bool(truth_value)
            outcome = _outcome(found, field, truth_value, in_list)
            bucket = group.setdefault(field, Counter())
            bucket["n"] += 1
            bucket[outcome] += 1
            if found is not None:
                bucket["matched"] += 1
            record[field] = outcome
        per_scan.append(record)
    return {"stats": stats, "scans": per_scan, "timing": dict(timing)}


def _line(group: str, field: str, b: Counter[str]) -> str:
    n = b["n"]
    matched = b["matched"]
    acc = b["correct"] / matched if matched else float("nan")
    return (
        f"{group:12s} {field:9s} n={n:3d} верно={b['correct']:3d} ошибочно={b['wrong']:2d} "
        f"пусто_верно={b['empty_ok']:2d} пропуск={b['miss']:3d} | "
        f"верных среди найденных {acc:5.1%}, верно всего {(b['correct'] + b['empty_ok']) / n:5.1%}"
    )


def summarize(stats: dict[str, dict[str, Counter[str]]]) -> list[str]:
    lines = []
    total: dict[str, Counter[str]] = {f: Counter() for f in FIELDS}
    for group in sorted(stats):
        for field in FIELDS:
            bucket = stats[group].get(field)
            if bucket:
                total[field].update(bucket)
                lines.append(_line(group, field, bucket))
        if "align_failed" in stats[group]:
            lines.append(f"{group:12s} align_failed={stats[group]['align_failed']['n']}")
    for field in FIELDS:
        if total[field]:
            lines.append(_line("ВСЕГО", field, total[field]))
    return lines


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=(__doc__ or "").splitlines()[0])
    parser.add_argument(
        "--split", default="test", choices=("train", "val", "test", "valtest", "all")
    )
    parser.add_argument(
        "--window-days",
        type=int,
        default=0,
        help="судно ищем среди строк выгрузки ±N дней от ваучера (0 — вся история)",
    )
    parser.add_argument("--n", type=int, default=40, help="сколько сканов (0 — все)")
    parser.add_argument("--seed", type=int, default=29)
    parser.add_argument("--legacy", action="store_true", help="прежний трафарет из долей страницы")
    parser.add_argument("--psm", type=int, default=7, help="режим Tesseract (0 — по умолчанию)")
    parser.add_argument(
        "--sweep",
        action="store_true",
        help="перебрать пороги сверки судна и агента (на одном и том же чтении)",
    )
    parser.add_argument("--upscale", type=float, default=PRINTED_UPSCALE)
    parser.add_argument(
        "--truth", type=Path, default=DATASET_CSV, help="выгрузка истины (OCR_DATASET_CSV)"
    )
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args(argv)
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[union-attr]

    resources = ocr_pipeline.get_resources(
        models_dir=settings.ocr_models_dir,
        layouts_dir=settings.ocr_layouts_dir,
        model_version=settings.ocr_model_version,
    )
    vessels, agents, dated = known_counts(args.truth)
    samples = build_samples(args.truth, args.split, args.n, args.seed)
    result = evaluate(
        samples,
        vessels,
        agents,
        dated,
        window_days=args.window_days,
        sweep=args.sweep,
        legacy=args.legacy,
        psm=args.psm or None,
        upscale=args.upscale,
        resources=resources,
    )
    way = "прежний трафарет" if args.legacy else "выровненные строки"
    window = (
        f", судно среди ±{args.window_days} дн."
        if args.window_days
        else ", судно среди всей истории"
    )
    print(f"выборка {args.split}: {len(samples)} сканов, {way}{window}")
    for line in summarize(result["stats"]):
        print(line)
    print("время, с:", {k: round(v, 1) for k, v in result["timing"].items()})
    name = f"printed_fields_{'legacy' if args.legacy else 'aligned'}_{args.split}.json"
    out = args.out or ensure_dir(REPORTS_DIR) / name
    out.write_text(json.dumps(result, ensure_ascii=False, indent=1), encoding="utf-8")
    print("отчёт:", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
