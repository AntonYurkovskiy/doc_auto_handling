"""Обезличенный пакет для облачных сессий (вариант А из `cloud_alternative.md`).

В пакет попадают только кропы цифровых подполей и номера, метки, сплит и веса моделей.
Названий файлов заявок, путей к сканам и свободного текста из заявок в нём нет.
Структура повторяет `data/ocr`, поэтому в облаке достаточно `OCR_WORK_DIR=<пакет>` и
`OCR_CACHE_DIR=<пакет>/cache`: код `ocr_lab` читает те же файлы по тем же путям.

    python -m ocr_lab.cloud_bundle build --out data/ocr/cloud_bundle
    python -m ocr_lab.cloud_bundle verify --bundle data/ocr/cloud_bundle
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import shutil
import sys
from collections.abc import Iterable, Sequence
from datetime import date
from pathlib import Path

from ocr_lab import paths

#: Колонки манифеста, которые идут в пакет. Всё остальное отбрасывается: список разрешённых,
#: а не запрещённых, чтобы новая колонка (например, судно) не просочилась сама.
MANIFEST_COLUMNS: tuple[str, ...] = (
    "scan_id",
    "tug_code",
    "year",
    "voucher_number",
    "voucher_file",
    "work_type",
    "app_dt",
    "truth_corrected",
    "dedup_conflict",
    "chain_ok",
    "chain_violation",
    "crosses_midnight",
    "minutes_mult10",
    "has_hour24",
    "pair_id",
    "split",
    *(
        f"{row}_{part}"
        for row in ("left_base", "arrived_base", "started_work", "finished_work")
        for part in ("dt", "day", "month", "hour", "minute", "year")
    ),
)

#: Колонки, которых в манифесте пакета быть не должно ни при каких условиях.
FORBIDDEN_COLUMNS: frozenset[str] = frozenset(
    {
        "application_file",
        "scan_path",
        "scan_exists",
        "ext",
        "work_type_raw",
        "agent_group",
        "vessel",
        "agent",
        "email",
        "path",
    }
)

#: Допустимые значения нормализованного вида работ (проверка на утечку свободного текста).
WORK_TYPES: frozenset[str] = frozenset(
    {
        "",
        "отшвартовка",
        "швартовка",
        "перестановка",
        "обслуживание морских сооружений",
        "сопровождение",
        "околка льда",
        "обслуживание судна",
        "буксировка",
    }
)

MODEL_FILES = (
    "best.pt",
    "config.json",
    "history.csv",
    "train.log",
    "val_predictions.jsonl",
    "test_predictions.jsonl",
)
MNIST_FILES = (
    "train-images-idx3-ubyte.gz",
    "train-labels-idx1-ubyte.gz",
    "t10k-images-idx3-ubyte.gz",
    "t10k-labels-idx1-ubyte.gz",
)


def _read_csv(path: Path) -> tuple[list[str], list[dict[str, str]]]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        reader = csv.DictReader(fh)
        return list(reader.fieldnames or []), list(reader)


def _write_csv(path: Path, fields: Sequence[str], rows: Iterable[dict[str, str]]) -> int:
    path.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(fields), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
            count += 1
    return count


def anonymize_manifest(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    """Оставить в строках манифеста только разрешённые колонки."""
    return [{col: row.get(col, "") for col in MANIFEST_COLUMNS} for row in rows]


def check_manifest(fields: Sequence[str], rows: Sequence[dict[str, str]]) -> list[str]:
    """Вернуть список нарушений обезличенности манифеста (пустой — всё в порядке)."""
    problems: list[str] = []
    bad = sorted(set(fields) & FORBIDDEN_COLUMNS)
    if bad:
        problems.append(f"запрещённые колонки: {', '.join(bad)}")
    extra = sorted(set(fields) - set(MANIFEST_COLUMNS))
    if extra:
        problems.append(f"колонки вне списка разрешённых: {', '.join(extra)}")
    unknown = sorted({r.get("work_type", "") for r in rows} - WORK_TYPES)
    if unknown:
        problems.append(f"неожиданные значения work_type: {unknown}")
    for row in rows:
        name = row.get("voucher_file", "")
        stem = name.rsplit(".", 1)[0]
        if len(stem) > 12 or "\\" in name or "/" in name:
            problems.append(f"подозрительное voucher_file: {name!r}")
            break
    return problems


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def build(out: Path, *, work: Path = paths.WORK_DIR, mnist: Path | None = None) -> dict[str, int]:
    """Собрать пакет в `out` из производных данных `work` и вернуть сводку."""
    if out.exists() and any(out.iterdir()):
        raise SystemExit(f"{out} не пуст: удалите или выберите другой каталог")
    out.mkdir(parents=True, exist_ok=True)
    stats: dict[str, int] = {}

    fields, rows = _read_csv(work / "manifest.csv")
    anon = anonymize_manifest(rows)
    problems = check_manifest(MANIFEST_COLUMNS, anon)
    if problems:
        raise SystemExit("манифест не прошёл проверку: " + "; ".join(problems))
    stats["manifest_rows"] = _write_csv(out / "manifest.csv", MANIFEST_COLUMNS, anon)

    idx_fields, idx_rows = _read_csv(work / "crops_index.csv")
    stats["crops_index_rows"] = _write_csv(out / "crops_index.csv", idx_fields, idx_rows)
    copied = 0
    for row in idx_rows:
        src = work / row["path"]
        if not src.is_file():
            continue
        dst = out / row["path"]
        dst.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src, dst)
        copied += 1
    stats["crops_copied"] = copied

    for rel in ("labels/printed_flags.csv", "review/crops_qc.csv", "review/crops_blank.csv"):
        src = work / rel
        if src.is_file():
            (out / rel).parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, out / rel)

    models_src = work / "models" / "digits_v0"
    for name in MODEL_FILES:
        src = models_src / name
        if src.is_file():
            dst = out / "models" / "digits_v0" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    mnist_raw = (mnist or paths.CACHE_DIR / "mnist") / "MNIST" / "raw"
    for name in MNIST_FILES:
        src = mnist_raw / name
        if src.is_file():
            dst = out / "cache" / "mnist" / "MNIST" / "raw" / name
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)

    manifest_hash = _sha256(out / "manifest.csv")
    (out / "README.md").write_text(
        "# Обезличенный пакет OCR для облачных сессий\n\n"
        f"Собран: {date.today().isoformat()}. Хэш `manifest.csv` (sha256): `{manifest_hash}`.\n\n"
        "Внутри: кропы цифровых подполей и номера (`crops/`), `crops_index.csv`, `manifest.csv`\n"
        "без названий файлов заявок и путей, метки печатного (`labels/`), разметка дефектных\n"
        "кропов (`review/`), веса и предсказания `digits_v0` (`models/`), кэш MNIST (`cache/`).\n\n"
        "Использование: `export OCR_WORK_DIR=<пакет> OCR_CACHE_DIR=<пакет>/cache`.\n"
        "Правила: не пересылать, не публиковать, удалить после работ. Полных сканов здесь нет.\n",
        encoding="utf-8",
    )
    return stats


def verify(bundle: Path) -> list[str]:
    """Проверить готовый пакет: колонки манифеста и наличие файлов, на которые ссылается индекс."""
    problems: list[str] = []
    fields, rows = _read_csv(bundle / "manifest.csv")
    problems += check_manifest(fields, rows)
    _, idx_rows = _read_csv(bundle / "crops_index.csv")
    ok_rows = [r for r in idx_rows if r.get("align_ok") == "True"]
    missing = [r["path"] for r in ok_rows if not (bundle / r["path"]).is_file()]
    if missing:
        problems.append(f"нет {len(missing)} кропов из индекса, например {missing[0]}")
    known = {r["scan_id"] for r in rows}
    orphan = {r["scan_id"] for r in idx_rows} - known
    if orphan:
        problems.append(f"в индексе {len(orphan)} сканов без строки манифеста")
    return problems


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_build = sub.add_parser("build", help="собрать пакет")
    p_build.add_argument("--out", type=Path, required=True)
    p_build.add_argument("--work", type=Path, default=paths.WORK_DIR)
    p_verify = sub.add_parser("verify", help="проверить пакет")
    p_verify.add_argument("--bundle", type=Path, required=True)
    args = parser.parse_args(argv)

    if args.cmd == "build":
        stats = build(args.out, work=args.work)
        for key, value in stats.items():
            print(f"{key}: {value}")
        problems = verify(args.out)
    else:
        problems = verify(args.bundle)
    for problem in problems:
        print("НАРУШЕНИЕ:", problem, file=sys.stderr)
    print("проверка:", "не пройдена" if problems else "пройдена")
    return 1 if problems else 0


if __name__ == "__main__":
    raise SystemExit(main())
