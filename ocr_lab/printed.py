"""Разметка «напечатано / от руки» для номера, дня и месяца (T12).

Размечаем три группы подполей: ``voucher_number``, ``day`` и ``month``. Для дня и месяца
берём подполя строки ``left_base`` — внутри одного ваучера способ записи одинаков во всех
четырёх строках (проверено на 79 ваучерах T04, см. журнал).

Шаги CLI (``python -m ocr_lab.printed <шаг>``):

- ``score`` — эвристика печатности по каждому кропу группы (``printedness_score``), запись
  отсортированных по убыванию оценок в ``data/ocr/labels/printed_scores_<group>.csv``
  (``scan_id, score, empty``). Оценка — доля «сильных» градиентов кропа, направленных
  вдоль осей 0°/90°: печатные цифры бланка — прямые штрихи, рукопись — наклонные и
  изогнутые. Нужна только для сортировки листов, не для итоговой метки.
- ``sheets`` — листы по 100 кропов на группу в порядке убывания оценки (печатные сверху),
  ``data/ocr/sheets/printed/<group>_<n>.png`` + CSV «индекс → scan_id, score».
- ``apply`` — применить правила блоков (``data/ocr/labels/printed_rules.csv``:
  ``group, sheet, from_idx, to_idx, kind`` — диапазон индексов внутри листа) и исключения
  (``data/ocr/labels/printed_exceptions.csv``: ``group, scan_id, kind``) к отсортированным
  оценкам → ``data/ocr/labels/printed_flags.csv`` (``scan_id, group, kind``).
- ``crosscheck`` — сверка с метками ``printed`` из ``data/ocr/review/truth_check.csv`` (T04):
  доля несовпадений по каждой группе → ``data/ocr/reports/printed_crosscheck.md``.
- ``stats`` — доля печатного по группе × буксиру × году (манифест) →
  ``data/ocr/reports/printed_stats.md``.

В консоль и отчёты идут только агрегаты и ``scan_id``.
"""

from __future__ import annotations

import argparse
import csv
import re
import sys
from pathlib import Path

import cv2
import numpy as np
import pandas as pd

from ocr_lab.paths import (
    CROPS_DIR,
    LABELS_DIR,
    MANIFEST,
    PRINTED_FLAGS_CSV,
    REPORTS_DIR,
    REVIEW_DIR,
    SHEETS_DIR,
    WORK_DIR,
    ensure_dir,
)
from ocr_lab.sheets import make_sheet

INDEX_CSV = WORK_DIR / "crops_index.csv"
TRUTH_CHECK_CSV = REVIEW_DIR / "truth_check.csv"

#: Три группы разметки и подполе ``crops_index.csv``, по которому они размечаются
#: (день и месяц — по строке ``left_base``, см. докстринг модуля).
GROUPS = ("voucher_number", "day", "month")
GROUP_SUBFIELD = {
    "voucher_number": "voucher_number",
    "day": "left_base.day",
    "month": "left_base.month",
}

SCORES_CSV = {g: LABELS_DIR / f"printed_scores_{g}.csv" for g in GROUPS}
RULES_CSV = LABELS_DIR / "printed_rules.csv"
EXCEPTIONS_CSV = LABELS_DIR / "printed_exceptions.csv"
PRINTED_SHEETS_DIR = SHEETS_DIR / "printed"
CROSSCHECK_MD = REPORTS_DIR / "printed_crosscheck.md"
STATS_MD = REPORTS_DIR / "printed_stats.md"

KINDS = ("printed", "handwritten", "empty", "unclear")
SHEET_SIZE = 100

#: Критерии приёмки промпта.
COVERAGE_TARGET = 0.98
UNCLEAR_MAX = 0.02
CROSSCHECK_MAX_MISMATCH = 0.01


def _read_gray(path: Path) -> np.ndarray | None:
    """Чтение серого PNG по пути, устойчивое к кириллице в пути (как в T04/T10)."""
    if not path.exists():
        return None
    data = np.fromfile(str(path), np.uint8)
    if data.size == 0:
        return None
    return cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)


def printedness_score(img: np.ndarray) -> float | None:
    """Эвристика «похоже на печатный шрифт бланка», от 0 до 1.

    Берём долю «сильных» градиентов (верхние 10% по модулю), направленных вдоль осей
    0°/90° (с допуском 12°). У печатных цифр бланка — прямые вертикальные/горизонтальные
    штрихи (и общая для обеих групп подчёркивающая линия поля), у рукописи — наклонные и
    изогнутые линии, поэтому доля ниже. ``None``, если на кропе нет чернил (поле пустое).

    Толщину штриха (разброс по distance transform) проверяли отдельно — на калибровочной
    выборке T04 (79 ваучеров, см. журнал) она не разделяла классы, поэтому в итоговую
    оценку не включена.
    """
    blurred = cv2.GaussianBlur(img, (3, 3), 0)
    _, mask = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    if int(mask.sum()) == 0:
        return None
    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
    mag = cv2.magnitude(gx, gy)
    threshold = float(np.percentile(mag, 90))
    strong = mag > threshold
    if not bool(strong.any()):
        return None
    angle = np.degrees(np.arctan2(gy[strong], gx[strong])) % 90
    return float(np.mean((angle < 12) | (angle > 78)))


def load_crops_index(path: Path = INDEX_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype=str, keep_default_na=False, encoding="utf-8-sig")


def build_scores(index: pd.DataFrame, group: str, crops_dir: Path = CROPS_DIR) -> pd.DataFrame:
    """Оценка печатности для всех выровненных кропов группы, по убыванию.

    Пустые кропы (``score is None``) уходят в конец списка отдельным блоком — на листах
    их видно сразу по подписи ``score=-1``.
    """
    subfield = GROUP_SUBFIELD[group]
    rows = index[(index["subfield"] == subfield) & (index["align_ok"] == "True")]
    records = []
    for rec in rows.itertuples():
        img = _read_gray(crops_dir / subfield / f"{rec.scan_id}.png")
        score = None if img is None else printedness_score(img)
        records.append({"scan_id": rec.scan_id, "score": score, "empty": score is None})
    df = pd.DataFrame.from_records(records, columns=["scan_id", "score", "empty"])
    df["score"] = df["score"].astype(float).fillna(-1.0)
    return df.sort_values("score", ascending=False, kind="stable").reset_index(drop=True)


def write_scores(df: pd.DataFrame, group: str) -> None:
    ensure_dir(SCORES_CSV[group].parent)
    df.to_csv(SCORES_CSV[group], index=False)


def read_scores(group: str) -> pd.DataFrame:
    return pd.read_csv(SCORES_CSV[group], dtype={"scan_id": str})


def build_sheets(
    group: str, scores: pd.DataFrame, crops_dir: Path = CROPS_DIR, sheets_dir: Path | None = None
) -> list[Path]:
    """Листы по ``SHEET_SIZE`` кропов группы, в порядке убывания оценки печатности."""
    subfield = GROUP_SUBFIELD[group]
    out_dir = sheets_dir if sheets_dir is not None else PRINTED_SHEETS_DIR
    blank = np.full((40, 100), 255, np.uint8)
    paths = []
    for start in range(0, len(scores), SHEET_SIZE):
        chunk = scores.iloc[start : start + SHEET_SIZE]
        images = []
        captions = []
        for rec in chunk.itertuples():
            img = _read_gray(crops_dir / subfield / f"{rec.scan_id}.png")
            images.append(img if img is not None else blank)
            captions.append(f"{rec.scan_id} {rec.score:.3f}")
        out_path = out_dir / f"{group}_{start // SHEET_SIZE}.png"
        make_sheet(images, captions, cols=10, cell_w=160, out_path=out_path)
        paths.append(out_path)
    return paths


# --- Применение правил разметки --------------------------------------------------------


def read_rules(path: Path = RULES_CSV) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def read_exceptions(path: Path = EXCEPTIONS_CSV) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def apply_rules(
    scores_by_group: dict[str, pd.DataFrame],
    rules: list[dict[str, str]],
    exceptions: list[dict[str, str]],
) -> pd.DataFrame:
    """Применить блочные правила листов и точечные исключения к оценкам.

    Правило ``{group, sheet, from_idx, to_idx, kind}`` размечает диапазон индексов
    ``[from_idx, to_idx]`` (включительно, внутри листа ``sheet`` по ``SHEET_SIZE``)
    значением ``kind``. Исключения ``{group, scan_id, kind}`` применяются последними и
    переопределяют правила блоков. Сканы, не попавшие ни в одно правило, получают
    ``unclear`` — это сигнал, что листы разобраны не полностью.
    """
    kind_by_group: dict[str, dict[str, str]] = {g: {} for g in GROUPS}
    for rule in rules:
        group = rule["group"]
        sheet = int(rule["sheet"])
        lo, hi = int(rule["from_idx"]), int(rule["to_idx"])
        scores = scores_by_group[group]
        start = sheet * SHEET_SIZE
        chunk = scores.iloc[start + lo : start + hi + 1]
        for scan_id in chunk["scan_id"]:
            kind_by_group[group][str(scan_id)] = rule["kind"]
    for exc in exceptions:
        kind_by_group[exc["group"]][str(exc["scan_id"])] = exc["kind"]

    out_rows = []
    for group, scores in scores_by_group.items():
        kinds = kind_by_group[group]
        for scan_id in scores["scan_id"]:
            out_rows.append(
                {"scan_id": scan_id, "group": group, "kind": kinds.get(str(scan_id), "unclear")}
            )
    return pd.DataFrame(out_rows, columns=["scan_id", "group", "kind"])


def write_printed_flags(df: pd.DataFrame, path: Path = PRINTED_FLAGS_CSV) -> None:
    ensure_dir(path.parent)
    df.to_csv(path, index=False)


def read_printed_flags(path: Path = PRINTED_FLAGS_CSV) -> pd.DataFrame:
    return pd.read_csv(path, dtype={"scan_id": str})


# --- Сверка с T04 ------------------------------------------------------------------------


def _parse_row_printed(value: str) -> tuple[str | None, str | None]:
    """Разобрать ``day=<kind>;month=<kind>`` из ``truth_check.csv`` (T04)."""
    match = re.match(r"day=(\w+);month=(\w+)", value)
    if not match:
        return None, None
    return match.group(1), match.group(2)


def truth_check_labels(path: Path = TRUTH_CHECK_CSV) -> pd.DataFrame:
    """Метки T04 (``truth_check.csv``), сведённые к формату ``scan_id, group, kind``.

    Номер — из строки ``field == voucher_number``; день/месяц — из строки
    ``field == left_base`` (та же строка, по которой размечает и T12).
    """
    truth = pd.read_csv(path, dtype=str, keep_default_na=False)
    rows = []
    for rec in truth.itertuples():
        if rec.field == "voucher_number" and rec.printed:
            rows.append({"scan_id": rec.scan_id, "group": "voucher_number", "kind": rec.printed})
        elif rec.field == "left_base":
            day, month = _parse_row_printed(rec.printed)
            if day:
                rows.append({"scan_id": rec.scan_id, "group": "day", "kind": day})
            if month:
                rows.append({"scan_id": rec.scan_id, "group": "month", "kind": month})
    return pd.DataFrame(rows, columns=["scan_id", "group", "kind"])


def crosscheck(flags: pd.DataFrame, truth: pd.DataFrame) -> pd.DataFrame:
    """Сверка меток T12 с эталоном T04: по строке на (group, scan_id) из эталона."""
    merged = truth.merge(
        flags, on=["scan_id", "group"], how="left", suffixes=("_truth", "_t12")
    )
    merged["match"] = merged["kind_truth"] == merged["kind_t12"]
    return merged


# --- Статистика --------------------------------------------------------------------------


def printed_share(flags: pd.DataFrame, manifest: pd.DataFrame) -> pd.DataFrame:
    """Доля ``printed`` среди размеченных (``printed``/``handwritten``), группа × буксир × год."""
    merged = flags.merge(manifest[["scan_id", "tug_code", "year"]], on="scan_id", how="left")
    merged = merged[merged["kind"].isin(("printed", "handwritten"))]
    out = (
        merged.groupby(["group", "tug_code", "year"])["kind"]
        .apply(lambda s: float((s == "printed").mean()))
        .reset_index(name="printed_share")
    )
    counts = merged.groupby(["group", "tug_code", "year"]).size().reset_index(name="n")
    return out.merge(counts, on=["group", "tug_code", "year"])


# --- CLI -----------------------------------------------------------------------------------


def _cmd_score(args: argparse.Namespace) -> None:
    index = load_crops_index()
    for group in GROUPS:
        df = build_scores(index, group)
        write_scores(df, group)
        n_empty = int(df["empty"].sum())
        print(f"{group}: {len(df)} кропов, пустых {n_empty}")


def _cmd_sheets(args: argparse.Namespace) -> None:
    for group in GROUPS:
        scores = read_scores(group)
        paths = build_sheets(group, scores)
        print(f"{group}: {len(paths)} листов")


def _cmd_apply(args: argparse.Namespace) -> None:
    scores_by_group = {g: read_scores(g) for g in GROUPS}
    rules = read_rules()
    exceptions = read_exceptions()
    flags = apply_rules(scores_by_group, rules, exceptions)
    write_printed_flags(flags)
    total = len(flags)
    counts = flags["kind"].value_counts()
    for kind in KINDS:
        print(f"{kind}: {counts.get(kind, 0)} / {total}")
    unclear_share = counts.get("unclear", 0) / total if total else 0.0
    print(f"доля unclear: {unclear_share:.3f} (цель <= {UNCLEAR_MAX})")


def _cmd_crosscheck(args: argparse.Namespace) -> None:
    flags = read_printed_flags()
    truth = truth_check_labels()
    merged = crosscheck(flags, truth)
    ensure_dir(CROSSCHECK_MD.parent)
    lines = ["# Сверка разметки T12 с эталоном T04\n"]
    overall_n = len(merged)
    overall_mismatch = int((~merged["match"]).sum())
    lines.append(
        f"Всего сопоставлено: {overall_n}. Несовпадений: {overall_mismatch} "
        f"({overall_mismatch / overall_n:.3%}).\n"
    )
    for group in GROUPS:
        sub = merged[merged["group"] == group]
        n = len(sub)
        mismatch = int((~sub["match"]).sum())
        share = mismatch / n if n else 0.0
        lines.append(f"- `{group}`: {n} сопоставлено, несовпадений {mismatch} ({share:.3%})")
    CROSSCHECK_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def _cmd_stats(args: argparse.Namespace) -> None:
    flags = read_printed_flags()
    manifest = pd.read_csv(MANIFEST, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    share = printed_share(flags, manifest)
    ensure_dir(STATS_MD.parent)
    lines = ["# Доля печатного по группе × буксиру × году (T12)\n"]
    lines.append("| group | tug_code | year | printed_share | n |")
    lines.append("|---|---|---|---|---|")
    for rec in share.sort_values(["group", "tug_code", "year"]).itertuples():
        lines.append(
            f"| {rec.group} | {rec.tug_code} | {rec.year} | {rec.printed_share:.3f} | {rec.n} |"
        )
    STATS_MD.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="T12: разметка «напечатано / от руки»")
    sub = parser.add_subparsers(dest="step", required=True)
    for name, fn in (
        ("score", _cmd_score),
        ("sheets", _cmd_sheets),
        ("apply", _cmd_apply),
        ("crosscheck", _cmd_crosscheck),
        ("stats", _cmd_stats),
    ):
        p = sub.add_parser(name)
        p.set_defaults(fn=fn)
    args = parser.parse_args(argv)
    args.fn(args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
