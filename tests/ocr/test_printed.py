"""Тесты разметки «напечатано / от руки» `ocr_lab.printed` — только синтетика."""

from __future__ import annotations

import csv

import numpy as np
import pandas as pd
import pytest

from ocr_lab.printed import (
    GROUPS,
    apply_rules,
    build_scores,
    build_sheets,
    crosscheck,
    load_crops_index,
    printed_share,
    printedness_score,
    truth_check_labels,
)


def _printed_like(size: int = 80) -> np.ndarray:
    """Синтетический «печатный» кроп: прямые вертикальные/горизонтальные штрихи."""
    img = np.full((size, size), 255, np.uint8)
    img[10:70, 20:28] = 0
    img[10:70, 52:60] = 0
    img[10:18, 20:60] = 0
    img[62:70, 20:60] = 0
    return img


def _handwritten_like(size: int = 80) -> np.ndarray:
    """Синтетический «рукописный» кроп: наклонная изогнутая линия (синусоида)."""
    img = np.full((size, size), 255, np.uint8)
    xs = np.arange(10, 70)
    ys = (size / 2 + 20 * np.sin((xs - 10) / 60 * 2 * np.pi) + 0.6 * (xs - 10)).astype(int)
    for x, y in zip(xs, ys, strict=True):
        img[max(0, y - 2) : y + 3, x : x + 2] = 0
    return img


def _empty_crop(size: int = 80) -> np.ndarray:
    return np.full((size, size), 255, np.uint8)


def test_printedness_score_separates_printed_and_handwritten():
    printed_score = printedness_score(_printed_like())
    handwritten_score = printedness_score(_handwritten_like())
    assert printed_score is not None and handwritten_score is not None
    assert printed_score > handwritten_score


def test_printedness_score_none_for_empty_crop():
    assert printedness_score(_empty_crop()) is None


def _write_png(path, img: np.ndarray) -> None:
    import cv2

    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".png", img)
    assert ok
    path.write_bytes(buf.tobytes())


def _make_crops_index(tmp_path, scan_ids: list[str]) -> tuple[pd.DataFrame, object]:
    """Готовит кропы на диске (печатные/рукописные/пустые по шаблону) и индекс в памяти."""
    crops_dir = tmp_path / "crops"
    rows = []
    for subfield in ("voucher_number", "left_base.day", "left_base.month"):
        for i, scan_id in enumerate(scan_ids):
            if scan_id.endswith("_empty"):
                img = _empty_crop()
            elif i % 3 == 0:
                img = _printed_like()
            else:
                img = _handwritten_like()
            _write_png(crops_dir / subfield / f"{scan_id}.png", img)
            rows.append(
                {
                    "scan_id": scan_id,
                    "variant": "kommunar_v1",
                    "subfield": subfield,
                    "path": f"crops/{subfield}/{scan_id}.png",
                    "w": img.shape[1],
                    "h": img.shape[0],
                    "align_ok": "True",
                    "score": "0.9",
                    "inliers": "100",
                    "rotated180": "False",
                    "source": "line",
                }
            )
    index = pd.DataFrame(rows)
    return index, crops_dir


def test_build_scores_sorts_descending_and_flags_empty(tmp_path):
    scan_ids = [f"2025_{i}k" for i in range(5)] + ["2025_9k_empty"]
    index, crops_dir = _make_crops_index(tmp_path, scan_ids)
    df = build_scores(index, "day", crops_dir=crops_dir)

    assert list(df["scan_id"]) != []
    assert df["score"].is_monotonic_decreasing
    empty_row = df[df["scan_id"] == "2025_9k_empty"].iloc[0]
    assert bool(empty_row["empty"]) is True
    assert empty_row["score"] == -1.0


def test_build_sheets_chunks_by_sheet_size(tmp_path):
    scan_ids = [f"2025_{i}k" for i in range(120)]
    index, crops_dir = _make_crops_index(tmp_path, scan_ids)
    scores = build_scores(index, "voucher_number", crops_dir=crops_dir)

    sheets_dir = tmp_path / "sheets"
    paths = build_sheets("voucher_number", scores, crops_dir=crops_dir, sheets_dir=sheets_dir)

    assert len(paths) == 2  # 120 кропов -> листы по 100
    for p in paths:
        assert p.exists()
        with p.with_suffix(".csv").open(encoding="utf-8") as fh:
            rows = list(csv.DictReader(fh))
        assert len(rows) <= 100


def test_apply_rules_blocks_and_exceptions():
    scores = {
        g: pd.DataFrame(
            {"scan_id": [f"s{i}" for i in range(6)], "score": [0.9, 0.8, 0.7, 0.3, 0.2, 0.1]}
        )
        for g in GROUPS
    }
    rules = [
        {"group": "day", "sheet": "0", "from_idx": "0", "to_idx": "2", "kind": "printed"},
        {"group": "day", "sheet": "0", "from_idx": "3", "to_idx": "5", "kind": "handwritten"},
    ]
    exceptions = [{"group": "day", "scan_id": "s1", "kind": "unclear"}]

    flags = apply_rules(scores, rules, exceptions)
    day = flags[flags["group"] == "day"].set_index("scan_id")["kind"]
    assert day["s0"] == "printed"
    assert day["s1"] == "unclear"  # исключение переопределило правило блока
    assert day["s2"] == "printed"
    assert day["s3"] == "handwritten"

    # Группы без правил -> все unclear (сигнал, что листы не разобраны).
    other = flags[flags["group"] == "voucher_number"].set_index("scan_id")["kind"]
    assert set(other.unique()) == {"unclear"}


def test_truth_check_labels_parses_row_and_voucher_number(tmp_path):
    truth_csv = tmp_path / "truth_check.csv"
    truth_csv.write_text(
        "scan_id,field,truth,seen,verdict,printed,note\n"
        "2025_1k,voucher_number,1k,1,match,handwritten,\n"
        "2025_1k,left_base,01.01 10:00,01.01 10:00,match,day=printed;month=handwritten,\n"
        "2025_1k,arrived_base,01.01 11:00,01.01 11:00,match,day=printed;month=handwritten,\n",
        encoding="utf-8",
    )
    labels = truth_check_labels(truth_csv).set_index("group")["kind"]
    assert labels["voucher_number"] == "handwritten"
    assert labels["day"] == "printed"
    assert labels["month"] == "handwritten"
    # Строка arrived_base не парсится (только left_base, как и в разметке T12).
    assert len(labels) == 3


def test_crosscheck_counts_mismatches():
    flags = pd.DataFrame(
        {
            "scan_id": ["s1", "s2", "s3"],
            "group": ["day", "day", "day"],
            "kind": ["printed", "handwritten", "printed"],
        }
    )
    truth = pd.DataFrame(
        {
            "scan_id": ["s1", "s2", "s3"],
            "group": ["day", "day", "day"],
            "kind": ["printed", "printed", "printed"],
        }
    )
    merged = crosscheck(flags, truth)
    assert merged["match"].sum() == 2
    assert (~merged["match"]).sum() == 1


def test_printed_share_by_group_tug_year():
    flags = pd.DataFrame(
        {
            "scan_id": ["a", "b", "c", "d"],
            "group": ["day", "day", "day", "day"],
            "kind": ["printed", "printed", "handwritten", "unclear"],
        }
    )
    manifest = pd.DataFrame(
        {
            "scan_id": ["a", "b", "c", "d"],
            "tug_code": ["k", "k", "p", "p"],
            "year": ["2025", "2025", "2025", "2026"],
        }
    )
    share = printed_share(flags, manifest)
    k2025 = share[(share.group == "day") & (share.tug_code == "k") & (share.year == "2025")]
    assert k2025.iloc[0]["printed_share"] == pytest.approx(1.0)
    assert k2025.iloc[0]["n"] == 2
    # unclear не входит ни в числитель, ни в знаменатель доли.
    p2025 = share[(share.group == "day") & (share.tug_code == "p") & (share.year == "2025")]
    assert p2025.iloc[0]["n"] == 1


def test_load_crops_index_reads_csv(tmp_path):
    path = tmp_path / "crops_index.csv"
    path.write_text(
        "scan_id,variant,subfield,path,w,h,align_ok,score,inliers,rotated180,source\n"
        "2025_1k,kommunar_v1,voucher_number,crops/voucher_number/2025_1k.png,100,50,True,0.9,"
        "100,False,line\n",
        encoding="utf-8",
    )
    df = load_crops_index(path)
    assert df.loc[0, "scan_id"] == "2025_1k"
