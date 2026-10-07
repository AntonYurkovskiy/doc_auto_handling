"""Тесты массовой нарезки кропов (T10): прогон, идемпотентность, листы, отчёт. Только синтетика."""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

from app.ocr import layouts as boxes
from app.ocr.variants import save_layout as save_variant
from ocr_lab import cut_crops as cc
from tests.ocr.synth import PAGE_H, PAGE_W, make_form

VARIANT = "syn_v1"


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())


def _box_layout() -> boxes.Layout:
    """17 неподвижных боксов (без ``line``) в кадре синтетического бланка."""
    items = [boxes.Box(boxes.VOUCHER_NUMBER, 900, 300, 1150, 400, "number")]
    for i, row in enumerate(boxes.DATE_ROWS):
        for j, part in enumerate(boxes.DATE_PARTS):
            x0, y0 = 300 + 200 * j, 1200 + 100 * i
            items.append(boxes.Box(f"{row}.{part}", x0, y0, x0 + 150, y0 + 90, "two_digit"))
    return boxes.Layout(VARIANT, "s", (PAGE_W, PAGE_H), tuple(items))


@pytest.fixture()
def env(tmp_path: Path) -> dict[str, Path]:
    """Манифест из трёх сканов: хороший (с кириллицей в id), шум и отсутствующая страница."""
    layouts_dir = tmp_path / "layouts"
    image, mask = make_form(layout_seed=0)
    save_variant(layouts_dir / VARIANT, image, mask, {"tug_code": "s"})
    boxes_dir = tmp_path / "boxes"
    boxes_dir.mkdir()
    boxes.save_layout(_box_layout(), boxes_dir / f"{VARIANT}.json")

    pages = tmp_path / "pages"
    good, _ = make_form(layout_seed=0, fill_seed=3)
    _write_png(pages / "2025_67к.png", good)
    noise = np.random.default_rng(0).integers(0, 256, (PAGE_H, PAGE_W), dtype=np.uint8)
    _write_png(pages / "2025_2k.png", noise)

    ids = [("2025_67к", "2025"), ("2025_2k", "2025"), ("2026_9k", "2026")]
    manifest = tmp_path / "manifest.csv"
    _write_csv(manifest, [{"scan_id": s, "year": y} for s, y in ids])
    assignment = tmp_path / "assignment.csv"
    _write_csv(assignment, [{"scan_id": s, "variant": VARIANT} for s, _ in ids])
    return {
        "manifest": manifest, "assignment": assignment, "pages_dir": pages,
        "crops_dir": tmp_path / "crops", "layouts_dir": layouts_dir, "boxes_dir": boxes_dir,
        "index_csv": tmp_path / "crops_index.csv",
        "failures_csv": tmp_path / "reports" / "align_failures.csv",
    }


def test_build_cuts_crops_lists_failures_and_is_idempotent(env: dict[str, Path]) -> None:
    results = {r.scan_id: r for r in cc.build(workers=1, **env)}  # type: ignore[arg-type]
    assert results["2025_67к"].status == "ok"
    assert results["2025_2k"].status == "failed"
    assert results["2026_9k"].status == "error"  # нет страницы — прогон не падает

    index = cc.read_csv_rows(env["index_csv"])
    assert list(index[0]) == list(cc.INDEX_COLUMNS)
    assert {r["scan_id"] for r in index} == {"2025_67к"}
    assert [r["subfield"] for r in index] == list(boxes.SUBFIELD_NAMES)
    for row in index:
        crop = cc.read_gray(cc.crop_path(env["crops_dir"], row["subfield"], row["scan_id"]))
        # Серый кроп в разрешении эталона: размер — ровно бокс макета.
        assert crop.ndim == 2
        assert crop.shape == (int(row["h"]), int(row["w"]))
    assert int(index[0]["w"]) == 250 and int(index[0]["h"]) == 100

    failures = cc.read_csv_rows(env["failures_csv"])
    assert [r["scan_id"] for r in failures] == ["2025_2k"]
    assert not (env["crops_dir"] / "left_base.day" / "2025_2k.png").exists()

    again = {r.scan_id: r.status for r in cc.build(workers=1, **env)}  # type: ignore[arg-type]
    assert again == {"2025_67к": "skipped", "2025_2k": "skipped", "2026_9k": "error"}
    assert cc.read_csv_rows(env["index_csv"]) == index

    forced = {r.scan_id: r.status for r in cc.build(workers=1, force=True, **env)}  # type: ignore[arg-type]
    assert forced["2025_67к"] == "ok"


def test_build_redoes_scan_with_missing_crop(env: dict[str, Path]) -> None:
    cc.build(workers=1, **env)  # type: ignore[arg-type]
    cc.crop_path(env["crops_dir"], "finished_work.minute", "2025_67к").unlink()
    statuses = {r.scan_id: r.status for r in cc.build(workers=1, **env)}  # type: ignore[arg-type]
    assert statuses["2025_67к"] == "ok"
    assert cc.crop_path(env["crops_dir"], "finished_work.minute", "2025_67к").exists()


def test_merge_results_failure_replaces_index_rows() -> None:
    index = {"a": [{"scan_id": "a", "subfield": "voucher_number"}]}
    failures: dict[str, dict[str, str]] = {}
    cc.merge_results(index, failures, [
        cc.ScanResult(scan_id="a", year="2025", status="failed", variant="v", reason="r"),
        cc.ScanResult(scan_id="b", year="2025", status="error", reason="e"),
    ])
    assert index == {}
    assert set(failures) == {"a"} and failures["a"]["reason"] == "r"


def _index_rows(n_scans: int) -> list[dict[str, str]]:
    return [
        {"scan_id": f"s{i:03d}", "subfield": name, "align_ok": "True"}
        for i in range(n_scans)
        for name in boxes.SUBFIELD_NAMES
    ]


def test_sample_for_sheets_is_reproducible_and_bounded() -> None:
    rows = _index_rows(300)
    first = cc.sample_for_sheets(rows, n=200, seed=0)
    assert first == cc.sample_for_sheets(list(reversed(rows)), n=200, seed=0)
    assert all(len(ids) == 200 == len(set(ids)) for ids in first.values())
    # Выборки подполей разные — под контролем больше сканов.
    assert first["left_base.day"] != first["left_base.month"]
    small = cc.sample_for_sheets(_index_rows(30), n=200, seed=0)
    assert all(len(ids) == 30 for ids in small.values())


def test_sheet_paths_split_by_hundred(tmp_path: Path) -> None:
    names = [p.name for p in cc.sheet_paths("left_base.day", 200, tmp_path)]
    assert names == ["left_base.day_1.png", "left_base.day_2.png"]
    assert len(cc.sheet_paths("x", 101, tmp_path)) == 2


def test_build_sheets_writes_index_csv(env: dict[str, Path], tmp_path: Path) -> None:
    cc.build(workers=1, **env)  # type: ignore[arg-type]
    out = tmp_path / "sheets"
    written = cc.build_sheets(
        n=5, subfields=["voucher_number"], index_csv=env["index_csv"],
        crops_dir=env["crops_dir"], out_dir=out,
    )
    assert [p.name for p in written] == ["voucher_number_1.png"]
    mapping = cc.read_csv_rows(out / "voucher_number_1.csv")
    assert mapping == [{"index": "0", "caption": "2025_67к"}]


def test_render_report_counts_failures_visibility_and_t11() -> None:
    jobs = [(f"s{i}", "2025" if i < 50 else "2026", "v1") for i in range(100)]
    index = [
        {"scan_id": sid, "variant": "v1", "subfield": name, "source": "line"}
        for sid, _, _ in jobs[6:]
        for name in boxes.SUBFIELD_NAMES
    ]
    failures = [
        {"scan_id": f"s{i}", "variant": "v1", "score": "0.5", "inliers": "10", "reason": "x"}
        for i in range(6)
    ]
    qc = [{"subfield": "left_base.day", "scan_id": f"s{i}", "problem": "cut_digit"}
          for i in range(10, 15)]
    text = cc.render_report(jobs, index, failures, qc, sample_n=200)
    assert "неудачных выравниваний: 6 (6.0 %)" in text
    assert "| 2025 | 50 | 6 | 12.0 % |" in text
    assert "| left_base.day | 195 | 5 | нет |" in text
    assert "| left_base.month | 200 | 0 | да |" in text
    assert "cut_digit" in text
    assert "T11 по этому критерию нужна" in text
