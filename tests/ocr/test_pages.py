"""Тесты кэша страниц `ocr_lab.pages` — только синтетика."""

from __future__ import annotations

import numpy as np
from PIL import Image

from ocr_lab.pages import PageStat, _process_one, update_index, write_sample_sheet


def _make_pdf(path, size=(120, 80)):
    Image.new("RGB", size, color=(180, 180, 180)).save(path, "PDF")
    return path


def test_process_one_writes_png(tmp_path):
    src = _make_pdf(tmp_path / "scan.pdf")
    out_dir = tmp_path / "pages"
    out_dir.mkdir()

    stat = _process_one("2025_1k", str(src), str(out_dir))

    assert stat.status == "ok"
    assert stat.n_pages == 1
    assert stat.src_width > 0 and stat.src_height > 0
    out = out_dir / "2025_1k.png"
    assert out.exists()
    with Image.open(out) as img:
        assert img.size == (1654, img.size[1])


def test_process_one_cyrillic_scan_id(tmp_path):
    """Регрессия: cv2.imwrite на Windows не пишет в не-ASCII пути — файл должен
    получиться ровно под именем scan_id, без искажения."""
    src = _make_pdf(tmp_path / "scan.pdf")
    out_dir = tmp_path / "pages"
    out_dir.mkdir()

    stat = _process_one("2025_67к", str(src), str(out_dir))

    assert stat.status == "ok"
    assert (out_dir / "2025_67к.png").exists()


def test_process_one_skips_existing(tmp_path):
    src = _make_pdf(tmp_path / "scan.pdf")
    out_dir = tmp_path / "pages"
    out_dir.mkdir()
    (out_dir / "2025_1k.png").write_bytes(b"already")

    stat = _process_one("2025_1k", str(src), str(out_dir))

    assert stat.status == "skipped"
    assert stat.n_pages == 1  # число страниц считается и для пропущенных


def test_process_one_missing_source(tmp_path):
    out_dir = tmp_path / "pages"
    out_dir.mkdir()

    stat = _process_one("2025_9k", str(tmp_path / "none.pdf"), str(out_dir))

    assert stat.status == "error"
    assert "none.pdf" in stat.error


def test_update_index_merges_stats(tmp_path):
    stats = [
        PageStat(scan_id="a", status="ok", n_pages=1, src_width=100, src_height=140, dpi=200),
        PageStat(scan_id="b", status="skipped", n_pages=2),
        PageStat(scan_id="c", status="error", error="boom"),
    ]
    rows = update_index(stats, tmp_path / "pages")

    assert rows["a"]["src_width"] == "100"
    assert rows["b"]["n_pages"] == "2"
    assert "c" not in rows

    # Повторный прогон: ok-строка не затирается «пустым» skipped.
    stats2 = [PageStat(scan_id="a", status="skipped", n_pages=1)]
    rows2 = update_index(stats2, tmp_path / "pages")
    assert rows2["a"]["src_width"] == "100"


def test_write_sample_sheet(tmp_path):
    pages_dir = tmp_path / "pages"
    pages_dir.mkdir()
    stats = []
    for i in range(5):
        scan_id = f"2025_{i}k"
        Image.fromarray(np.zeros((300, 200), dtype=np.uint8)).save(pages_dir / f"{scan_id}.png")
        stats.append(PageStat(scan_id=scan_id, status="ok"))

    sheet_path = tmp_path / "sheets" / "sample.png"
    n = write_sample_sheet(stats, pages_dir, sheet_path, n=4, seed=1, cols=2)

    assert n == 4
    with Image.open(sheet_path) as sheet:
        assert max(sheet.size) <= 2000
