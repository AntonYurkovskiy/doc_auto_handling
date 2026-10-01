"""Тесты загрузчика сканов `app.ocr.io` — только синтетика."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image

from app.ocr.io import (
    PAGE_WIDTH_200DPI,
    ScanLoadError,
    load_scan,
    load_scan_pil,
    normalize_width,
    page_count,
    page_size_px,
)


def test_load_scan_pdf_single_page(tmp_path):
    path = tmp_path / "one.pdf"
    Image.new("RGB", (120, 80), color=(200, 100, 50)).save(path, "PDF")

    gray = load_scan(path)

    assert gray.dtype == np.uint8
    assert gray.ndim == 2  # серое HxW, без канала цвета
    assert page_count(path) == 1


def test_load_scan_pdf_two_pages(tmp_path):
    path = tmp_path / "two.pdf"
    first = Image.new("RGB", (90, 60), color=(255, 255, 255))
    second = Image.new("RGB", (90, 60), color=(0, 0, 0))
    first.save(path, "PDF", save_all=True, append_images=[second])

    assert page_count(path) == 2
    page0 = load_scan(path, page=0)
    page1 = load_scan(path, page=1)
    assert float(page0.mean()) > float(page1.mean())

    with pytest.raises(ScanLoadError):
        load_scan(path, page=5)


def test_load_scan_jpeg_exif_orientation(tmp_path):
    # Сохранено 30x10, EXIF Orientation=6 — при показе повернуть на 90° по часовой.
    img = Image.new("RGB", (30, 10), color=(0, 255, 0))
    exif = img.getexif()
    exif[274] = 6
    path = tmp_path / "rot.jpg"
    img.save(path, exif=exif)

    gray = load_scan(path)

    assert gray.shape == (30, 10)  # H x W после поворота


def test_load_scan_png_gray_and_rgb(tmp_path):
    gray_path = tmp_path / "gray.png"
    Image.new("L", (40, 25), color=128).save(gray_path)
    rgb_path = tmp_path / "rgb.png"
    Image.new("RGB", (40, 25), color=(10, 200, 90)).save(rgb_path)

    gray = load_scan(gray_path)
    rgb = load_scan(rgb_path)

    assert gray.shape == (25, 40)
    assert rgb.shape == (25, 40)  # RGB приведён к серому
    assert rgb.dtype == np.uint8
    assert page_count(gray_path) == 1


def test_load_scan_missing_file(tmp_path):
    path = tmp_path / "нет_файла.pdf"
    with pytest.raises(FileNotFoundError) as exc_info:
        load_scan(path)
    assert "нет_файла.pdf" in str(exc_info.value)

    with pytest.raises(FileNotFoundError):
        page_count(path)


def test_load_scan_unsupported_suffix(tmp_path):
    path = tmp_path / "scan.xyz"
    path.write_bytes(b"not an image")
    with pytest.raises(ScanLoadError) as exc_info:
        load_scan(path)
    assert "scan.xyz" in str(exc_info.value)


def test_load_scan_corrupt_image(tmp_path):
    path = tmp_path / "broken.png"
    path.write_bytes(b"this is not a png")
    with pytest.raises(ScanLoadError):
        load_scan(path)


def test_load_scan_pil_returns_rgb(tmp_path):
    path = tmp_path / "gray.png"
    Image.new("L", (20, 20), color=200).save(path)
    image = load_scan_pil(path)
    try:
        assert image.mode == "RGB"
    finally:
        image.close()


def test_page_size_px_matches_render(tmp_path):
    """`page_size_px` должен давать тот же размер, что и реальный `load_scan`."""
    pdf_path = tmp_path / "doc.pdf"
    Image.new("RGB", (120, 80), color=(100, 100, 100)).save(pdf_path, "PDF")
    w, h = page_size_px(pdf_path, dpi=200)
    assert (h, w) == load_scan(pdf_path, dpi=200).shape

    img = Image.new("RGB", (30, 10), color=(0, 255, 0))
    exif = img.getexif()
    exif[274] = 6
    jpg_path = tmp_path / "rot.jpg"
    img.save(jpg_path, exif=exif)
    w, h = page_size_px(jpg_path)
    assert (h, w) == load_scan(jpg_path).shape  # EXIF-поворот учтён


def test_normalize_width_shrink_and_grow():
    img = np.zeros((100, 200), dtype=np.uint8)

    shrunk = normalize_width(img, 100)
    assert shrunk.shape == (50, 100)

    grown = normalize_width(img, 400)
    assert grown.shape == (200, 400)

    same = normalize_width(img, 200)
    assert same.shape == (100, 200)


def test_normalize_width_a4_proportions():
    # Портрет A4 при 300 dpi -> ширина 1654 (200 dpi), пропорции сохраняются.
    img = np.zeros((3508, 2480), dtype=np.uint8)
    out = normalize_width(img, PAGE_WIDTH_200DPI)
    assert out.shape[1] == PAGE_WIDTH_200DPI
    assert abs(out.shape[0] / out.shape[1] - 3508 / 2480) < 0.01


def test_normalize_width_rejects_bad_shape():
    with pytest.raises(ValueError):
        normalize_width(np.zeros((4, 4, 4, 4), dtype=np.uint8))


def test_voucher_ocr_wrapper_uses_shared_loader(tmp_path):
    """`load_voucher_image` остаётся тонкой обёрткой: PDF -> PIL RGB, dpi=300."""
    from app.models import Voucher
    from app.services.voucher_ocr import load_voucher_image

    path = tmp_path / "v.pdf"
    Image.new("RGB", (72, 72), color=(255, 0, 0)).save(path, "PDF", resolution=72)

    voucher = Voucher(file_path=str(path))
    image = load_voucher_image(voucher)
    assert image is not None
    assert image.mode == "RGB"
    assert image.size == (300, 300)  # 72 pt при 300 dpi

    assert load_voucher_image(Voucher(file_path=None)) is None
    assert load_voucher_image(Voucher(file_path=str(tmp_path / "none.pdf"))) is None
