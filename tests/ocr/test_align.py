"""Тесты выравнивания скана по эталону (только синтетика)."""

from __future__ import annotations

import time

import cv2
import numpy as np
import pytest

from app.ocr.align import (
    AlignParams,
    Reference,
    align,
    alignment_score,
    build_reference,
    map_points,
    map_points_to_scan,
)
from tests.ocr.synth import PAGE_H, PAGE_W, corner_error, degrade, make_form, random_homography

MAX_CORNER_ERR_PX = 1.5


@pytest.fixture(scope="module")
def ref() -> Reference:
    image, mask = make_form(layout_seed=0)
    return build_reference(image, mask, name="synth")


def _scan(seed: int) -> tuple[np.ndarray, np.ndarray]:
    """Скан бланка с новой «рукописью» и случайной H эталон → скан."""
    rng = np.random.default_rng(1000 + seed)
    page, _ = make_form(layout_seed=0, fill_seed=seed)
    H_true = random_homography(rng)
    return degrade(page, H_true, rng), H_true


@pytest.mark.parametrize("seed", range(6))
def test_random_homography_corner_error(ref: Reference, seed: int) -> None:
    scan, H_true = _scan(seed)
    res = align(scan, ref)
    assert res.ok, res.reason
    assert res.H is not None and res.warped is not None
    assert res.warped.shape == ref.image.shape
    assert not res.rotated180
    assert res.method.startswith("orb")
    assert res.score >= ref.params.min_score
    err = corner_error(res.H, H_true, (PAGE_W, PAGE_H))
    assert err.max() < MAX_CORNER_ERR_PX, err


def test_rotated_180(ref: Reference) -> None:
    scan, H_true = _scan(40)
    rotated = cv2.rotate(scan, cv2.ROTATE_180)
    res = align(rotated, ref)
    assert res.ok, res.reason
    assert res.rotated180
    assert res.H is not None
    # Истинная H: эталон → скан → повёрнутый скан.
    h, w = scan.shape
    rot = np.array([[-1.0, 0.0, w - 1.0], [0.0, -1.0, h - 1.0], [0.0, 0.0, 1.0]])
    err = corner_error(res.H, rot @ H_true, (PAGE_W, PAGE_H))
    assert err.max() < MAX_CORNER_ERR_PX, err


def test_rotated_180_without_explicit_retry(ref: Reference) -> None:
    """ORB инвариантен к повороту: 180° находится и без второй попытки."""
    scan, _ = _scan(41)
    params = AlignParams(try_rotate180=False)
    res = align(cv2.rotate(scan, cv2.ROTATE_180), ref, params)
    assert res.ok, res.reason
    assert res.rotated180


@pytest.mark.parametrize("kind", ["noise", "other_form", "blank"])
def test_foreign_image_rejected(ref: Reference, kind: str) -> None:
    rng = np.random.default_rng(5)
    if kind == "noise":
        img = rng.integers(0, 256, (PAGE_H, PAGE_W)).astype(np.uint8)
    elif kind == "other_form":
        page, _ = make_form(layout_seed=5, fill_seed=1)
        img = degrade(page, random_homography(rng), rng)
    else:
        img = np.full((PAGE_H, PAGE_W), 255, np.uint8)
    res = align(img, ref)
    assert not res.ok
    assert res.reason
    assert "инлайер" in res.reason or "признак" in res.reason or "пар" in res.reason


def test_map_points_to_scan(ref: Reference) -> None:
    scan, H_true = _scan(7)
    res = align(scan, ref)
    assert res.ok and res.H is not None
    rng = np.random.default_rng(0)
    pts_ref = rng.uniform([100, 100], [PAGE_W - 100, PAGE_H - 100], size=(50, 2))
    got = map_points_to_scan(pts_ref, res.H)
    want = map_points(pts_ref, H_true)
    assert np.linalg.norm(got - want, axis=1).max() < MAX_CORNER_ERR_PX


def test_bgr_input_keeps_channels(ref: Reference) -> None:
    scan, _ = _scan(8)
    res = align(cv2.cvtColor(scan, cv2.COLOR_GRAY2BGR), ref)
    assert res.ok, res.reason
    assert res.warped is not None and res.warped.shape == (PAGE_H, PAGE_W, 3)


def test_score_drops_with_misalignment(ref: Reference) -> None:
    scan, H_true = _scan(9)
    H = np.linalg.inv(H_true)
    exact = alignment_score(scan, H, ref)
    shifted = alignment_score(scan, np.array([[1, 0, 10], [0, 1, 5], [0, 0, 1.0]]) @ H, ref)
    other, _ = make_form(layout_seed=5, fill_seed=1)
    assert exact > 0.9
    assert shifted < ref.params.min_score
    assert alignment_score(other, np.eye(3), ref) < ref.params.min_score


def test_mask_size_mismatch() -> None:
    image, _ = make_form(layout_seed=0)
    with pytest.raises(ValueError):
        build_reference(image, np.zeros((10, 10), np.uint8))


def test_time_per_page(ref: Reference) -> None:
    scan, _ = _scan(11)
    align(scan, ref)  # прогрев
    t0 = time.perf_counter()
    res = align(scan, ref)
    elapsed = time.perf_counter() - t0
    assert res.ok, res.reason
    assert elapsed < 2.0, f"{elapsed:.2f} с"


def test_sift_fallback(ref: Reference) -> None:
    """ORB почти без признаков → запасной путь SIFT."""
    scan, H_true = _scan(12)
    res = align(scan, ref, AlignParams(orb_nfeatures=20))
    assert res.ok, res.reason
    assert res.method.startswith("sift")
    assert res.H is not None
    assert corner_error(res.H, H_true, (PAGE_W, PAGE_H)).max() < MAX_CORNER_ERR_PX
