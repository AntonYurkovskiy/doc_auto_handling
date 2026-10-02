"""Тесты вариантов бланка: запись/чтение эталона и ``detect_variant`` (только синтетика)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from app.ocr.align import AlignParams
from app.ocr.variants import (
    Layout,
    detect_variant,
    load_layout,
    load_layouts,
    params_from_meta,
    save_layout,
)
from tests.ocr.synth import degrade, make_form, random_homography

VARIANT_SEEDS = {"form_a": 0, "form_b": 1}


@pytest.fixture(scope="module")
def layouts_dir(tmp_path_factory: pytest.TempPathFactory) -> Path:
    root = tmp_path_factory.mktemp("layouts")
    for name, seed in VARIANT_SEEDS.items():
        image, mask = make_form(layout_seed=seed)
        save_layout(root / name, image, mask, {"source_scan_ids": ["s1", "s2"]})
    (root / "_work").mkdir()  # служебный каталог без эталона вариантом не считается
    return root


@pytest.fixture(scope="module")
def layouts(layouts_dir: Path) -> dict[str, Layout]:
    return load_layouts(layouts_dir)


def _scan(layout_seed: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(500 + seed)
    page, _ = make_form(layout_seed=layout_seed, fill_seed=seed)
    return degrade(page, random_homography(rng), rng)


def test_save_and_load_layout_roundtrip(tmp_path: Path) -> None:
    image, mask = make_form(layout_seed=0)
    meta = {"source_scan_ids": ["a", "b"], "align_params": {"min_score": 0.45, "unknown": 1}}
    save_layout(tmp_path / "кириллица" / "form_x", image, mask, meta)

    layout = load_layout(tmp_path / "кириллица" / "form_x")
    assert layout.name == "form_x"
    assert layout.reference.size == (image.shape[1], image.shape[0])
    assert np.array_equal(layout.reference.image, image)
    assert np.array_equal(layout.reference.mask, mask)
    assert layout.meta["source_scan_ids"] == ["a", "b"]
    assert layout.meta["width"] == image.shape[1]
    assert layout.meta["height"] == image.shape[0]
    # Порог из меты применён, неизвестный ключ проигнорирован, остальное — по умолчанию.
    assert layout.reference.params.min_score == 0.45
    assert layout.reference.params.min_inliers == AlignParams().min_inliers


def test_load_layout_without_mask_and_meta(tmp_path: Path) -> None:
    image, _ = make_form(layout_seed=0)
    save_layout(tmp_path / "v", image, None, {})
    (tmp_path / "v" / "meta.json").unlink()
    layout = load_layout(tmp_path / "v")
    assert layout.name == "v"
    assert int(layout.reference.mask.min()) == 255
    assert layout.reference.params == AlignParams()


def test_load_layout_missing_reference(tmp_path: Path) -> None:
    (tmp_path / "empty").mkdir()
    with pytest.raises(FileNotFoundError):
        load_layout(tmp_path / "empty")


def test_params_from_meta_keeps_base() -> None:
    base = AlignParams(min_inliers=55)
    params = params_from_meta({"align_params": {"min_score": 0.5}}, base)
    assert params.min_inliers == 55
    assert params.min_score == 0.5
    assert params_from_meta({}, base) == base


def test_load_layouts_lists_only_variants(layouts_dir: Path, layouts: dict[str, Layout]) -> None:
    assert list(layouts) == ["form_a", "form_b"]
    assert load_layouts(layouts_dir / "нет_такого") == {}
    meta = json.loads((layouts_dir / "form_a" / "meta.json").read_text(encoding="utf-8"))
    assert meta["name"] == "form_a"


@pytest.mark.parametrize(("name", "seed"), [("form_a", 3), ("form_b", 4), ("form_b", 5)])
def test_detect_variant_picks_own_form(layouts: dict[str, Layout], name: str, seed: int) -> None:
    match = detect_variant(_scan(VARIANT_SEEDS[name], seed), layouts)
    assert match.ok, match.result.reason
    assert match.variant == name
    assert set(match.results) == set(layouts)
    assert match.result is match.results[name]
    assert match.margin > 0.2
    other = next(n for n in layouts if n != name)
    assert not match.results[other].ok


def test_detect_variant_accepts_iterable(layouts: dict[str, Layout]) -> None:
    match = detect_variant(_scan(0, 6), list(layouts.values()))
    assert match.ok
    assert match.variant == "form_a"


def test_detect_variant_single_layout_margin_zero(layouts: dict[str, Layout]) -> None:
    match = detect_variant(_scan(0, 7), {"form_a": layouts["form_a"]})
    assert match.ok
    assert match.margin == 0.0


def _grouped(layouts: dict[str, Layout]) -> dict[str, Layout]:
    """Те же эталоны, но с буксирами в мете: form_a — буксир x, form_b и её копия — y."""
    a, b = layouts["form_a"], layouts["form_b"]
    return {
        "form_a": Layout("form_a", a.reference, {"tug_code": "x"}),
        "form_b": Layout("form_b", b.reference, {"tug_code": "y"}),
        "form_b2": Layout("form_b2", b.reference, {"tug_code": "y"}),
    }


def test_detect_variant_early_exit_skips_other_tugs(layouts: dict[str, Layout]) -> None:
    grouped = _grouped(layouts)
    match = detect_variant(_scan(0, 8), grouped)
    assert match.ok and match.variant == "form_a"
    assert list(match.results) == ["form_a"]  # чужие буксиры не проверялись
    assert match.margin == 0.0

    full = detect_variant(_scan(0, 8), grouped, confident_score=None)
    assert set(full.results) == set(grouped)
    assert full.variant == "form_a"


def test_detect_variant_checks_all_layouts_of_same_tug(layouts: dict[str, Layout]) -> None:
    grouped = _grouped(layouts)
    match = detect_variant(_scan(1, 9), grouped)
    # form_a проверена первой и не подошла; оба варианта буксира y проверены.
    assert set(match.results) == {"form_a", "form_b", "form_b2"}
    assert match.variant in ("form_b", "form_b2")

    hinted = detect_variant(_scan(1, 9), grouped, prefer="y")
    assert set(hinted.results) == {"form_b", "form_b2"}
    assert hinted.ok
    by_name = detect_variant(_scan(1, 9), grouped, prefer="form_b2")
    assert next(iter(by_name.results)) == "form_b2"


def test_detect_variant_wrong_hint_still_finds_form(layouts: dict[str, Layout]) -> None:
    match = detect_variant(_scan(0, 10), _grouped(layouts), prefer="y")
    assert match.ok and match.variant == "form_a"


def test_detect_variant_rejects_foreign_page(layouts: dict[str, Layout]) -> None:
    rng = np.random.default_rng(9)
    noise = rng.integers(0, 256, layouts["form_a"].reference.image.shape, dtype=np.uint8)
    match = detect_variant(noise, layouts)
    assert not match.ok
    assert match.variant in layouts


def test_detect_variant_requires_layouts() -> None:
    with pytest.raises(ValueError):
        detect_variant(np.zeros((10, 10), np.uint8), {})
