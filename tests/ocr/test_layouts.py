"""Тесты лаборатории вариантов бланка (T07): метрики, обезличивание, статичный эталон."""

from __future__ import annotations

import cv2
import numpy as np
import pandas as pd
import pytest

from app.ocr.align import AlignParams, build_reference
from ocr_lab import layouts as lab
from tests.ocr.synth import PAGE_H, PAGE_W, make_form


def _shift(img: np.ndarray, dx: float, dy: float) -> np.ndarray:
    M = np.array([[1.0, 0.0, dx], [0.0, 1.0, dy]])
    return cv2.warpAffine(img, M, (img.shape[1], img.shape[0]), borderValue=255)


@pytest.mark.parametrize("angle", [-1.5, 0.0, 1.0])
def test_estimate_skew_recovers_rotation(angle: float) -> None:
    form, _ = make_form(layout_seed=0)
    small = cv2.resize(form, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
    h, w = small.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.0)
    rotated = cv2.warpAffine(small, M, (w, h), borderValue=255)
    # Оценка — поворот, который выпрямляет строки, то есть обратный сделанному.
    assert lab.estimate_skew(rotated < 128) == pytest.approx(-angle, abs=0.25)


def test_page_metrics_values_and_profiles() -> None:
    form, _ = make_form(layout_seed=0, fill_seed=1)
    metrics, rows, cols = lab.page_metrics(form)
    assert metrics["contrast"] > 100
    assert 0 < metrics["ink"] < 0.2
    assert abs(metrics["skew_deg"]) <= 0.25
    assert metrics["sharpness"] > 0
    assert rows.shape == (round(PAGE_H * 0.25),)
    assert cols.shape == (round(PAGE_W * 0.25),)

    blank, _, _ = lab.page_metrics(np.full((400, 300), 255, np.uint8))
    assert blank["contrast"] == 0.0


def test_profile_shift_finds_offset() -> None:
    rng = np.random.default_rng(0)
    base = rng.random(200)
    shift, corr = lab.profile_shift(np.roll(base, 7), base, max_shift=10)
    assert shift == 7
    assert corr == pytest.approx(1.0)


def test_pick_candidates_prefers_typical_and_contrast() -> None:
    rows = []
    for i in range(40):
        rows.append(
            {
                "scan_id": f"s{i:02d}", "tug_code": "k", "contrast": 100.0 + i,
                "sharpness": 500.0 + i, "skew_deg": 0.0, "shift_y": 0.0, "shift_x": 0.0,
                "corr_y": 0.6,
            }
        )
    rows[39]["skew_deg"] = 2.0  # самый контрастный, но с наклоном — не кандидат
    rows[38]["shift_y"] = 0.05  # сдвинут — не кандидат
    picked = lab.pick_candidates(pd.DataFrame(rows), n=5)
    assert list(picked["scan_id"]) == ["s37", "s36", "s35", "s34", "s33"]


def test_pick_candidates_relaxes_filter_when_few() -> None:
    rows = [
        {
            "scan_id": f"s{i}", "tug_code": "p", "contrast": 100.0 + i, "sharpness": 1.0,
            "skew_deg": 1.0, "shift_y": 0.0, "shift_x": 0.0, "corr_y": 0.5,
        }
        for i in range(4)
    ]
    assert len(lab.pick_candidates(pd.DataFrame(rows), n=3)) == 3


def test_anonymize_whitens_zones_in_page_coordinates() -> None:
    page = np.zeros((1000, 800), np.uint8)
    out = lab.anonymize(page, "kommunar_v1", margin=0.0)
    assert out is not page and int(page.max()) == 0
    for x0, y0, x1, y1 in lab.SENSITIVE_ZONES["kommunar_v1"]:
        cx, cy = int((x0 + x1) / 2 * 800), int((y0 + y1) / 2 * 1000)
        assert out[min(cy, 999), min(cx, 799)] == 255
    # Блок строк дат остаётся видимым.
    assert out[550, 400] == 0
    assert out[550, 10] == 0


def test_anonymize_union_of_variants_covers_both() -> None:
    page = np.zeros((1000, 800), np.uint8)
    out = lab.anonymize(page, ["pioneer_v1", "pioneer_v2"], margin=0.0)
    for name in ("pioneer_v1", "pioneer_v2"):
        assert np.all(lab.anonymize(page, name, margin=0.0)[out == 0] == 0)


def test_anonymize_with_homography_follows_scan() -> None:
    ref_size = (800, 1000)
    scan = np.zeros((1000, 800), np.uint8)
    # Скан сдвинут на (+40, +60) относительно эталона: H (скан → эталон) сдвигает обратно.
    H = np.array([[1.0, 0.0, -40.0], [0.0, 1.0, -60.0], [0.0, 0.0, 1.0]])
    out = lab.anonymize(scan, "kommunar_v1", H, ref_size, margin=0.0)
    plain = lab.anonymize(scan, "kommunar_v1", margin=0.0)
    inner = (slice(100, 900), slice(100, 700))
    assert np.array_equal(out[inner], _shift(plain, 40, 60)[inner])
    # Нижняя зона (подписи) уходит за край эталона и закрывает скан до самого низа.
    assert np.all(out[900:, :] == 255)
    with pytest.raises(ValueError):
        lab.anonymize(scan, "kommunar_v1", H)


def test_pixelate_removes_fine_detail() -> None:
    form, _ = make_form(layout_seed=0, fill_seed=2)
    out = lab.pixelate(form, blocks=40)
    assert out.shape == form.shape
    block = form.shape[1] // 40
    assert len(np.unique(out[:block, :block])) == 1
    assert len(np.unique(out)) <= 40 * 60


def test_build_static_removes_handwriting() -> None:
    clean, field_mask = make_form(layout_seed=0)
    pages = [make_form(layout_seed=0, fill_seed=seed)[0] for seed in range(9)]
    reference, mask = lab.build_static(pages, zones=[])
    fields = field_mask == 0
    # В полях рукописи нет ни в эталоне, ни в маске; печать формы осталась.
    extra_ink = (reference < 128) & (clean >= 128)
    assert float(extra_ink[fields].mean()) < 0.001
    near_print = cv2.dilate((clean < 128).astype(np.uint8), np.ones((15, 15), np.uint8)) > 0
    assert float((mask[fields & ~near_print] > 0).mean()) < 0.001
    static_ink = (clean < 128) & ~fields
    assert float((mask[static_ink] > 0).mean()) > 0.99
    assert float((np.abs(reference.astype(int) - clean)[~fields] > 60).mean()) < 0.002


def test_build_static_cleans_frequent_text_in_zones() -> None:
    clean, _ = make_form(layout_seed=0)
    stamped = clean.copy()
    cv2.putText(stamped, "AGENT", (700, 1200), cv2.FONT_HERSHEY_DUPLEX, 2.0, 0, 4)
    pages = [stamped] * 9 + [clean]  # «частый агент»: тёмный в 90 % сканов
    zone = (0.35, 0.45, 0.95, 0.56)
    loose_ref, loose_mask = lab.build_static(pages, zones=[])
    strict_ref, strict_mask = lab.build_static(pages, zones=[zone])
    text = (stamped < 128) & (clean >= 128)
    assert float((loose_ref[text] < 128).mean()) > 0.9
    assert float((loose_mask[text] > 0).mean()) > 0.9
    # В зоне переменного текста эталон белый, а маска пустая.
    assert not (strict_ref[text] < 255).any()
    assert not (strict_mask[text] > 0).any()
    # Вне зоны всё как было.
    y0 = round(0.56 * PAGE_H) + 10
    assert np.array_equal(strict_ref[y0:], loose_ref[y0:])
    assert np.array_equal(strict_mask[y0:], loose_mask[y0:])


def test_pick_sources_limits_one_month() -> None:
    ranked = pd.DataFrame({"scan_id": [f"s{i}" for i in range(12)]})
    month = {f"s{i}": "2025-3" if i < 8 else f"2025-{i}" for i in range(12)}
    picked = lab.pick_sources(ranked, month, n=5)
    assert picked == ["s0", "s1", "s2", "s8", "s9", "s10", "s11"]


def test_anchor_residuals_measure_shift() -> None:
    clean, _ = make_form(layout_seed=0)
    _, mask = lab.build_static([clean] * 3, zones=[])
    anchors = lab.find_anchors(clean, mask)
    assert len(anchors) >= 8
    zero = lab.anchor_residuals(clean, clean, anchors)
    assert len(zero) == len(anchors)
    assert max(zero) == 0.0
    moved = lab.anchor_residuals(_shift(clean, 4, -3), clean, anchors)
    assert len(moved) >= 0.8 * len(anchors)
    assert float(np.median(moved)) == pytest.approx(5.0, abs=0.01)
    blank = lab.anchor_residuals(np.full_like(clean, 255), clean, anchors)
    assert blank == []


def test_h_string_roundtrip_and_result_row() -> None:
    H = np.array([[1.01, 0.002, -3.5], [0.001, 0.99, 7.25], [1e-6, -2e-6, 1.0]])
    restored = lab.h_from_str(lab._h_to_str(H))
    assert restored is not None
    assert np.allclose(restored, H, rtol=1e-8)
    assert lab.h_from_str("") is None

    form, mask = make_form(layout_seed=0)
    ref = build_reference(form, mask, name="synth")
    from app.ocr.align import align

    row = lab.result_row("scan_1", "synth", align(form, ref))
    assert row["scan_id"] == "scan_1" and row["variant"] == "synth"
    assert row["ok"] is True and row["has_warp"] is True
    assert float(row["score"]) > 0.9
    assert row["H"]


def test_ok_mask_applies_thresholds() -> None:
    df = pd.DataFrame(
        {
            "has_warp": [True, True, True, False, True],
            "inliers": [100, 100, 30, 100, 100],
            "inlier_ratio": [0.5, 0.5, 0.5, 0.5, 0.5],
            "reproj_err": [1.0, 1.0, 1.0, 1.0, 5.0],
            "score": [0.7, 0.5, 0.7, 0.7, 0.7],
        }
    )
    assert list(lab.ok_mask(df, AlignParams())) == [True, False, False, False, False]
    assert list(lab.ok_mask(df, AlignParams(min_score=0.4))) == [True, True, False, False, False]


def test_own_and_best_rows_pick_by_tug() -> None:
    manifest = pd.DataFrame({"scan_id": ["a", "b"], "tug_code": ["k", "p"]})
    rows = []
    for scan_id, scores in (("a", (0.3, 0.6, 0.2)), ("b", (0.1, 0.4, 0.7))):
        for variant, score in zip(("kommunar_v1", "pioneer_v1", "pioneer_v2"), scores, strict=True):
            rows.append(
                {"scan_id": scan_id, "variant": variant, "has_warp": True, "score": score,
                 "inliers": 100}
            )
    df = pd.DataFrame(rows)
    own = lab.own_rows(df, manifest)
    assert own.loc["a", "variant"] == "kommunar_v1"  # свой буксир, хотя чужой бланк лучше
    assert own.loc["b", "variant"] == "pioneer_v2"
    best = lab.best_rows(df)
    assert best.loc["a", "variant"] == "pioneer_v1"
    assert best.loc["b", "variant"] == "pioneer_v2"


def test_zones_and_bands_defined_for_every_variant() -> None:
    names = {v.name for v in lab.VARIANTS}
    assert names == set(lab.SENSITIVE_ZONES) == set(lab.OVERLAY_BANDS)
    assert set(lab.variants_for_tug("p")) == {"pioneer_v1", "pioneer_v2"}
    for rects in lab.SENSITIVE_ZONES.values():
        for x0, y0, x1, y1 in rects:
            assert 0.0 <= x0 < x1 <= 1.0 and 0.0 <= y0 < y1 <= 1.0


def test_block_fit_row_separates_bands() -> None:
    clean, _ = make_form(layout_seed=0)
    _, mask = lab.build_static([clean] * 3, zones=[])
    layout = lab.Layout("synth", build_reference(clean, mask, name="synth"))
    head, date = (0.0, 0.05, 1.0, 0.45), (0.0, 0.55, 1.0, 0.95)
    anchors = {
        "head": lab.find_anchors(clean, mask, head, lab.BLOCK_GRID),
        "date": lab.find_anchors(clean, mask, date, lab.BLOCK_GRID),
    }
    h = clean.shape[0]
    for key, (_, y0, _, y1) in zip(("head", "date"), (head, date), strict=True):
        assert anchors[key], key
        # Окна целиком внутри своей полосы.
        assert all(y0 * h <= y and y + lab.ANCHOR_SIZE <= y1 * h for _, y in anchors[key])

    # Нижняя половина страницы съехала на 10 px вниз: шапка на месте, блок дат — нет.
    warped = clean.copy()
    cut = h // 2
    warped[cut:] = _shift(clean, 0, 10)[cut:]
    row = lab.block_fit_row("s1", "synth", warped, layout, anchors)
    assert row["head_n"] >= 3 and float(row["head_med"]) == 0.0
    assert row["date_n"] >= 3 and float(row["date_med"]) == pytest.approx(10.0, abs=0.5)
