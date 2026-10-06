"""Тесты лаборатории вариантов бланка (T07): метрики, обезличивание, статичный эталон.

В конце — тесты макетов боксов подполей и кропов (T08): `app.ocr.layouts`, `app.ocr.crops`.
"""

from __future__ import annotations

import json
from collections.abc import Callable
from dataclasses import replace
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
import pytest

from app.ocr import layouts as boxes
from app.ocr.align import AlignParams, build_reference
from app.ocr.crops import SEARCH_DY_PX, crop_box, crop_subfields, find_lines, locate_boxes
from app.ocr.crops import _fit_line_len as fit_line_len
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


# --- T08: макеты боксов подполей и кропы ---------------------------------------------------

#: Синтетический бланк для боксов: подчёркивания частей строки ``(x0, x1)`` и высоты строк.
SYN_SIZE = (1000, 700)
SYN_PARTS = {"day": (100, 200), "month": (230, 400), "hour": (500, 640), "minute": (700, 860)}
SYN_ROW_Y = {row: 220 + 100 * i for i, row in enumerate(boxes.DATE_ROWS)}
SYN_NUMBER = (300, 500, 100)


def _syn_box(name: str, line: tuple[int, int, int]) -> boxes.Box:
    x0, x1, y = line
    return boxes.Box(
        name=name,
        x0=x0 - 20,
        y0=y - 70,
        x1=x1 + 20,
        y1=y + 12,
        kind=boxes.EXPECTED_KIND[name],
        printed_by_template=name == boxes.VOUCHER_NUMBER or name.endswith((".day", ".month")),
        line=line,
    )


def _syn_layout() -> boxes.Layout:
    items = [_syn_box(boxes.VOUCHER_NUMBER, SYN_NUMBER)]
    for row, y in SYN_ROW_Y.items():
        for part, (x0, x1) in SYN_PARTS.items():
            items.append(_syn_box(f"{row}.{part}", (x0, x1, y)))
    return boxes.Layout("syn_v1", "s", SYN_SIZE, tuple(items))


def _line(layout: boxes.Layout, name: str) -> tuple[int, int, int]:
    line = layout.box(name).line
    assert line is not None
    return line


def _syn_page(
    layout: boxes.Layout,
    dx: int = 0,
    dy: int = 0,
    skip: tuple[str, ...] = (),
    ends: dict[str, tuple[int, int]] | None = None,
) -> np.ndarray:
    """Белая страница с подчёркиваниями макета, сдвинутыми на ``(dx, dy)``.

    ``skip`` — подполя без линии; ``ends`` — поправки концов линии ``(слева, справа)``.
    """
    page = np.full((SYN_SIZE[1], SYN_SIZE[0]), 255, np.uint8)
    for box in layout.boxes:
        if box.line is None or box.name in skip:
            continue
        x0, x1, y = box.line
        e0, e1 = (ends or {}).get(box.name, (0, 0))
        # Толщина 3 px: середина по толщине — ровно y + dy.
        page[y + dy - 1 : y + dy + 2, x0 + dx + e0 : x1 + dx + e1] = 0
    return page


def test_box_layouts_committed_for_all_variants() -> None:
    # Варианты бланка из T07: макеты боксов есть у каждого (T08 — Коммунар, T09 — Пионер).
    loaded = boxes.load_layouts()
    expected = {"kommunar_v1": "k", "pioneer_v1": "p", "pioneer_v2": "p"}
    assert set(expected) <= set(loaded), "нет макета варианта в app/ocr/layouts/"
    for name, tug_code in expected.items():
        layout = loaded[name]
        assert len(layout.boxes) == 17
        assert set(layout.names) == set(boxes.SUBFIELD_NAMES)
        assert layout.tug_code == tug_code
        assert layout.ref_size == (1654, 2340)
        assert all(box.line is not None for box in layout.boxes)


def test_box_layout_save_load_roundtrip(tmp_path: Path) -> None:
    layout = _syn_layout()
    path = tmp_path / "syn_v1.json"
    boxes.save_layout(layout, path)
    assert boxes.load_layout(path) == layout
    data = json.loads(path.read_text(encoding="utf-8"))
    assert [b["name"] for b in data["boxes"]] == list(boxes.SUBFIELD_NAMES)
    assert boxes.load_layouts(tmp_path) == {"syn_v1": layout}


@pytest.mark.parametrize(
    ("mutate", "message"),
    [
        (lambda d: d["boxes"][1].update(name="left_base.year"), "неизвестное подполе"),
        (lambda d: d["boxes"][2].update(name="left_base.day"), "дубль подполя"),
        (lambda d: d["boxes"][1].update(x1=5000), "выходит за эталон"),
        (lambda d: d["boxes"][1].update(y0=-1), "выходит за эталон"),
        (lambda d: d["boxes"][1].update(x1=50), "пустой бокс"),
        (lambda d: d["boxes"][0].update(kind="two_digit"), "ожидался 'number'"),
        (lambda d: d["boxes"][1].update(kind="letters"), "неизвестный вид"),
        (lambda d: d["boxes"].pop(), "нет подполей: finished_work.minute"),
        (lambda d: d["boxes"][1].update(x0=1.5), "ожидалось целое"),
        (lambda d: d["boxes"][1].update(line=[1, 2]), "line должен быть"),
        (lambda d: d["boxes"][1].update(line=[300, 200, 10]), "подчёркивание"),
        (lambda d: d["boxes"][1].update(printed_by_template="да"), "должен быть bool"),
        (lambda d: d["boxes"][1].pop("kind"), "нет ключа 'kind'"),
        (lambda d: d.update(ref_size=[1000]), "ref_size"),
    ],
)
def test_box_layout_validation_errors(
    mutate: Callable[[dict[str, Any]], object], message: str, tmp_path: Path
) -> None:
    data = boxes.layout_to_dict(_syn_layout())
    mutate(data)
    path = tmp_path / "bad.json"
    path.write_text(json.dumps(data), encoding="utf-8")
    with pytest.raises(boxes.LayoutError, match=message):
        boxes.load_layout(path)


def test_box_layout_rejects_non_json_and_duplicate_variant(tmp_path: Path) -> None:
    (tmp_path / "a.json").write_text("{не json", encoding="utf-8")
    with pytest.raises(boxes.LayoutError, match="не JSON"):
        boxes.load_layout(tmp_path / "a.json")
    layout = _syn_layout()
    boxes.save_layout(layout, tmp_path / "a.json")
    boxes.save_layout(layout, tmp_path / "b.json")
    with pytest.raises(boxes.LayoutError, match="уже загружен"):
        boxes.load_layouts(tmp_path)
    assert boxes.load_layouts(tmp_path / "нет") == {}


def test_crop_subfields_shapes_and_padding() -> None:
    layout = _syn_layout()
    page = np.arange(SYN_SIZE[0] * SYN_SIZE[1], dtype=np.uint32).reshape(SYN_SIZE[::-1])
    page = (page % 251).astype(np.uint8)
    crops = crop_subfields(page, layout, refine=False)
    assert set(crops) == set(boxes.SUBFIELD_NAMES)
    box = layout.box("left_base.hour")
    assert crops[box.name].shape == (box.height, box.width)
    assert np.array_equal(crops[box.name], page[box.y0 : box.y1, box.x0 : box.x1])
    padded = crop_subfields(page, layout, pad=7, names=[box.name], refine=False)
    assert list(padded) == [box.name]
    assert padded[box.name].shape == (box.height + 14, box.width + 14)
    assert np.array_equal(padded[box.name][7:-7, 7:-7], crops[box.name])


def test_crop_box_fills_outside_image() -> None:
    page = np.zeros((50, 80, 3), np.uint8)
    box = boxes.Box("left_base.day", 0, 40, 30, 50, "two_digit")
    crop = crop_box(page, box, pad=5)
    assert crop.shape == (20, 40, 3)
    assert (crop[:5, 5:] == 0).all()  # над боксом — изображение
    assert (crop[-5:] == 255).all()  # ниже края — заливка
    assert (crop[:, :5] == 255).all()  # левее края — заливка
    assert (crop[5:15, 5:35] == 0).all()
    assert (crop_box(page, box, pad=5, fill=128)[-1] == 128).all()
    with pytest.raises(ValueError, match="pad"):
        crop_box(page, box, pad=-1)


def test_crop_subfields_checks_size_and_names() -> None:
    layout = _syn_layout()
    with pytest.raises(ValueError, match="не совпадает с эталоном"):
        crop_subfields(np.zeros((10, 10), np.uint8), layout)
    with pytest.raises(KeyError, match="left_base.year"):
        crop_subfields(_syn_page(layout), layout, names=["left_base.year"])


def test_find_lines_ignores_short_and_vertical_strokes() -> None:
    page = np.full((100, 300), 255, np.uint8)
    page[50:53, 20:220] = 0
    page[20:23, 240:270] = 0  # короче MIN_LINE_PX
    page[10:90, 150:153] = 0  # вертикальный штрих
    lines = find_lines(page, offset=(5, 7))
    assert [(s.x0, s.x1, round(s.y)) for s in lines] == [(25, 225, 58)]


def test_locate_boxes_follows_shifted_lines() -> None:
    layout = _syn_layout()
    placed = locate_boxes(_syn_page(layout, dx=15, dy=6), layout)
    for box in layout.boxes:
        p = placed[box.name]
        assert p.source == "line", box.name
        assert (p.dx0, p.dx1, p.dy) == (15, 15, 6), box.name
        assert p.box == replace(box, x0=box.x0 + 15, x1=box.x1 + 15, y0=box.y0 + 6, y1=box.y1 + 6)
    # Без линий боксы остаются на месте.
    empty = locate_boxes(_syn_page(layout, skip=layout.names), layout)
    assert {p.source for p in empty.values()} == {"static"}
    assert all(p.box == layout.box(name) for name, p in empty.items())


def test_locate_boxes_fallbacks_column_and_row() -> None:
    layout = _syn_layout()
    # Минуты не найдены в одной строке — сдвиг минут других строк.
    placed = locate_boxes(_syn_page(layout, dx=-20, dy=4, skip=("arrived_base.minute",)), layout)
    p = placed["arrived_base.minute"]
    assert p.source == "column"
    assert (p.dx0, p.dx1, p.dy) == (-20, -20, 4)
    # Минут нет нигде — сдвиг ближайшего найденного бокса строки (часы).
    skip_all = tuple(f"{row}.minute" for row in boxes.DATE_ROWS)
    placed = locate_boxes(_syn_page(layout, dx=-20, dy=4, skip=skip_all), layout)
    p = placed["left_base.minute"]
    assert p.source == "row"
    assert (p.dx0, p.dx1, p.dy) == (-20, -20, 4)


def test_locate_boxes_edge_rules() -> None:
    layout = _syn_layout()
    ends = {
        "left_base.day": (0, -20),  # короткое подчёркивание дня: бокс сужается
        "left_base.hour": (0, 30),  # длинное подчёркивание часов: бокс расширяется
        "arrived_base.month": (25, 0),  # короче слева: бокс месяца не сужается
    }
    placed = locate_boxes(_syn_page(layout, ends=ends), layout)
    assert (placed["left_base.day"].dx0, placed["left_base.day"].dx1) == (0, -20)
    assert (placed["left_base.hour"].dx0, placed["left_base.hour"].dx1) == (0, 30)
    assert (placed["arrived_base.month"].dx0, placed["arrived_base.month"].dx1) == (0, 25)


def test_locate_boxes_ignores_handwriting_stroke_near_line() -> None:
    layout = _syn_layout()
    page = _syn_page(layout, dx=-30)
    x0, _, y = _line(layout, "finished_work.day")
    # Низ рукописной «2» — горизонтальный штрих 55 px чуть выше подчёркивания; его левый
    # конец ближе к эталонному краю, чем настоящий край линии.
    page[y - 10 : y - 7, x0 + 10 : x0 + 65] = 0
    p = locate_boxes(page, layout)["finished_work.day"]
    assert (p.dx0, p.dx1, p.dy) == (-30, -30, 0)


def test_locate_boxes_wide_search_for_shifted_header() -> None:
    layout = _syn_layout()
    page = _syn_page(layout, skip=(boxes.VOUCHER_NUMBER,))
    x0, x1, y = SYN_NUMBER
    # Шапка съехала на 60 px вверх и 100 px вправо: обычное окно её не видит.
    page[y - 61 : y - 58, x0 + 100 : x1 + 100] = 0
    p = locate_boxes(page, layout)[boxes.VOUCHER_NUMBER]
    assert p.source == "line_wide"
    assert (p.dx0, p.dx1, p.dy) == (100, 100, -60)


def test_locate_boxes_does_not_cross_neighbour_line() -> None:
    layout = _syn_layout()
    # Подчёркивание часов уехало вправо на 50 px и заходит в бокс минут (до линии минут
    # остаётся зазор 10 px, линии не сливаются).
    page = _syn_page(layout, skip=("left_base.hour",))
    x0, x1, y = _line(layout, "left_base.hour")
    page[y - 1 : y + 2, x0 + 50 : x1 + 50] = 0
    placed = locate_boxes(page, layout)
    hour, minute = placed["left_base.hour"].box, placed["left_base.minute"].box
    assert hour.x0 == layout.box("left_base.hour").x0 + 50
    # Левая сторона минут не заходит левее правого конца подчёркивания часов.
    assert minute.x0 == x1 + 50
    assert minute.x1 == layout.box("left_base.minute").x1


def test_locate_boxes_merged_lines_split_by_template_length() -> None:
    layout = _syn_layout()
    # Подчёркивания дня и месяца слились в одну линию 100…360: месяц уехал влево на 40 px
    # (Коммунар, строка «Начало работ»). У дня найден только левый конец, у месяца — правый.
    page = _syn_page(layout, skip=("left_base.day", "left_base.month"))
    y = _line(layout, "left_base.day")[2]
    page[y - 1 : y + 2, 100:360] = 0
    placed = locate_boxes(page, layout)
    day, month = placed["left_base.day"], placed["left_base.month"]
    assert (day.source, month.source) == ("line_left", "line_right")
    # Стык — по длине подчёркиваний эталона: день кончается на 100 + 100, месяц
    # начинается на 360 − 170. Без этого бокс месяца (170…380) захватывал бы хвост дня.
    assert (month.box.x0, month.box.x1) == (200, 380)
    assert (day.box.x0, day.box.x1) == (80, 190)


def test_fit_line_len_skips_line_at_neighbour_place() -> None:
    layout = _syn_layout()
    month = layout.box("left_base.month")
    y = _line(layout, "left_base.month")[2]
    # Линия длиной как подчёркивание месяца (170 px), но на месте дня (центр 170 при
    # ожидаемом центре дня 150 и месяца 315): так у Пионера месяц брал линию ненайденного
    # дня (2025_185p).
    page = np.full((SYN_SIZE[1], SYN_SIZE[0]), 255, np.uint8)
    page[y - 1 : y + 2, 85:255] = 0
    seg = fit_line_len(page, month, SEARCH_DY_PX)
    assert seg is not None and (seg.x0, seg.x1) == (85, 255)
    assert fit_line_len(page, month, SEARCH_DY_PX, expected=(315.0, [150.0, 570.0])) is None
    # Та же линия при ожидаемом месте месяца рядом — своя.
    found = fit_line_len(page, month, SEARCH_DY_PX, expected=(200.0, [100.0, 570.0]))
    assert found is not None


def test_locate_boxes_prefers_line_at_row_height() -> None:
    layout = _syn_layout()
    page = _syn_page(layout, skip=("left_base.hour",))
    x0, x1, y = _line(layout, "left_base.hour")
    # Подчёркивание часов уехало вправо на 50 px, а на месте эталонного, на 12 px выше, —
    # сплошной низ печатного слова (засечки «Time» у Courier) той же длины.
    page[y - 1 : y + 2, x0 + 50 : x1 + 50] = 0
    page[y - 13 : y - 10, x0 - 5 : x1 - 5] = 0
    p = locate_boxes(page, layout)["left_base.hour"]
    assert p.source == "line"
    # Правая сторона обрезана по левому концу подчёркивания минут (п. 7): 20 + 50 → 40.
    assert (p.dx0, p.dx1, p.dy) == (50, 40, 0)


def test_locate_boxes_drops_foreign_line_off_row_height() -> None:
    layout = _syn_layout()
    page = _syn_page(layout, skip=("left_base.hour",))
    x0, x1, y = _line(layout, "left_base.hour")
    # Своего подчёркивания нет, есть только линия засечек выше строки: она не берётся,
    # бокс идёт на запасной путь (часы других строк).
    page[y - 13 : y - 10, x0 - 5 : x1 - 5] = 0
    p = locate_boxes(page, layout)["left_base.hour"]
    assert p.source == "column"
    assert (p.dx0, p.dx1, p.dy) == (0, 0, 0)


def test_locate_boxes_month_without_line_follows_day() -> None:
    layout = _syn_layout()
    months = tuple(f"{row}.month" for row in boxes.DATE_ROWS)
    # Печатный месяц без подчёркивания («» _03_ 2025» у Пионера). Подчёркивание дня
    # длиннее эталонного на 70 px и длиной как подчёркивание месяца: поиск по длине не
    # должен отдать месяцу линию дня.
    page = _syn_page(layout, skip=months, ends={"left_base.day": (0, 70)})
    placed = locate_boxes(page, layout)
    day, month = placed["left_base.day"], placed["left_base.month"]
    assert (day.source, day.dx0, day.dx1) == ("line", 0, 70)
    # Месяц едет целиком за правым краем дня.
    assert month.source == "row"
    assert (month.dx0, month.dx1, month.dy) == (70, 70, 0)
    assert placed["arrived_base.month"].source == "row"
    assert (placed["arrived_base.month"].dx0, placed["arrived_base.month"].dx1) == (0, 0)


def test_locate_boxes_wide_search_by_ends_for_odd_length() -> None:
    layout = _syn_layout()
    page = _syn_page(layout, skip=(boxes.VOUCHER_NUMBER,))
    x0, x1, y = SYN_NUMBER
    # Шапка съехала на 50 px вверх, подчёркивание номера на 35 % длиннее эталонного:
    # поиск по длине его не берёт, берёт поиск по обоим концам.
    page[y - 51 : y - 48, x0 - 10 : x1 + 60] = 0
    p = locate_boxes(page, layout)[boxes.VOUCHER_NUMBER]
    assert p.source == "line_wide"
    assert (p.dx0, p.dx1, p.dy) == (-10, 60, -50)
