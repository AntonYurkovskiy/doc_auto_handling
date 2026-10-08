"""Тесты датасета двузначных подполей `ocr_lab.dataset` (T15) — только синтетика.

`torch` нужен только здесь (в `ocr_lab/`, не в `app/ocr/`, это разрешено `_common.md`), но
модуль пропускается, если его нет в окружении (`pytest.importorskip`).
"""

from __future__ import annotations

import csv
from pathlib import Path

import cv2
import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocr_lab import augment as aug  # noqa: E402
from ocr_lab import dataset as ds  # noqa: E402

ROWS = ("left_base", "arrived_base", "started_work", "finished_work")
PARTS = ("day", "month", "hour", "minute")


# --- encode_target -------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("value", "tens", "units", "ambiguous"),
    [
        (0, 0, 0, True),
        (7, 0, 7, True),
        (10, 1, 0, False),
        (24, 2, 4, False),
        (31, 3, 1, False),
        (59, 5, 9, False),
    ],
)
def test_encode_target(value: int, tens: int, units: int, ambiguous: bool) -> None:
    target = ds.encode_target(value)
    assert (target.tens, target.units, target.leading_zero_ambiguous) == (tens, units, ambiguous)


def test_encode_target_rejects_out_of_range() -> None:
    with pytest.raises(ValueError, match="недопустимое значение"):
        ds.encode_target(60)
    with pytest.raises(ValueError, match="недопустимое значение"):
        ds.encode_target(-1)


# --- fit_with_aspect -------------------------------------------------------------------------


def _crop_with_mark(h: int, w: int) -> np.ndarray:
    """Серый кроп h×w: белый фон с чёрным прямоугольником-«цифрой» по центру."""
    img = np.full((h, w), 255, np.uint8)
    cv2.rectangle(img, (w // 4, h // 4), (3 * w // 4, 3 * h // 4), 0, -1)
    return img


@pytest.mark.parametrize(("h", "w"), [(92, 150), (112, 267), (99, 99), (92, 99)])
def test_fit_with_aspect_preserves_proportions(h: int, w: int) -> None:
    crop = _crop_with_mark(h, w)
    out = ds.fit_with_aspect(crop, size=(64, 160))
    assert out.shape == (64, 160)
    # Масштаб общий для обеих осей (без сплющивания): вычисляем по меньшей из двух долей.
    scale = min(64 / h, 160 / w)
    expected_h = max(1, round(h * scale))
    expected_w = max(1, round(w * scale))
    # Непустая (не полностью белая) область изображения — вписанный кроп.
    ys, xs = np.where(out < 255)
    assert ys.size > 0 and xs.size > 0
    assert ys.max() - ys.min() + 1 <= expected_h + 1
    assert xs.max() - xs.min() + 1 <= expected_w + 1
    # Канва вокруг кропа остаётся белой (letterbox), размеры по центру совпадают.
    assert out[0, 0] == 255
    assert out[-1, -1] == 255


def test_fit_with_aspect_rejects_non_gray() -> None:
    with pytest.raises(ValueError, match="серое изображение"):
        ds.fit_with_aspect(np.zeros((10, 10, 3), np.uint8))


# --- детерминизм аугментаций по seed --------------------------------------------------------


def test_augment_deterministic_by_seed() -> None:
    crop = _crop_with_mark(100, 160)
    out1 = aug.augment(crop, seed=42)
    out2 = aug.augment(crop, seed=42)
    np.testing.assert_array_equal(out1, out2)


def test_augment_different_seed_differs() -> None:
    crop = _crop_with_mark(100, 160)
    out1 = aug.augment(crop, seed=1)
    out2 = aug.augment(crop, seed=2)
    assert not np.array_equal(out1, out2)


def test_augment_single_op_is_deterministic_and_known() -> None:
    crop = _crop_with_mark(100, 160)
    for op in aug.OP_ORDER:
        out1 = aug.augment(crop, seed=7, ops=(op,))
        out2 = aug.augment(crop, seed=7, ops=(op,))
        np.testing.assert_array_equal(out1, out2)
        assert out1.shape == crop.shape


def test_augment_unknown_op_rejected() -> None:
    with pytest.raises(ValueError, match="неизвестные техники"):
        aug.augment(_crop_with_mark(50, 80), seed=0, ops=("nope",))


def test_stroke_thickness_direction() -> None:
    """Положительная дельта утолщает (больше тёмных пикселей), отрицательная — утончает."""
    crop = _crop_with_mark(60, 60)
    base_ink = int((crop < 128).sum())
    thicker = aug.stroke_thickness(crop, 2)
    thinner = aug.stroke_thickness(crop, -2)
    assert int((thicker < 128).sum()) > base_ink
    assert int((thinner < 128).sum()) < base_ink
    np.testing.assert_array_equal(aug.stroke_thickness(crop, 0), crop)


# --- build_samples / TwoDigitFieldDataset ---------------------------------------------------


def _write_manifest(path: Path, rows: list[dict[str, object]]) -> None:
    cols = ["scan_id", "split", "tug_code", "year", "voucher_number", "has_hour24"]
    for r in ROWS:
        cols += [f"{r}_dt", f"{r}_day", f"{r}_month", f"{r}_year", f"{r}_hour", f"{r}_minute"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def _manifest_row(scan_id: str, split: str, *, started_hour_missing: bool = False) -> dict:
    row: dict[str, object] = {
        "scan_id": scan_id, "split": split, "tug_code": "k", "year": 2026,
        "voucher_number": 101, "has_hour24": False,
    }
    specs = {
        "left_base": (5, 7, 9, 10),
        "arrived_base": (5, 7, 13, 40),
        "started_work": (5, 7, 10, 0),
        "finished_work": (5, 7, 12, 30),
    }
    for r, (day, month, hour, minute) in specs.items():
        if started_hour_missing and r == "started_work":
            row[f"{r}_dt"] = ""
            row[f"{r}_hour"] = ""
            row[f"{r}_minute"] = ""
        else:
            row[f"{r}_dt"] = f"2026-{month:02d}-{day:02d}T{hour:02d}:{minute:02d}"
            row[f"{r}_hour"] = hour
            row[f"{r}_minute"] = minute
        row[f"{r}_day"] = day
        row[f"{r}_month"] = month
        row[f"{r}_year"] = 2026
    return row


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())


@pytest.fixture
def env(tmp_path: Path) -> dict[str, Path | str]:
    """Манифест с двумя сканами (train/val) + индекс кропов с разными исключениями.

    `2026_1k` — train, все подполя валидны.
    `2026_2k` — val, у started_work нет времени (исключается целиком).
    В индекс также добавлены: подполе с `align_ok=False`, подполе `voucher_number`
    (не входит в датасет) и подполе с пометкой проблемы в `crops_qc.csv`.
    """
    manifest = tmp_path / "manifest.csv"
    _write_manifest(
        manifest,
        [
            _manifest_row("2026_1k", "train"),
            _manifest_row("2026_2k", "val", started_hour_missing=True),
        ],
    )

    crops_dir = tmp_path / "crops"
    unaligned_subfield = "arrived_base.hour"  # для 2026_1k этот кроп не прошёл выравнивание
    index_rows = []
    for scan_id in ("2026_1k", "2026_2k"):
        for subfield in ds.SUBFIELDS:
            if scan_id == "2026_1k" and subfield == unaligned_subfield:
                continue
            h, w = (92, 130) if subfield.endswith("day") else (92, 170)
            _write_png(crops_dir / subfield / f"{scan_id}.png", _crop_with_mark(h, w))
            path = f"crops/{subfield}/{scan_id}.png"
            index_rows.append(
                {"scan_id": scan_id, "variant": "k", "subfield": subfield, "path": path,
                 "w": w, "h": h, "align_ok": "True", "score": "0.99", "inliers": "500",
                 "rotated180": "False", "source": "line"}
            )
    # Бокс, который не прошёл выравнивание — должен быть исключён (на практике T10 такую
    # строку в индекс не пишет вовсе, но исключение поддержано на всякий случай).
    index_rows.append(
        {"scan_id": "2026_1k", "variant": "k", "subfield": unaligned_subfield,
         "path": "crops/bad.png", "w": 10, "h": 10, "align_ok": "False", "score": "0",
         "inliers": "0", "rotated180": "False", "source": "line"}
    )
    # Номер ваучера — не входит в двузначные подполя.
    _write_png(crops_dir / "voucher_number" / "2026_1k.png", _crop_with_mark(110, 250))
    index_rows.append(
        {"scan_id": "2026_1k", "variant": "k", "subfield": "voucher_number",
         "path": "crops/voucher_number/2026_1k.png", "w": 250, "h": 110, "align_ok": "True",
         "score": "0.99", "inliers": "500", "rotated180": "False", "source": "line"}
    )
    index_csv = tmp_path / "crops_index.csv"
    with index_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(index_rows[0]))
        writer.writeheader()
        writer.writerows(index_rows)

    qc_csv = tmp_path / "crops_qc.csv"
    with qc_csv.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["subfield", "scan_id", "problem"])
        writer.writerow(["finished_work.minute", "2026_1k", "cut_digit"])

    return {
        "manifest": manifest, "index_csv": index_csv, "qc_csv": qc_csv,
        "work_dir": tmp_path,
    }


def test_build_samples_filters_split_align_qc_and_missing_truth(env: dict) -> None:
    samples = ds.build_samples(
        split="train", manifest=env["manifest"], index_csv=env["index_csv"],
        qc_csv=env["qc_csv"], work_dir=env["work_dir"],
    )
    subfields = {s.subfield for s in samples}
    scan_ids = {s.scan_id for s in samples}
    assert scan_ids == {"2026_1k"}  # второй скан — val, не train
    assert "voucher_number" not in subfields  # не двузначное подполе
    assert "arrived_base.hour" not in subfields  # не прошёл выравнивание
    assert "finished_work.minute" not in subfields  # помечено в crops_qc.csv
    assert len(samples) == len(ds.SUBFIELDS) - 2  # все 16 минус невыровненный и qc-исключение


def test_build_samples_excludes_only_subfields_without_truth(env: dict) -> None:
    """У `2026_2k` пропало время (час/минута) только у `started_work` — остальные
    12 двузначных подполей и день/месяц `started_work` остаются в датасете."""
    samples = ds.build_samples(
        split="val", manifest=env["manifest"], index_csv=env["index_csv"],
        qc_csv=env["qc_csv"], work_dir=env["work_dir"],
    )
    subfields = {s.subfield for s in samples}
    assert {s.scan_id for s in samples} == {"2026_2k"}
    assert "started_work.hour" not in subfields
    assert "started_work.minute" not in subfields
    assert "started_work.day" in subfields
    assert "started_work.month" in subfields
    assert len(samples) == len(ds.SUBFIELDS) - 2


def test_build_samples_missing_time_recovers_when_truth_added_back(env: dict) -> None:
    import pandas as pd

    manifest = pd.read_csv(env["manifest"])
    manifest.loc[manifest["scan_id"] == "2026_2k", "started_work_hour"] = 10
    manifest.loc[manifest["scan_id"] == "2026_2k", "started_work_minute"] = 0
    manifest.loc[manifest["scan_id"] == "2026_2k", "started_work_dt"] = "2026-07-05T10:00"
    manifest.to_csv(env["manifest"], index=False)

    samples = ds.build_samples(
        split="val", manifest=env["manifest"], index_csv=env["index_csv"],
        qc_csv=env["qc_csv"], work_dir=env["work_dir"],
    )
    assert {s.scan_id for s in samples} == {"2026_2k"}
    assert len(samples) == len(ds.SUBFIELDS)


def test_dataset_getitem_shape_and_targets(env: dict) -> None:
    dataset = ds.TwoDigitFieldDataset(
        split="train", manifest=env["manifest"], index_csv=env["index_csv"],
        qc_csv=env["qc_csv"], work_dir=env["work_dir"],
    )
    assert len(dataset) > 0
    tensor, targets = dataset[0]
    assert tensor.shape == (1, 64, 160)
    assert tensor.dtype == torch.float32
    assert 0.0 <= float(tensor.min()) and float(tensor.max()) <= 1.0
    assert targets["part_id"] in range(4)
    assert 0 <= targets["units"] <= 9
    assert 0 <= targets["tens"] <= ds.TENS_EMPTY
    assert isinstance(targets["leading_zero_ambiguous"], bool)
    # День 5 и месяц 7 из фикстуры — однозначно неоднозначны (значение < 10).
    day_sample = next(s for s in dataset.samples if s.subfield == "left_base.day")
    assert day_sample.value == 5
    target = ds.encode_target(day_sample.value)
    assert target.leading_zero_ambiguous is True


def test_dataset_augment_changes_pixels(env: dict) -> None:
    plain = ds.TwoDigitFieldDataset(
        split="train", manifest=env["manifest"], index_csv=env["index_csv"],
        qc_csv=env["qc_csv"], work_dir=env["work_dir"], augment=False,
    )
    augmented = ds.TwoDigitFieldDataset(
        split="train", manifest=env["manifest"], index_csv=env["index_csv"],
        qc_csv=env["qc_csv"], work_dir=env["work_dir"], augment=True, seed=0,
    )
    tensor_plain, _ = plain[0]
    tensor_aug, _ = augmented[0]
    assert not torch.equal(tensor_plain, tensor_aug)


# --- part_weights / make_balanced_sampler ---------------------------------------------------


def test_part_weights_balance_uneven_counts() -> None:
    part_ids = [0, 0, 0, 1]  # «день» ×3, «месяц» ×1 — месяц должен весить втрое больше
    weights = ds.part_weights(part_ids)
    assert weights == pytest.approx([1 / 3, 1 / 3, 1 / 3, 1.0])


def test_make_balanced_sampler_draws_expected_length(env: dict) -> None:
    dataset = ds.TwoDigitFieldDataset(
        split="train", manifest=env["manifest"], index_csv=env["index_csv"],
        qc_csv=env["qc_csv"], work_dir=env["work_dir"],
    )
    sampler = ds.make_balanced_sampler(dataset, seed=0)
    drawn = list(sampler)
    assert len(drawn) == len(dataset)
    assert all(0 <= i < len(dataset) for i in drawn)


# --- синтетика MNIST (инфраструктура, без реальной загрузки) --------------------------------


def _fake_glyph(digit: int, h: int = 28, w: int = 20) -> np.ndarray:
    img = np.full((h, w), 255, np.uint8)
    cv2.putText(img, str(digit), (2, h - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.7, 0, 2, cv2.LINE_AA)
    return img


def test_paste_digit_pair_with_and_without_tens() -> None:
    units = _fake_glyph(7)
    tens = _fake_glyph(2)
    both = ds.paste_digit_pair(tens, units)
    assert both.shape[0] == units.shape[0]
    assert both.shape[1] == tens.shape[1] + ds.MNIST_DIGIT_GAP + units.shape[1]

    only_units = ds.paste_digit_pair(None, units)
    np.testing.assert_array_equal(only_units, units)


def test_synthetic_digits_dataset_targets() -> None:
    bank = {d: [_fake_glyph(d)] for d in range(10)}
    dataset = ds.SyntheticDigitsDataset(bank, n=20, seed=0, empty_tens_prob=1.0)
    tensor, targets = dataset[0]
    assert tensor.shape == (1, *ds.TARGET_SIZE)
    assert targets["tens"] == ds.TENS_EMPTY
    assert targets["leading_zero_ambiguous"] is True

    dataset_full = ds.SyntheticDigitsDataset(bank, n=20, seed=0, empty_tens_prob=0.0)
    _tensor, targets_full = dataset_full[0]
    assert targets_full["leading_zero_ambiguous"] is False
    assert 0 <= targets_full["tens"] <= 9


def test_synthetic_digits_dataset_requires_full_bank() -> None:
    with pytest.raises(ValueError, match="все классы"):
        ds.SyntheticDigitsDataset({0: [_fake_glyph(0)]}, n=1)
