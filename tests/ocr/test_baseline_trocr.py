"""Тесты бейзлайна TrOCR (T14): текст→число, подготовка кропа, кандидаты, прогон на сплите.

Только синтетика. Модель и процессор TrOCR не грузятся: ``beam_predict``/``load_model``
подменяются, либо (в одном тесте) проверяются на фейковых объектах с контрактом
``transformers`` без реальных весов.
"""

from __future__ import annotations

import csv
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import pytest
from PIL import Image

from ocr_lab import baseline_trocr as bt
from ocr_lab.predictions import read_predictions

# --- text_to_int --------------------------------------------------------------------------


@pytest.mark.parametrize(
    "text,expected",
    [
        ("12", 12),
        ("007", 7),
        ("", None),
        ("—", None),
        ("l2", 12),  # l -> 1
        ("I9", 19),  # I -> 1
        ("O5", 5),   # O -> 0 (ведущий 0 не меняет число)
        ("S0", 50),  # S -> 5
        ("2 4", 24),  # пробел внутри — просто не цифра, цифры собираются подряд
        ("б3", 3),  # посторонняя буква игнорируется (не входит в минимальный набор замен)
    ],
)
def test_text_to_int(text: str, expected: int | None) -> None:
    assert bt.text_to_int(text) == expected


# --- prepare_image --------------------------------------------------------------------------


def test_pad_to_square_centers_on_white_background() -> None:
    image = Image.new("RGB", (100, 40), (10, 20, 30))
    padded = bt.pad_to_square(image)
    assert padded.size == (100, 100)
    # Углы — белый фон (паддинг), центр — исходный цвет.
    assert padded.getpixel((0, 0)) == (255, 255, 255)
    assert padded.getpixel((50, 50)) == (10, 20, 30)


def test_prepare_image_plain_returns_image_unchanged() -> None:
    image = Image.new("RGB", (100, 40))
    assert bt.prepare_image(image, "plain") is image


def test_prepare_image_padded_makes_square() -> None:
    image = Image.new("RGB", (100, 40))
    assert bt.prepare_image(image, "padded").size == (100, 100)


def test_prepare_image_unknown_variant_raises() -> None:
    with pytest.raises(ValueError):
        bt.prepare_image(Image.new("RGB", (10, 10)), "huge")


# --- to_candidates --------------------------------------------------------------------------


def test_to_candidates_merges_equal_values_and_sorts_desc() -> None:
    # "12" и "l2" (l -> 1) дают одно и то же значение 12 — вероятности складываются.
    beam = [("12", 0.5), ("l2", 0.2), ("99", 0.2), ("O9", 0.1)]
    cands = bt.to_candidates(beam)
    assert cands[0] == (12, pytest.approx(0.7))
    values = [v for v, _ in cands]
    assert values == sorted(values, key=lambda v: -dict(cands)[v])
    assert set(values) == {12, 99, 9}


def test_to_candidates_drops_empty_text_and_respects_top_k() -> None:
    beam = [("1", 0.4), ("", 0.3), ("2", 0.15), ("3", 0.1), ("4", 0.05)]
    cands = bt.to_candidates(beam, top_k=2)
    assert len(cands) == 2
    assert cands[0][0] == 1


def test_to_candidates_empty_beam_returns_empty() -> None:
    assert bt.to_candidates([]) == []


# --- beam_predict (фейковые процессор/модель, без реальных весов) --------------------------


class _FakeProcessor:
    """Минимальный контракт TrOCRProcessor, нужный ``beam_predict``."""

    def __call__(self, image: Image.Image, return_tensors: str = "pt") -> SimpleNamespace:
        import torch

        return SimpleNamespace(pixel_values=torch.zeros(1, 3, 2, 2))

    def batch_decode(self, sequences: object, skip_special_tokens: bool = True) -> list[str]:
        return ["12", "l2", "99", "12", "O9"]


class _FakeModel:
    """Минимальный контракт VisionEncoderDecoderModel.generate с beam search."""

    def generate(self, pixel_values: object, **kwargs: object) -> SimpleNamespace:
        import torch

        assert kwargs["num_return_sequences"] == 5
        assert kwargs["num_beams"] >= kwargs["num_return_sequences"]
        scores = torch.tensor([-0.1, -0.2, -0.5, -0.6, -0.9])
        return SimpleNamespace(
            sequences=torch.zeros(5, 1, dtype=torch.long), sequences_scores=scores
        )


def test_beam_predict_returns_softmax_pseudo_probabilities() -> None:
    result = bt.beam_predict(
        Image.new("RGB", (10, 10)), _FakeProcessor(), _FakeModel(), "cpu",
        num_return_sequences=5, max_new_tokens=8,
    )
    texts = [t for t, _ in result]
    probs = [p for _, p in result]
    assert texts == ["12", "l2", "99", "12", "O9"]
    assert probs == sorted(probs, reverse=True)  # оценки beam search уже убывают
    assert sum(probs) == pytest.approx(1.0, abs=1e-6)


def test_beam_predict_handles_missing_sequence_scores() -> None:
    class _NoScoreModel(_FakeModel):
        def generate(self, pixel_values: object, **kwargs: object) -> SimpleNamespace:
            res = super().generate(pixel_values, **kwargs)
            res.sequences_scores = None
            return res

    result = bt.beam_predict(
        Image.new("RGB", (10, 10)), _FakeProcessor(), _NoScoreModel(), "cpu",
    )
    assert len(result) == 5
    assert all(p == pytest.approx(0.2) for _, p in result)


# --- манифест и индекс кропов ----------------------------------------------------------------


def _write_png(path: Path, value: int = 200) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    img = np.full((20, 40), value, dtype=np.uint8)
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scan_id", "split"])
        writer.writeheader()
        writer.writerows(rows)


def _write_index(path: Path, rows: list[dict[str, str]]) -> None:
    cols = ["scan_id", "variant", "subfield", "path", "w", "h", "align_ok", "score",
            "inliers", "rotated180", "source"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        for row in rows:
            writer.writerow({c: row.get(c, "") for c in cols})


def test_split_scan_ids_filters_by_split(tmp_path: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"scan_id": "a", "split": "train"},
        {"scan_id": "b", "split": "val"},
        {"scan_id": "c", "split": "val"},
    ])
    assert bt.split_scan_ids("val", manifest) == ["b", "c"]
    assert bt.split_scan_ids("all", manifest) == ["a", "b", "c"]


def test_load_crop_paths_skips_failed_alignment(tmp_path: Path) -> None:
    index = tmp_path / "crops_index.csv"
    _write_index(index, [
        {"scan_id": "a", "subfield": "voucher_number", "path": "crops/voucher_number/a.png",
         "align_ok": "True"},
        {"scan_id": "b", "subfield": "voucher_number", "path": "crops/voucher_number/b.png",
         "align_ok": "False"},
    ])
    paths = bt.load_crop_paths(index, work_dir=tmp_path)
    assert set(paths) == {("a", "voucher_number")}
    assert paths[("a", "voucher_number")] == tmp_path / "crops/voucher_number/a.png"


def test_load_crop_paths_missing_file_returns_empty(tmp_path: Path) -> None:
    assert bt.load_crop_paths(tmp_path / "nope.csv", work_dir=tmp_path) == {}


# --- infer_split: прогон, пропуск готового, дозапись ----------------------------------------


@pytest.fixture()
def infer_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, Path]:
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, [
        {"scan_id": "s1", "split": "val"},
        {"scan_id": "s2", "split": "val"},
    ])
    index = tmp_path / "crops_index.csv"
    rows = []
    for sid in ("s1", "s2"):
        for sub in ("voucher_number", "left_base.day"):
            rel = f"crops/{sub}/{sid}.png"
            _write_png(tmp_path / rel)
            rows.append({"scan_id": sid, "subfield": sub, "path": rel, "align_ok": "True"})
    _write_index(index, rows)

    monkeypatch.setattr(bt, "load_model", lambda device=None: (None, None, "cpu"))
    monkeypatch.setattr(
        bt, "beam_predict",
        lambda image, processor, model, device, **kw: [("5", 0.9), ("6", 0.1)],
    )
    return {
        "manifest": manifest, "index_csv": index, "out": tmp_path / "out.jsonl",
        "work_dir": tmp_path,
    }


def test_infer_split_writes_predictions_for_all_subfields(infer_env: dict[str, Path]) -> None:
    out = bt.infer_split(
        "val", "plain", manifest=infer_env["manifest"], index_csv=infer_env["index_csv"],
        work_dir=infer_env["work_dir"], out_path=infer_env["out"],
    )
    preds = {p.scan_id: p for p in read_predictions(out)}
    assert set(preds) == {"s1", "s2"}
    for pred in preds.values():
        assert pred.source == "trocr_baseline_plain"
        assert set(pred.fields) == {"voucher_number", "left_base.day"}
        assert pred.fields["voucher_number"] == [(5, pytest.approx(0.9)), (6, pytest.approx(0.1))]


def test_infer_split_limit_then_resume_skips_done(infer_env: dict[str, Path]) -> None:
    first = bt.infer_split(
        "val", "plain", manifest=infer_env["manifest"], index_csv=infer_env["index_csv"],
        work_dir=infer_env["work_dir"], out_path=infer_env["out"], limit=1,
    )
    assert {p.scan_id for p in read_predictions(first)} == {"s1"}

    second = bt.infer_split(
        "val", "plain", manifest=infer_env["manifest"], index_csv=infer_env["index_csv"],
        work_dir=infer_env["work_dir"], out_path=infer_env["out"],
    )
    assert {p.scan_id for p in read_predictions(second)} == {"s1", "s2"}


def test_infer_split_no_remaining_scans_is_noop(infer_env: dict[str, Path]) -> None:
    bt.infer_split(
        "val", "plain", manifest=infer_env["manifest"], index_csv=infer_env["index_csv"],
        work_dir=infer_env["work_dir"], out_path=infer_env["out"],
    )
    before = infer_env["out"].read_text(encoding="utf-8")
    bt.infer_split(
        "val", "plain", manifest=infer_env["manifest"], index_csv=infer_env["index_csv"],
        work_dir=infer_env["work_dir"], out_path=infer_env["out"],
    )
    assert infer_env["out"].read_text(encoding="utf-8") == before


def test_predict_scan_skips_missing_crop(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(
        bt, "beam_predict",
        lambda image, processor, model, device, **kw: [("3", 1.0)],
    )
    png = tmp_path / "a.png"
    _write_png(png)
    crop_paths = {("s1", "voucher_number"): png}  # left_base.day отсутствует
    fields = bt.predict_scan("s1", "plain", crop_paths, None, None, "cpu")
    assert set(fields) == {"voucher_number"}
    assert fields["voucher_number"] == [(3, 1.0)]


# --- CLI --------------------------------------------------------------------------------------


def test_main_infer_calls_infer_split_with_parsed_args(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    calls: list[dict[str, object]] = []

    def fake_infer_split(split: str, variant: str, **kwargs: object) -> Path:
        calls.append({"split": split, "variant": variant, **kwargs})
        return tmp_path / "out.jsonl"

    monkeypatch.setattr(bt, "infer_split", fake_infer_split)
    rc = bt.main(["infer", "--split", "test", "--variant", "padded", "--limit", "5"])
    assert rc == 0
    assert calls == [{
        "split": "test", "variant": "padded", "device": None, "limit": 5,
        "max_minutes": None, "out_path": None,
    }]


def test_main_timing_calls_time_estimate(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        bt, "time_estimate",
        lambda n, variant, device=None: {
            "n": n, "device": "cpu", "seconds": 1.0, "seconds_per_crop": 1.0 / n
        },
    )
    assert bt.main(["timing", "--n", "10", "--variant", "plain"]) == 0
