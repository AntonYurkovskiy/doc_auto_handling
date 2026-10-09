"""Тесты T17: модели номера, функции потерь, синтетика и помощники обучения.

Только синтетика: настоящие веса не нужны — бэкбон подменяется крошечной свёрточной сетью.
Модуль пропускается, если нет torch.
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import cv2
import numpy as np
import pytest

from ocr_lab import number_scores as ns

torch = pytest.importorskip("torch")

from torch import nn  # noqa: E402

from ocr_lab import number_models as nm  # noqa: E402
from ocr_lab import train_number as tn  # noqa: E402
from ocr_lab.predictions import VOUCHER_NUMBER, parse_prediction  # noqa: E402

CHANNELS = 8


def _tiny_features(ctc: bool) -> nn.Module:
    """Крошечная свёрточная часть: `(B,1,H,W)` → `(B,8,H/8,W/8)` или `(B,8,H/16,W/8)`."""
    stride = (2, 1) if ctc else (2, 2)
    return nn.Sequential(
        nn.Conv2d(1, CHANNELS, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(CHANNELS, CHANNELS, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(CHANNELS, CHANNELS, 3, stride=2, padding=1), nn.ReLU(),
        nn.Conv2d(CHANNELS, CHANNELS, 3, stride=stride, padding=1), nn.ReLU(),
    )


def _tiny_model(kind: str) -> nm.NumberNet:
    cfg = nm.NumberModelConfig(kind=kind, lstm_hidden=8, lstm_layers=1, dropout=0.0)
    cls = nm.NumberHeadsNet if kind == "heads" else nm.NumberCRNN
    return cls(cfg, features=_tiny_features(kind == "ctc"), channels=CHANNELS)


def _fake_bank() -> dict[int, list[np.ndarray]]:
    """Поддельные «глифы MNIST»: 28×28, белый фон, тёмная цифра cv2."""
    bank: dict[int, list[np.ndarray]] = {}
    for d in range(10):
        g = np.full((28, 28), 255, np.uint8)
        cv2.putText(g, str(d), (6, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.8, 0, 2)
        bank[d] = [g]
    return bank


# --- модели и потери ---------------------------------------------------------------------------


def test_heads_model_output_shape() -> None:
    model = _tiny_model("heads").eval()
    out = model(torch.rand(3, 1, *tn.TARGET_SIZE))
    assert out.shape == (3, 3, ns.NUM_CLASSES)


def test_crnn_output_shape_frames_along_width() -> None:
    model = _tiny_model("ctc").eval()
    out = model(torch.rand(2, 1, *tn.TARGET_SIZE))
    assert out.shape == (2, tn.TARGET_SIZE[1] // 8, ns.NUM_CLASSES)


def test_resnet_ctc_features_keep_width() -> None:
    """У CRNN на ResNet18 по ширине остаётся W/8 кадров (шаг layer3/layer4 — (2, 1))."""
    pytest.importorskip("torchvision")
    model = nm.build_number_model(nm.NumberModelConfig(kind="ctc")).eval()
    with torch.no_grad():
        out = model(torch.rand(1, 1, *tn.TARGET_SIZE))
    assert out.shape == (1, tn.TARGET_SIZE[1] // 8, ns.NUM_CLASSES)
    heads = nm.build_number_model(nm.NumberModelConfig(kind="heads")).eval()
    with torch.no_grad():
        assert heads(torch.rand(1, 1, *tn.TARGET_SIZE)).shape == (1, 3, ns.NUM_CLASSES)


def test_heads_loss_is_sum_of_cross_entropies() -> None:
    logits = torch.randn(4, 3, ns.NUM_CLASSES)
    targets = torch.tensor([ns.encode_heads(v) for v in (243, 43, 7, 100)])
    loss, value = nm.heads_loss(logits, targets)
    lp = torch.log_softmax(logits, dim=2)
    expected = -lp.gather(2, targets.unsqueeze(2)).squeeze(2).sum(dim=1).mean()
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)
    assert value == pytest.approx(float(expected), rel=1e-6)


def test_ctc_loss_matches_numpy_likelihood() -> None:
    torch.manual_seed(0)
    logits = torch.randn(3, 16, ns.NUM_CLASSES)
    values = [5, 243, 111]
    loss, _ = nm.ctc_loss(logits, values)
    expected = -np.mean([
        ns.score_candidates(logits[i].numpy(), [str(v)], kind="ctc")[str(v)]
        for i, v in enumerate(values)
    ])
    assert float(loss) == pytest.approx(expected, rel=1e-5)


def test_load_digits_backbone_copies_feature_weights(tmp_path: Path) -> None:
    src = _tiny_model("heads")
    ckpt = tmp_path / "best.pt"
    # Чекпоинт T16: ключи `features.*` плюс посторонние (головы DigitNet).
    state = {f"features.{k}": v for k, v in src.features.state_dict().items()}
    state["head_tens.weight"] = torch.zeros(11, 4)
    torch.save({"model": state}, ckpt)
    dst = _tiny_model("ctc")
    assert nm.load_digits_backbone(dst, ckpt) == len(src.features.state_dict())
    for (k, a), b in zip(src.features.state_dict().items(),
                         dst.features.state_dict().values(), strict=True):
        assert torch.equal(a, b), k


# --- данные и синтетика ------------------------------------------------------------------------


def test_render_synthetic_number_deterministic_and_inked() -> None:
    bank = _fake_bank()
    a = tn.render_synthetic_number(243, bank, np.random.default_rng(7))
    b = tn.render_synthetic_number(243, bank, np.random.default_rng(7))
    assert a.dtype == np.uint8 and a.ndim == 2
    assert 100 <= a.shape[0] <= 116 and 190 <= a.shape[1] <= 275
    assert np.array_equal(a, b)
    assert (a < 128).sum() > 50  # цифры нарисованы


def test_synth_value_ranges() -> None:
    rng = np.random.default_rng(0)
    values = [tn.synth_value(rng) for _ in range(2000)]
    assert min(values) >= 1 and max(values) <= 399
    three = sum(v >= 100 for v in values) / len(values)
    assert 0.5 < three < 0.7


def test_train_set_real_repeats_and_synthetic() -> None:
    images = [np.full((110, 250), 255, np.uint8) for _ in range(3)]
    ds = tn.NumberTrainSet(images, [12, 243, 7], augment=True, seed=1, size=tn.TARGET_SIZE,
                           real_repeats=2, bank=_fake_bank(), n_synth=4)
    assert len(ds) == 3 * 2 + 4
    x, v = ds[4]
    assert x.shape == (1, *tn.TARGET_SIZE) and v == 243
    xs, vs = ds[7]
    xs2, vs2 = ds[7]
    assert 1 <= vs <= 399 and vs == vs2 and torch.equal(xs, xs2)
    # Следующая «эпоха» того же элемента — другая синтетика.
    assert not torch.equal(ds[7][0], ds[7 + len(ds)][0])


def test_epoch_permutation_covers_all_with_offset() -> None:
    sampler = tn.EpochPermutation(10, seed=3)
    sampler.set_epoch(2)
    idx = list(sampler)
    assert sorted(idx) == list(range(20, 30))
    assert idx == list(sampler)


def _write_fixture(tmp_path: Path) -> tuple[Path, Path]:
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scan_id", "split", "tug_code", "year", "voucher_number"])
        w.writerows([["s1", "val", "k", "2026", "243"], ["s2", "val", "p", "2026", ""],
                     ["s3", "train", "k", "2025", "12"], ["s4", "val", "k", "2026", "5"],
                     ["s5 ", "val", "k", "2026", "77"]])
    index = tmp_path / "crops_index.csv"
    with index.open("w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["scan_id", "subfield", "path", "align_ok"])
        w.writerows([["s4", VOUCHER_NUMBER, "c/s4.png", "True"],
                     ["s1", VOUCHER_NUMBER, "c/s1.png", "True"],
                     ["s1", "left_base.day", "c/d.png", "True"],
                     ["s2", VOUCHER_NUMBER, "c/s2.png", "True"],
                     ["s3", VOUCHER_NUMBER, "c/s3.png", "True"],
                     ["s5 ", VOUCHER_NUMBER, "c/s5 .png", "True"]])
    return manifest, index


def test_number_crops_filters_split_and_truth(tmp_path: Path) -> None:
    manifest, index = _write_fixture(tmp_path)
    crops = tn.number_crops("val", manifest=manifest, index_csv=index, work_dir=tmp_path)
    assert [(c.scan_id, c.value) for c in crops] == [("s1", 243), ("s4", 5), ("s5", 77)]
    assert crops[0].path == tmp_path / "c" / "s1.png"
    # Пробел в конце `scan_id` (файл «223k .pdf»): истина срезает его, индекс — нет.
    assert crops[2].path == tmp_path / "c" / "s5 .png"
    all_crops = tn.number_crops("val", manifest=manifest, index_csv=index, work_dir=tmp_path,
                                with_truth=False)
    assert [c.scan_id for c in all_crops] == ["s1", "s2", "s4", "s5"]


# --- метрики и предсказания --------------------------------------------------------------------


def _confident_heads(values: list[int]) -> np.ndarray:
    out = np.zeros((len(values), 3, ns.NUM_CLASSES))
    for i, v in enumerate(values):
        for h, cls in enumerate(ns.encode_heads(v)):
            out[i, h, cls] = 9.0
    return out


def test_score_outputs_perfect_and_wrong() -> None:
    values = [243, 17, 5]
    m = tn.score_outputs(_confident_heads(values), values, "heads")
    assert m["top1"] == m["top5"] == m["rank1"] == 1.0
    assert m["mean_rank"] == 1.0 and m["nll"] < 0.01
    wrong = tn.score_outputs(_confident_heads([248, 17, 5]), values, "heads")
    assert wrong["top1"] == pytest.approx(2 / 3)
    assert wrong["mean_rank"] > 1.0
    assert tn.score_outputs(np.zeros((0, 3, 11)), [], "heads")["n"] == 0


def test_build_predictions_t13_format() -> None:
    crops = [tn.NumberCrop("s1", Path("a.png"), 243), tn.NumberCrop("s2", Path("b.png"), None)]
    preds = tn.build_predictions(crops, _confident_heads([243, 9]), "heads", "number_test")
    for pred in preds:
        parsed = parse_prediction(json.loads(json.dumps(pred.to_json())))
        assert len(parsed.fields[VOUCHER_NUMBER]) == tn.TOP_K
    assert preds[0].top1(VOUCHER_NUMBER) == 243
    assert preds[1].top1(VOUCHER_NUMBER) == 9


def test_train_config_roundtrip() -> None:
    cfg = tn.TrainConfig(run="x", model=nm.NumberModelConfig(kind="ctc", lstm_hidden=64),
                         synth=500)
    back = tn.TrainConfig.from_dict(json.loads(json.dumps(cfg.to_dict())))
    assert back == cfg


# --- обучение целиком на крошечной модели ------------------------------------------------------


def test_train_resume_and_predict_smoke(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Две эпохи с остановкой по времени, `--resume`, затем `predict` на крошечной модели."""
    rng = np.random.default_rng(0)
    crops = []
    for i, v in enumerate([12, 243, 7, 100, 55, 31]):
        path = tmp_path / f"c{i}.png"
        cv2.imwrite(str(path), rng.integers(0, 256, size=(110, 250), dtype=np.uint8))
        crops.append(tn.NumberCrop(f"s{i}", path, v))
    monkeypatch.setattr(tn, "number_crops", lambda split, **_: crops)
    monkeypatch.setattr(tn, "build_model", lambda cfg, init_from: _tiny_model(cfg.model.kind))
    monkeypatch.setattr(tn, "load_model", lambda run_dir, device=None: (
        _load_tiny(run_dir), tn.TrainConfig.from_dict(
            torch.load(run_dir / "best.pt", weights_only=False)["config"]), {}))
    cfg = tn.TrainConfig(run="r", model=nm.NumberModelConfig(kind="ctc", lstm_hidden=8,
                                                             lstm_layers=1),
                         epochs=2, batch_size=2, num_workers=0, init_digits=False)
    assert tn.train(cfg, runs_dir=tmp_path, max_minutes=0.0) == 3
    run_dir = tmp_path / "r"
    assert (run_dir / "last.pt").exists() and (run_dir / "best.pt").exists()
    with pytest.raises(SystemExit):
        tn.train(cfg, runs_dir=tmp_path)
    assert tn.train(cfg, runs_dir=tmp_path, resume=True) == 0
    rows = list(csv.DictReader((run_dir / "history.csv").open(encoding="utf-8")))
    assert [r["epoch"] for r in rows] == ["0", "1"]
    out = tn.predict(run_dir, "val")
    assert len(out.read_text(encoding="utf-8").splitlines()) == len(crops)
    npz = np.load(run_dir / "val_outputs.npz")
    assert npz["output"].shape[0] == len(crops) and str(npz["kind"]) == "ctc"


def _load_tiny(run_dir: Path) -> nm.NumberNet:
    state = torch.load(run_dir / "best.pt", weights_only=False)
    model = _tiny_model("ctc")
    model.load_state_dict(state["model"])
    return model.eval()


def test_stress_tensors_copies_and_truth(tmp_path: Path) -> None:
    path = tmp_path / "c.png"
    img = np.full((110, 250), 255, np.uint8)
    cv2.putText(img, "243", (60, 80), cv2.FONT_HERSHEY_SIMPLEX, 2.0, 0, 4)
    cv2.imwrite(str(path), img)
    crops = [tn.NumberCrop("s1", path, 243), tn.NumberCrop("s2", path, 17)]
    x, values = tn.stress_tensors(crops, tn.TARGET_SIZE, 3)
    assert x.shape == (6, 1, *tn.TARGET_SIZE)
    assert values == [243, 243, 243, 17, 17, 17]
    assert not torch.equal(x[0], x[1])  # копии различаются
    x2, _ = tn.stress_tensors(crops, tn.TARGET_SIZE, 3)
    assert torch.equal(x, x2)  # и детерминированы
