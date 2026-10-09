"""Тесты T18: `ocr_lab.export_onnx` — экспорт в ONNX, паритет с torch, `meta.json` (синтетика)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
pytest.importorskip("onnx")
pytest.importorskip("onnxruntime")

from torch import nn  # noqa: E402

from app.ocr import runtime as rt  # noqa: E402
from app.ocr.digit_values import value_log_probs  # noqa: E402
from ocr_lab import export_onnx as ex  # noqa: E402
from ocr_lab.models import DigitNet, ModelConfig  # noqa: E402
from ocr_lab.number_models import NumberCRNN, NumberModelConfig  # noqa: E402

DIGIT_SIZE = (64, 160)
NUMBER_SIZE = (64, 176)


@pytest.mark.parametrize("in_w", [5, 6, 7, 11, 22])
@pytest.mark.parametrize("out_w", [1, 4])
def test_static_pool_equals_adaptive_pool(in_w: int, out_w: int) -> None:
    x = torch.randn(3, 8, 2, in_w)
    want = nn.AdaptiveAvgPool2d((1, out_w))(x)
    got = ex.StaticAdaptivePool(in_w, out_w)(x)
    torch.testing.assert_close(got, want)


def _tiny_digit_net() -> DigitNet:
    torch.manual_seed(0)
    net = DigitNet(ModelConfig(), pretrained=False)
    # случайные, но не вырожденные BN-статистики и головы
    for m in net.modules():
        if isinstance(m, nn.BatchNorm2d):
            m.running_mean.normal_(0, 0.1)
            m.running_var.uniform_(0.5, 1.5)
    return net.eval()


def test_digits_export_dynamic_batch_and_parity(tmp_path: Path) -> None:
    import onnxruntime as ort  # noqa: PLC0415

    net = _tiny_digit_net()
    out = tmp_path / "model.onnx"
    ex.export_model(net, "digits", DIGIT_SIZE, out)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["image", "part_id"]
    for batch in (1, 5):
        x = torch.rand(batch, 1, *DIGIT_SIZE)
        pid = torch.arange(batch) % 4
        with torch.no_grad():
            want_t, want_u = net(x, pid)
        got_t, got_u = sess.run(None, {"image": x.numpy(), "part_id": pid.numpy()})
        assert got_t.shape == (batch, 11) and got_u.shape == (batch, 10)
        np.testing.assert_allclose(got_t, want_t.numpy(), atol=1e-4)
        np.testing.assert_allclose(got_u, want_u.numpy(), atol=1e-4)


def test_number_ctc_export_parity(tmp_path: Path) -> None:
    import onnxruntime as ort  # noqa: PLC0415

    from ocr_lab.number_models import build_features  # noqa: PLC0415

    torch.manual_seed(1)
    cfg = NumberModelConfig(kind="ctc")
    features, channels = build_features(cfg)
    net = NumberCRNN(cfg, features=features, channels=channels).eval()
    out = tmp_path / "model.onnx"
    ex.export_model(net, "number", NUMBER_SIZE, out)
    sess = ort.InferenceSession(str(out), providers=["CPUExecutionProvider"])
    for batch in (1, 3):
        x = torch.rand(batch, 1, *NUMBER_SIZE)
        with torch.no_grad():
            want = net(x).numpy()
        got = sess.run(None, {"image": x.numpy()})[0]
        assert got.shape == want.shape == (batch, 22, 11)
        np.testing.assert_allclose(got, want, atol=1e-4)


def _write_model_files(directory: Path, kind: str, temps: dict[str, float]) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / "config.json").write_text(json.dumps({
        "run": "r", "git_commit": "abc1234", "manifest_sha256": "m" * 16,
        "crops_index_sha256": "i" * 16, "torch": "2.7.1",
    }), encoding="utf-8")
    (directory / "best.pt").write_bytes(b"weights")
    (directory / "model.onnx").write_bytes(b"onnx")
    metric = {"n": 3, "acc": 0.9, "mean_conf": 0.95, "ece": 0.05, "nll": 0.3, "bins": [],
              "by_part": {}}
    (directory / "calibration.json").write_text(json.dumps({
        "model": directory.name, "fitted_on": "val", "n_fit": 3, "temperatures": temps,
        "metrics": {"val": {"before": metric, "after": {**metric, "ece": 0.01}}},
    }), encoding="utf-8")


def test_meta_contract_digits_roundtrips_through_runtime(tmp_path: Path) -> None:
    d = tmp_path / "digits_v0"
    _write_model_files(d, "digits", {"tens": 1.2, "units": 0.9})
    meta = ex.build_meta("digits", d, size=DIGIT_SIZE, trained_at="2026-10-09", parity=None)
    ex.write_meta(d, meta)
    loaded = rt.load_meta(d)
    assert loaded.size == DIGIT_SIZE and loaded.temperatures == {"tens": 1.2, "units": 0.9}
    assert loaded.normalize_in_graph and loaded.input_name == "image"
    raw = loaded.raw
    assert raw["train"]["git_commit"] == "abc1234"
    assert raw["train"]["manifest_sha256"] == "m" * 16
    assert raw["train"]["trained_at"] == "2026-10-09"
    assert raw["part_ids"] == {"day": 0, "month": 1, "hour": 2, "minute": 3}
    assert raw["opset"] == 17 and raw["outputs"][0]["classes"] == 11
    assert "bins" not in json.dumps(raw["calibration"])  # корзины диаграммы в meta не нужны


def test_meta_contract_number_kinds(tmp_path: Path) -> None:
    d = tmp_path / "number_v0"
    _write_model_files(d, "number", {"ctc": 1.7})
    meta = ex.build_meta("number", d, size=NUMBER_SIZE, trained_at="2026-10-09", parity=None,
                         number_kind="ctc")
    assert meta["kind"] == "ctc" and meta["outputs"][0]["blank"] == 0
    assert meta["temperatures"] == {"ctc": 1.7}
    heads = ex.build_meta("number", d, size=NUMBER_SIZE, trained_at="x", parity=None,
                          number_kind="heads")
    assert heads["outputs"][0]["heads"] == ["hundreds", "tens", "units"]


def test_meta_requires_calibration(tmp_path: Path) -> None:
    d = tmp_path / "digits_v0"
    d.mkdir()
    with pytest.raises(FileNotFoundError, match="calibrate"):
        ex.build_meta("digits", d, size=DIGIT_SIZE, trained_at="x", parity=None)


def test_runtime_matches_torch_end_to_end(tmp_path: Path) -> None:
    """Экспорт → meta.json → рантайм: распределения значений совпадают с torch (< 1e-4)."""
    net = _tiny_digit_net()
    d = tmp_path / "digits_v0"
    temps = {"tens": 1.3, "units": 0.85}
    _write_model_files(d, "digits", temps)
    ex.export_model(net, "digits", DIGIT_SIZE, d / "model.onnx")
    meta: dict[str, Any] = ex.build_meta("digits", d, size=DIGIT_SIZE, trained_at="x", parity=None)
    ex.write_meta(d, meta)
    model = rt.OnnxModel(d)

    rng = np.random.default_rng(0)
    names = ["left_base.day", "left_base.month", "started_work.hour", "arrived_base.minute"]
    crops = {n: rng.integers(0, 256, (95, 130 + 20 * i), dtype=np.uint8)
             for i, n in enumerate(names)}
    got = rt.predict_digits(crops, model=model, k=5)

    from ocr_lab.dataset import PART_ID, fit_with_aspect  # noqa: PLC0415

    for name, crop in crops.items():
        part = rt.part_of(name)
        x = torch.from_numpy(fit_with_aspect(crop, DIGIT_SIZE).astype(np.float32) / 255.0)
        with torch.no_grad():
            t, u = net(x[None, None], torch.tensor([PART_ID[part]]))
        values, logp = value_log_probs(t[0].numpy(), u[0].numpy(), part,
                                       t_tens=temps["tens"], t_units=temps["units"])
        order = np.argsort(-logp, kind="stable")[:5]
        for (v, p), i in zip(got[name], order, strict=True):
            assert v == int(values[i])
            assert abs(p - float(np.exp(logp[i]))) < 1e-4

