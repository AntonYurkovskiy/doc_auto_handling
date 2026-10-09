"""Тесты T18: `app/ocr/runtime.py` без реальных моделей.

Сессию onnxruntime подменяет заглушка; отдельный тест собирает крошечную ONNX-модель через
`onnx.helper` и прогоняет её настоящим onnxruntime.
"""

from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from app.ocr import runtime as rt
from app.ocr.digit_values import value_log_probs
from app.ocr.number_scores import score_candidates, top_numbers

SIZE = (8, 16)


def write_model_dir(
    root: Path, name: str, task: str, *, temperatures: dict[str, float] | None = None,
    in_graph: bool = True, extra: dict[str, Any] | None = None, version: int = 1,
) -> Path:
    """Каталог модели с `meta.json` (и пустым `model.onnx`) по контракту T18."""
    directory = root / name
    directory.mkdir(parents=True)
    meta: dict[str, Any] = {
        "format_version": version, "name": name, "task": task, "onnx": "model.onnx",
        "input": {"name": "image", "size": list(SIZE), "background": 255},
        "normalization": {"mean": 0.449, "std": 0.226, "in_graph": in_graph},
        "temperatures": temperatures or {},
    }
    if task == "digits":
        meta["part_input"] = {"name": "part_id"}
        meta["part_ids"] = {"day": 0, "month": 1, "hour": 2, "minute": 3}
    else:
        meta["kind"] = "ctc"
        meta["heads"] = ["hundreds", "tens", "units"]
    meta.update(extra or {})
    (directory / "meta.json").write_text(json.dumps(meta), encoding="utf-8")
    (directory / "model.onnx").write_bytes(b"stub")
    return directory


class FakeSession:
    """Заглушка `InferenceSession`: запоминает вход, отдаёт заданные логиты."""

    def __init__(self, outputs: list[np.ndarray]) -> None:
        self.outputs = outputs
        self.feeds: list[dict[str, np.ndarray]] = []

    def run(self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
            ) -> list[np.ndarray]:
        self.feeds.append(input_feed)
        n = len(next(iter(input_feed.values())))
        return [np.repeat(o[:1], n, axis=0) if len(o) != n else o for o in self.outputs]


def digit_logits(tens: int, units: int, strength: float = 6.0) -> tuple[np.ndarray, np.ndarray]:
    lt = np.zeros((1, 11), dtype=np.float32)
    lu = np.zeros((1, 10), dtype=np.float32)
    lt[0, tens] = strength
    lu[0, units] = strength
    return lt, lu


def ctc_path_logits(path: list[int], strength: float = 6.0) -> np.ndarray:
    out = np.zeros((1, len(path), 11), dtype=np.float32)
    out[0, np.arange(len(path)), path] = strength
    return out


def crop(h: int = 40, w: int = 90, value: int = 120) -> np.ndarray:
    return np.full((h, w), value, dtype=np.uint8)


@pytest.fixture(autouse=True)
def _fresh_registry() -> Any:
    rt.reset_models()
    yield
    rt.reset_models()


# --- предобработка -------------------------------------------------------------------------


def test_fit_with_aspect_keeps_aspect_and_white_background() -> None:
    img = np.zeros((40, 20), dtype=np.uint8)  # узкий чёрный кроп
    out = rt.fit_with_aspect(img, (8, 16))
    assert out.shape == (8, 16)
    ink_cols = np.nonzero((out == 0).any(axis=0))[0]
    assert ink_cols.min() > 0 and ink_cols.max() < 15  # по бокам белые поля
    assert out[0, 0] == 255


def test_fit_with_aspect_matches_lab_version() -> None:
    pytest.importorskip("torch")
    from ocr_lab.dataset import fit_with_aspect as lab_fit  # noqa: PLC0415

    rng = np.random.default_rng(0)
    for shape in ((92, 150), (112, 99), (100, 267), (30, 30)):
        img = rng.integers(0, 256, shape, dtype=np.uint8)
        np.testing.assert_array_equal(rt.fit_with_aspect(img, (64, 160)),
                                      lab_fit(img, (64, 160)))


def test_fit_with_aspect_rejects_non_gray() -> None:
    with pytest.raises(rt.OcrRuntimeError):
        rt.fit_with_aspect(np.zeros((4, 4, 3), dtype=np.uint8), SIZE)


def test_prepare_normalizes_outside_graph_only_when_asked(tmp_path: Path) -> None:
    inside = rt.OnnxModel(write_model_dir(tmp_path, "a", "digits", in_graph=True))
    outside = rt.OnnxModel(write_model_dir(tmp_path, "b", "digits", in_graph=False))
    img = crop(value=255)
    x_in = inside.prepare([img])
    x_out = outside.prepare([img])
    assert x_in.shape == (1, 1, *SIZE) and x_in.dtype == np.float32
    np.testing.assert_allclose(x_in, 1.0)
    np.testing.assert_allclose(x_out, (1.0 - 0.449) / 0.226, rtol=1e-5)


def test_prepare_accepts_color_and_float_crops(tmp_path: Path) -> None:
    model = rt.OnnxModel(write_model_dir(tmp_path, "a", "digits"))
    gray = crop(value=100)
    color = np.dstack([gray] * 3)
    np.testing.assert_allclose(model.prepare([gray]), model.prepare([color]), atol=1 / 255)
    np.testing.assert_allclose(model.prepare([gray]), model.prepare([gray.astype(np.float32)]))


# --- контракт ------------------------------------------------------------------------------


def test_load_meta_errors(tmp_path: Path) -> None:
    with pytest.raises(rt.OcrRuntimeError, match="нет meta.json"):
        rt.load_meta(tmp_path / "missing")
    bad_version = write_model_dir(tmp_path, "v2", "digits", version=2)
    with pytest.raises(rt.OcrRuntimeError, match="версия контракта"):
        rt.load_meta(bad_version)
    broken = write_model_dir(tmp_path, "broken", "digits")
    (broken / "meta.json").write_text("{\"format_version\": 1}", encoding="utf-8")
    with pytest.raises(rt.OcrRuntimeError, match="не по контракту"):
        rt.load_meta(broken)


def test_models_dir_from_environment(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCR_MODELS_DIR", raising=False)
    monkeypatch.setenv("OCR_WORK_DIR", str(tmp_path / "work"))
    assert rt.models_dir() == tmp_path / "work" / "models"
    monkeypatch.setenv("OCR_MODELS_DIR", str(tmp_path / "m"))
    assert rt.models_dir() == tmp_path / "m"


def test_missing_onnx_file(tmp_path: Path) -> None:
    directory = write_model_dir(tmp_path, "a", "digits")
    (directory / "model.onnx").unlink()
    model = rt.OnnxModel(directory, session_factory=lambda p: FakeSession([]))
    with pytest.raises(rt.OcrRuntimeError, match="нет файла модели"):
        model.session  # noqa: B018


# --- ленивая загрузка и потокобезопасность --------------------------------------------------


def test_session_is_lazy_and_created_once_under_threads(tmp_path: Path) -> None:
    calls: list[Path] = []
    lock = threading.Lock()

    def factory(path: Path) -> FakeSession:
        with lock:
            calls.append(path)
        return FakeSession([*digit_logits(1, 2)])

    model = rt.OnnxModel(write_model_dir(tmp_path, "a", "digits"), session_factory=factory)
    assert calls == []  # создание объекта сессию не открывает
    barrier = threading.Barrier(8)
    errors: list[BaseException] = []

    def work() -> None:
        try:
            barrier.wait()
            rt.predict_digits({"left_base.hour": crop()}, model=model)
        except BaseException as exc:  # noqa: BLE001
            errors.append(exc)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert not errors
    assert len(calls) == 1


def test_get_model_is_cached_per_directory(tmp_path: Path) -> None:
    write_model_dir(tmp_path, "digits_v0", "digits")
    a = rt.get_model("digits_v0", directory=tmp_path)
    assert rt.get_model("digits_v0", directory=tmp_path) is a
    rt.reset_models()
    assert rt.get_model("digits_v0", directory=tmp_path) is not a


# --- цифры ----------------------------------------------------------------------------------


def test_predict_digits_top_values_and_batching(tmp_path: Path) -> None:
    session = FakeSession([*digit_logits(1, 7)])  # «17» для всех кропов
    model = rt.OnnxModel(write_model_dir(tmp_path, "d", "digits"),
                         session_factory=lambda p: session)
    crops = {"left_base.day": crop(), "left_base.hour": crop(), "arrived_base.minute": crop()}
    out = rt.predict_digits(crops, model=model, k=3)
    assert set(out) == set(crops)
    assert len(session.feeds) == 1  # все подполя — одним батчем
    feed = session.feeds[0]
    assert feed["image"].shape == (3, 1, *SIZE)
    assert feed["part_id"].dtype == np.int64
    assert feed["part_id"].tolist() == [0, 2, 3]  # day, hour, minute
    for top in out.values():
        assert top[0][0] == 17 and len(top) == 3
        assert top[0][1] > 0.9
        probs = [p for _, p in top]
        assert probs == sorted(probs, reverse=True)
    # диапазон части: для месяца 17 недопустимо, значение берётся из 1..12
    month = rt.predict_digits({"left_base.month": crop()}, model=model)["left_base.month"]
    assert 1 <= month[0][0] <= 12


def test_predict_digits_applies_temperatures_from_meta(tmp_path: Path) -> None:
    logits = digit_logits(2, 3, strength=4.0)
    plain = rt.OnnxModel(write_model_dir(tmp_path, "p", "digits"),
                         session_factory=lambda p: FakeSession([*logits]))
    soft = rt.OnnxModel(
        write_model_dir(tmp_path, "s", "digits", temperatures={"tens": 4.0, "units": 2.0}),
        session_factory=lambda p: FakeSession([*logits]),
    )
    a = rt.predict_digits({"left_base.minute": crop()}, model=plain)["left_base.minute"]
    b = rt.predict_digits({"left_base.minute": crop()}, model=soft)["left_base.minute"]
    assert a[0][0] == b[0][0] == 23
    assert b[0][1] < a[0][1]  # температура > 1 смягчает
    values, logp = value_log_probs(logits[0][0], logits[1][0], "minute", t_tens=4.0, t_units=2.0)
    assert b[0][1] == pytest.approx(float(np.exp(logp[list(values).index(23)])))


def test_predict_digits_errors_and_empty(tmp_path: Path) -> None:
    model = rt.OnnxModel(write_model_dir(tmp_path, "d", "digits"),
                         session_factory=lambda p: FakeSession([*digit_logits(0, 1)]))
    assert rt.predict_digits({}, model=model) == {}
    with pytest.raises(rt.OcrRuntimeError, match="неизвестная часть"):
        rt.predict_digits({"left_base.second": crop()}, model=model)
    number = rt.OnnxModel(write_model_dir(tmp_path, "n", "number"))
    with pytest.raises(rt.OcrRuntimeError, match="не модель цифр"):
        rt.predict_digits({"left_base.day": crop()}, model=number)


# --- номер ----------------------------------------------------------------------------------


def number_model(tmp_path: Path, temperature: float = 1.0) -> rt.OnnxModel:
    path = [3, 0, 5, 0, 8]  # классы CTC: цифра d → d+1, т. е. «2», «4», «7»
    logits = ctc_path_logits([c for c in path])
    return rt.OnnxModel(
        write_model_dir(tmp_path, f"n{temperature}", "number", temperatures={"ctc": temperature}),
        session_factory=lambda p: FakeSession([logits]),
    )


def test_score_number_candidates_matches_number_scores(tmp_path: Path) -> None:
    model = number_model(tmp_path, temperature=2.5)
    out = ctc_path_logits([3, 0, 5, 0, 8])[0]
    cands = ["247", "248", "24", "0247", "abc", 9999]
    got = rt.score_number_candidates(crop(), cands, model=model)
    want = score_candidates(out, cands, kind="ctc", temperature=2.5)
    assert set(got) == {str(c) for c in cands}
    for key, value in want.items():
        assert got[key] == pytest.approx(value)
    assert got["abc"] == float("-inf") and got["9999"] == float("-inf")
    assert max(got, key=lambda k: got[k]) in {"247", "0247"}  # ведущий ноль не мешает


def test_predict_number_top_k_is_calibrated(tmp_path: Path) -> None:
    sharp = number_model(tmp_path, temperature=1.0)
    soft = number_model(tmp_path, temperature=4.0)
    a = rt.predict_number(crop(), k=3, model=sharp)
    b = rt.predict_number(crop(), k=3, model=soft)
    assert a[0][0] == b[0][0] == 247 and len(a) == 3
    assert b[0][1] < a[0][1]
    out = ctc_path_logits([3, 0, 5, 0, 8])[0]
    assert a == top_numbers(out, kind="ctc", k=3)


def test_number_distribution_is_normalized(tmp_path: Path) -> None:
    values, logp = rt.number_distribution(crop(), model=number_model(tmp_path))
    assert values[0] == 1 and values[-1] == 999
    assert float(np.exp(logp).sum()) == pytest.approx(1.0)


def test_number_wrong_task(tmp_path: Path) -> None:
    digits = rt.OnnxModel(write_model_dir(tmp_path, "d", "digits"))
    with pytest.raises(rt.OcrRuntimeError, match="не модель номера"):
        rt.predict_number(crop(), model=digits)


def test_number_heads_temperature_per_head(tmp_path: Path) -> None:
    logits = np.zeros((1, 3, 11), dtype=np.float32)
    logits[0, 0, 2] = logits[0, 1, 4] = logits[0, 2, 7] = 5.0
    model = rt.OnnxModel(
        write_model_dir(tmp_path, "h", "number", extra={"kind": "heads"},
                        temperatures={"hundreds": 1.0, "tens": 2.0, "units": 3.0}),
        session_factory=lambda p: FakeSession([logits]),
    )
    assert rt.number_temperature(model).tolist() == [1.0, 2.0, 3.0]  # type: ignore[union-attr]
    got = rt.score_number_candidates(crop(), [247], model=model)["247"]
    want = score_candidates(logits[0], [247], kind="heads",
                            temperature=np.array([1.0, 2.0, 3.0]))["247"]
    assert got == pytest.approx(want)


# --- настоящий onnxruntime на крошечной модели ----------------------------------------------


def build_tiny_digits_onnx(path: Path, w_tens: np.ndarray, w_units: np.ndarray) -> None:
    """ONNX: логиты = среднее изображения · W + part_id · W2 (динамический батч)."""
    onnx = pytest.importorskip("onnx")
    from onnx import TensorProto, helper, numpy_helper  # noqa: PLC0415

    nodes = [
        helper.make_node("ReduceMean", ["image"], ["mean"], axes=[1, 2, 3], keepdims=0),
        helper.make_node("Unsqueeze", ["mean", "ax1"], ["mean2"]),  # (B,) → (B,1)
        helper.make_node("Cast", ["part_id"], ["pid_f"], to=TensorProto.FLOAT),
        helper.make_node("Unsqueeze", ["pid_f", "ax1"], ["pid2"]),
        helper.make_node("MatMul", ["mean2", "w_tens"], ["tens_a"]),
        helper.make_node("MatMul", ["pid2", "w_tens_p"], ["tens_b"]),
        helper.make_node("Add", ["tens_a", "tens_b"], ["tens_logits"]),
        helper.make_node("MatMul", ["mean2", "w_units"], ["units_logits"]),
    ]
    inits = [
        numpy_helper.from_array(np.array([1], dtype=np.int64), "ax1"),
        numpy_helper.from_array(w_tens.astype(np.float32), "w_tens"),
        numpy_helper.from_array(np.full((1, 11), 0.1, dtype=np.float32), "w_tens_p"),
        numpy_helper.from_array(w_units.astype(np.float32), "w_units"),
    ]
    graph = helper.make_graph(
        nodes, "tiny",
        [helper.make_tensor_value_info("image", TensorProto.FLOAT, ["batch", 1, *SIZE]),
         helper.make_tensor_value_info("part_id", TensorProto.INT64, ["batch"])],
        [helper.make_tensor_value_info("tens_logits", TensorProto.FLOAT, ["batch", 11]),
         helper.make_tensor_value_info("units_logits", TensorProto.FLOAT, ["batch", 10])],
        inits,
    )
    model = helper.make_model(graph, opset_imports=[helper.make_opsetid("", 17)])
    model.ir_version = 9
    onnx.checker.check_model(model)
    path.write_bytes(model.SerializeToString())


def test_real_onnxruntime_on_tiny_model(tmp_path: Path) -> None:
    pytest.importorskip("onnxruntime")
    w_tens = np.linspace(-3, 3, 11).reshape(1, 11)
    w_units = np.linspace(2, -2, 10).reshape(1, 10)
    directory = write_model_dir(tmp_path, "d", "digits", temperatures={"tens": 1.5, "units": 0.8})
    build_tiny_digits_onnx(directory / "model.onnx", w_tens, w_units)
    model = rt.get_model("d", directory=tmp_path)
    crops = {"left_base.day": crop(value=255), "left_base.minute": crop(value=0),
             "started_work.hour": crop(value=128)}
    out = rt.predict_digits(crops, model=model)
    # ожидание — теми же numpy-функциями по ручному расчёту логитов
    for name, img in crops.items():
        part = rt.part_of(name)
        mean = float(model.prepare([img]).mean())
        tens = mean * w_tens[0] + 0.1 * model.meta.raw["part_ids"][part]
        units = mean * w_units[0]
        values, logp = value_log_probs(tens, units, part, t_tens=1.5, t_units=0.8)
        best = int(np.argmax(logp))
        assert out[name][0][0] == int(values[best])
        assert out[name][0][1] == pytest.approx(float(np.exp(logp[best])), rel=1e-4)
    # батч любого размера
    one = rt.predict_digits({"left_base.day": crop(value=255)}, model=model)
    assert one["left_base.day"] == out["left_base.day"]
