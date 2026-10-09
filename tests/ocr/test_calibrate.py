"""Тесты T18: `ocr_lab.calibrate` — температура, ECE, диаграмма надёжности (синтетика)."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

torch = pytest.importorskip("torch")

from ocr_lab import calibrate as cal  # noqa: E402


def _sample_logits(n: int, classes: int, scale: float, seed: int = 0
                   ) -> tuple[np.ndarray, np.ndarray]:
    """Логиты `scale · z`, метки выбираются из `softmax(z)`: истинная калибровка — T = scale."""
    rng = np.random.default_rng(seed)
    z = rng.normal(0.0, 1.5, (n, classes))
    p = np.exp(z - z.max(1, keepdims=True))
    p /= p.sum(1, keepdims=True)
    labels = np.array([rng.choice(classes, p=row) for row in p])
    return z * scale, labels


def test_ece_perfect_and_overconfident() -> None:
    conf = np.full(1000, 0.8)
    rng = np.random.default_rng(0)
    correct = (rng.random(1000) < 0.8).astype(float)
    assert cal.ece(conf, correct) < 0.03
    assert cal.ece(np.full(1000, 0.99), correct) > 0.15
    assert cal.ece(np.array([]), np.array([])) == 0.0


def test_reliability_bins_counts_and_edges() -> None:
    conf = np.array([0.0, 0.05, 0.5, 0.99, 1.0])
    correct = np.array([0, 1, 1, 1, 0])
    bins = cal.reliability_bins(conf, correct, n_bins=10)
    assert sum(b["n"] for b in bins) == 5  # 1.0 попадает в последнюю корзину
    assert bins[-1]["hi"] == pytest.approx(1.0) and bins[-1]["n"] == 2
    assert bins[0]["n"] == 2 and bins[0]["acc"] == pytest.approx(0.5)


def test_fit_temperature_recovers_scale() -> None:
    logits, labels = _sample_logits(4000, 10, scale=3.0)
    lg = torch.from_numpy(logits)
    y = torch.from_numpy(labels)
    t = cal.fit_temperature(lambda temp: cal.units_nll(lg, y, temp))
    assert t == pytest.approx(3.0, rel=0.15)  # модель втрое самоуверенна


def test_fit_temperature_is_clamped() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    y = torch.tensor([0, 1])  # идеальная модель: NLL убывает при T → 0
    t = cal.fit_temperature(lambda temp: cal.units_nll(logits, y, temp))
    assert t == pytest.approx(cal.T_MIN, rel=1e-3)


def test_fit_guarded_falls_back_to_one_at_bound() -> None:
    logits = torch.tensor([[3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    y = torch.tensor([0, 1])  # ошибок нет: NLL монотонно падает при T → 0
    notes: dict[str, str] = {}
    assert cal.fit_guarded("units", lambda t: cal.units_nll(logits, y, t), notes) == 1.0
    assert "units" in notes and "T=1" in notes["units"]
    noisy, labels = _sample_logits(2000, 10, scale=2.0, seed=3)
    notes = {}
    t = cal.fit_guarded("units", lambda temp: cal.units_nll(
        torch.from_numpy(noisy), torch.from_numpy(labels), temp), notes)
    assert t > 1.5 and not notes


def test_fit_digit_temperatures_improves_nll_and_ece() -> None:
    n = 3000
    tens_l, tens_y = _sample_logits(n, 11, scale=2.5, seed=1)
    units_l, units_y = _sample_logits(n, 10, scale=2.0, seed=2)
    values = tens_y * 10 + units_y
    values = np.where(values > 59, values % 60, values)
    temps = cal.fit_digit_temperatures(tens_l, units_l, values)
    assert temps["tens"] > 1.2 and temps["units"] > 1.2
    parts = ["minute"] * n
    before = cal.digit_metrics(tens_l, units_l, parts, values.tolist())
    after = cal.digit_metrics(tens_l, units_l, parts, values.tolist(), temps)
    assert after["nll"] < before["nll"]
    assert after["ece"] <= before["ece"]
    assert set(before["by_part"]) == {"minute"}
    assert before["acc"] == pytest.approx(after["acc"], abs=0.05)


def test_tens_nll_marginalizes_leading_zero() -> None:
    logits = torch.zeros(2, 11, dtype=torch.float64)
    logits[:, 0] = 2.0
    logits[:, 10] = 2.0  # «0» и «пусто» поровну
    t1 = torch.tensor(1.0, dtype=torch.float64)
    ambiguous = torch.tensor([True, True])
    nll = cal.tens_nll(logits, torch.tensor([0, 0]), ambiguous, t1)
    exact = cal.tens_nll(logits, torch.tensor([0, 0]), torch.tensor([False, False]), t1)
    assert nll < exact  # маргинал «0 или пусто» вероятнее одного «0»


def test_fit_number_temperature_ctc_and_heads() -> None:
    # CTC: кадры уверенно выдают путь «2 4 7», но в 30 % случаев истина — другая → T > 1
    rng = np.random.default_rng(0)
    n, frames = 300, 7
    outputs = np.zeros((n, frames, 11))
    values = []
    for i in range(n):
        outputs[i, np.arange(frames), [3, 0, 5, 0, 8, 0, 0]] = 6.0
        values.append(247 if rng.random() < 0.7 else 248)
    temps = cal.fit_number_temperatures(outputs, values, "ctc")
    assert set(temps) == {"ctc"} and temps["ctc"] > 1.1

    heads = np.zeros((n, 3, 11))
    heads[:, 0, 2] = heads[:, 1, 4] = heads[:, 2, 7] = 6.0
    temps_h = cal.fit_number_temperatures(heads, values, "heads")
    assert set(temps_h) == {"hundreds", "tens", "units"}
    assert temps_h["units"] > 1.5  # единицы ошибочны в 30 % случаев
    assert temps_h["hundreds"] <= 1.0  # сотни всегда верны

    with pytest.raises(ValueError):
        cal.fit_number_temperatures(heads, values, "other")


def test_number_metrics_temperature_changes_confidence() -> None:
    out = np.zeros((1, 5, 11))
    out[0, np.arange(5), [3, 0, 5, 0, 8]] = 6.0
    base = cal.number_metrics(out, [247], "ctc")
    soft = cal.number_metrics(out, [247], "ctc", {"ctc": 5.0})
    assert base["acc"] == soft["acc"] == 1.0
    assert soft["mean_conf"] < base["mean_conf"]
    assert cal.temperature_arg({"ctc": 2.0}, "ctc") == 2.0
    assert cal.temperature_arg({"tens": 2.0}, "heads").tolist() == [1.0, 2.0, 1.0]


def test_plot_reliability_writes_png(tmp_path: Path) -> None:
    pytest.importorskip("matplotlib")
    metric = {"n": 4, "acc": 0.75, "mean_conf": 0.9, "ece": 0.15, "nll": 0.4,
              "bins": cal.reliability_bins(np.array([0.9, 0.95, 0.99, 0.6]),
                                           np.array([1, 1, 0, 1]))}
    result = {"model": "digits_v0", "temperatures": {"tens": 1.3, "units": 1.1},
              "metrics": {"val": {"before": metric, "after": metric}}}
    png = cal.plot_reliability(result, tmp_path / "sub" / "calibration_digits.png")
    assert png.read_bytes()[:8] == b"\x89PNG\r\n\x1a\n"
    json.dumps(cal.summary_lines({**result, "n_fit": 4}))
