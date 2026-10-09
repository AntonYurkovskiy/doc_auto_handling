"""Тесты T16: функция потерь, распределение по значениям, модель и помощники обучения.

Только синтетика: настоящие веса и `torchvision` не нужны (бэкбон подменяется крошечной
свёрточной сетью). `torch` в `.venv` есть (CPU), но модуль пропускается, если его нет.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest

from ocr_lab import digit_values as dv

torch = pytest.importorskip("torch")

from ocr_lab import models  # noqa: E402
from ocr_lab import train_digits as td  # noqa: E402
from ocr_lab.dataset import Sample  # noqa: E402
from ocr_lab.predictions import parse_prediction  # noqa: E402


def _softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max())
    return e / e.sum()


def _brute_force(tens_logits: np.ndarray, units_logits: np.ndarray, part: str) -> dict[int, float]:
    """Распределение «по определению» — прямой перебор для сверки."""
    pt, pu = _softmax(tens_logits), _softmax(units_logits)
    lo, hi = dv.PART_RANGES[part]
    raw = {}
    for v in range(lo, hi + 1):
        p_tens = pt[0] + pt[dv.TENS_EMPTY] if v < 10 else pt[v // 10]
        raw[v] = p_tens * pu[v % 10]
    z = sum(raw.values())
    return {v: p / z for v, p in raw.items()}


# --- распределение по значениям ------------------------------------------------------------


@pytest.mark.parametrize("part", ["day", "month", "hour", "minute"])
def test_value_distribution_matches_definition_and_is_normalized(part: str) -> None:
    rng = np.random.default_rng(0)
    for _ in range(5):
        tl, ul = rng.normal(size=11) * 3, rng.normal(size=10) * 3
        values, logp = dv.value_log_probs(tl, ul, part)
        lo, hi = dv.PART_RANGES[part]
        assert values.tolist() == list(range(lo, hi + 1))
        probs = np.exp(logp)
        assert probs.sum() == pytest.approx(1.0, abs=1e-12)
        expected = _brute_force(tl, ul, part)
        for v, p in zip(values.tolist(), probs, strict=True):
            assert p == pytest.approx(expected[v], rel=1e-9)


def test_ranges_exclude_impossible_values() -> None:
    tl, ul = np.zeros(11), np.zeros(10)
    assert 0 not in dv.value_log_probs(tl, ul, "day")[0]
    assert 32 not in dv.value_log_probs(tl, ul, "day")[0]
    assert dv.value_log_probs(tl, ul, "month")[0].max() == 12
    assert dv.value_log_probs(tl, ul, "minute")[0].tolist() == list(range(60))


def test_hour_24_is_allowed_and_25_is_not() -> None:
    tl = np.full(11, -10.0)
    tl[2] = 10.0  # десятки «2»
    ul = np.full(10, -10.0)
    ul[4] = 5.0  # единицы «4»
    ul[5] = 6.0  # «5» чуть вероятнее, но 25 часов не бывает
    values, logp = dv.value_log_probs(tl, ul, "hour")
    assert values.max() == 24
    top = dv.top_values(values, logp, 1)
    assert top[0][0] == 24
    assert dv.truth_log_prob(tl, ul, "hour", 25) == float("-inf")
    # Для минут то же самое «2»+«5» даёт 25.
    assert dv.top_values(*dv.value_log_probs(tl, ul, "minute"), 1)[0][0] == 25


def test_leading_zero_is_marginalized() -> None:
    """Значение < 10: «0» и «пусто» у десятков взаимозаменяемы, складываются вероятности."""
    ul = np.full(10, -5.0)
    ul[9] = 5.0
    as_zero = np.full(11, -5.0)
    as_zero[0] = 5.0
    as_empty = np.full(11, -5.0)
    as_empty[dv.TENS_EMPTY] = 5.0
    split = np.full(11, -5.0)
    # Та же суммарная масса «0 + пусто», что у as_zero/as_empty, но поровну.
    split[0] = split[dv.TENS_EMPTY] = math.log((math.exp(5.0) + math.exp(-5.0)) / 2)
    results = [dv.value_log_probs(t, ul, "minute")[1] for t in (as_zero, as_empty, split)]
    np.testing.assert_allclose(results[0], results[1], atol=1e-12)
    np.testing.assert_allclose(results[0], results[2], atol=1e-12)
    # «пусто» не даёт вероятности двузначным значениям: 19, 29, … маловероятны.
    values, logp = dv.value_log_probs(as_empty, ul, "minute")
    assert dv.top_values(values, logp, 1)[0] == (9, pytest.approx(math.exp(logp[9])))
    assert np.exp(logp[values == 19][0]) < 1e-3


def test_empty_tens_mass_moves_to_single_digit_values() -> None:
    """Больше «пусто» у десятков — больше вероятность однозначных значений дня."""
    ul = np.zeros(10)
    low = np.zeros(11)
    high = np.zeros(11)
    high[dv.TENS_EMPTY] = 3.0
    p_low = np.exp(dv.value_log_probs(low, ul, "day")[1])
    p_high = np.exp(dv.value_log_probs(high, ul, "day")[1])
    assert p_high[:9].sum() > p_low[:9].sum()


def test_batch_and_temperature() -> None:
    rng = np.random.default_rng(1)
    tl, ul = rng.normal(size=(3, 11)), rng.normal(size=(3, 10))
    values, batch = dv.value_log_probs(tl, ul, "day")
    for i in range(3):
        np.testing.assert_allclose(batch[i], dv.value_log_probs(tl[i], ul[i], "day")[1])
    hot = np.exp(dv.value_log_probs(tl[0], ul[0], "day", t_tens=0.5, t_units=0.5)[1])
    cold = np.exp(dv.value_log_probs(tl[0], ul[0], "day", t_tens=2.0, t_units=2.0)[1])
    assert hot.max() > cold.max()
    assert values[np.argmax(hot)] == values[np.argmax(np.exp(batch[0]))]


def test_top_values_and_predict_values_format() -> None:
    rng = np.random.default_rng(2)
    tl, ul = rng.normal(size=(4, 11)) * 2, rng.normal(size=(4, 10)) * 2
    parts = ["day", "month", "hour", "minute"]
    tops = dv.predict_values(tl, ul, parts)
    for cands, part in zip(tops, parts, strict=True):
        assert len(cands) == 5
        assert all(isinstance(v, int) for v, _ in cands)
        probs = [p for _, p in cands]
        assert probs == sorted(probs, reverse=True)
        assert sum(probs) <= 1 + 1e-9
        lo, hi = dv.PART_RANGES[part]
        assert all(lo <= v <= hi for v, _ in cands)


def test_value_log_probs_rejects_wrong_heads() -> None:
    with pytest.raises(ValueError, match="11 и 10"):
        dv.value_log_probs(np.zeros(10), np.zeros(10), "day")


# --- функция потерь ------------------------------------------------------------------------


def test_loss_exact_for_two_digit_values() -> None:
    tl = torch.randn(4, 11)
    ul = torch.randn(4, 10)
    tens = torch.tensor([1, 2, 3, 5])
    units = torch.tensor([0, 4, 1, 9])
    amb = torch.zeros(4, dtype=torch.bool)
    loss, parts = models.digit_loss(tl, ul, tens, units, amb)
    expected = (torch.nn.functional.cross_entropy(tl, tens)
                + torch.nn.functional.cross_entropy(ul, units))
    assert float(loss) == pytest.approx(float(expected), rel=1e-6)
    assert parts["tens"] + parts["units"] == pytest.approx(float(loss), rel=1e-6)


def test_loss_marginal_for_single_digit_values() -> None:
    ul = torch.zeros(1, 10)
    units = torch.tensor([7])
    amb = torch.tensor([True])
    # Масса десятков целиком на «0», целиком на «пусто» или поровну — потеря одинаковая.
    losses = []
    for idx in ([0], [dv.TENS_EMPTY], [0, dv.TENS_EMPTY]):
        tl = torch.full((1, 11), -20.0)
        tl[0, idx] = 10.0
        loss, _ = models.digit_loss(tl, ul, torch.tensor([0]), units, amb)
        losses.append(float(loss))
    assert losses[0] == pytest.approx(losses[1], abs=1e-6)
    assert losses[0] == pytest.approx(losses[2], abs=1e-6)
    # Явная формула: -log(p0 + p_empty) - log p_units.
    tl = torch.randn(1, 11)
    loss, parts = models.digit_loss(tl, ul, torch.tensor([0]), units, amb)
    p = torch.softmax(tl, dim=1)[0]
    assert parts["tens"] == pytest.approx(-math.log(float(p[0] + p[dv.TENS_EMPTY])), rel=1e-5)
    assert parts["units"] == pytest.approx(math.log(10.0), rel=1e-6)
    # А мимо «0/пусто» — штраф.
    wrong = torch.full((1, 11), -20.0)
    wrong[0, 3] = 10.0
    assert float(models.digit_loss(wrong, ul, torch.tensor([0]), units, amb)[0]) > losses[0] + 10


def test_loss_gradient_does_not_separate_zero_and_empty() -> None:
    tl = torch.zeros(1, 11, requires_grad=True)
    loss, _ = models.digit_loss(tl, torch.zeros(1, 10), torch.tensor([0]), torch.tensor([3]),
                                torch.tensor([True]))
    loss.backward()
    assert tl.grad is not None
    grad = tl.grad[0]
    assert float(grad[0]) == pytest.approx(float(grad[dv.TENS_EMPTY]))
    assert float(grad[0]) < 0 < float(grad[5])


# --- модель --------------------------------------------------------------------------------


def _tiny_net(part_embed: bool) -> models.DigitNet:
    features = torch.nn.Sequential(torch.nn.Conv2d(1, 8, 3, stride=4, padding=1),
                                   torch.nn.ReLU())
    cfg = models.ModelConfig(backbone="tiny", part_embed=part_embed)
    return models.DigitNet(cfg, features=features, channels=8)


@pytest.mark.parametrize("part_embed", [True, False])
def test_digitnet_shapes(part_embed: bool) -> None:
    net = _tiny_net(part_embed)
    x = torch.rand(3, 1, 64, 160)
    tens, units = net(x, torch.tensor([0, 3, -1]))
    assert tens.shape == (3, 11)
    assert units.shape == (3, 10)
    n_head = sum(p.numel() for p in net.head_parameters())
    assert n_head == (sum(p.numel() for p in net.head_tens.parameters())
                      + sum(p.numel() for p in net.head_units.parameters())
                      + (5 * 16 if part_embed else 0))


def test_gray_first_conv_equals_rgb_with_equal_channels() -> None:
    conv = torch.nn.Conv2d(3, 4, 3, padding=1, bias=True)
    gray = models._gray_first_conv(conv)
    x = torch.rand(2, 1, 10, 12)
    np.testing.assert_allclose(gray(x).detach().numpy(),
                               conv(x.repeat(1, 3, 1, 1)).detach().numpy(), atol=1e-5)


def test_model_config_roundtrip() -> None:
    cfg = models.ModelConfig(backbone="mobilenet_v3_small", part_embed=False)
    assert models.ModelConfig.from_dict(cfg.to_dict() | {"extra": 1}) == cfg
    tcfg = td.TrainConfig(run="x", model=cfg, mnist=100)
    again = td.TrainConfig.from_dict(json.loads(json.dumps(tcfg.to_dict())))
    assert again == tcfg


# --- помощники обучения --------------------------------------------------------------------


def test_lr_lambda_warmup_and_cosine() -> None:
    assert td.lr_lambda(0, warmup=10, total=100) == pytest.approx(0.1)
    assert td.lr_lambda(9, warmup=10, total=100) == pytest.approx(1.0)
    assert td.lr_lambda(10, warmup=10, total=100) == pytest.approx(1.0)
    assert td.lr_lambda(55, warmup=10, total=100) == pytest.approx(0.5)
    assert td.lr_lambda(100, warmup=10, total=100) == pytest.approx(0.0, abs=1e-12)


def test_epoch_sampler_is_deterministic_and_offsets_epochs() -> None:
    sampler = td.EpochSampler([1.0, 1.0, 2.0, 4.0], seed=3)
    sampler.set_epoch(0)
    first = list(sampler)
    assert list(sampler) == first
    assert all(0 <= i < 4 for i in first)
    sampler.set_epoch(2)
    second = list(sampler)
    assert all(8 <= i < 12 for i in second)
    assert len(sampler) == 4


def _samples(tmp_path: Path, n_per_part: dict[str, int]) -> tuple[list[Sample], list[np.ndarray]]:
    samples, images = [], []
    for part, n in n_per_part.items():
        for i in range(n):
            samples.append(Sample(scan_id=f"s{i}", subfield=f"left_base.{part}",
                                  path=tmp_path / f"{part}_{i}.png", value=5 + i))
            img = np.full((92, 150), 255, np.uint8)
            img[30:60, 40 + i : 70 + i] = 0
            images.append(img)
    return samples, images


def test_train_weights_balance_parts_and_synthetic_share(tmp_path: Path) -> None:
    samples, _ = _samples(tmp_path, {"day": 6, "month": 2})
    weights = td.train_weights(samples, n_synthetic=4)
    assert len(weights) == 12
    assert sum(weights[:6]) == pytest.approx(sum(weights[6:8]))
    assert sum(weights[:8]) == pytest.approx(8.0)
    assert weights[8:] == [1.0] * 4


def test_trainset_virtual_index(tmp_path: Path) -> None:
    samples, images = _samples(tmp_path, {"day": 2, "hour": 1})
    plain = td.TrainSet(images, samples, augment=False, seed=0, size=(64, 160))
    x0, tens, units, amb, part = plain[0]
    assert x0.shape == (1, 64, 160)
    assert (int(tens), int(units), bool(amb), int(part)) == (0, 5, True, 0)
    # Тот же кроп в другой эпохе без аугментации — тот же тензор.
    assert torch.equal(plain[3][0], x0)
    aug = td.TrainSet(images, samples, augment=True, seed=0, size=(64, 160))
    assert torch.equal(aug[0][0], aug[0][0])
    assert not torch.equal(aug[0][0], aug[3][0])
    assert int(plain[2][4]) == 2  # hour


def test_score_logits_and_build_predictions(tmp_path: Path) -> None:
    samples = [
        Sample("a", "left_base.day", tmp_path / "1.png", 7),
        Sample("a", "left_base.hour", tmp_path / "2.png", 24),
        Sample("b", "left_base.day", tmp_path / "3.png", 15),
    ]
    tl = np.full((3, 11), -10.0)
    ul = np.full((3, 10), -10.0)
    tl[0, dv.TENS_EMPTY], ul[0, 7] = 5.0, 5.0  # «7» без десятков — верно
    tl[1, 2], ul[1, 4] = 5.0, 5.0  # 24 — верно
    tl[2, 1], ul[2, 6] = 5.0, 5.0  # 16 вместо 15 — ошибка
    metrics = td.score_logits(tl, ul, samples)
    assert metrics["subfield_acc"] == {"left_base.day": 0.5, "left_base.hour": 1.0}
    assert metrics["mean_subfield_acc"] == pytest.approx(0.75)
    assert metrics["micro_acc"] == pytest.approx(2 / 3)
    assert metrics["part_acc"] == {"day": 0.5, "hour": 1.0}

    crops = [td.CropRef(s.scan_id, s.subfield, s.path) for s in samples]
    preds = td.build_predictions(crops, tl, ul, source="digits_test")
    assert [p.scan_id for p in preds] == ["a", "b"]
    assert preds[0].top1("left_base.day") == 7
    assert preds[0].top1("left_base.hour") == 24
    assert preds[1].top1("left_base.day") == 16
    for p in preds:
        parsed = parse_prediction(json.loads(json.dumps(p.to_json())))
        assert all(len(c) == 5 for c in parsed.fields.values())
