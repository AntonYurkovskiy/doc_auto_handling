"""Тесты T17: `ocr_lab.number_scores` — оценка кандидатов номера на синтетических логитах.

Модуль без torch; сверка с `torch.nn.functional.ctc_loss` — отдельный тест, пропускается,
если torch нет.
"""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from ocr_lab import number_scores as ns


def _onehot_heads(value: int, strength: float = 8.0) -> np.ndarray:
    """Логиты трёх голов, уверенно указывающие на `value`."""
    logits = np.zeros((3, ns.NUM_CLASSES))
    for h, cls in enumerate(ns.encode_heads(value)):
        logits[h, cls] = strength
    return logits


def _ctc_path_logits(path: list[int], strength: float = 8.0) -> np.ndarray:
    """Логиты CTC `(T, 11)`, уверенно указывающие на путь классов `path` (0 — blank)."""
    logits = np.zeros((len(path), ns.NUM_CLASSES))
    logits[np.arange(len(path)), path] = strength
    return logits


def _brute_ctc(log_probs: np.ndarray, labels: list[int], blank: int = 0) -> float:
    """CTC по определению: сумма вероятностей всех путей, которые сжимаются в `labels`."""
    n_frames, n_classes = log_probs.shape
    total = 0.0
    for path in itertools.product(range(n_classes), repeat=n_frames):
        collapsed = [c for i, c in enumerate(path) if c != blank and (i == 0 or c != path[i - 1])]
        if collapsed == labels:
            total += math.exp(sum(log_probs[t, c] for t, c in enumerate(path)))
    return math.log(total) if total > 0 else float("-inf")


# --- кодирование и кандидаты -----------------------------------------------------------------


def test_encode_heads_right_aligned() -> None:
    e = ns.EMPTY
    assert ns.encode_heads(243) == (2, 4, 3)
    assert ns.encode_heads(43) == (e, 4, 3)
    assert ns.encode_heads(7) == (e, e, 7)
    assert ns.encode_heads(100) == (1, 0, 0)
    assert ns.encode_heads(0) == (e, e, 0)
    with pytest.raises(ValueError):
        ns.encode_heads(1000)


def test_encode_ctc_shifts_digits_past_blank() -> None:
    assert ns.encode_ctc(243) == [3, 5, 4]
    assert ns.encode_ctc(100) == [2, 1, 1]
    assert ns.encode_ctc(0) == [1]


@pytest.mark.parametrize(
    ("raw", "expected"),
    [("243", 243), (" 17 ", 17), ("043", 43), (5, 5), (np.int64(12), 12), ("0", 0),
     ("", None), ("1a", None), ("1000", None), (-1, None), (True, None), ("٣", None)],
)
def test_canonical(raw: object, expected: int | None) -> None:
    assert ns.canonical(raw) == expected  # type: ignore[arg-type]


# --- подход А: головы ------------------------------------------------------------------------


def test_heads_score_is_sum_of_head_log_probs() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(size=(3, 11)) * 2
    lp = ns.log_softmax(logits)
    scores = ns.score_candidates(logits, ["243", "43", "7", 100], kind="heads")
    e = ns.EMPTY
    assert scores["243"] == pytest.approx(lp[0, 2] + lp[1, 4] + lp[2, 3])
    assert scores["43"] == pytest.approx(lp[0, e] + lp[1, 4] + lp[2, 3])
    assert scores["7"] == pytest.approx(lp[0, e] + lp[1, e] + lp[2, 7])
    assert scores["100"] == pytest.approx(lp[0, 1] + lp[1, 0] + lp[2, 0])


def test_heads_confident_logits_rank_truth_first() -> None:
    logits = _onehot_heads(248)
    scores = ns.score_candidates(logits, ns.window_candidates(248), kind="heads")
    assert max(scores, key=scores.__getitem__) == "248"
    assert ns.truth_rank(scores, "248") == 1
    assert ns.top_numbers(logits, kind="heads")[0][0] == 248


def test_heads_full_space_sums_to_one() -> None:
    """Три головы задают распределение на 11³ сочетаниях — сумма по ним равна 1."""
    rng = np.random.default_rng(1)
    logits = rng.normal(size=(3, 11))
    lp = ns.log_softmax(logits)
    total = sum(math.exp(lp[0, a] + lp[1, b] + lp[2, c])
                for a in range(11) for b in range(11) for c in range(11))
    assert total == pytest.approx(1.0)


def test_temperature_flattens_heads() -> None:
    logits = _onehot_heads(120, strength=4.0)
    sharp = ns.score_candidates(logits, ["120"], kind="heads")["120"]
    flat = ns.score_candidates(logits, ["120"], kind="heads", temperature=4.0)["120"]
    assert flat < sharp < 0


def test_heads_shape_checked() -> None:
    with pytest.raises(ValueError):
        ns.score_candidates(np.zeros((2, 11)), ["1"], kind="heads")


# --- подход Б: CTC ---------------------------------------------------------------------------


@pytest.mark.parametrize("labels", [[1], [2, 3], [2, 2], [1, 2, 1], [3, 3, 3], []])
def test_ctc_matches_brute_force(labels: list[int]) -> None:
    rng = np.random.default_rng(len(labels) * 7 + sum(labels))
    log_probs = ns.log_softmax(rng.normal(size=(6, 4)))
    got = ns.ctc_log_likelihoods(log_probs, np.array([labels], dtype=np.int64))[0]
    assert got == pytest.approx(_brute_ctc(log_probs, labels), abs=1e-9)


def test_ctc_too_long_for_frames_is_impossible() -> None:
    """«111» требует 5 кадров (blank между повторами): на 4 кадрах — `-inf`, на 5 — нет."""
    lp4 = ns.log_softmax(np.zeros((4, 11)))
    lp5 = ns.log_softmax(np.zeros((5, 11)))
    labels = np.array([ns.encode_ctc(111)])
    assert ns.ctc_log_likelihoods(lp4, labels)[0] == -np.inf
    assert np.isfinite(ns.ctc_log_likelihoods(lp5, labels)[0])


def test_ctc_batch_equals_one_by_one() -> None:
    rng = np.random.default_rng(3)
    logits = rng.normal(size=(12, 11))
    values = [5, 17, 243, 248, 111, 100, 7]
    batch = ns.ctc_number_log_likelihoods(logits, values)
    lp = ns.log_softmax(logits)
    for v, s in zip(values, batch, strict=True):
        single = ns.ctc_log_likelihoods(lp, np.array([ns.encode_ctc(v)]))[0]
        assert s == pytest.approx(single)


def test_ctc_confident_path_ranks_truth_first() -> None:
    # Путь «2 2 blank 4 blank 8» сжимается в «248».
    logits = _ctc_path_logits([3, 3, 0, 5, 0, 9, 9, 0])
    scores = ns.score_candidates(logits, ns.window_candidates(248), kind="ctc")
    assert ns.truth_rank(scores, "248") == 1
    assert ns.top_numbers(logits, kind="ctc")[0][0] == 248
    # Повтор без blank сжимается в одну цифру: «4 4» — это «4», а не «44».
    single = _ctc_path_logits([0, 5, 5, 0])
    s = ns.score_candidates(single, ["4", "44"], kind="ctc")
    assert s["4"] > s["44"]


def test_ctc_matches_torch_ctc_loss() -> None:
    torch = pytest.importorskip("torch")
    rng = np.random.default_rng(4)
    logits = rng.normal(size=(20, 11)).astype(np.float32)
    values = [3, 42, 243, 111, 300]
    ours = ns.ctc_number_log_likelihoods(logits, values)
    lp = torch.log_softmax(torch.from_numpy(logits).double(), dim=1)
    for v, s in zip(values, ours, strict=True):
        target = torch.tensor([ns.encode_ctc(v)])
        nll = torch.nn.functional.ctc_loss(
            lp.unsqueeze(1), target, torch.tensor([20]), torch.tensor([target.shape[1]]),
            blank=ns.CTC_BLANK, reduction="none",
        )
        assert s == pytest.approx(-float(nll[0]), abs=1e-8)


# --- общий интерфейс ---------------------------------------------------------------------------


@pytest.mark.parametrize("kind", ["heads", "ctc"])
def test_score_candidates_keys_invalid_and_duplicates(kind: str) -> None:
    rng = np.random.default_rng(5)
    logits = rng.normal(size=(3, 11) if kind == "heads" else (10, 11))
    scores = ns.score_candidates(logits, ["243", 243, "043", "x", "", "1000"], kind=kind)
    assert set(scores) == {"243", "043", "x", "", "1000"}
    assert scores["043"] == pytest.approx(
        ns.score_candidates(logits, ["43"], kind=kind)["43"]
    )
    assert scores["x"] == scores[""] == scores["1000"] == float("-inf")
    assert np.isfinite(scores["243"])
    assert ns.score_candidates(logits, [], kind=kind) == {}


@pytest.mark.parametrize("kind", ["heads", "ctc"])
def test_number_log_probs_normalized_and_consistent(kind: str) -> None:
    rng = np.random.default_rng(6)
    logits = rng.normal(size=(3, 11) if kind == "heads" else (16, 11)) * 2
    values, logp = ns.number_log_probs(logits, kind=kind)
    assert values[0] == 1 and values[-1] == ns.MAX_NUMBER
    assert np.exp(logp).sum() == pytest.approx(1.0)
    # Разности логарифмов те же, что у score_candidates: нормировка — общий сдвиг.
    s = ns.score_candidates(logits, ["17", "243"], kind=kind)
    assert logp[242] - logp[16] == pytest.approx(s["243"] - s["17"])
    top = ns.top_numbers(logits, kind=kind)
    assert len(top) == ns.TOP_K
    assert [p for _, p in top] == sorted((p for _, p in top), reverse=True)


def test_unknown_kind_rejected() -> None:
    with pytest.raises(ValueError):
        ns.score_candidates(np.zeros((3, 11)), ["1"], kind="beam")


def test_truth_rank_is_pessimistic_on_ties() -> None:
    assert ns.truth_rank({"1": -1.0, "2": -1.0, "3": -2.0}, "1") == 2
    assert ns.truth_rank({"1": -0.5, "2": -1.0}, "1") == 1


def test_window_candidates_clipped_at_one() -> None:
    assert ns.window_candidates(3, 5) == [str(v) for v in range(1, 9)]
    assert len(ns.window_candidates(200)) == 31
