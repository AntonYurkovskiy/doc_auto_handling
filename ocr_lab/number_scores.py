"""T17: оценка кандидатов номера ваучера — реэкспорт из `app.ocr.number_scores`.

Реализация (только numpy) переехала в рантайм в T18: её вызывает `app.ocr.runtime` на
выходах onnxruntime, и приложению нельзя зависеть от лаборатории. Имена и поведение прежние.
"""

from __future__ import annotations

from app.ocr.number_scores import (
    CTC_BLANK,
    EMPTY,
    HEADS,
    KINDS,
    MAX_DIGITS,
    MAX_NUMBER,
    NUM_CLASSES,
    NUMBER_RANGE,
    TOP_K,
    canonical,
    ctc_log_likelihoods,
    ctc_number_log_likelihoods,
    encode_ctc,
    encode_heads,
    heads_log_likelihoods,
    log_softmax,
    number_log_probs,
    score_candidates,
    top_numbers,
    truth_rank,
    value_log_likelihoods,
    window_candidates,
)

__all__ = [
    "CTC_BLANK",
    "EMPTY",
    "HEADS",
    "KINDS",
    "MAX_DIGITS",
    "MAX_NUMBER",
    "NUMBER_RANGE",
    "NUM_CLASSES",
    "TOP_K",
    "canonical",
    "ctc_log_likelihoods",
    "ctc_number_log_likelihoods",
    "encode_ctc",
    "encode_heads",
    "heads_log_likelihoods",
    "log_softmax",
    "number_log_probs",
    "score_candidates",
    "top_numbers",
    "truth_rank",
    "value_log_likelihoods",
    "window_candidates",
]
