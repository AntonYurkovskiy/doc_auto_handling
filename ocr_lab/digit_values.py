"""T16: распределение по значениям двузначного подполя — реэкспорт из `app.ocr.digit_values`.

Реализация (только numpy) переехала в рантайм в T18: её вызывает `app.ocr.runtime` на
выходах onnxruntime, и приложению нельзя зависеть от лаборатории. Имена и поведение прежние.
"""

from __future__ import annotations

from app.ocr.digit_values import (
    NUM_TENS_CLASSES,
    NUM_UNITS_CLASSES,
    PART_RANGES,
    TENS_EMPTY,
    TOP_K,
    log_softmax,
    part_values,
    predict_values,
    top_values,
    truth_log_prob,
    value_log_probs,
)

__all__ = [
    "NUM_TENS_CLASSES",
    "NUM_UNITS_CLASSES",
    "PART_RANGES",
    "TENS_EMPTY",
    "TOP_K",
    "log_softmax",
    "part_values",
    "predict_values",
    "top_values",
    "truth_log_prob",
    "value_log_probs",
]
