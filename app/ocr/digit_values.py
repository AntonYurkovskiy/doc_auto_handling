"""T16: распределение вероятностей по значениям двузначного подполя (только numpy).

Модель цифр (`ocr_lab.models.DigitNet`) выдаёт две головы:

- `tens` — 11 классов: цифра десятков 0..9 и :data:`TENS_EMPTY` (10) — «пусто», то есть
  на бланке написан один знак («9» вместо «09»);
- `units` — 10 классов: цифра единиц 0..9.

Отсюда распределение по значениям части строки (план, «Этап 2», п. 3 промпта T16):

    P(v) ∝ P(tens(v)) · P(units(v)),   v в допустимом диапазоне части,

где для `v < 10` десятки — это «0» **или** «пусто»: `P(tens(v)) = P(tens=0) + P(tens=empty)`.
Нормировка — по допустимому диапазону части (:data:`PART_RANGES`): день 1–31, месяц 1–12,
часы 0–24 (24 допустимо, декодер T19 разрешит его только с минутами 00), минуты 0–59.

Модуль без torch и живёт в рантайме (`app/ocr`): им пользуются `app.ocr.runtime` (onnxruntime)
и лаборатория (`ocr_lab.digit_values` — реэкспорт). Температуры голов (калибровка T18,
`ocr_lab.calibrate`) — необязательные аргументы, по умолчанию 1 (без калибровки).
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Класс «пусто» у головы десятков.
TENS_EMPTY = 10
NUM_TENS_CLASSES = 11
NUM_UNITS_CLASSES = 10

#: Допустимый диапазон значений части строки (включительно).
PART_RANGES: dict[str, tuple[int, int]] = {
    "day": (1, 31),
    "month": (1, 12),
    "hour": (0, 24),
    "minute": (0, 59),
}

#: Сколько значений отдавать в формат T13.
TOP_K = 5


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Численно устойчивый `log_softmax` по оси."""
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def part_values(part: str) -> np.ndarray:
    """Все допустимые значения части строки по возрастанию."""
    lo, hi = PART_RANGES[part]
    return np.arange(lo, hi + 1)


def value_log_probs(
    tens_logits: np.ndarray,
    units_logits: np.ndarray,
    part: str,
    *,
    t_tens: float = 1.0,
    t_units: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """Логарифмы нормированных вероятностей значений части строки.

    `tens_logits` — `(11,)` или `(N, 11)`, `units_logits` — `(10,)` или `(N, 10)` (сырые
    логиты голов). Возвращает `(values, logp)`: `values` — `(V,)` допустимые значения
    части, `logp` — `(V,)` или `(N, V)`, `logsumexp(logp) = 0` по последней оси.
    """
    lt = log_softmax(np.asarray(tens_logits, dtype=np.float64) / t_tens)
    lu = log_softmax(np.asarray(units_logits, dtype=np.float64) / t_units)
    if lt.shape[-1] != NUM_TENS_CLASSES or lu.shape[-1] != NUM_UNITS_CLASSES:
        raise ValueError(f"ожидались головы 11 и 10 классов, получено {lt.shape}, {lu.shape}")
    values = part_values(part)
    # Десятки «0 или пусто» для однозначных значений: истина не различает «09» и «9».
    lead = np.logaddexp(lt[..., 0], lt[..., TENS_EMPTY])
    tens_idx = values // 10
    lt_v = lt[..., tens_idx]
    lt_v = np.where(values < 10, lead[..., None], lt_v)
    joint = lt_v + lu[..., values % 10]
    m = np.max(joint, axis=-1, keepdims=True)
    logz = m + np.log(np.sum(np.exp(joint - m), axis=-1, keepdims=True))
    return values, joint - logz


def top_values(
    values: np.ndarray, logp: np.ndarray, k: int = TOP_K
) -> list[tuple[int, float]]:
    """Top-k пар «значение, вероятность» по убыванию вероятности (одна строка `logp`)."""
    if logp.ndim != 1:
        raise ValueError("top_values: ожидается одна строка вероятностей")
    order = np.argsort(-logp, kind="stable")[:k]
    return [(int(values[i]), float(min(1.0, np.exp(logp[i])))) for i in order]


def predict_values(
    tens_logits: np.ndarray,
    units_logits: np.ndarray,
    parts: Sequence[str],
    *,
    k: int = TOP_K,
    t_tens: float = 1.0,
    t_units: float = 1.0,
) -> list[list[tuple[int, float]]]:
    """Top-k значений для батча: `parts[i]` — часть строки i-го кропа."""
    tens = np.asarray(tens_logits)
    units = np.asarray(units_logits)
    out: list[list[tuple[int, float]]] = []
    for i, part in enumerate(parts):
        values, logp = value_log_probs(tens[i], units[i], part, t_tens=t_tens, t_units=t_units)
        out.append(top_values(values, logp, k))
    return out


def truth_log_prob(
    tens_logits: np.ndarray,
    units_logits: np.ndarray,
    part: str,
    value: int,
    *,
    t_tens: float = 1.0,
    t_units: float = 1.0,
) -> float:
    """`log P(value)` по распределению одного кропа; `-inf`, если значение вне диапазона."""
    values, logp = value_log_probs(tens_logits, units_logits, part, t_tens=t_tens, t_units=t_units)
    hit = np.nonzero(values == value)[0]
    return float(logp[hit[0]]) if hit.size else float("-inf")
