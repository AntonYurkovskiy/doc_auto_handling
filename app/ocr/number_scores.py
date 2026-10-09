"""T17: оценка кандидатов номера ваучера по выходам модели — только numpy, без torch.

Модель номера (`ocr_lab.number_models`) бывает двух видов, у каждого свой выход:

- **`heads`** (подход А) — три головы «сотни», «десятки», «единицы», каждая по
  :data:`NUM_CLASSES` = 11 классов: цифра 0..9 и :data:`EMPTY` (10) — «пусто». Выход одного
  кропа — логиты `(3, 11)` в порядке :data:`HEADS`. Номер выравнивается по правому краю:
  `243` → (2, 4, 3), `43` → (пусто, 4, 3), `7` → (пусто, пусто, 7). Оценка кандидата —
  сумма логарифмов вероятностей его классов по трём головам (произведение вероятностей);
- **`ctc`** (подход Б, CRNN + CTC) — логиты `(T, 11)` по кадрам: класс 0 — blank CTC,
  класс `d + 1` — цифра `d`. Оценка кандидата — CTC-правдоподобие строки цифр
  (forward-алгоритм по всем выравниваниям, :func:`ctc_log_likelihoods`).

Главная функция — :func:`score_candidates`: `(выход модели, ["243", "248", …]) → {кандидат:
log P}`. Её вызывает декодер номера (T20) и рантайм на выходах onnxruntime (`app.ocr.runtime`) —
поэтому модуль без torch и живёт в `app/ocr` (`ocr_lab.number_scores` — реэкспорт).

Кандидат — строка из цифр или целое. Ведущие нули отбрасываются (`"043"` — номер 43):
номера на бланке пишут без них. Кандидат вне `0..999`, пустой или с не-цифрами получает
`-inf`. Оценки — ненормированные по набору кандидатов логарифмы вероятностей модели
(у `heads` — по всем 11³ сочетаниям классов, у `ctc` — по всем строкам), поэтому их можно
складывать с приорами декодера как есть. Нормированное распределение по всем номерам
`1..999` (для top-k формата T13) даёт :func:`number_log_probs`.

Температура (калибровка T18) — необязательный аргумент, по умолчанию 1. Это число (одна
температура на все логиты) либо, для `heads`, массив из трёх чисел — по одной на голову.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np

#: Температура: одно число либо (для голов подхода А) по числу на голову.
Temperature = float | Sequence[float] | np.ndarray

#: Головы подхода А, слева направо.
HEADS: tuple[str, ...] = ("hundreds", "tens", "units")
#: Класс «пусто» у голов подхода А.
EMPTY = 10
#: Число классов головы (А) и кадра CTC (Б): 10 цифр + «пусто» / blank.
NUM_CLASSES = 11
#: Класс blank в CTC; цифра `d` — класс `d + 1`.
CTC_BLANK = 0
#: Больше трёх цифр номер не бывает (номера по буксиру за год — до ~340).
MAX_DIGITS = 3
MAX_NUMBER = 10**MAX_DIGITS - 1
#: Диапазон номеров для распределения top-k (номера начинаются с 1 каждый год).
NUMBER_RANGE: tuple[int, int] = (1, MAX_NUMBER)
#: Виды выхода модели.
KINDS: tuple[str, ...] = ("heads", "ctc")
#: Сколько номеров отдавать в формат T13.
TOP_K = 5


def log_softmax(logits: np.ndarray, axis: int = -1) -> np.ndarray:
    """Численно устойчивый `log_softmax` по оси (float64)."""
    x = np.asarray(logits, dtype=np.float64)
    m = np.max(x, axis=axis, keepdims=True)
    z = x - m
    return z - np.log(np.sum(np.exp(z), axis=axis, keepdims=True))


def canonical(candidate: str | int) -> int | None:
    """Кандидат → номер `0..999` или `None`, если это не номер.

    Строка допускается только из цифр (пробелы по краям срезаются), ведущие нули
    отбрасываются. `bool` — не номер.
    """
    if isinstance(candidate, bool):
        return None
    if isinstance(candidate, int | np.integer):
        value = int(candidate)
    else:
        text = str(candidate).strip()
        if not text or not text.isascii() or not text.isdigit():
            return None
        value = int(text)
    return value if 0 <= value <= MAX_NUMBER else None


def encode_heads(value: int) -> tuple[int, int, int]:
    """Номер → классы голов (сотни, десятки, единицы), выравнивание по правому краю."""
    if not 0 <= value <= MAX_NUMBER:
        raise ValueError(f"номер вне 0..{MAX_NUMBER}: {value}")
    hundreds = value // 100 if value >= 100 else EMPTY
    tens = (value // 10) % 10 if value >= 10 else EMPTY
    return hundreds, tens, value % 10


def encode_ctc(value: int) -> list[int]:
    """Номер → метки CTC (цифра `d` → класс `d + 1`), без ведущих нулей."""
    if not 0 <= value <= MAX_NUMBER:
        raise ValueError(f"номер вне 0..{MAX_NUMBER}: {value}")
    return [int(ch) + 1 for ch in str(value)]


# --- подход А: три головы -------------------------------------------------------------------


def _head_temperature(temperature: Temperature) -> float | np.ndarray:
    """Число или массив `(3,)` → делитель логитов `(3, 11)` (по температуре на голову)."""
    t = np.asarray(temperature, dtype=np.float64)
    if t.ndim == 0:
        return float(t)
    if t.shape != (len(HEADS),):
        raise ValueError(f"температура голов: ожидалось число или {len(HEADS)} чисел")
    return t[:, None]


def heads_log_likelihoods(
    logits: np.ndarray, values: Sequence[int], *, temperature: Temperature = 1.0
) -> np.ndarray:
    """`log P(номер)` для каждого из `values` по логитам голов `(3, 11)` одного кропа."""
    lp = log_softmax(np.asarray(logits, dtype=np.float64) / _head_temperature(temperature))
    if lp.shape != (len(HEADS), NUM_CLASSES):
        raise ValueError(f"ожидались логиты голов (3, 11), получено {lp.shape}")
    vals = np.asarray(values, dtype=np.int64)
    if vals.size == 0:
        return np.zeros(0)
    hundreds = np.where(vals >= 100, vals // 100, EMPTY)
    tens = np.where(vals >= 10, (vals // 10) % 10, EMPTY)
    units = vals % 10
    return lp[0, hundreds] + lp[1, tens] + lp[2, units]


# --- подход Б: CTC --------------------------------------------------------------------------


def ctc_log_likelihoods(
    log_probs: np.ndarray, labels: np.ndarray, *, blank: int = CTC_BLANK
) -> np.ndarray:
    """CTC-правдоподобие `log P(labels | log_probs)` для батча строк одной длины.

    `log_probs` — `(T, C)`, логарифмы вероятностей по кадрам (после `log_softmax`);
    `labels` — `(n, L)`, метки без blank, `L ≥ 0`. Forward-алгоритм в логарифмах по
    расширенной строке `blank, l1, blank, l2, …, lL, blank`; переход через одну позицию
    разрешён, если соседние метки различны. Строка длиннее, чем позволяет `T`, получает
    `-inf`. Совпадает с `torch.nn.functional.ctc_loss(..., reduction="none")` со знаком минус.
    """
    lp = np.asarray(log_probs, dtype=np.float64)
    lab = np.asarray(labels, dtype=np.int64)
    if lab.ndim != 2:
        raise ValueError("ctc_log_likelihoods: labels должны быть (n, L)")
    n, length = lab.shape
    n_frames = lp.shape[0]
    if n == 0:
        return np.zeros(0)
    if length == 0:
        return np.full(n, float(np.sum(lp[:, blank])))
    size = 2 * length + 1
    ext = np.full((n, size), blank, dtype=np.int64)
    ext[:, 1::2] = lab
    # Переход s-2 → s: только на метку (не blank), отличную от метки через одну позицию.
    skip = np.zeros((n, size), dtype=bool)
    skip[:, 3::2] = lab[:, 1:] != lab[:, :-1]
    neg_inf = -np.inf
    alpha = np.full((n, size), neg_inf)
    alpha[:, 0] = lp[0, blank]
    alpha[:, 1] = lp[0, ext[:, 1]]
    for t in range(1, n_frames):
        prev1 = np.concatenate([np.full((n, 1), neg_inf), alpha[:, :-1]], axis=1)
        prev2 = np.concatenate([np.full((n, 2), neg_inf), alpha[:, :-2]], axis=1)
        prev2 = np.where(skip, prev2, neg_inf)
        stacked = np.stack([alpha, prev1, prev2])
        m = np.max(stacked, axis=0)
        finite = np.isfinite(m)
        safe = np.where(finite, m, 0.0)
        with np.errstate(divide="ignore"):
            acc = safe + np.log(np.sum(np.exp(stacked - safe), axis=0))
        alpha = np.where(finite, acc, neg_inf) + lp[t][ext]
    return np.logaddexp(alpha[:, -1], alpha[:, -2])


def ctc_number_log_likelihoods(
    logits: np.ndarray, values: Sequence[int], *, temperature: float = 1.0
) -> np.ndarray:
    """`log P(номер)` для каждого из `values` по логитам CTC `(T, 11)` одного кропа."""
    lp = log_softmax(np.asarray(logits, dtype=np.float64) / temperature)
    if lp.ndim != 2 or lp.shape[1] != NUM_CLASSES:
        raise ValueError(f"ожидались логиты CTC (T, 11), получено {lp.shape}")
    vals = [int(v) for v in values]
    out = np.full(len(vals), -np.inf)
    by_len: dict[int, list[int]] = {}
    for i, v in enumerate(vals):
        by_len.setdefault(len(str(v)), []).append(i)
    for _length, idx in by_len.items():
        labels = np.array([encode_ctc(vals[i]) for i in idx], dtype=np.int64)
        out[idx] = ctc_log_likelihoods(lp, labels)
    return out


# --- общий интерфейс ------------------------------------------------------------------------


def _check_kind(kind: str) -> None:
    if kind not in KINDS:
        raise ValueError(f"неизвестный вид выхода модели: {kind} (есть {', '.join(KINDS)})")


def value_log_likelihoods(
    output: np.ndarray, values: Sequence[int], *, kind: str, temperature: Temperature = 1.0
) -> np.ndarray:
    """`log P(номер)` для списка номеров `0..999` по выходу модели одного кропа."""
    _check_kind(kind)
    if kind == "heads":
        return heads_log_likelihoods(output, values, temperature=temperature)
    t = np.asarray(temperature, dtype=np.float64)
    if t.size != 1:
        raise ValueError("у CTC одна температура на все логиты")
    return ctc_number_log_likelihoods(output, values, temperature=float(t.reshape(-1)[0]))


def score_candidates(
    output: np.ndarray,
    candidates: Sequence[str | int],
    *,
    kind: str,
    temperature: Temperature = 1.0,
) -> dict[str, float]:
    """Оценить кандидатов номера по выходу модели одного кропа: `{кандидат: log P}`.

    `output` — логиты `(3, 11)` для `kind="heads"` или `(T, 11)` для `kind="ctc"`.
    Ключ словаря — кандидат как передан (`str(c)`); недопустимый кандидат получает
    `-inf`. Повторяющиеся кандидаты дают один ключ.
    """
    _check_kind(kind)
    keys = [str(c) for c in candidates]
    numbers = [canonical(c) for c in candidates]
    valid = [n for n in numbers if n is not None]
    scores = value_log_likelihoods(output, valid, kind=kind, temperature=temperature)
    by_number = dict(zip(valid, (float(s) for s in scores), strict=True))
    return {
        key: (by_number[n] if n is not None else float("-inf"))
        for key, n in zip(keys, numbers, strict=True)
    }


def number_log_probs(
    output: np.ndarray,
    *,
    kind: str,
    temperature: Temperature = 1.0,
    value_range: tuple[int, int] = NUMBER_RANGE,
) -> tuple[np.ndarray, np.ndarray]:
    """Нормированное по `value_range` распределение номеров: `(values, logp)`.

    `logsumexp(logp) = 0`. Это перенормировка оценок :func:`score_candidates` на все
    допустимые номера — вход top-k для формата T13 и метрик.
    """
    lo, hi = value_range
    values = np.arange(lo, hi + 1)
    scores = value_log_likelihoods(output, values.tolist(), kind=kind, temperature=temperature)
    m = np.max(scores)
    logz = m + np.log(np.sum(np.exp(scores - m)))
    return values, scores - logz


def top_numbers(
    output: np.ndarray, *, kind: str, k: int = TOP_K, temperature: Temperature = 1.0
) -> list[tuple[int, float]]:
    """Top-k номеров с вероятностями (по убыванию) — для поля `voucher_number` формата T13."""
    values, logp = number_log_probs(output, kind=kind, temperature=temperature)
    order = np.argsort(-logp, kind="stable")[:k]
    return [(int(values[i]), float(min(1.0, np.exp(logp[i])))) for i in order]


def truth_rank(scores: dict[str, float], truth: str) -> int:
    """Ранг истины среди кандидатов (1 — лучший), при равенстве оценок — пессимистично."""
    s_true = scores[truth]
    return 1 + sum(1 for key, s in scores.items() if key != truth and s >= s_true)


def window_candidates(value: int, radius: int = 15) -> list[str]:
    """Кандидаты `value ± radius` (не меньше 1) — окно метрики ранжирования T17."""
    lo = max(NUMBER_RANGE[0], value - radius)
    hi = min(NUMBER_RANGE[1], value + radius)
    return [str(v) for v in range(lo, hi + 1)]
