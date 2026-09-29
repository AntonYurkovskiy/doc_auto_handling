"""Совместный декодер записи ваучера (T19): ядро.

Декодер не читает подполя по отдельности, а ищет самую вероятную запись ваучера целиком:
четыре метки времени (`left_base`, `started_work`, `finished_work`, `arrived_base`),
которые не противоречат жёстким правилам.

Модель. Запись задаётся базовой датой B (дата выхода на бланке), шаблоном смещений дней
`o` по строкам в хронологическом порядке (`o_left = 0`, смещения не убывают) и парами
«час, минута» для каждой строки. На бланке строка r записана как дата `B + o_r` и время
`hour:minute`; час 24 допустим только с минутами 00 и означает 00:00 следующих суток.
Оценка записи — сумма логарифмов:

- картинка: `Σ w_part · log P_img(значение)` по 16 подполям; пропущенное подполе даёт
  равномерное распределение, значение вне top-k — остаток массы поровну (с полом);
- приоры: минуты, часы по строкам, длительности соседних участков цепочки, шаблон
  смещений дней, отклонение «Начало − время заявки» (без окна, только с полом),
  штраф за год вне контекста;
- жёсткие правила: `Выход ≤ Начало ≤ Окончание ≤ Приход`, час 24 только с минутами 00,
  дата валидна (получается сама — даты строятся календарной арифметикой от B).

Поиск. При фиксированных B и шаблоне смещений оценка раскладывается на унарные члены
строк и парные члены соседних строк цепочки. Поэтому нормировка и маргиналы считаются
точно (сумма-произведение по цепочке, батчем по всем B), а top-N записей — точным
k-лучшим динамическим программированием с отсечением по верхней границе цепочки.
Вероятности — softmax по всему перечисленному пространству кандидатов.

Зависимости — только numpy и stdlib.
"""

from __future__ import annotations

import heapq
import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field, replace
from datetime import date, datetime, time, timedelta
from pathlib import Path
from typing import Any

import numpy as np

# Строки бланка в порядке из `_common.md` и в хронологическом порядке цепочки.
ROWS: tuple[str, ...] = ("left_base", "arrived_base", "started_work", "finished_work")
CHAIN: tuple[str, ...] = ("left_base", "started_work", "finished_work", "arrived_base")
PARTS: tuple[str, ...] = ("day", "month", "hour", "minute")
PART_RANGE: dict[str, tuple[int, int]] = {
    "day": (1, 31),
    "month": (1, 12),
    "hour": (0, 24),
    "minute": (0, 59),
}
# Участки цепочки: (строка-начало, строка-конец).
LEGS: tuple[tuple[str, str], ...] = tuple(zip(CHAIN[:-1], CHAIN[1:], strict=True))
LEG_NAMES: tuple[str, ...] = tuple(f"{a}->{b}" for a, b in LEGS)

MINUTES_PER_DAY = 1440
_NEG_INF = float("-inf")

SubfieldDist = Sequence[tuple[int, float]] | Sequence[Sequence[float]]
DecoderInputs = Mapping[str, SubfieldDist]
FormRow = tuple[date, int, int]
"""Строка записи в написании бланка: дата на бланке, час (0–24), минута."""


def subfield(row: str, part: str) -> str:
    """Имя подполя, например `left_base.hour`."""
    return f"{row}.{part}"


SUBFIELDS: tuple[str, ...] = tuple(subfield(r, p) for r in ROWS for p in PARTS)


# ---------------------------------------------------------------------------
# Приоры: плотности и таблицы
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class BinnedLogDensity:
    """Кусочно-постоянная лог-плотность на минуту.

    Бин i — полуинтервал `[edges[i], edges[i+1])`, в нём лог-плотность `logp[i]`
    (лог массы на одну минуту). Вне бинов и ниже пола — `floor`: хвостов не отрезаем.
    """

    edges: tuple[float, ...]
    logp: tuple[float, ...]
    floor: float

    def __post_init__(self) -> None:
        if len(self.edges) != len(self.logp) + 1:
            raise ValueError("число границ должно быть на 1 больше числа бинов")
        if any(b <= a for a, b in zip(self.edges[:-1], self.edges[1:], strict=True)):
            raise ValueError("границы бинов должны строго возрастать")
        object.__setattr__(self, "_edges", np.asarray(self.edges, dtype=float))
        object.__setattr__(
            self, "_logp", np.maximum(np.asarray(self.logp, dtype=float), self.floor)
        )

    def __call__(self, x: np.ndarray | float) -> np.ndarray:
        arr = np.asarray(x, dtype=float)
        edges: np.ndarray = self._edges  # type: ignore[attr-defined]
        logp: np.ndarray = self._logp  # type: ignore[attr-defined]
        idx = np.searchsorted(edges, arr, side="right") - 1
        inside = (idx >= 0) & (idx < len(logp))
        return np.where(inside, logp[np.clip(idx, 0, len(logp) - 1)], self.floor)

    @classmethod
    def from_masses(
        cls, edges: Sequence[float], masses: Sequence[float], floor_per_minute: float
    ) -> BinnedLogDensity:
        """Собрать плотность из масс по бинам (нормируются) и пола на минуту."""
        m = np.asarray(masses, dtype=float)
        if m.shape[0] != len(edges) - 1 or np.any(m < 0) or m.sum() <= 0:
            raise ValueError("массы бинов некорректны")
        m = m / m.sum()
        widths = np.diff(np.asarray(edges, dtype=float))
        with np.errstate(divide="ignore"):
            logp = np.log(m / widths)
        floor = math.log(floor_per_minute)
        return cls(tuple(float(e) for e in edges), tuple(float(v) for v in logp), floor)

    def to_json(self) -> dict[str, Any]:
        return {
            "edges": list(self.edges),
            "logp": [v if math.isfinite(v) else None for v in self.logp],
            "floor": self.floor,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> BinnedLogDensity:
        floor = float(data["floor"])
        logp = tuple(floor if v is None else float(v) for v in data["logp"])
        return cls(tuple(float(e) for e in data["edges"]), logp, floor)


# Бины длительностей участков, минуты. Шаг 10 минут до 6 часов, дальше шире.
DURATION_EDGES: tuple[float, ...] = (
    0.0,
    1.0,
    *(float(v) for v in range(10, 361, 10)),
    420.0,
    480.0,
    600.0,
    720.0,
    960.0,
    1440.0,
    2160.0,
    2880.0,
    4320.0,
)
# Бины отклонения «Начало − время заявки», минуты: от −10 до +10 суток.
APP_EDGES: tuple[float, ...] = (
    -14400.0,
    -4320.0,
    -1440.0,
    -720.0,
    -360.0,
    -180.0,
    -120.0,
    -60.0,
    0.0,
    60.0,
    120.0,
    180.0,
    240.0,
    360.0,
    480.0,
    720.0,
    1440.0,
    2880.0,
    4320.0,
    8640.0,
    14400.0,
)


def _lognormal_cdf(x: float, median: float, sigma: float) -> float:
    if x <= 0:
        return 0.0
    return 0.5 * (1.0 + math.erf(math.log(x / median) / (sigma * math.sqrt(2.0))))


def default_duration_masses(median: float) -> list[float]:
    """Массы бинов длительности по умолчанию: смесь «ноль», основной и тяжёлый хвост.

    Параметры подобраны под цифры плана (99 % ваучеров укладываются в 5,3 ч от выхода
    до прихода) и заменяются эмпирикой `train` в `ocr_lab/fit_decoder_priors.py`.
    """
    zero, core, heavy = 0.01, 0.92, 0.07
    masses: list[float] = []
    for lo, hi in zip(DURATION_EDGES[:-1], DURATION_EDGES[1:], strict=True):
        mass = 0.0
        if lo <= 0.0 < hi:
            mass += zero
        mass += core * (_lognormal_cdf(hi, median, 0.8) - _lognormal_cdf(lo, median, 0.8))
        mass += heavy * (_lognormal_cdf(hi, 3 * median, 1.5) - _lognormal_cdf(lo, 3 * median, 1.5))
        masses.append(mass)
    return masses


# Медианы участков по умолчанию, минуты.
_DEFAULT_LEG_MEDIANS: dict[str, float] = dict(zip(LEG_NAMES, (40.0, 90.0, 40.0), strict=True))

# Массы бинов APP_EDGES по умолчанию: медиана +4 ч, в ±3 ч — 43 %, p99 — +6 суток.
DEFAULT_APP_MASSES: tuple[float, ...] = (
    0.003,  # −10…−3 сут
    0.007,  # −3…−1 сут
    0.007,  # −24…−12 ч
    0.006,  # −12…−6 ч
    0.007,  # −6…−3 ч
    0.04,  # −3…−2 ч
    0.06,  # −2…−1 ч
    0.09,  # −1…0 ч
    0.09,  # 0…1 ч
    0.08,  # 1…2 ч
    0.07,  # 2…3 ч
    0.04,  # 3…4 ч
    0.09,  # 4…6 ч
    0.07,  # 6…8 ч
    0.07,  # 8…12 ч
    0.08,  # 12…24 ч
    0.08,  # 1…2 сут
    0.05,  # 2…3 сут
    0.05,  # 3…6 сут
    0.01,  # 6…10 сут
)


def default_minute_probs() -> np.ndarray:
    """Приор минут по умолчанию: кратные 10 — 99,8 %, кратные только 5 — 0,19 %."""
    probs = np.empty(60)
    for m in range(60):
        if m % 10 == 0:
            probs[m] = 0.998 / 6
        elif m % 5 == 0:
            probs[m] = 0.0019 / 6
        else:
            probs[m] = 0.0001 / 48
    return probs / probs.sum()


# Шаблоны смещений дней по строкам CHAIN: через полночь переходят 2,2 % ваучеров.
DEFAULT_PATTERN_PROBS: dict[tuple[int, ...], float] = {
    (0, 0, 0, 0): 0.978,
    (0, 0, 0, 1): 0.010,
    (0, 0, 1, 1): 0.006,
    (0, 1, 1, 1): 0.006,
}


def chain_patterns(max_day_offset: int) -> list[tuple[int, ...]]:
    """Все неубывающие шаблоны смещений с `o_left = 0` и `o ≤ max_day_offset`."""
    out: list[tuple[int, ...]] = []

    def rec(prefix: tuple[int, ...]) -> None:
        if len(prefix) == len(CHAIN):
            out.append(prefix)
            return
        for o in range(prefix[-1], max_day_offset + 1):
            rec((*prefix, o))

    rec((0,))
    return out


def pattern_key(pattern: Sequence[int]) -> str:
    return ",".join(str(int(o)) for o in pattern)


def parse_pattern_key(key: str) -> tuple[int, ...]:
    return tuple(int(v) for v in key.split(","))


@dataclass(frozen=True)
class DecoderPriors:
    """Приоры декодера. Все значения — натуральные логарифмы."""

    minute_logp: tuple[float, ...]
    hour_logp: Mapping[str, tuple[float, ...]]
    durations: Mapping[str, BinnedLogDensity]
    durations_by_work_type: Mapping[str, Mapping[str, BinnedLogDensity]]
    pattern_logp: Mapping[tuple[int, ...], float]
    pattern_floor: float
    max_day_offset: int
    app_deviation: BinnedLogDensity
    other_year_logp: float = -3.0

    def __post_init__(self) -> None:
        if len(self.minute_logp) != 60:
            raise ValueError("minute_logp: нужно 60 значений")
        for row in CHAIN:
            if len(self.hour_logp.get(row, ())) != 25:
                raise ValueError(f"hour_logp[{row}]: нужно 25 значений (0–24)")
        # Кэш массивов для векторных вычислений.
        object.__setattr__(self, "_minute", np.asarray(self.minute_logp, dtype=float))
        object.__setattr__(
            self,
            "_hour",
            {row: np.asarray(self.hour_logp[row], dtype=float) for row in CHAIN},
        )
        for name in LEG_NAMES:
            if name not in self.durations:
                raise ValueError(f"нет плотности длительности {name}")

    @property
    def minute_array(self) -> np.ndarray:
        return self._minute  # type: ignore[attr-defined]

    def hour_array(self, row: str) -> np.ndarray:
        return self._hour[row]  # type: ignore[attr-defined]

    def leg_densities(self, work_type: str | None) -> Mapping[str, BinnedLogDensity]:
        """Плотности длительностей: по виду работ, если он есть в приорах, иначе общие."""
        if work_type and work_type in self.durations_by_work_type:
            specific = self.durations_by_work_type[work_type]
            return {name: specific.get(name, self.durations[name]) for name in LEG_NAMES}
        return self.durations

    def pattern_value(self, pattern: tuple[int, ...]) -> float:
        return self.pattern_logp.get(pattern, self.pattern_floor)

    @classmethod
    def default(cls) -> DecoderPriors:
        """Приоры по цифрам плана — чтобы декодер работал без файла приоров."""
        durations = {
            name: BinnedLogDensity.from_masses(
                DURATION_EDGES, default_duration_masses(_DEFAULT_LEG_MEDIANS[name]), 1e-9
            )
            for name in LEG_NAMES
        }
        hour = tuple([math.log(1.0 / 25)] * 25)
        return cls(
            minute_logp=tuple(float(v) for v in np.log(default_minute_probs())),
            hour_logp={row: hour for row in CHAIN},
            durations=durations,
            durations_by_work_type={},
            pattern_logp={p: math.log(v) for p, v in DEFAULT_PATTERN_PROBS.items()},
            pattern_floor=math.log(1e-4),
            max_day_offset=1,
            app_deviation=BinnedLogDensity.from_masses(APP_EDGES, DEFAULT_APP_MASSES, 1e-8),
        )

    def to_json(self) -> dict[str, Any]:
        return {
            "minute_logp": list(self.minute_logp),
            "hour_logp": {row: list(v) for row, v in self.hour_logp.items()},
            "durations": {k: v.to_json() for k, v in self.durations.items()},
            "durations_by_work_type": {
                wt: {k: v.to_json() for k, v in legs.items()}
                for wt, legs in self.durations_by_work_type.items()
            },
            "pattern_logp": {pattern_key(p): v for p, v in self.pattern_logp.items()},
            "pattern_floor": self.pattern_floor,
            "max_day_offset": self.max_day_offset,
            "app_deviation": self.app_deviation.to_json(),
            "other_year_logp": self.other_year_logp,
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> DecoderPriors:
        return cls(
            minute_logp=tuple(float(v) for v in data["minute_logp"]),
            hour_logp={row: tuple(float(x) for x in v) for row, v in data["hour_logp"].items()},
            durations={k: BinnedLogDensity.from_json(v) for k, v in data["durations"].items()},
            durations_by_work_type={
                wt: {k: BinnedLogDensity.from_json(v) for k, v in legs.items()}
                for wt, legs in data.get("durations_by_work_type", {}).items()
            },
            pattern_logp={parse_pattern_key(k): float(v) for k, v in data["pattern_logp"].items()},
            pattern_floor=float(data["pattern_floor"]),
            max_day_offset=int(data["max_day_offset"]),
            app_deviation=BinnedLogDensity.from_json(data["app_deviation"]),
            other_year_logp=float(data.get("other_year_logp", -3.0)),
        )


@dataclass(frozen=True)
class DecoderWeights:
    """Веса членов оценки: подполя картинки по частям и множители приоров."""

    day: float = 1.0
    month: float = 1.0
    hour: float = 1.0
    minute: float = 1.0
    prior_minute: float = 1.0
    prior_hour: float = 1.0
    prior_duration: float = 1.0
    prior_pattern: float = 1.0
    prior_app: float = 1.0


WEIGHT_NAMES: tuple[str, ...] = tuple(DecoderWeights.__dataclass_fields__)


@dataclass(frozen=True)
class SearchParams:
    """Параметры перечисления кандидатов."""

    top_k: Mapping[str, int] = field(
        default_factory=lambda: {"day": 3, "month": 2, "hour": 4, "minute": 3}
    )
    min_prob: float = 0.01
    # Минуты, которые всегда добавляются в кандидаты (их поддерживает приор).
    extra_minutes: tuple[int, ...] = (0, 10, 20, 30, 40, 50)
    prob_floor: float = 1e-6
    top_n: int = 10
    max_base_dates: int = 40
    marginal_top: int = 3
    # Окно вокруг времени заявки для базовых дат, если дни не прочитаны вовсе.
    app_window_days: tuple[int, int] = (-10, 3)

    def to_json(self) -> dict[str, Any]:
        data = asdict(self)
        data["top_k"] = dict(self.top_k)
        data["extra_minutes"] = list(self.extra_minutes)
        data["app_window_days"] = list(self.app_window_days)
        return data

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> SearchParams:
        base = cls()
        return cls(
            top_k={**base.top_k, **{k: int(v) for k, v in data.get("top_k", {}).items()}},
            min_prob=float(data.get("min_prob", base.min_prob)),
            extra_minutes=tuple(int(v) for v in data.get("extra_minutes", base.extra_minutes)),
            prob_floor=float(data.get("prob_floor", base.prob_floor)),
            top_n=int(data.get("top_n", base.top_n)),
            max_base_dates=int(data.get("max_base_dates", base.max_base_dates)),
            marginal_top=int(data.get("marginal_top", base.marginal_top)),
            app_window_days=tuple(data.get("app_window_days", base.app_window_days)),  # type: ignore[arg-type]
        )


@dataclass(frozen=True)
class DecoderModel:
    """Приоры + веса + параметры поиска. Сериализуется в `decoder_priors_v0.json`."""

    priors: DecoderPriors
    weights: DecoderWeights = field(default_factory=DecoderWeights)
    search: SearchParams = field(default_factory=SearchParams)
    meta: Mapping[str, Any] = field(default_factory=dict)

    @classmethod
    def default(cls) -> DecoderModel:
        return cls(priors=DecoderPriors.default(), meta={"source": "defaults_from_plan"})

    def with_weights(self, **changes: float) -> DecoderModel:
        return replace(self, weights=replace(self.weights, **changes))

    def to_json(self) -> dict[str, Any]:
        return {
            "version": 1,
            "meta": dict(self.meta),
            "weights": asdict(self.weights),
            "search": self.search.to_json(),
            "priors": self.priors.to_json(),
        }

    @classmethod
    def from_json(cls, data: Mapping[str, Any]) -> DecoderModel:
        if int(data.get("version", 0)) != 1:
            raise ValueError("неизвестная версия файла приоров декодера")
        known = set(WEIGHT_NAMES)
        weights = DecoderWeights(
            **{k: float(v) for k, v in data.get("weights", {}).items() if k in known}
        )
        return cls(
            priors=DecoderPriors.from_json(data["priors"]),
            weights=weights,
            search=SearchParams.from_json(data.get("search", {})),
            meta=dict(data.get("meta", {})),
        )

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_json(), ensure_ascii=False, indent=1), encoding="utf-8")

    @classmethod
    def load(cls, path: Path | None) -> DecoderModel:
        """Загрузить модель из JSON; без файла — приоры по умолчанию."""
        if path is None or not Path(path).exists():
            return cls.default()
        return cls.from_json(json.loads(Path(path).read_text(encoding="utf-8")))


# ---------------------------------------------------------------------------
# Контракт входа и выхода
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeContext:
    """Контекст ваучера.

    - `year` — год из имени файла или печати. Базовая дата берётся в этом году;
      переход через Новый год внутри ваучера получается календарной арифметикой.
    - `extra_years` — дополнительные допустимые годы базовой даты (со штрафом
      `other_year_logp`), например соседний год, если ваучер на стыке лет.
      Год времени заявки добавляется автоматически.
    - `app_dt` — время из заявки (наивное локальное), слабый приор без окна.
    - `work_type` — нормализованный вид работ для приоров длительностей.
    """

    year: int
    app_dt: datetime | None = None
    work_type: str | None = None
    extra_years: tuple[int, ...] = ()

    def allowed_years(self) -> tuple[int, ...]:
        years = [self.year, *self.extra_years]
        if self.app_dt is not None:
            years.append(self.app_dt.year)
        return tuple(sorted(set(years)))


@dataclass(frozen=True)
class RowValue:
    """Значение строки: написание на бланке и фактическое время."""

    form_date: date
    hour: int
    minute: int

    @property
    def day(self) -> int:
        return self.form_date.day

    @property
    def month(self) -> int:
        return self.form_date.month

    @property
    def hour24(self) -> bool:
        return self.hour == 24

    @property
    def dt(self) -> datetime:
        """Фактическое время: `24:00` дня D — это 00:00 дня D+1."""
        return datetime.combine(self.form_date, time()) + timedelta(
            minutes=self.hour * 60 + self.minute
        )

    def as_form(self) -> FormRow:
        return (self.form_date, self.hour, self.minute)

    def subfield_values(self) -> dict[str, int]:
        return {"day": self.day, "month": self.month, "hour": self.hour, "minute": self.minute}


@dataclass(frozen=True)
class Record:
    """Запись ваучера: четыре строки, оценка и нормированная вероятность."""

    rows: Mapping[str, RowValue]
    score: float
    p: float

    @property
    def hour24(self) -> tuple[str, ...]:
        return tuple(r for r in ROWS if self.rows[r].hour24)

    def forms(self) -> dict[str, FormRow]:
        return {r: v.as_form() for r, v in self.rows.items()}

    def to_json(self) -> dict[str, Any]:
        """Запись в формате T13: время ISO до минут, `24:00` — как 00:00 следующих суток."""
        data: dict[str, Any] = {r: self.rows[r].dt.isoformat(timespec="minutes") for r in ROWS}
        data["hour24"] = list(self.hour24)
        data["p"] = self.p
        return data


@dataclass(frozen=True)
class RowOption:
    """Маргинальный вариант строки для кнопок в интерфейсе."""

    value: RowValue
    p: float


# Коды флагов DecodeResult.flags.
FLAG_NO_CANDIDATES = "no_candidates"
FLAG_CROSSES_MIDNIGHT = "crosses_midnight"
FLAG_HOUR24 = "hour24"
FLAG_MINUTE_NOT_MULT10 = "minute_not_mult10"
FLAG_OVERRIDE = "decoder_override"
FLAG_TOP1_CHAIN_VIOLATION = "fields_top1_chain_violation"
FLAG_TOP1_BAD_HOUR24 = "fields_top1_bad_hour24"
FLAG_TOP1_INVALID_DATE = "fields_top1_invalid_date"
FLAG_TOP1_DAY_MISMATCH = "fields_top1_day_mismatch"
FLAG_APP_FAR = "app_far"
FLAG_OTHER_YEAR = "other_year"
FLAG_MISSING = "missing_subfields"


@dataclass(frozen=True)
class DecodeResult:
    """Итог декодирования.

    - `records` — top-N записей, `p` — softmax по всему перечисленному пространству;
    - `confidence = p(top1)`, `margin = p(top1) − p(top2)`;
    - `marginals` — top-3 значений каждой строки с маргинальной вероятностью;
    - `flags` — сработавшие правила (коды `FLAG_*`), `missing` — пропущенные подполя,
      `overridden` — подполя, где декодер выбрал не top-1 картинки;
    - `log_z` — лог нормировки (нужен для подбора весов).
    """

    records: tuple[Record, ...]
    confidence: float
    margin: float
    marginals: Mapping[str, tuple[RowOption, ...]]
    flags: tuple[str, ...]
    missing: tuple[str, ...]
    overridden: tuple[str, ...]
    log_z: float

    @property
    def top(self) -> Record | None:
        return self.records[0] if self.records else None

    def to_json(self) -> dict[str, Any]:
        """Часть строки предсказаний T13, которую даёт декодер."""
        return {
            "records": [r.to_json() for r in self.records],
            "confidence": self.confidence,
            "margin": self.margin,
            "flags": list(self.flags),
            "missing": list(self.missing),
            "overridden": list(self.overridden),
        }


# ---------------------------------------------------------------------------
# Свидетельства картинки
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Evidence:
    logp: Mapping[str, np.ndarray]  # подполе -> массив, индекс = значение
    ranked: Mapping[str, list[tuple[int, float]]]  # перечисленные значения по убыванию
    missing: tuple[str, ...]


def _image_logp(dist: list[tuple[int, float]], part: str, floor: float) -> np.ndarray:
    lo, hi = PART_RANGE[part]
    k = hi - lo + 1
    arr = np.full(hi + 1, _NEG_INF)
    if not dist:
        arr[lo:] = math.log(1.0 / k)
        return arr
    listed = sum(p for _, p in dist)
    rest = max(0.0, 1.0 - listed)
    n_rest = k - len(dist)
    p_rest = rest / n_rest if n_rest > 0 else 0.0
    arr[lo:] = math.log(max(p_rest, floor))
    for value, p in dist:
        arr[value] = math.log(max(p, floor))
    return arr


def _normalize_dist(raw: SubfieldDist | None, part: str) -> list[tuple[int, float]]:
    """Отфильтровать значения вне диапазона, слить дубли, отсортировать по убыванию."""
    if not raw:
        return []
    lo, hi = PART_RANGE[part]
    merged: dict[int, float] = {}
    for item in raw:
        value, prob = int(item[0]), float(item[1])
        if not lo <= value <= hi or not math.isfinite(prob):
            continue
        merged[value] = merged.get(value, 0.0) + min(max(prob, 0.0), 1.0)
    total = sum(merged.values())
    if total > 1.0:
        merged = {v: p / total for v, p in merged.items()}
    return sorted(merged.items(), key=lambda vp: (-vp[1], vp[0]))


def _evidence(inputs: DecoderInputs, floor: float) -> _Evidence:
    logp: dict[str, np.ndarray] = {}
    ranked: dict[str, list[tuple[int, float]]] = {}
    missing: list[str] = []
    for row in ROWS:
        for part in PARTS:
            name = subfield(row, part)
            dist = _normalize_dist(inputs.get(name), part)
            if not dist:
                missing.append(name)
            ranked[name] = dist
            logp[name] = _image_logp(dist, part, floor)
    return _Evidence(logp=logp, ranked=ranked, missing=tuple(missing))


# ---------------------------------------------------------------------------
# Пространство кандидатов
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class CandidateSpace:
    """Перечисляемое пространство: базовые даты × шаблоны × (час, минута) по строкам."""

    base_dates: tuple[date, ...]
    patterns: tuple[tuple[int, ...], ...]
    hours: Mapping[str, tuple[int, ...]]
    minutes: Mapping[str, tuple[int, ...]]

    def states(self, row: str) -> tuple[np.ndarray, np.ndarray]:
        """Допустимые пары (час, минута) строки: час 24 только с минутами 00."""
        hs, ms = [], []
        for h in self.hours[row]:
            for m in self.minutes[row]:
                if h == 24 and m != 0:
                    continue
                hs.append(h)
                ms.append(m)
        return np.asarray(hs, dtype=np.int64), np.asarray(ms, dtype=np.int64)

    def contains(self, forms: Mapping[str, FormRow]) -> bool:
        """Входит ли запись (в написании бланка) в перечисляемое пространство."""
        base = forms[CHAIN[0]][0]
        if base not in self.base_dates:
            return False
        pattern = tuple((forms[r][0] - base).days for r in CHAIN)
        if pattern not in self.patterns:
            return False
        for row in CHAIN:
            _, h, m = forms[row]
            if h not in self.hours[row] or m not in self.minutes[row] or (h == 24 and m != 0):
                return False
        return True


def _top_values(dist: list[tuple[int, float]], k: int, min_prob: float) -> list[int]:
    out = [v for i, (v, p) in enumerate(dist) if i < k and (i == 0 or p >= min_prob)]
    return out


def _safe_date(year: int, month: int, day: int) -> date | None:
    try:
        return date(year, month, day)
    except ValueError:
        return None


def _date_part_score(
    ev: _Evidence, weights: DecoderWeights, dates: Sequence[date], row: str
) -> np.ndarray:
    days = np.fromiter((d.day for d in dates), dtype=np.int64, count=len(dates))
    months = np.fromiter((d.month for d in dates), dtype=np.int64, count=len(dates))
    return (
        weights.day * ev.logp[subfield(row, "day")][days]
        + weights.month * ev.logp[subfield(row, "month")][months]
    )


def _candidate_space(ev: _Evidence, context: DecodeContext, model: DecoderModel) -> CandidateSpace:
    search = model.search
    priors = model.priors
    max_off = max(0, int(priors.max_day_offset))
    patterns = tuple(chain_patterns(max_off))
    allowed_years = set(context.allowed_years())

    hours: dict[str, tuple[int, ...]] = {}
    minutes: dict[str, tuple[int, ...]] = {}
    for row in CHAIN:
        h_dist = ev.ranked[subfield(row, "hour")]
        m_dist = ev.ranked[subfield(row, "minute")]
        if h_dist:
            hs = set(_top_values(h_dist, search.top_k["hour"], search.min_prob))
        else:
            hs = set(range(25))
        ms = set(_top_values(m_dist, search.top_k["minute"], search.min_prob))
        ms.update(search.extra_minutes)
        if not m_dist and not ms:
            ms = set(range(60))
        if hs == {24}:
            ms.add(0)  # иначе у строки нет ни одного допустимого состояния
        hours[row] = tuple(sorted(hs))
        minutes[row] = tuple(sorted(ms))

    day_set: set[int] = set()
    month_set: set[int] = set()
    for row in CHAIN:
        day_set.update(
            _top_values(ev.ranked[subfield(row, "day")], search.top_k["day"], search.min_prob)
        )
        month_set.update(
            _top_values(ev.ranked[subfield(row, "month")], search.top_k["month"], search.min_prob)
        )

    bases: set[date] = set()
    if day_set:
        months = month_set
        if not months:
            if context.app_dt is not None:
                app_month = context.app_dt.month
                months = {(app_month + d - 1) % 12 + 1 for d in (-1, 0, 1)}
            else:
                months = set(range(1, 13))
        for y in range(min(allowed_years) - 1, max(allowed_years) + 2):
            for m in months:
                for d in day_set:
                    written = _safe_date(y, m, d)
                    if written is None:
                        continue
                    for o in range(max_off + 1):
                        base = written - timedelta(days=o)
                        if base.year in allowed_years:
                            bases.add(base)
    elif context.app_dt is not None:
        lo, hi = search.app_window_days
        app_day = context.app_dt.date()
        for o in range(lo, hi + 1):
            base = app_day + timedelta(days=o)
            if base.year in allowed_years:
                bases.add(base)

    base_list = sorted(bases)
    if len(base_list) > search.max_base_dates:
        # Оставляем базовые даты с лучшей оценкой дат (максимум по шаблонам).
        best = np.full(len(base_list), _NEG_INF)
        for pattern in patterns:
            total = np.zeros(len(base_list))
            for row, o in zip(CHAIN, pattern, strict=True):
                shifted = [b + timedelta(days=o) for b in base_list]
                total += _date_part_score(ev, model.weights, shifted, row)
            best = np.maximum(best, total)
        keep = np.argsort(-best, kind="stable")[: search.max_base_dates]
        base_list = sorted(base_list[i] for i in keep)
    return CandidateSpace(
        base_dates=tuple(base_list), patterns=patterns, hours=hours, minutes=minutes
    )


def candidate_space(
    inputs: DecoderInputs, context: DecodeContext, model: DecoderModel | None = None
) -> CandidateSpace:
    """Пространство кандидатов, которое перечисляет `decode` (для тестов и подбора весов)."""
    model = model or DecoderModel.default()
    return _candidate_space(_evidence(inputs, model.search.prob_floor), context, model)


# ---------------------------------------------------------------------------
# Оценка одной записи (скалярно; эталон для проверки поиска и для подбора весов)
# ---------------------------------------------------------------------------


def _score_scalar(
    forms: Mapping[str, FormRow], ev: _Evidence, context: DecodeContext, model: DecoderModel
) -> float:
    w = model.weights
    priors = model.priors
    base = forms[CHAIN[0]][0]
    pattern = tuple((forms[r][0] - base).days for r in CHAIN)
    if pattern[0] != 0 or any(b < a for a, b in zip(pattern[:-1], pattern[1:], strict=True)):
        return _NEG_INF
    if pattern[-1] > priors.max_day_offset:
        return _NEG_INF
    score = w.prior_pattern * priors.pattern_value(pattern)
    if base.year != context.year:
        score += priors.other_year_logp
    t: list[int] = []
    for row, o in zip(CHAIN, pattern, strict=True):
        written, h, m = forms[row]
        if not (0 <= h <= 24 and 0 <= m <= 59) or (h == 24 and m != 0):
            return _NEG_INF
        t.append(o * MINUTES_PER_DAY + h * 60 + m)
        score += w.day * float(ev.logp[subfield(row, "day")][written.day])
        score += w.month * float(ev.logp[subfield(row, "month")][written.month])
        score += w.hour * float(ev.logp[subfield(row, "hour")][h])
        score += w.minute * float(ev.logp[subfield(row, "minute")][m])
        score += w.prior_hour * float(priors.hour_array(row)[h])
        score += w.prior_minute * float(priors.minute_array[m])
    if any(b < a for a, b in zip(t[:-1], t[1:], strict=True)):
        return _NEG_INF
    legs = priors.leg_densities(context.work_type)
    for name, a, b in zip(LEG_NAMES, t[:-1], t[1:], strict=True):
        score += w.prior_duration * float(legs[name](float(b - a)))
    if context.app_dt is not None:
        start = datetime.combine(base, time()) + timedelta(minutes=t[1])
        dev = (start - context.app_dt).total_seconds() / 60.0
        score += w.prior_app * float(priors.app_deviation(dev))
    return score


def score_record(
    forms: Mapping[str, FormRow],
    inputs: DecoderInputs,
    context: DecodeContext,
    model: DecoderModel | None = None,
) -> float:
    """Ненормированная оценка записи; `-inf`, если запись нарушает жёсткие правила."""
    model = model or DecoderModel.default()
    return _score_scalar(forms, _evidence(inputs, model.search.prob_floor), context, model)


# ---------------------------------------------------------------------------
# Поиск по цепочке
# ---------------------------------------------------------------------------


def _lse_step(alpha: np.ndarray, pair: np.ndarray) -> np.ndarray:
    """`out[b, j] = logsumexp_i(alpha[b, i] + pair[i, j])` через матричное умножение."""
    m = alpha.max(axis=1, keepdims=True)
    m = np.where(np.isfinite(m), m, 0.0)
    c = pair.max(axis=0, keepdims=True)
    c = np.where(np.isfinite(c), c, 0.0)
    with np.errstate(divide="ignore", under="ignore"):
        prod = np.exp(alpha - m) @ np.exp(pair - c)
        return np.log(prod) + m + c


def _max_step(alpha: np.ndarray, pair: np.ndarray) -> np.ndarray:
    return (alpha[:, :, None] + pair[None, :, :]).max(axis=1)


def _logsumexp(values: np.ndarray) -> float:
    if values.size == 0:
        return _NEG_INF
    m = float(np.max(values))
    if not math.isfinite(m):
        return _NEG_INF
    return m + math.log(float(np.sum(np.exp(values - m))))


def _kbest_chain(
    unaries: Sequence[np.ndarray], pairs: Sequence[np.ndarray], k: int
) -> list[tuple[float, tuple[int, ...]]]:
    """Точные k лучших путей по цепочке: унарные (S_r,) и парные (S_r, S_{r+1})."""
    scores = unaries[0][:, None]  # (S0, K0)
    backs: list[tuple[np.ndarray, int]] = []
    for pair, unary in zip(pairs, unaries[1:], strict=True):
        s_prev, k_prev = scores.shape
        cand = (scores[:, :, None] + pair[:, None, :]).reshape(s_prev * k_prev, -1)
        kk = min(k, cand.shape[0])
        if kk < cand.shape[0]:
            idx = np.argpartition(-cand, kk - 1, axis=0)[:kk]
        else:
            idx = np.broadcast_to(np.arange(cand.shape[0])[:, None], cand.shape).copy()
        top = np.take_along_axis(cand, idx, axis=0)
        order = np.argsort(-top, axis=0, kind="stable")
        idx = np.take_along_axis(idx, order, axis=0)
        top = np.take_along_axis(top, order, axis=0)
        scores = top.T + unary[:, None]  # (S_next, kk)
        backs.append((idx.T, k_prev))
    flat = scores.ravel()
    kk = min(k, flat.size)
    best = np.argpartition(-flat, kk - 1)[:kk] if kk < flat.size else np.arange(flat.size)
    best = best[np.argsort(-flat[best], kind="stable")]
    out: list[tuple[float, tuple[int, ...]]] = []
    k_last = scores.shape[1]
    for f in best:
        value = float(flat[f])
        if not math.isfinite(value):
            continue
        state, kidx = divmod(int(f), k_last)
        path = [state]
        for back, k_prev in reversed(backs):
            prev = int(back[state, kidx])
            state, kidx = divmod(prev, k_prev)
            path.append(state)
        out.append((value, tuple(reversed(path))))
    return out


@dataclass
class _PatternBatch:
    pattern: tuple[int, ...]
    unaries: list[np.ndarray]  # по строкам CHAIN, (nB, S_r), дата уже в первой строке
    pairs: list[np.ndarray]  # (S_r, S_{r+1})
    alphas: list[np.ndarray]
    betas: list[np.ndarray]
    log_z: np.ndarray  # (nB,)
    best: np.ndarray  # (nB,)


def _pair_matrix(
    density: BinnedLogDensity, weight: float, t_prev: np.ndarray, t_next: np.ndarray
) -> np.ndarray:
    dt = t_next[None, :] - t_prev[:, None]
    return np.where(dt >= 0, weight * density(dt.astype(float)), _NEG_INF)


def _argmax_forms(ev: _Evidence, year: int) -> tuple[dict[str, int], list[str]]:
    """Top-1 картинки по подполям и флаги нарушений у такой «наивной» записи."""
    top: dict[str, int] = {}
    for name, dist in ev.ranked.items():
        if dist:
            top[name] = dist[0][0]
    flags: list[str] = []
    times: list[datetime | None] = []
    for row in CHAIN:
        vals = [top.get(subfield(row, p)) for p in PARTS]
        if any(v is None for v in vals):
            times.append(None)
            continue
        d, mo, h, mi = (int(v) for v in vals)  # type: ignore[arg-type]
        written = _safe_date(year, mo, d)
        if written is None:
            flags.append(FLAG_TOP1_INVALID_DATE)
            times.append(None)
            continue
        if h == 24 and mi != 0:
            flags.append(FLAG_TOP1_BAD_HOUR24)
            times.append(None)
            continue
        first = next((t for t in times if t is not None), None)
        if first is not None and (first.date() - written).days > 180:
            written = _safe_date(year + 1, mo, d) or written  # переход через Новый год
        times.append(datetime.combine(written, time()) + timedelta(minutes=h * 60 + mi))
    known = [t for t in times if t is not None]
    if any(b < a for a, b in zip(known[:-1], known[1:], strict=True)):
        flags.append(FLAG_TOP1_CHAIN_VIOLATION)
    return top, sorted(set(flags))


def decode(
    inputs: DecoderInputs,
    context: DecodeContext,
    model: DecoderModel | None = None,
) -> DecodeResult:
    """Найти top-N записей ваучера по распределениям подполей.

    `inputs` — формат `runtime.predict_digits`: `{подполе: [(значение, вероятность), …]}`,
    например `{"left_base.hour": [(9, 0.93), (8, 0.05)]}`. Отсутствующее подполе или
    пустой список — пропуск (равномерное распределение).
    """
    model = model or DecoderModel.default()
    search = model.search
    w = model.weights
    priors = model.priors
    ev = _evidence(inputs, search.prob_floor)
    space = _candidate_space(ev, context, model)
    top_fields, top_flags = _argmax_forms(ev, context.year)
    base_flags = list(top_flags)
    if ev.missing:
        base_flags.append(FLAG_MISSING)

    def empty(flag: str) -> DecodeResult:
        return DecodeResult(
            records=(),
            confidence=0.0,
            margin=0.0,
            marginals={r: () for r in ROWS},
            flags=tuple(sorted({*base_flags, flag})),
            missing=ev.missing,
            overridden=(),
            log_z=_NEG_INF,
        )

    bases = space.base_dates
    if not bases:
        return empty(FLAG_NO_CANDIDATES)
    n_b = len(bases)
    base_ord = np.asarray([b.toordinal() for b in bases], dtype=np.int64)

    # Состояния строк и B-независимые унарные члены.
    states = {row: space.states(row) for row in CHAIN}
    if any(states[row][0].size == 0 for row in CHAIN):
        return empty(FLAG_NO_CANDIDATES)
    base_t = {row: states[row][0] * 60 + states[row][1] for row in CHAIN}
    unary: dict[str, np.ndarray] = {}
    for row in CHAIN:
        hs, ms = states[row]
        unary[row] = (
            w.hour * ev.logp[subfield(row, "hour")][hs]
            + w.minute * ev.logp[subfield(row, "minute")][ms]
            + w.prior_hour * priors.hour_array(row)[hs]
            + w.prior_minute * priors.minute_array[ms]
        )

    # Оценка дат по смещению: (строка, o) -> (nB,).
    max_off = max(p[-1] for p in space.patterns)
    date_score: dict[tuple[str, int], np.ndarray] = {}
    for o in range(max_off + 1):
        shifted = [b + timedelta(days=o) for b in bases]
        for row in CHAIN:
            date_score[(row, o)] = _date_part_score(ev, w, shifted, row)
    year_term = np.asarray(
        [0.0 if b.year == context.year else priors.other_year_logp for b in bases]
    )

    # Приор заявки для строки «Начало»: (nB, S) по смещению.
    start_row = CHAIN[1]
    app_term: dict[int, np.ndarray] = {}
    if context.app_dt is not None:
        app_min = (
            np.asarray(
                [(datetime.combine(b, time()) - context.app_dt).total_seconds() for b in bases]
            )
            / 60.0
        )
        for o in range(max_off + 1):
            dev = app_min[:, None] + o * MINUTES_PER_DAY + base_t[start_row][None, :]
            app_term[o] = w.prior_app * priors.app_deviation(dev)

    legs = priors.leg_densities(context.work_type)
    pair_cache: dict[tuple[int, int], np.ndarray] = {}

    def pair(leg: int, diff: int) -> np.ndarray:
        key = (leg, diff)
        if key not in pair_cache:
            a, b = LEGS[leg]
            pair_cache[key] = _pair_matrix(
                legs[LEG_NAMES[leg]],
                w.prior_duration,
                base_t[a],
                base_t[b] + diff * MINUTES_PER_DAY,
            )
        return pair_cache[key]

    batches: list[_PatternBatch] = []
    for pattern in space.patterns:
        d_score = w.prior_pattern * priors.pattern_value(pattern) + year_term
        for row, o in zip(CHAIN, pattern, strict=True):
            d_score = d_score + date_score[(row, o)]
        unaries: list[np.ndarray] = []
        for i, (row, o) in enumerate(zip(CHAIN, pattern, strict=True)):
            u = np.broadcast_to(unary[row], (n_b, unary[row].shape[0])).copy()
            if i == 0:
                u += d_score[:, None]
            if row == start_row and o in app_term:
                u += app_term[o]
            unaries.append(u)
        pairs = [pair(i, pattern[i + 1] - pattern[i]) for i in range(len(LEGS))]
        alphas = [unaries[0]]
        maxes = unaries[0]
        for p_mat, u in zip(pairs, unaries[1:], strict=True):
            alphas.append(_lse_step(alphas[-1], p_mat) + u)
            maxes = _max_step(maxes, p_mat) + u
        betas = [np.zeros_like(unaries[-1])]
        for p_mat, u in zip(reversed(pairs), reversed(unaries[1:]), strict=True):
            betas.append(_lse_step(u + betas[-1], p_mat.T))
        betas.reverse()
        with np.errstate(under="ignore", divide="ignore"):
            last = alphas[-1]
            m = last.max(axis=1)
            safe = np.where(np.isfinite(m), m, 0.0)
            log_z_b = np.log(np.exp(last - safe[:, None]).sum(axis=1)) + safe
        batches.append(
            _PatternBatch(
                pattern=pattern,
                unaries=unaries,
                pairs=pairs,
                alphas=alphas,
                betas=betas,
                log_z=np.where(np.isfinite(m), log_z_b, _NEG_INF),
                best=maxes.max(axis=1),
            )
        )

    log_z = _logsumexp(np.concatenate([b.log_z for b in batches]))
    if not math.isfinite(log_z):
        return empty(FLAG_NO_CANDIDATES)

    # Top-N: цепочки (шаблон, B) по убыванию точного максимума, отсечение по N-й оценке.
    chains = sorted(
        (
            (float(batch.best[b]), pi, b)
            for pi, batch in enumerate(batches)
            for b in range(n_b)
            if math.isfinite(batch.best[b])
        ),
        key=lambda x: -x[0],
    )
    heap: list[tuple[float, int, tuple[int, int, tuple[int, ...]]]] = []
    counter = 0
    for best, pi, b in chains:
        if len(heap) >= search.top_n and best < heap[0][0]:
            break
        batch = batches[pi]
        for value, path in _kbest_chain([u[b] for u in batch.unaries], batch.pairs, search.top_n):
            item = (value, -counter, (pi, b, path))
            counter += 1
            if len(heap) < search.top_n:
                heapq.heappush(heap, item)
            elif value > heap[0][0]:
                heapq.heapreplace(heap, item)
            else:
                break
    ranked = sorted(heap, key=lambda x: (-x[0], -x[1]))

    def make_row(row: str, b: int, o: int, s: int) -> RowValue:
        hs, ms = states[row]
        return RowValue(bases[b] + timedelta(days=o), int(hs[s]), int(ms[s]))

    records: list[Record] = []
    for value, _, (pi, b, path) in ranked:
        pattern = batches[pi].pattern
        rows = {row: make_row(row, b, o, s) for row, o, s in zip(CHAIN, pattern, path, strict=True)}
        records.append(
            Record(
                rows={r: rows[r] for r in ROWS},
                score=value,
                p=math.exp(value - log_z),
            )
        )

    marginals = _marginals(batches, states, base_ord, log_z, search.marginal_top)

    top = records[0]
    p1 = top.p
    p2 = records[1].p if len(records) > 1 else 0.0
    flags = set(base_flags)
    if any(top.rows[r].form_date != top.rows[CHAIN[0]].form_date for r in CHAIN):
        flags.add(FLAG_CROSSES_MIDNIGHT)
    if top.hour24:
        flags.add(FLAG_HOUR24)
    if any(top.rows[r].minute % 10 for r in ROWS):
        flags.add(FLAG_MINUTE_NOT_MULT10)
    if top.rows[CHAIN[0]].form_date.year != context.year:
        flags.add(FLAG_OTHER_YEAR)
    if context.app_dt is not None:
        dev = abs((top.rows[start_row].dt - context.app_dt).total_seconds()) / 60.0
        if dev > MINUTES_PER_DAY:
            flags.add(FLAG_APP_FAR)
    overridden: list[str] = []
    for row in ROWS:
        values = top.rows[row].subfield_values()
        for part in PARTS:
            name = subfield(row, part)
            if name in top_fields and top_fields[name] != values[part]:
                overridden.append(name)
    if any(name.endswith(".day") for name in overridden):
        flags.add(FLAG_TOP1_DAY_MISMATCH)
    if overridden:
        flags.add(FLAG_OVERRIDE)
    return DecodeResult(
        records=tuple(records),
        confidence=p1,
        margin=p1 - p2,
        marginals=marginals,
        flags=tuple(sorted(flags)),
        missing=ev.missing,
        overridden=tuple(overridden),
        log_z=log_z,
    )


def _marginals(
    batches: Sequence[_PatternBatch],
    states: Mapping[str, tuple[np.ndarray, np.ndarray]],
    base_ord: np.ndarray,
    log_z: float,
    top: int,
) -> dict[str, tuple[RowOption, ...]]:
    """Маргинальные top-N значений каждой строки (по написанию на бланке)."""
    out: dict[str, tuple[RowOption, ...]] = {}
    for ri, row in enumerate(CHAIN):
        hs, ms = states[row]
        keys: list[np.ndarray] = []
        probs: list[np.ndarray] = []
        for batch in batches:
            o = batch.pattern[ri]
            with np.errstate(under="ignore", invalid="ignore"):
                prob = np.exp(batch.alphas[ri] + batch.betas[ri] - log_z)
            prob = np.nan_to_num(prob, nan=0.0)
            # Ключ — написание: (порядковый номер даты на бланке, час, минута).
            written_ord = base_ord[:, None] + o
            key = (written_ord * 25 + hs[None, :]) * 60 + ms[None, :]
            keys.append(key.ravel())
            probs.append(prob.ravel())
        all_keys = np.concatenate(keys)
        all_probs = np.concatenate(probs)
        uniq, inv = np.unique(all_keys, return_inverse=True)
        summed = np.bincount(inv, weights=all_probs, minlength=uniq.size)
        order = np.argsort(-summed, kind="stable")[:top]
        options: list[RowOption] = []
        for i in order:
            if summed[i] <= 0:
                continue
            key = int(uniq[i])
            rest, minute = divmod(key, 60)
            day_ord, hour = divmod(rest, 25)
            value = RowValue(date.fromordinal(day_ord), hour, minute)
            options.append(RowOption(value=value, p=float(summed[i])))
        out[row] = tuple(options)
    return {r: out[r] for r in ROWS}


def forms_from_values(values: Mapping[str, Mapping[str, int]], year: int) -> dict[str, FormRow]:
    """Собрать запись в написании бланка из значений подполей строк и года.

    `values[row] = {"day", "month", "hour", "minute"}` (+ необязательный `"year"`).
    Удобно для истины из манифеста и для тестов.
    """
    out: dict[str, FormRow] = {}
    for row in CHAIN:
        v = values[row]
        out[row] = (
            date(int(v.get("year", year)), int(v["month"]), int(v["day"])),
            int(v["hour"]),
            int(v["minute"]),
        )
    return out
