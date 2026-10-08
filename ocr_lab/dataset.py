"""T15: torch `Dataset` для двузначных подполей дат (день/месяц/час/минута всех строк).

Источники:

- `data/ocr/crops_index.csv` (T10) — путь к кропу каждого подполя и `align_ok`;
- `data/ocr/manifest.csv` (T02) — истина и сплит; читается только через
  `ocr_lab.evaluate.load_truth` (контракт T02/T13 — разбор дат манифеста живёт там один
  раз, здесь он не дублируется);
- `data/ocr/review/crops_qc.csv` (T10) — ручная разметка дефектных кропов (`cut_digit`,
  `empty_misaligned`, `neighbour_digits`, `rotated`, `other`) — такие подполя сканов
  исключаются из датасета.

Номер ваучера сюда не входит: у него другой вид бокса (`"number"`, 1–4 знака, бывает с
буквой), не `"two_digit"` (см. `app.ocr.layouts.EXPECTED_KIND`) — отдельная задача.

Цели (план «Этап 2», п. 1 промпта T15):

- `units` — цифра единиц, 0..9;
- `tens` — цифра десятков, 0..9, либо :data:`TENS_EMPTY` (10) — «пусто». Вычисляется как
  `value // 10` (для `value < 10` это 0), а флаг `leading_zero_ambiguous` отмечает, что
  для таких значений десятки неоднозначны: на бланке мог быть один знак без десятков
  («9» вместо «09»), и истина (просто число) этого не различает. Сама функция потерь,
  которая решает, что делать с этой неоднозначностью (например, засчитывать оба класса
  десятков верными), — в T16, здесь только разметка;
- `hour = 24` — обычное двузначное число по той же формуле: `tens=2, units=4`, особого
  случая не требуется.
- `part_id` — вспомогательный признак типа подполя (день/месяц/час/минута,
  :data:`PART_ID`), чтобы общая на все подполя модель (план: «одна модель на все
  подполя») могла учитывать его при предсказании.

Предобработка кропа — без сплющивания: :func:`fit_with_aspect` вписывает серый кроп в
канву :data:`TARGET_SIZE` (H, W) с сохранением пропорций, фон белый, затем — нормализация
в `[0, 1]`. Аугментации (:mod:`ocr_lab.augment`) применяются раньше, на исходном кропе
(так геометрические искажения и обрывки у края считаются в масштабе реального бокса, а
не канвы) — только если `augment=True` у датасета.

Сэмплер :func:`make_balanced_sampler` балансирует вклад `part_id` в эпоху: без него
минуты (на практике почти всегда кратны 10 — фактически около 6 частых значений) не
должны перевешивать дни (1..31) просто потому, что после фильтров (`align_ok`,
`crops_qc.csv`) для одной из частей осталось больше валидных кропов, чем для другой.
Баланса классов *внутри* подполя сэмплер не делает (решение промпта: это сломало бы
калибровку приоров декодера T19, который ждёт настоящее распределение значений).

Флаг ненадёжности истины (упомянут в промпте — «если такой флаг есть») сейчас не
применяется: в манифесте T02 (`ocr_lab.evaluate.ScanTruth`) нет такой колонки на уровне
подполя. Если она появится позже, добавлять фильтр нужно сюда и в `load_truth`.
"""

from __future__ import annotations

import csv
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
import torch
from torch.utils.data import Dataset, WeightedRandomSampler

from ocr_lab.augment import augment as augment_crop
from ocr_lab.cut_crops import read_gray
from ocr_lab.evaluate import load_truth
from ocr_lab.paths import MANIFEST, REVIEW_DIR, WORK_DIR
from ocr_lab.predictions import PARTS, ROWS

#: Двузначные подполя датасета (16 штук) — без номера ваучера.
SUBFIELDS: tuple[str, ...] = tuple(f"{r}.{p}" for r in ROWS for p in PARTS)
#: Индекс части подполя для вспомогательного признака `part_id`.
PART_ID: dict[str, int] = {p: i for i, p in enumerate(PARTS)}

INDEX_CSV = WORK_DIR / "crops_index.csv"
QC_CSV = REVIEW_DIR / "crops_qc.csv"

#: Класс «пусто» у цифры десятков: на бланке не нарисована вторая цифра.
TENS_EMPTY = 10
NUM_TENS_CLASSES = 11  # 0..9 + «пусто»
NUM_UNITS_CLASSES = 10

#: Канва (H, W), в которую вписывается кроп без искажения пропорций (белый фон).
#:
#: Подобрано по распределению кропов T10 (`data/ocr/crops_index.csv`, 12 304 кропа
#: двузначных подполей, без номера ваучера): высота принимает только два значения —
#: 92 px (Коммунар) или 112 px (Пионер), ширина — 99..267 px (99-й перцентиль — 209 px).
#: Поскольку высота всегда больше 64, масштаб всегда ограничен высотой (`64/h < 1`) —
#: при высоте канвы 64 px 99-й перцентиль ширины после масштабирования — 145 px, максимум
#: — 186 px. При ширине канвы 128 px (значение-пример из плана) дополнительно сжимать
#: пришлось бы 22,6 % кропов (уменьшая цифру по высоте меньше 64 px); при 160 px —
#: только 0,07 % (9 из 12 304, самые широкие кропы минут). Выбрано 160, чтобы подавляющее
#: большинство цифр занимало канву по максимальной высоте.
TARGET_SIZE: tuple[int, int] = (64, 160)


@dataclass(frozen=True)
class Target:
    """Цели одного кропа (см. докстринг модуля)."""

    tens: int
    units: int
    leading_zero_ambiguous: bool


def encode_target(value: int) -> Target:
    """Истинное значение подполя (0..59) → цели `tens`/`units` и флаг неоднозначности."""
    if not 0 <= value <= 59:
        raise ValueError(f"недопустимое значение двузначного подполя: {value}")
    return Target(tens=value // 10, units=value % 10, leading_zero_ambiguous=value < 10)


def fit_with_aspect(image: np.ndarray, size: tuple[int, int] = TARGET_SIZE) -> np.ndarray:
    """Вписать серый кроп в канву `size` (H, W) без искажения пропорций, фон белый.

    Кроп масштабируется так, чтобы он целиком поместился в канву (`min` по обеим осям —
    только уменьшение, кропы T10 всегда крупнее 64 px по высоте), затем центрируется на
    белом фоне.
    """
    if image.ndim != 2:
        raise ValueError(f"fit_with_aspect: ожидалось серое изображение, форма {image.shape}")
    target_h, target_w = size
    h, w = image.shape
    if h <= 0 or w <= 0:
        raise ValueError("fit_with_aspect: пустое изображение")
    scale = min(target_h / h, target_w / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w), 255, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


# --- сбор сэмплов --------------------------------------------------------------------------


def _read_csv_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        return list(csv.DictReader(fh))


def _bad_crops(qc_csv: Path) -> set[tuple[str, str]]:
    """`(subfield, scan_id)` с пометкой проблемы из ручной проверки листов T10."""
    return {(r["subfield"], r["scan_id"]) for r in _read_csv_rows(qc_csv)}


@dataclass(frozen=True)
class Sample:
    """Один кроп двузначного подполя с истиной — элемент :data:`TwoDigitFieldDataset`."""

    scan_id: str
    subfield: str
    path: Path
    value: int


def build_samples(
    *,
    split: str,
    manifest: Path = MANIFEST,
    index_csv: Path = INDEX_CSV,
    qc_csv: Path = QC_CSV,
    work_dir: Path = WORK_DIR,
) -> list[Sample]:
    """Собрать кропы двузначных подполей для сплита — вход `TwoDigitFieldDataset`.

    Исключения: подполя не из :data:`SUBFIELDS` (номер ваучера), `align_ok=False` в
    индексе T10, сканы не из запрошенного сплита или без значения истины (пустое время
    строки, см. T02 — `load_truth` отдаёт `None`), кропы с пометкой проблемы в
    `crops_qc.csv` (T10, любой `problem`, не только «серьёзные»).
    """
    truth = load_truth(manifest, split)
    bad = _bad_crops(qc_csv)
    samples: list[Sample] = []
    for row in _read_csv_rows(index_csv):
        subfield = row["subfield"]
        if subfield not in SUBFIELDS:
            continue
        if (row.get("align_ok") or "True") != "True":
            continue
        scan_id = row["scan_id"]
        scan_truth = truth.get(scan_id)
        if scan_truth is None:
            continue
        if (subfield, scan_id) in bad:
            continue
        value = scan_truth.parts.get(subfield)
        if value is None:
            continue
        samples.append(
            Sample(scan_id=scan_id, subfield=subfield, path=work_dir / row["path"], value=value)
        )
    return samples


# --- Dataset ---------------------------------------------------------------------------


class TwoDigitFieldDataset(Dataset):
    """Кропы двузначных подполей дат для обучения общей на все подполя CNN (T16).

    Элемент — `(image, targets)`:

    - `image` — `torch.float32` тензор `1 x H x W`, 0..1 (канва :data:`TARGET_SIZE`);
    - `targets` — словарь `tens`, `units`, `leading_zero_ambiguous`, `part_id` (цели и
      признак для модели/функции потерь T16), и `value`, `scan_id`, `subfield` (для
      разбора ошибок, не для функции потерь).
    """

    def __init__(
        self,
        *,
        split: str,
        manifest: Path = MANIFEST,
        index_csv: Path = INDEX_CSV,
        qc_csv: Path = QC_CSV,
        work_dir: Path = WORK_DIR,
        target_size: tuple[int, int] = TARGET_SIZE,
        augment: bool = False,
        seed: int | None = None,
        samples: Sequence[Sample] | None = None,
    ) -> None:
        self.split = split
        self.target_size = target_size
        self.augment_enabled = augment
        self.samples: list[Sample] = (
            list(samples)
            if samples is not None
            else build_samples(
                split=split, manifest=manifest, index_csv=index_csv, qc_csv=qc_csv,
                work_dir=work_dir,
            )
        )
        self._rng = np.random.default_rng(seed)

    def __len__(self) -> int:
        return len(self.samples)

    @property
    def part_ids(self) -> list[int]:
        """`part_id` каждого сэмпла в порядке датасета — вход :func:`make_balanced_sampler`."""
        return [PART_ID[s.subfield.split(".")[1]] for s in self.samples]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, object]]:
        sample = self.samples[index]
        image = read_gray(sample.path)
        if self.augment_enabled:
            aug_seed = int(self._rng.integers(0, 2**31 - 1))
            image = augment_crop(image, aug_seed)
        image = fit_with_aspect(image, self.target_size)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        target = encode_target(sample.value)
        _row, part = sample.subfield.split(".")
        targets: dict[str, object] = {
            "tens": target.tens,
            "units": target.units,
            "leading_zero_ambiguous": target.leading_zero_ambiguous,
            "part_id": PART_ID[part],
            "value": sample.value,
            "scan_id": sample.scan_id,
            "subfield": sample.subfield,
        }
        return tensor, targets


def part_weights(part_ids: Sequence[int]) -> list[float]:
    """Вес сэмпла, обратный частоте его `part_id` в выборке (вход `WeightedRandomSampler`).

    Сумма весов по каждому `part_id` одинакова — при `replacement=True` и
    `num_samples=len(part_ids)` это даёт равный (в среднем) вклад каждой части строки
    в эпоху, независимо от того, сколько сэмплов реально прошло фильтры.
    """
    counts = Counter(part_ids)
    return [1.0 / counts[p] for p in part_ids]


def make_balanced_sampler(
    dataset: TwoDigitFieldDataset, *, seed: int | None = None
) -> WeightedRandomSampler:
    """Сэмплер с равным (в среднем) вкладом части подполя — день/месяц/час/минута.

    Балансировки по классам *внутри* подполя нет: это решение промпта (не ломать
    калибровку приоров декодера T19 — ему нужно настоящее распределение значений).
    """
    weights = part_weights(dataset.part_ids)
    generator = torch.Generator().manual_seed(seed) if seed is not None else None
    return WeightedRandomSampler(weights, num_samples=len(dataset), replacement=True,
                                  generator=generator)


# --- синтетика из MNIST (п. 4 промпта, выключена по умолчанию) ---------------------------

#: Решение по умолчанию: синтетика не используется — решит T16 по результату на `val`
#: (п. 4 промпта T15). `torchvision` (нужен для загрузки MNIST) пока не установлен и не
#: объявлен ни в одном requirements-файле — это тоже на совести того, кто включит опцию.
SYNTHETIC_ENABLED_DEFAULT = False
#: Зазор между цифрами десятков и единиц при склейке (в пикселях глифа).
MNIST_DIGIT_GAP = 2


def paste_digit_pair(
    tens_glyph: np.ndarray | None, units_glyph: np.ndarray, *, gap: int = MNIST_DIGIT_GAP
) -> np.ndarray:
    """Склеить изображение из одного-двух глифов цифр (белый фон, тёмная цифра).

    `tens_glyph=None` — только единицы (имитация «9» вместо «09»); иначе десятки
    рисуются слева от единиц. Только компоновка изображения — какой цифре соответствует
    каждый глиф, знает вызывающий код (:class:`SyntheticDigitsDataset`), он же строит
    :class:`Target`. Глифы должны быть одной высоты (как цифры MNIST после загрузки).
    """
    if tens_glyph is None:
        return units_glyph.copy()
    if tens_glyph.shape[0] != units_glyph.shape[0]:
        raise ValueError("paste_digit_pair: глифы должны быть одной высоты")
    h = tens_glyph.shape[0]
    w = tens_glyph.shape[1] + gap + units_glyph.shape[1]
    canvas = np.full((h, w), 255, dtype=np.uint8)
    canvas[:, : tens_glyph.shape[1]] = tens_glyph
    canvas[:, tens_glyph.shape[1] + gap :] = units_glyph
    return canvas


def load_mnist_digit_bank(root: Path) -> dict[int, list[np.ndarray]]:
    """Цифры MNIST (белый фон, тёмная цифра) по классам 0..9 — для :func:`paste_digit_pair`.

    Ленивый импорт `torchvision`: он не нужен, пока опция выключена (решение по
    умолчанию — :data:`SYNTHETIC_ENABLED_DEFAULT`). Если `torchvision` не установлен,
    исключение подскажет, что сначала нужно добавить его в `requirements-train.txt`.
    """
    from torchvision.datasets import MNIST  # noqa: PLC0415 — опциональная тяжёлая зависимость

    dataset = MNIST(root=str(root), train=True, download=True)
    bank: dict[int, list[np.ndarray]] = {d: [] for d in range(10)}
    for image, label in dataset:
        # MNIST: чёрный фон, белая цифра — инвертируем под конвенцию бланка (белый фон).
        bank[int(label)].append(255 - np.asarray(image, dtype=np.uint8))
    return bank


class SyntheticDigitsDataset(Dataset):
    """Двузначные числа, склеенные из цифр MNIST (:func:`paste_digit_pair`).

    Не подключается к `TwoDigitFieldDataset` автоматически — решение, нужна ли синтетика,
    принимает T16 по результату на `val`. Для обучения совместно с реальными кропами
    используйте `torch.utils.data.ConcatDataset([real_ds, synthetic_ds])`.
    """

    def __init__(
        self,
        digit_bank: dict[int, list[np.ndarray]],
        *,
        n: int,
        seed: int = 0,
        empty_tens_prob: float = 0.15,
        target_size: tuple[int, int] = TARGET_SIZE,
    ) -> None:
        if set(digit_bank) != set(range(10)) or any(not v for v in digit_bank.values()):
            raise ValueError("SyntheticDigitsDataset: нужны все классы 0..9 хотя бы по 1 глифу")
        self.digit_bank = digit_bank
        self.n = n
        self.seed = seed
        self.empty_tens_prob = empty_tens_prob
        self.target_size = target_size

    def __len__(self) -> int:
        return self.n

    def _pick(self, rng: np.random.Generator, digit: int) -> np.ndarray:
        glyphs = self.digit_bank[digit]
        return glyphs[int(rng.integers(0, len(glyphs)))]

    def __getitem__(self, index: int) -> tuple[torch.Tensor, dict[str, object]]:
        rng = np.random.default_rng(self.seed * 1_000_003 + index)
        units_digit = int(rng.integers(0, 10))
        empty_tens = bool(rng.random() < self.empty_tens_prob)
        tens_digit = None if empty_tens else int(rng.integers(0, 10))
        units_glyph = self._pick(rng, units_digit)
        tens_glyph = None if tens_digit is None else self._pick(rng, tens_digit)
        image = paste_digit_pair(tens_glyph, units_glyph)
        target = (
            Target(tens=TENS_EMPTY, units=units_digit, leading_zero_ambiguous=True)
            if tens_digit is None
            else Target(tens=tens_digit, units=units_digit, leading_zero_ambiguous=False)
        )
        image = fit_with_aspect(image, self.target_size)
        tensor = torch.from_numpy(image.astype(np.float32) / 255.0).unsqueeze(0)
        targets: dict[str, object] = {
            "tens": target.tens,
            "units": target.units,
            "leading_zero_ambiguous": target.leading_zero_ambiguous,
            "part_id": -1,  # синтетика не привязана к конкретной части строки
            "value": None,
            "scan_id": f"synthetic_{index}",
            "subfield": "synthetic",
        }
        return tensor, targets
