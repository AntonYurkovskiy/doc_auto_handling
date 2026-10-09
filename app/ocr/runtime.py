"""T18: рантайм моделей OCR на onnxruntime — только CPU, без torch.

Две модели лежат в каталоге моделей (по умолчанию `<OCR_WORK_DIR>/models`, либо `OCR_MODELS_DIR`):

- `digits_v0/` — CNN двузначных подполей дат (день, месяц, часы, минуты);
- `number_v0/` — CRNN (CTC) номера ваучера.

В каждом лежат `model.onnx` и `meta.json` — контракт рантайма: размер входа, фон, нормализация,
порядок голов и классов, температуры калибровки (`ocr_lab.calibrate`), хэши обучения. Граф
отдаёт **сырые логиты**, температуры применяются здесь — их можно перекалибровать без
повторного экспорта. Предобработка (вписывание кропа в канву, нормализация) повторяет
`ocr_lab.dataset.fit_with_aspect` и параметры из `meta.json` на numpy.

Публичный интерфейс:

- :func:`predict_digits` — `{подполе: [(значение, вероятность), …]}` для словаря кропов
  (все подполя ваучера идут одним батчем);
- :func:`predict_number` / :func:`score_number_candidates` — top-k номеров и оценка списка
  кандидатов (`log P` модели с температурой, их складывает с приорами декодер).

Сессия onnxruntime создаётся лениво, один раз на процесс и модель; `InferenceSession.run`
потокобезопасен, создание сессии защищено блокировкой.
"""

from __future__ import annotations

import json
import os
import threading
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from app.ocr.digit_values import TOP_K, top_values, value_log_probs
from app.ocr.number_scores import (
    number_log_probs,
    score_candidates,
    top_numbers,
)

META_NAME = "meta.json"
FORMAT_VERSION = 1
DIGITS_MODEL = "digits_v0"
NUMBER_MODEL = "number_v0"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


class OcrRuntimeError(RuntimeError):
    """Модель не найдена, `meta.json` не соответствует контракту или вход некорректен."""


class SessionLike(Protocol):
    """Часть интерфейса `onnxruntime.InferenceSession`, которую использует рантайм."""

    def run(
        self, output_names: list[str] | None, input_feed: dict[str, np.ndarray]
    ) -> list[np.ndarray]: ...


SessionFactory = Callable[[Path], SessionLike]


def models_dir() -> Path:
    """Каталог моделей: `OCR_MODELS_DIR`, иначе `<OCR_WORK_DIR>/models` (`data/ocr`)."""
    explicit = os.environ.get("OCR_MODELS_DIR")
    if explicit:
        return Path(explicit)
    work = os.environ.get("OCR_WORK_DIR")
    return (Path(work) if work else _REPO_ROOT / "data" / "ocr") / "models"


# --- предобработка ----------------------------------------------------------------------------


def fit_with_aspect(image: np.ndarray, size: tuple[int, int], background: int = 255) -> np.ndarray:
    """Вписать серый кроп в канву `size` (H, W) без искажения пропорций, фон `background`.

    Копия `ocr_lab.dataset.fit_with_aspect` (рантайм не импортирует лабораторию): масштаб
    `min(H/h, W/w)` (и уменьшение, и увеличение), центрирование.
    """
    if image.ndim != 2:
        raise OcrRuntimeError(f"ожидался серый кроп (H, W), форма {image.shape}")
    target_h, target_w = size
    h, w = image.shape
    if h <= 0 or w <= 0:
        raise OcrRuntimeError("пустой кроп")
    scale = min(target_h / h, target_w / w)
    new_h = max(1, round(h * scale))
    new_w = max(1, round(w * scale))
    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_LINEAR
    resized = cv2.resize(image, (new_w, new_h), interpolation=interp)
    canvas = np.full((target_h, target_w), background, dtype=np.uint8)
    y0 = (target_h - new_h) // 2
    x0 = (target_w - new_w) // 2
    canvas[y0 : y0 + new_h, x0 : x0 + new_w] = resized
    return canvas


def _as_gray_u8(image: np.ndarray) -> np.ndarray:
    """Кроп → серый `uint8` (H, W): цветной (BGR) переводится в серый, `float` обрезается."""
    arr = np.asarray(image)
    if arr.ndim == 3 and arr.shape[2] in (3, 4):
        code = cv2.COLOR_BGR2GRAY if arr.shape[2] == 3 else cv2.COLOR_BGRA2GRAY
        arr = cv2.cvtColor(arr.astype(np.uint8), code)
    if arr.ndim != 2:
        raise OcrRuntimeError(f"ожидался кроп (H, W) или (H, W, 3), форма {np.shape(image)}")
    if arr.dtype != np.uint8:
        arr = np.clip(arr, 0, 255).astype(np.uint8)
    return arr


# --- контракт ---------------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelMeta:
    """Разобранный `meta.json` (сырой словарь — в `raw`)."""

    name: str
    task: str
    onnx_file: str
    size: tuple[int, int]
    background: int
    mean: float
    std: float
    normalize_in_graph: bool
    input_name: str
    raw: dict[str, Any]

    @property
    def temperatures(self) -> dict[str, float]:
        return {k: float(v) for k, v in self.raw.get("temperatures", {}).items()}


def load_meta(model_dir: Path) -> ModelMeta:
    """Прочитать и проверить `meta.json` каталога модели."""
    path = model_dir / META_NAME
    if not path.is_file():
        raise OcrRuntimeError(f"нет {META_NAME} в каталоге модели {model_dir.name}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
        if raw["format_version"] != FORMAT_VERSION:
            raise OcrRuntimeError(
                f"{model_dir.name}: версия контракта {raw['format_version']}, "
                f"рантайм понимает {FORMAT_VERSION}"
            )
        inp = raw["input"]
        norm = raw["normalization"]
        h, w = (int(v) for v in inp["size"])
        return ModelMeta(
            name=str(raw["name"]),
            task=str(raw["task"]),
            onnx_file=str(raw.get("onnx", "model.onnx")),
            size=(h, w),
            background=int(inp.get("background", 255)),
            mean=float(norm["mean"]),
            std=float(norm["std"]),
            normalize_in_graph=bool(norm["in_graph"]),
            input_name=str(inp["name"]),
            raw=raw,
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise OcrRuntimeError(f"{model_dir.name}: {META_NAME} не по контракту: {exc!r}") from exc


def _default_session_factory(path: Path) -> SessionLike:
    import onnxruntime as ort  # noqa: PLC0415 — тяжёлый импорт, нужен только при первом вызове

    opts = ort.SessionOptions()
    threads = os.environ.get("OCR_ORT_THREADS")
    if threads:
        opts.intra_op_num_threads = int(threads)
    opts.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
    return ort.InferenceSession(  # type: ignore[no-any-return]
        str(path), sess_options=opts, providers=["CPUExecutionProvider"]
    )


class OnnxModel:
    """Модель с ленивой сессией: `meta.json` читается сразу, onnxruntime — при первом `run`."""

    def __init__(self, model_dir: Path, *, session_factory: SessionFactory | None = None) -> None:
        self.model_dir = model_dir
        self.meta = load_meta(model_dir)
        self._factory = session_factory or _default_session_factory
        self._session: SessionLike | None = None
        self._lock = threading.Lock()

    @property
    def session(self) -> SessionLike:
        if self._session is None:
            with self._lock:
                if self._session is None:
                    path = self.model_dir / self.meta.onnx_file
                    if not path.is_file():
                        raise OcrRuntimeError(f"нет файла модели {self.meta.name}/{path.name}")
                    self._session = self._factory(path)
        return self._session

    def prepare(self, crops: Sequence[np.ndarray]) -> np.ndarray:
        """Кропы → `float32 (B, 1, H, W)`: вписывание, `0..1`, нормализация, если она снаружи."""
        meta = self.meta
        batch = np.stack(
            [fit_with_aspect(_as_gray_u8(c), meta.size, meta.background) for c in crops]
        ).astype(np.float32)
        batch = batch[:, None, :, :] / np.float32(255.0)
        if not meta.normalize_in_graph:
            batch = (batch - np.float32(meta.mean)) / np.float32(meta.std)
        return np.ascontiguousarray(batch, dtype=np.float32)

    def run(self, feed: dict[str, np.ndarray]) -> list[np.ndarray]:
        return self.session.run(None, feed)


_MODELS: dict[tuple[str, str], OnnxModel] = {}
_REGISTRY_LOCK = threading.Lock()


def get_model(
    name: str, *, directory: Path | None = None, session_factory: SessionFactory | None = None
) -> OnnxModel:
    """Модель `name` из каталога моделей — один объект на процесс (кэш по пути и имени)."""
    base = directory if directory is not None else models_dir()
    key = (str(base.resolve()), name)
    with _REGISTRY_LOCK:
        model = _MODELS.get(key)
        if model is None:
            model = OnnxModel(base / name, session_factory=session_factory)
            _MODELS[key] = model
        return model


def reset_models() -> None:
    """Забыть загруженные модели (тесты, смена каталога моделей на лету)."""
    with _REGISTRY_LOCK:
        _MODELS.clear()


# --- цифры ------------------------------------------------------------------------------------


def digit_logits(
    model: OnnxModel, crops: Sequence[np.ndarray], parts: Sequence[str]
) -> tuple[np.ndarray, np.ndarray]:
    """Сырые логиты голов модели цифр: `(tens (N, 11), units (N, 10))`. `parts[i]` — часть кропа."""
    if model.meta.task != "digits":
        raise OcrRuntimeError(f"{model.meta.name}: это не модель цифр")
    ids = model.meta.raw["part_ids"]
    try:
        part_id = np.array([ids[p] for p in parts], dtype=np.int64)
    except KeyError as exc:
        raise OcrRuntimeError(f"неизвестная часть строки: {exc.args[0]}") from exc
    if not len(crops):
        return np.zeros((0, 11), dtype=np.float32), np.zeros((0, 10), dtype=np.float32)
    feed = {
        model.meta.input_name: model.prepare(crops),
        str(model.meta.raw["part_input"]["name"]): part_id,
    }
    tens, units = model.run(feed)[:2]
    return tens, units


def part_of(subfield: str) -> str:
    """`left_base.hour` → `hour`."""
    return subfield.rsplit(".", 1)[-1]


def digit_top_values(
    tens: np.ndarray,
    units: np.ndarray,
    parts: Sequence[str],
    temperatures: Mapping[str, float],
    k: int = TOP_K,
) -> list[list[tuple[int, float]]]:
    """Top-k значений по сырым логитам голов с температурами (`tens`, `units`)."""
    out = []
    for i, part in enumerate(parts):
        values, logp = value_log_probs(
            tens[i], units[i], part,
            t_tens=temperatures.get("tens", 1.0), t_units=temperatures.get("units", 1.0),
        )
        out.append(top_values(values, logp, k))
    return out


def predict_digits(
    crops: Mapping[str, np.ndarray],
    *,
    k: int = TOP_K,
    model: OnnxModel | None = None,
) -> dict[str, list[tuple[int, float]]]:
    """Top-k значений для кропов двузначных подполей: `{подполе: [(значение, вероятность)]}`.

    `crops` — серые кропы по именам подполей (`left_base.hour`, …). Все кропы идут одним
    батчем; вероятности откалиброваны температурами из `meta.json` и нормированы по
    допустимому диапазону части (день 1–31, месяц 1–12, час 0–24, минуты 0–59).
    """
    model = model or get_model(DIGITS_MODEL)
    names = list(crops)
    parts = [part_of(n) for n in names]
    tens, units = digit_logits(model, [crops[n] for n in names], parts)
    tops = digit_top_values(tens, units, parts, model.meta.temperatures, k)
    return dict(zip(names, tops, strict=True))


# --- номер ------------------------------------------------------------------------------------


def number_logits(model: OnnxModel, crops: Sequence[np.ndarray]) -> np.ndarray:
    """Сырые логиты модели номера: `(N, T, 11)` для `ctc`, `(N, 3, 11)` для `heads`."""
    if model.meta.task != "number":
        raise OcrRuntimeError(f"{model.meta.name}: это не модель номера")
    if not len(crops):
        return np.zeros((0, 0, 0), dtype=np.float32)
    return model.run({model.meta.input_name: model.prepare(crops)})[0]


def number_kind(model: OnnxModel) -> str:
    return str(model.meta.raw["kind"])


def number_temperature(model: OnnxModel) -> float | np.ndarray:
    """Температура номера из `meta.json`: число (`ctc`) или массив по головам (`heads`)."""
    temps = model.meta.temperatures
    if number_kind(model) == "ctc":
        return temps.get("ctc", 1.0)
    return np.array([temps.get(h, 1.0) for h in model.meta.raw["heads"]], dtype=np.float64)


def predict_number(
    crop: np.ndarray, *, k: int = TOP_K, model: OnnxModel | None = None
) -> list[tuple[int, float]]:
    """Top-k номеров `1..999` с откалиброванными вероятностями (нормировка по всем номерам)."""
    model = model or get_model(NUMBER_MODEL)
    out = number_logits(model, [crop])[0]
    return top_numbers(out, kind=number_kind(model), k=k, temperature=number_temperature(model))


def score_number_candidates(
    crop: np.ndarray,
    candidates: Sequence[str | int],
    *,
    model: OnnxModel | None = None,
) -> dict[str, float]:
    """`log P` каждого кандидата номера по кропу (с температурой): `{кандидат: logp}`.

    Оценки не нормированы по набору кандидатов (это вероятность строки под моделью), поэтому
    их складывают с приорами декодера как есть. Недопустимый кандидат получает `-inf`.
    """
    model = model or get_model(NUMBER_MODEL)
    out = number_logits(model, [crop])[0]
    return score_candidates(
        out, candidates, kind=number_kind(model), temperature=number_temperature(model)
    )


def number_distribution(
    crop: np.ndarray, *, model: OnnxModel | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Нормированное распределение по номерам `1..999`: `(values, logp)`."""
    model = model or get_model(NUMBER_MODEL)
    out = number_logits(model, [crop])[0]
    return number_log_probs(out, kind=number_kind(model), temperature=number_temperature(model))
