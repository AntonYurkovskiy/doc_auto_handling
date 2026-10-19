"""T25: рантайм-пайплайн распознавания ваучера — один вызов для приложения.

Собирает воедино то, что уже есть в `app/ocr/*`: «вариант бланка → выравнивание → кропы →
модель → декодер». Приложение (T26) вызывает только :func:`recognize_voucher`; инференс
идёт на CPU внутри процесса приложения, отдельный сервис не нужен.

Деградация без падений: нет моделей/эталонов — `status="no_models"`; выравнивание не
прошло пороги — `status="align_failed"`; что угодно ещё (битый файл, сбой модели) —
`status="error"`. Ни одно исключение наружу не выходит.

Производительность: модели и эталоны бланков загружаются один раз на процесс (кэш по
каталогам и версии модели, под блокировкой) — см. :func:`get_resources`.

Зависимости — только то, что уже использует `app/ocr/`: numpy, opencv-python-headless,
pypdfium2, Pillow, onnxruntime (лениво, внутри `app.ocr.runtime`). torch не импортируется.
"""

from __future__ import annotations

import json
import os
import threading
import time
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from typing import Protocol

import cv2
import numpy as np

from app.ocr import runtime as rt
from app.ocr.crops import crop_box, locate_boxes
from app.ocr.decoder import (
    PARTS,
    ROWS,
    DecodeContext,
    DecodeResult,
    DecoderModel,
    FormRow,
    Record,
    RowOption,
    decode,
    subfield,
)
from app.ocr.io import ScanLoadError, load_scan, normalize_width
from app.ocr.layouts import VOUCHER_NUMBER, LayoutError
from app.ocr.layouts import Layout as BoxLayout
from app.ocr.layouts import load_layouts as load_box_layouts
from app.ocr.number_decoder import NumberContext, NumberPriors, NumberResult, decode_number_output
from app.ocr.pair_decoder import PairContext, PairPriors, decode_pair
from app.ocr.variants import Layout as VariantLayout
from app.ocr.variants import detect_variant
from app.ocr.variants import load_layouts as load_variant_layouts
from app.services.voucher_files import safe_basename

# --- статусы VoucherRecognition.status ----------------------------------------------------

STATUS_OK = "ok"
STATUS_NO_MODELS = "no_models"
STATUS_ALIGN_FAILED = "align_failed"
STATUS_ERROR = "error"

#: Подполя дат (без номера), которые идут моделью цифр одним батчем.
DIGIT_SUBFIELDS: tuple[str, ...] = tuple(subfield(r, p) for r in ROWS for p in PARTS)
#: top-k модели цифр для декодера — лучше полное распределение (T19/T21).
INPUT_K = 60
#: Сид ГСЧ OpenCV перед выравниванием — воспроизводимость RANSAC (как в `ocr_lab`).
CV_SEED = 0
#: Консервативный порог автоприёма по умолчанию (T21/T22, `MODEL_CARD_v1.md`): выбор
#: окончательного порога для прода остаётся за человеком (H5), это лишь запасной вариант.
DEFAULT_AUTO_ACCEPT_THRESHOLD = 0.9995
#: Дополнительный флаг: партнёр для пары указан, но его не удалось распознать/прочитать.
FLAG_PARTNER_UNAVAILABLE = "partner_unavailable"

_REPO_ROOT = Path(__file__).resolve().parent.parent.parent


# ---------------------------------------------------------------------------
# Каталоги по умолчанию (как `app.ocr.runtime.models_dir`)
# ---------------------------------------------------------------------------


def default_layouts_dir() -> Path:
    """Каталог эталонов вариантов бланка: `OCR_LAYOUTS_DIR`, иначе `<OCR_WORK_DIR>/layouts`."""
    explicit = os.environ.get("OCR_LAYOUTS_DIR")
    if explicit:
        return Path(explicit)
    work = os.environ.get("OCR_WORK_DIR")
    return (Path(work) if work else _REPO_ROOT / "data" / "ocr") / "layouts"


def default_crops_dir() -> Path:
    """Каталог сохранённых кропов для UI: `OCR_CROPS_DIR`, иначе `data/files/ocr_crops`."""
    explicit = os.environ.get("OCR_CROPS_DIR")
    if explicit:
        return Path(explicit)
    return _REPO_ROOT / "data" / "files" / "ocr_crops"


# ---------------------------------------------------------------------------
# Контекст вызова (то, что знает приложение о ваучере)
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberHistory:
    """Что приложение знает об истории номеров (буксир, год) без текущего ваучера.

    Строится вызывающим кодом (T26) — `app.services.voucher_number.predict_next` даёт
    только `expected`; полную историю (`used`, `pause_days`) даёт
    `app.ocr.number_decoder.history_context` по уже подтверждённым ваучерам того же
    буксира и года. `None` в :attr:`RecognitionContext.number_history` — приора порядка
    нет (например, буксир ещё не определён).
    """

    expected: int | None = None
    used: frozenset[int] = frozenset()
    pause_days: float | None = None


@dataclass(frozen=True)
class ConfirmedPartner:
    """Второй ваучер той же заявки/работы уже подтверждён человеком (T20).

    `forms` — его записи в написании бланка (`{строка: (дата, час, минута)}`, `FormRow`
    из `app.ocr.decoder`) — сильный приор, не кандидат.
    """

    forms: Mapping[str, FormRow]


@dataclass(frozen=True)
class PendingPartner:
    """Второй ваучер той же заявки/работы ещё ждёт проверки — его тоже нужно распознать.

    `path`/`tug_hint` — как у основного вызова :func:`recognize_voucher`; `context` — его
    собственный контекст ядра (год, заявка, вид работ). Если партнёра распознать не
    удастся, пара не применяется (см. :data:`FLAG_PARTNER_UNAVAILABLE`), а не падает.
    """

    path: Path
    context: RecognitionContext
    tug_hint: str | None = None


@dataclass(frozen=True)
class RecognitionContext:
    """Контекст ваучера, который знает приложение, — для совместного декодера (T19/T20).

    - `year` — год ваучера (из имени файла или печати бланка); базовая дата ищется в этом
      году, переход через Новый год внутри ваучера — календарная арифметика;
    - `app_dt` — время из заявки (наивное локальное) — слабый приор начала работ;
    - `work_type` — нормализованный вид работ (см. `app.services.calculation.
      normalize_work_type`) — для приоров длительностей; `None` — общие приоры;
    - `filename` — исходное имя файла ваучера (для приора номера, T01); `None` — берётся
      имя файла `path` из :func:`recognize_voucher`;
    - `number_history` — см. :class:`NumberHistory`; `None` — приора порядка номера нет;
    - `partner` — :class:`ConfirmedPartner`, :class:`PendingPartner` или `None` (пары нет);
    - `extra_years` — дополнительные допустимые годы базовой даты (см. `DecodeContext`).
    """

    year: int
    app_dt: datetime | None = None
    work_type: str | None = None
    filename: str | None = None
    number_history: NumberHistory | None = None
    partner: ConfirmedPartner | PendingPartner | None = None
    extra_years: tuple[int, ...] = ()

    def decode_context(self) -> DecodeContext:
        return DecodeContext(
            year=self.year,
            app_dt=self.app_dt,
            work_type=self.work_type,
            extra_years=self.extra_years,
        )


# ---------------------------------------------------------------------------
# Результат
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class AlignmentInfo:
    """Метрики выравнивания выбранного варианта бланка (`app.ocr.align.AlignResult`)."""

    ok: bool
    score: float
    inliers: int
    reproj_err: float
    rotated180: bool
    method: str
    margin: float


@dataclass(frozen=True)
class VoucherRecognition:
    """Итог распознавания одного ваучера — контракт для приложения (T26/T27).

    - `status`/`reason` — `"ok"` | `"no_models"` | `"align_failed"` | `"error"` и текст
      причины (без данных ваучера, только тип/конфигурация);
    - `variant`/`alignment` — выбранный вариант бланка и метрики выравнивания;
    - `fields` — по каждой из 4 строк дат — top-3 вариантов `(дата, час, минута)` с
      вероятностью (`app.ocr.decoder.DecodeResult.marginals`);
    - `voucher_number` — top-3 номеров `(номер, вероятность)`;
    - `records` — top-N записей ваучера целиком (`app.ocr.decoder.Record`, со `score`/`p`);
    - `confidence`/`margin` — записи **ядра** (чтение одного ваучера, без пары): по ним
      плашка, автоприём и мониторинг (DECISIONS T21: пара поднимает уверенность и у ошибок,
      совпавших с партнёром, поэтому для автоприёма не годится). `fields`/`records` —
      после пары, если она была: лучшие подсказки оператору;
    - `pair_confidence` — уверенность записи после пары (`None`, если пары не было);
    - `auto_accept` — `confidence` ядра не ниже порога; только предзаполнение, не
      подтверждение (DECISIONS Q3) — решение о статусе ваучера принимает приложение;
    - `crop_paths` — пути сохранённых кропов подполей (для показа в UI, T27);
    - `timing` — время по этапам, секунды;
    - `flags` — коды из `app.ocr.decoder`/`number_decoder`/`pair_decoder`
      (`fields_top1_*`, `hour24`, `number_used`, `pair_same`, …) плюс
      :data:`FLAG_PARTNER_UNAVAILABLE`.
    """

    status: str
    reason: str
    variant: str | None
    alignment: AlignmentInfo | None
    fields: Mapping[str, tuple[RowOption, ...]]
    voucher_number: tuple[tuple[int, float], ...]
    records: tuple[Record, ...]
    confidence: float
    margin: float
    auto_accept: bool
    crop_paths: Mapping[str, Path]
    timing: Mapping[str, float]
    flags: tuple[str, ...]
    pair_confidence: float | None = None
    #: Выровненная по эталону страница (1654×2340, серая) — только при `keep_aligned=True`;
    #: из неё приложение вырезает печатные строки (`app.ocr.printed`).
    aligned_page: np.ndarray | None = field(default=None, repr=False, compare=False)

    @property
    def ok(self) -> bool:
        return self.status == STATUS_OK

    @property
    def top(self) -> Record | None:
        return self.records[0] if self.records else None


def _empty_result(
    status: str,
    reason: str,
    timing: Mapping[str, float],
    *,
    variant: str | None = None,
    alignment: AlignmentInfo | None = None,
) -> VoucherRecognition:
    return VoucherRecognition(
        status=status,
        reason=reason,
        variant=variant,
        alignment=alignment,
        fields={},
        voucher_number=(),
        records=(),
        confidence=0.0,
        margin=0.0,
        auto_accept=False,
        crop_paths={},
        timing=dict(timing),
        flags=(),
    )


def unavailable_result(reason: str) -> VoucherRecognition:
    """Результат «пайплайн недоступен» (`no_models`) для приложения, когда ресурсов нет."""
    return _empty_result(STATUS_NO_MODELS, reason, {})


# ---------------------------------------------------------------------------
# Распознаватель моделей — подменяется в тестах (без реальных весов)
# ---------------------------------------------------------------------------


class Recognizer(Protocol):
    """Модели цифр и номера в том виде, в каком их зовёт пайплайн (см. `OnnxRecognizer`)."""

    def digits(self, crops: Mapping[str, np.ndarray]) -> dict[str, list[tuple[int, float]]]: ...

    def number(self, crop: np.ndarray) -> tuple[np.ndarray, str, float | list[float]]:
        """Сырой выход модели номера, её вид (`ctc`/`heads`) и температура калибровки."""
        ...


@dataclass
class OnnxRecognizer:
    """Рантайм T18 (`app.ocr.runtime`) как :class:`Recognizer`."""

    digits_model: rt.OnnxModel
    number_model: rt.OnnxModel
    k: int = INPUT_K

    def digits(self, crops: Mapping[str, np.ndarray]) -> dict[str, list[tuple[int, float]]]:
        return rt.predict_digits(crops, k=self.k, model=self.digits_model)

    def number(self, crop: np.ndarray) -> tuple[np.ndarray, str, float | list[float]]:
        out = rt.number_logits(self.number_model, [crop])[0]
        raw_temperature = rt.number_temperature(self.number_model)
        temperature: float | list[float] = (
            raw_temperature.tolist() if isinstance(raw_temperature, np.ndarray) else raw_temperature
        )
        return out, rt.number_kind(self.number_model), temperature


# ---------------------------------------------------------------------------
# Ресурсы процесса: эталоны, макеты боксов, модели, приоры — загружаются один раз
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PipelineResources:
    """Всё, что грузится с диска один раз на процесс."""

    variants: Mapping[str, VariantLayout]
    boxes: Mapping[str, BoxLayout]
    recognizer: Recognizer
    decoder_model: DecoderModel
    number_priors: NumberPriors
    pair_priors: PairPriors


def _load_number_priors(path: Path) -> NumberPriors:
    if not path.is_file():
        return NumberPriors.default()
    return NumberPriors.from_json(json.loads(path.read_text(encoding="utf-8")))


def _load_pair_priors(path: Path) -> PairPriors:
    if not path.is_file():
        return PairPriors.default()
    return PairPriors.from_json(json.loads(path.read_text(encoding="utf-8")))


def _build_resources(models_dir: Path, layouts_dir: Path, model_version: str) -> PipelineResources:
    variants = load_variant_layouts(layouts_dir)
    if not variants:
        raise FileNotFoundError(f"нет эталонов бланков в {layouts_dir}")
    boxes = load_box_layouts()
    if not boxes:
        raise LayoutError("нет макетов боксов подполей (app/ocr/layouts/*.json)")
    digits_model = rt.OnnxModel(models_dir / f"digits_{model_version}")
    number_model = rt.OnnxModel(models_dir / f"number_{model_version}")
    return PipelineResources(
        variants=variants,
        boxes=boxes,
        recognizer=OnnxRecognizer(digits_model, number_model),
        decoder_model=DecoderModel.load(models_dir / f"decoder_priors_{model_version}.json"),
        number_priors=_load_number_priors(models_dir / f"number_priors_{model_version}.json"),
        pair_priors=_load_pair_priors(models_dir / f"pair_priors_{model_version}.json"),
    )


_RESOURCES: dict[tuple[str, str, str], PipelineResources] = {}
_RESOURCES_LOCK = threading.Lock()
#: Ключ -> (когда не удалось, класс ошибки, текст). Без этого на машине без моделей каждый
#: ваучер заново пытался загрузить эталоны (+1,4 с, дефект D4 приёмки T29).
_UNAVAILABLE: dict[tuple[str, str, str], tuple[float, type[Exception], str]] = {}
#: Сколько секунд помнить, что ресурсов нет, прежде чем попробовать загрузить заново.
UNAVAILABLE_TTL_S = 60.0


def get_resources(
    *,
    models_dir: Path | None = None,
    layouts_dir: Path | None = None,
    model_version: str | None = None,
) -> PipelineResources:
    """Ресурсы процесса — один раз на (каталог моделей, каталог эталонов, версия).

    По умолчанию каталоги и версия — те же, что использует `app.ocr.runtime`
    (`OCR_MODELS_DIR`/`OCR_WORK_DIR`, `OCR_MODEL_VERSION`) и :func:`default_layouts_dir`.
    Приложение (T26) должно передавать их явно из `Settings` (`ocr_models_dir`,
    `ocr_layouts_dir`, `ocr_model_version`), не полагаясь на переменные окружения.

    Бросает `FileNotFoundError`/`app.ocr.layouts.LayoutError`/`app.ocr.runtime.
    OcrRuntimeError`, если эталонов или моделей нет, — вызывающий (:func:`recognize_voucher`)
    превращает это в `status="no_models"`.
    """
    resolved_models = Path(models_dir) if models_dir is not None else rt.models_dir()
    resolved_layouts = Path(layouts_dir) if layouts_dir is not None else default_layouts_dir()
    version = model_version or rt.MODEL_VERSION
    key = (str(resolved_models.resolve()), str(resolved_layouts.resolve()), version)
    with _RESOURCES_LOCK:
        resources = _RESOURCES.get(key)
        if resources is None:
            failed = _UNAVAILABLE.get(key)
            if failed is not None and time.monotonic() - failed[0] < UNAVAILABLE_TTL_S:
                raise failed[1](failed[2])
            try:
                resources = _build_resources(resolved_models, resolved_layouts, version)
            except (FileNotFoundError, LayoutError, rt.OcrRuntimeError) as exc:
                _UNAVAILABLE[key] = (time.monotonic(), type(exc), str(exc))
                raise
            _UNAVAILABLE.pop(key, None)
            _RESOURCES[key] = resources
        return resources


def reset_resources() -> None:
    """Забыть загруженные ресурсы (тесты, смена каталогов на лету)."""
    with _RESOURCES_LOCK:
        _RESOURCES.clear()
        _UNAVAILABLE.clear()


# ---------------------------------------------------------------------------
# Ядро: скан -> вариант -> кропы -> распределения подполей
# ---------------------------------------------------------------------------


@dataclass
class _CoreResult:
    status: str  # "ok" | "align_failed"
    reason: str
    variant: str | None
    alignment: AlignmentInfo
    tug_code: str | None
    inputs: dict[str, list[tuple[int, float]]] = field(default_factory=dict)
    number_crop: np.ndarray | None = None
    crops: dict[str, np.ndarray] = field(default_factory=dict)
    timing: dict[str, float] = field(default_factory=dict)
    warped: np.ndarray | None = None


def _recognize_core(
    page: np.ndarray, resources: PipelineResources, *, tug_hint: str | None
) -> _CoreResult:
    """Скан (после `normalize_width`) -> вариант, кропы и распределения подполей дат."""
    timing: dict[str, float] = {}
    t0 = time.perf_counter()
    cv2.setRNGSeed(CV_SEED)
    match = detect_variant(page, resources.variants, prefer=tug_hint)
    timing["align"] = time.perf_counter() - t0
    res = match.result
    alignment = AlignmentInfo(
        ok=bool(match.ok),
        score=float(res.score),
        inliers=int(res.inliers),
        reproj_err=float(res.reproj_err),
        rotated180=bool(res.rotated180),
        method=res.method,
        margin=float(match.margin),
    )
    if not match.ok or res.warped is None:
        reason = res.reason or "выравнивание не прошло пороги"
        return _CoreResult("align_failed", reason, match.variant, alignment, None, timing=timing)
    layout = resources.boxes.get(match.variant)
    if layout is None:
        reason = f"нет макета боксов для варианта {match.variant}"
        return _CoreResult("align_failed", reason, match.variant, alignment, None, timing=timing)

    t0 = time.perf_counter()
    placements = locate_boxes(res.warped, layout)
    crops = {name: crop_box(res.warped, pl.box) for name, pl in placements.items()}
    timing["crops"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    digit_crops = {name: crops[name] for name in DIGIT_SUBFIELDS if name in crops}
    inputs = resources.recognizer.digits(digit_crops)
    timing["digits"] = time.perf_counter() - t0

    variant_layout = resources.variants.get(match.variant)
    tug_code = None
    if variant_layout is not None and variant_layout.meta.get("tug_code"):
        tug_code = str(variant_layout.meta["tug_code"])
    return _CoreResult(
        "ok",
        "",
        match.variant,
        alignment,
        tug_code,
        inputs=inputs,
        number_crop=crops.get(VOUCHER_NUMBER),
        crops=crops,
        timing=timing,
        warped=res.warped,
    )


# ---------------------------------------------------------------------------
# Номер и пара
# ---------------------------------------------------------------------------


def _number_context(
    context: RecognitionContext, *, tug_code: str | None, default_filename: str
) -> NumberContext:
    history = context.number_history
    return NumberContext(
        expected=history.expected if history else None,
        used=history.used if history else frozenset(),
        pause_days=history.pause_days if history else None,
        filename=context.filename or default_filename,
        tug_code=tug_code,
    )


def _decode_number(
    core: _CoreResult,
    context: RecognitionContext,
    resources: PipelineResources,
) -> NumberResult:
    number_ctx = _number_context(context, tug_code=core.tug_code, default_filename="")
    if core.number_crop is None:
        return decode_number_output(None, number_ctx, resources.number_priors)
    output, kind, temperature = resources.recognizer.number(core.number_crop)
    return decode_number_output(
        output, number_ctx, resources.number_priors, kind=kind, temperature=temperature
    )


def _decode_with_partner(
    a_inputs: Mapping[str, list[tuple[int, float]]],
    decode_ctx: DecodeContext,
    core_result: DecodeResult,
    partner: ConfirmedPartner | PendingPartner,
    resources: PipelineResources,
) -> tuple[DecodeResult, float | None]:
    """Совместный декодер с партнёром; при сбое партнёра — запись ядра и флаг-предупреждение."""
    try:
        if isinstance(partner, ConfirmedPartner):
            pair_ctx = PairContext(a=decode_ctx, b_confirmed=partner.forms)
            pair_result = decode_pair(
                a_inputs, None, pair_ctx, resources.decoder_model, resources.pair_priors
            )
        else:
            b_page = normalize_width(load_scan(partner.path))
            b_core = _recognize_core(b_page, resources, tug_hint=partner.tug_hint)
            if b_core.status != "ok":
                raise ScanLoadError(b_core.reason)
            pair_ctx = PairContext(a=decode_ctx, b=partner.context.decode_context())
            pair_result = decode_pair(
                a_inputs, b_core.inputs, pair_ctx, resources.decoder_model, resources.pair_priors
            )
    except (ScanLoadError, FileNotFoundError):
        flags = tuple(sorted({*core_result.flags, FLAG_PARTNER_UNAVAILABLE}))
        return _replace_flags(core_result, flags), None
    return pair_result.a, pair_result.p_same


def _replace_flags(result: DecodeResult, flags: tuple[str, ...]) -> DecodeResult:
    return replace(result, flags=flags)


# ---------------------------------------------------------------------------
# Сохранение кропов для UI (T27)
# ---------------------------------------------------------------------------


def _write_png(path: Path, image: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", image)
    if not ok:
        raise ValueError(f"не кодируется PNG: {path.name}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(buf.tobytes())


def crop_output_dir(scan_path: str | Path, crops_dir: Path) -> Path:
    """Каталог сохранённых кропов конкретного скана: `<crops_dir>/<безопасное имя без
    расширения>/`. Та же схема имён, что :func:`_save_crops` использует при сохранении —
    публичная версия нужна T27 (`/vouchers/{id}/crops/{subfield}`), чтобы не дублировать
    схему имён в приложении.
    """
    stem = Path(safe_basename(Path(scan_path).name)).stem
    return crops_dir / stem


def _save_crops(
    scan_path: Path, crops: Mapping[str, np.ndarray], crops_dir: Path
) -> dict[str, Path]:
    """Сохраняет кропы подполей в `<crops_dir>/<безопасное имя скана без расширения>/`."""
    out_dir = crop_output_dir(scan_path, crops_dir)
    out: dict[str, Path] = {}
    for name, crop in crops.items():
        dest = out_dir / f"{name}.png"
        _write_png(dest, crop)
        out[name] = dest
    return out


# ---------------------------------------------------------------------------
# Контракт приложения
# ---------------------------------------------------------------------------


def recognize_voucher(
    path: str | Path,
    *,
    tug_hint: str | None = None,
    context: RecognitionContext,
    resources: PipelineResources | None = None,
    auto_accept_threshold: float = DEFAULT_AUTO_ACCEPT_THRESHOLD,
    crops_dir: Path | None = None,
    save_crops: bool = True,
    keep_aligned: bool = False,
) -> VoucherRecognition:
    """Распознать один ваучер: файл скана -> вариант -> кропы -> модели -> декодер.

    `path` — путь к файлу скана (PDF или картинка), `tug_hint` — подсказка варианта бланка
    (код буксира из имени файла или ожидаемое имя варианта), `context` — то, что знает о
    ваучере приложение (:class:`RecognitionContext`).

    Не бросает исключений: любой сбой превращается в `VoucherRecognition.status`
    (`"no_models"` | `"align_failed"` | `"error"`) с понятным `reason`. `resources` —
    заранее загруженные ресурсы (по умолчанию — :func:`get_resources`, общие на процесс);
    `auto_accept_threshold` — порог `confidence` для `auto_accept` (порог для прода решает
    человек, см. `docs/ocr_tasks/DECISIONS.md`); `crops_dir`/`save_crops` — куда сохранять
    кропы подполей для UI (T27), по умолчанию :func:`default_crops_dir`; `keep_aligned` —
    вернуть выровненную страницу в `VoucherRecognition.aligned_page` (для печатных строк).
    """
    timing: dict[str, float] = {}
    t_all = time.perf_counter()
    scan_path = Path(path)

    try:
        active_resources = resources if resources is not None else get_resources()
    except (FileNotFoundError, LayoutError, rt.OcrRuntimeError) as exc:
        return _empty_result(STATUS_NO_MODELS, str(exc), timing)

    try:
        t0 = time.perf_counter()
        page = normalize_width(load_scan(scan_path))
        timing["load"] = time.perf_counter() - t0
    except (FileNotFoundError, ScanLoadError) as exc:
        return _empty_result(STATUS_ERROR, str(exc), timing)

    core: _CoreResult | None = None
    try:
        core = _recognize_core(page, active_resources, tug_hint=tug_hint)
        timing.update(core.timing)
        if core.status != "ok":
            return _empty_result(
                STATUS_ALIGN_FAILED, core.reason, timing, variant=core.variant,
                alignment=core.alignment,
            )

        decode_ctx = context.decode_context()
        t0 = time.perf_counter()
        core_result = decode(core.inputs, decode_ctx, active_resources.decoder_model)
        timing["decode"] = time.perf_counter() - t0

        t0 = time.perf_counter()
        number_result = _decode_number(core, context, active_resources)
        timing["number"] = time.perf_counter() - t0

        final_result = core_result
        if context.partner is not None:
            t0 = time.perf_counter()
            final_result, _p_same = _decode_with_partner(
                core.inputs, decode_ctx, core_result, context.partner, active_resources
            )
            timing["pair"] = time.perf_counter() - t0
    except (rt.OcrRuntimeError, ModuleNotFoundError, ImportError) as exc:
        return _empty_result(
            STATUS_NO_MODELS, str(exc), timing, variant=core.variant if core else None,
        )
    except Exception as exc:  # сбой одного ваучера не должен ронять приложение
        return _empty_result(STATUS_ERROR, type(exc).__name__, timing)

    crop_paths: dict[str, Path] = {}
    if save_crops:
        t0 = time.perf_counter()
        try:
            crop_paths = _save_crops(scan_path, core.crops, crops_dir or default_crops_dir())
        except OSError:
            crop_paths = {}
        timing["save_crops"] = time.perf_counter() - t0

    timing["total"] = time.perf_counter() - t_all
    # Плашка и автоприём — по ядру, без пары (решение T21, DECISIONS Q3, T29 В7).
    confidence = core_result.confidence if core_result.records else 0.0
    margin = core_result.margin if core_result.records else 0.0
    auto_accept = bool(core_result.records) and confidence >= auto_accept_threshold
    pair_confidence = (
        final_result.confidence if context.partner is not None and final_result.records else None
    )
    flags = tuple(sorted({*final_result.flags, *number_result.flags}))
    return VoucherRecognition(
        status=STATUS_OK,
        reason="",
        variant=core.variant,
        alignment=core.alignment,
        fields=final_result.marginals,
        voucher_number=number_result.candidates[:3],
        records=final_result.records,
        confidence=confidence,
        margin=margin,
        auto_accept=auto_accept,
        crop_paths=crop_paths,
        timing=timing,
        flags=flags,
        pair_confidence=pair_confidence,
        aligned_page=core.warped if keep_aligned else None,
    )
