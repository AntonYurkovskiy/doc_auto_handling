"""Варианты бланка ваучера: загрузка эталонов и выбор варианта по выравниванию.

У буксиров разные бланки (вёрстка, шрифт, положение строк). Каждый вариант — каталог
``<layouts_dir>/<variant>/`` с файлами:

- ``reference.png`` — статичный эталон (медиана выровненных сканов, только форма);
- ``static_mask.png`` — маска статичных зон (255 — печать бланка), необязательна;
- ``meta.json`` — размер, исходные ``scan_id``, параметры сборки и, в ключе
  ``align_params``, откалиброванные поля :class:`app.ocr.align.AlignParams`.

:func:`detect_variant` выравнивает скан по каждому эталону и берёт лучший ``score``:
в проде буксир (а значит и бланк) может быть неизвестен.

Зависимости — только numpy и opencv.
"""

from __future__ import annotations

import dataclasses
import json
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from app.ocr.align import AlignParams, AlignResult, Reference, align, build_reference

REFERENCE_FILE = "reference.png"
MASK_FILE = "static_mask.png"
META_FILE = "meta.json"

_PARAM_FIELDS = {f.name for f in dataclasses.fields(AlignParams)}

#: ``score``, начиная с которого вариант считается найденным уверенно. На реальных сканах
#: (T07) чужой бланк набирает не больше 0,57, свой выровненный — не меньше 0,70
#: (``min_score`` в ``meta.json`` вариантов), 98 % своих — от 0,75.
CONFIDENT_SCORE = 0.75


@dataclass(frozen=True, eq=False)
class Layout:
    """Вариант бланка: имя, подготовленный эталон и метаданные из ``meta.json``."""

    name: str
    reference: Reference
    meta: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True, eq=False)
class VariantMatch:
    """Результат :func:`detect_variant`.

    ``variant`` — лучший вариант (даже если его выравнивание не прошло пороги; тогда
    ``ok=False``), ``result`` — его :class:`AlignResult`, ``results`` — результаты по всем
    проверенным вариантам, ``margin`` — отрыв лучшего ``score`` от второго (0, если
    проверен один вариант).
    """

    variant: str
    ok: bool
    result: AlignResult
    results: dict[str, AlignResult]
    margin: float


def _read_png(path: Path) -> np.ndarray:
    """Серое изображение; путь может содержать не-ASCII (cv2.imread на Windows не читает)."""
    data = np.frombuffer(path.read_bytes(), np.uint8)
    img = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"не читается изображение эталона: {path.name}")
    return img


def _write_png(path: Path, img: np.ndarray) -> None:
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"не кодируется PNG: {path.name}")
    path.write_bytes(buf.tobytes())


def params_from_meta(meta: Mapping[str, Any], base: AlignParams | None = None) -> AlignParams:
    """``AlignParams`` варианта: значения по умолчанию плюс ``meta["align_params"]``.

    Неизвестные ключи игнорируются: файл мог быть записан более новой версией кода.
    """
    stored = dict(meta.get("align_params") or {})
    overrides = {k: v for k, v in stored.items() if k in _PARAM_FIELDS}
    return dataclasses.replace(base or AlignParams(), **overrides)


def save_layout(
    directory: Path,
    reference: np.ndarray,
    mask: np.ndarray | None,
    meta: Mapping[str, Any],
) -> None:
    """Пишет вариант бланка в каталог ``directory`` (создаётся при необходимости)."""
    directory.mkdir(parents=True, exist_ok=True)
    _write_png(directory / REFERENCE_FILE, reference)
    if mask is not None:
        _write_png(directory / MASK_FILE, mask)
    payload = dict(meta)
    payload.setdefault("name", directory.name)
    payload["width"] = int(reference.shape[1])
    payload["height"] = int(reference.shape[0])
    (directory / META_FILE).write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


def load_layout(directory: Path, params: AlignParams | None = None) -> Layout:
    """Загружает вариант из каталога. ``params`` — база, поверх неё идут пороги из меты."""
    ref_path = directory / REFERENCE_FILE
    if not ref_path.exists():
        raise FileNotFoundError(f"нет эталона варианта: {ref_path}")
    meta_path = directory / META_FILE
    meta: dict[str, Any] = {}
    if meta_path.exists():
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
    image = _read_png(ref_path)
    mask_path = directory / MASK_FILE
    mask = _read_png(mask_path) if mask_path.exists() else None
    name = str(meta.get("name") or directory.name)
    reference = build_reference(image, mask, name=name, params=params_from_meta(meta, params))
    return Layout(name=name, reference=reference, meta=meta)


def load_layouts(layouts_dir: Path, params: AlignParams | None = None) -> dict[str, Layout]:
    """Все варианты из ``layouts_dir`` (подкаталоги с ``reference.png``), по имени."""
    if not layouts_dir.is_dir():
        return {}
    out: dict[str, Layout] = {}
    for sub in sorted(p for p in layouts_dir.iterdir() if p.is_dir()):
        if (sub / REFERENCE_FILE).exists():
            layout = load_layout(sub, params)
            out[layout.name] = layout
    return out


def _group(layout: Layout) -> str | None:
    """Буксир варианта (``tug_code`` из меты): варианты одного буксира — одна группа."""
    value = layout.meta.get("tug_code")
    return str(value) if value else None


def detect_variant(
    scan: np.ndarray,
    layouts: Mapping[str, Layout] | Iterable[Layout],
    *,
    prefer: str | None = None,
    confident_score: float | None = CONFIDENT_SCORE,
) -> VariantMatch:
    """Определяет вариант бланка: выравнивает скан по эталонам, берёт лучший.

    Лучший — сначала прошедший пороги ``ok``, затем с большим ``score``, затем с большим
    числом инлайеров. Пороги у каждого варианта свои (из его ``meta.json``).

    ``prefer`` — подсказка, с чего начать: имя варианта или код буксира (например, из имени
    файла). Она влияет только на порядок, а не на выбор.

    Ранний выход: если вариант прошёл пороги со ``score ≥ confident_score``, варианты других
    буксиров уже не проверяются (чужой бланк столько не набирает, см. калибровку T07), а
    варианты того же буксира — проверяются, чтобы выбрать вёрстку. ``None`` отключает выход.
    В ``results`` попадают только проверенные варианты.
    """
    items = list(layouts.values()) if isinstance(layouts, Mapping) else list(layouts)
    if not items:
        raise ValueError("detect_variant: нет ни одного варианта бланка")
    if prefer is not None:
        items.sort(key=lambda layout: prefer not in (layout.name, _group(layout)))
    results: dict[str, AlignResult] = {}
    confident: str | None = None
    for layout in items:
        group = _group(layout)
        if confident is not None and group != confident:
            continue
        res = align(scan, layout.reference)
        results[layout.name] = res
        if (
            confident is None
            and confident_score is not None
            and group is not None
            and res.ok
            and res.score >= confident_score
        ):
            confident = group
    ranked = sorted(
        results.items(), key=lambda kv: (kv[1].ok, kv[1].score, kv[1].inliers), reverse=True
    )
    best_name, best = ranked[0]
    margin = best.score - ranked[1][1].score if len(ranked) > 1 else 0.0
    return VariantMatch(
        variant=best_name, ok=best.ok, result=best, results=results, margin=float(margin)
    )
