"""Сквозная оценка цепочки распознавания (T21): скан → вариант → кропы → ONNX → декодер.

Запуск (пути — из `ocr_lab.paths`, то есть от `OCR_WORK_DIR`):

    python -m ocr_lab.eval_pipeline run --split val      # скан → распределения подполей
    python -m ocr_lab.eval_pipeline decode --split val   # ядро + номер + пары → JSONL T13
    python -m ocr_lab.eval_pipeline report               # порог на val → test, отчёт
    python -m ocr_lab.eval_pipeline review               # «уверенные расхождения» для H3
    python -m ocr_lab.eval_pipeline propose              # вердикты просмотра → поправки

- `run` берёт **файл скана** (PDF/картинку из `scan_path` манифеста), а не готовые кропы T10, и
  проходит те же модули `app/ocr/*`, что будет звать приложение: `io.load_scan` +
  `normalize_width` → `variants.detect_variant` (подсказка — код буксира из имени файла) →
  `crops.locate_boxes` + `crop_box` → `runtime` (onnxruntime, температуры из `meta.json`).
  Пишет `e2e/<name>/<split>_scans.jsonl` (полные распределения подполей, сырой выход модели
  номера, метрики выравнивания, способ подгонки боксов, время по этапам) и полосы строк дат с
  номером для просмотра (`e2e/<name>/bands/<scan_id>/`). Готовые сканы пропускаются,
  `--max-minutes` останавливает прогон между сканами — его можно продолжить.
- `decode` — ядро T19, номер T20 (порядок по истории, без имени файла) и пары k/p T20, как в
  `ocr_lab.fit_number_pairs`. Контекст (год, заявка, вид работ) и история номеров — из
  манифеста. Пишет JSONL T13 (`<split>_predictions.jsonl`, источник `<name>`) и подробности
  (`<split>_details.jsonl`: уверенность ядра и пары, флаги, время декодера).
- `report` — уровни точности (`ocr_lab.evaluate`), порог на `val`, его проверка на `test`
  (покрытие, точность, интервал Уилсона), разбор ошибок по этапам, деньги, время →
  `reports/e2e_report.md` и `reports/e2e_metrics.json`.
- `review` — до 50 ваучеров, где уверенность не ниже порога, а истина говорит другое:
  `review/confident_disagreements/<scan_id>.png` (полосы строк, истина, предсказание,
  уверенность) и `index.csv`.
- `propose` — по вердиктам просмотра (`review/confident_disagreements/verdicts.csv`,
  `verdict=model`) дописывает в `truth_corrections.csv` поправки `status=proposed`,
  `source=T21`. Подтверждает их человек (H3).

В консоль и отчёты идут только агрегаты и `scan_id`.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, field, replace
from datetime import date, datetime
from pathlib import Path
from typing import Any, Protocol

import cv2
import numpy as np

from app.ocr.crops import Placement, crop_box, locate_boxes
from app.ocr.decoder import (
    CHAIN,
    ROWS,
    DecodeContext,
    DecodeResult,
    DecoderModel,
    FormRow,
    decode,
    subfield,
)
from app.ocr.io import load_scan, normalize_width
from app.ocr.layouts import VOUCHER_NUMBER, Box, load_layouts
from app.ocr.layouts import Layout as BoxLayout
from app.ocr.number_decoder import (
    NumberPriors,
    NumberResult,
    decode_number_output,
    filename_hint,
)
from app.ocr.pair_decoder import FLAG_PAIR_CHANGED, PairContext, PairPriors, decode_pair
from app.ocr.variants import Layout as VariantLayout
from app.ocr.variants import detect_variant
from app.ocr.variants import load_layouts as load_variants
from app.services.calculation import _is_night, minutes_between
from ocr_lab import paths
from ocr_lab.fit_decoder_priors import _int, context_for, read_manifest
from ocr_lab.fit_number_pairs import history_from_manifest, manifest_pairs, number_context
from ocr_lab.sheets import MAX_SIDE

NAME = "e2e_v0"
SPLITS = ("train", "val", "test")
DIGIT_SUBFIELDS: tuple[str, ...] = tuple(
    subfield(r, p) for r in ROWS for p in ("day", "month", "hour", "minute")
)
#: Все значения части (как `fit_decoder_priors inputs`): декодеру лучше полное распределение.
INPUT_K = 60
FIELDS_K = 5
TOP_N = 5
#: Сид ГСЧ OpenCV перед выравниванием — как в `ocr_lab.cut_crops` (воспроизводимость RANSAC).
CV_SEED = 0
#: Способы подгонки бокса «без своего подчёркивания» (`app.ocr.crops.Placement.source`).
FALLBACK_SOURCES = frozenset({"column", "row", "static"})
DEFAULT_TARGET = 0.995
MAX_REVIEW = 50

ROW_TITLES = {
    "left_base": "Выход из базы",
    "arrived_base": "Приход в базу",
    "started_work": "Начало работ",
    "finished_work": "Окончание работ",
}


def out_dir(name: str = NAME) -> Path:
    return paths.WORK_DIR / "e2e" / name


def scans_path(split: str, name: str = NAME) -> Path:
    return out_dir(name) / f"{split}_scans.jsonl"


def predictions_path(split: str, name: str = NAME) -> Path:
    return out_dir(name) / f"{split}_predictions.jsonl"


def details_path(split: str, name: str = NAME) -> Path:
    return out_dir(name) / f"{split}_details.jsonl"


def bands_dir(name: str = NAME) -> Path:
    return out_dir(name) / "bands"


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(encoding="utf-8") as fh:
        return [json.loads(line) for line in fh if line.strip()]


def _write_png(path: Path, img: np.ndarray) -> None:
    """PNG по пути с кириллицей (cv2.imwrite на Windows такие пути не пишет)."""
    ok, buf = cv2.imencode(".png", img)
    if not ok:
        raise ValueError(f"не кодируется PNG: {path.name}")
    paths.ensure_dir(path.parent)
    path.write_bytes(buf.tobytes())


def _read_png(path: Path) -> np.ndarray:
    img = cv2.imdecode(np.fromfile(str(path), np.uint8), cv2.IMREAD_UNCHANGED)
    if img is None:
        raise ValueError(f"не читается {path.name}")
    return img


# ---------------------------------------------------------------------------
# Распознаватель (onnxruntime) — подменяется в тестах
# ---------------------------------------------------------------------------


class Recognizer(Protocol):
    """Модели цифр и номера в том виде, в каком их зовёт приложение."""

    def digits(self, crops: Mapping[str, np.ndarray]) -> dict[str, list[tuple[int, float]]]: ...

    def number(self, crop: np.ndarray) -> tuple[np.ndarray, str, Any]:
        """Сырой выход модели номера, её вид (`ctc`/`heads`) и температура."""
        ...


@dataclass
class OnnxRecognizer:
    """Рантайм T18: `runtime.predict_digits` и сырые логиты номера для декодера T20."""

    models_dir: Path
    digits_name: str = "digits_v0"
    number_name: str = "number_v0"
    k: int = INPUT_K

    def __post_init__(self) -> None:
        from app.ocr import runtime as rt

        self._rt = rt
        self.digits_model = rt.OnnxModel(self.models_dir / self.digits_name)
        self.number_model = rt.OnnxModel(self.models_dir / self.number_name)

    def digits(self, crops: Mapping[str, np.ndarray]) -> dict[str, list[tuple[int, float]]]:
        return self._rt.predict_digits(crops, k=self.k, model=self.digits_model)

    def number(self, crop: np.ndarray) -> tuple[np.ndarray, str, Any]:
        rt = self._rt
        out = rt.number_logits(self.number_model, [crop])[0]
        temp = rt.number_temperature(self.number_model)
        temp_json = temp.tolist() if isinstance(temp, np.ndarray) else float(temp)
        return out, rt.number_kind(self.number_model), temp_json


# ---------------------------------------------------------------------------
# run: скан → распределения подполей
# ---------------------------------------------------------------------------


@dataclass
class Resources:
    variants: dict[str, VariantLayout]
    boxes: dict[str, BoxLayout]
    recognizer: Recognizer


def load_resources(
    layouts_dir: Path, models_dir: Path, *, digits: str = "digits_v0", number: str = "number_v0"
) -> Resources:
    variants = load_variants(layouts_dir)
    if not variants:
        raise FileNotFoundError(f"нет эталонов бланков в {layouts_dir}")
    return Resources(variants, load_layouts(), OnnxRecognizer(models_dir, digits, number))


def _union(boxes: Sequence[Box], name: str) -> Box:
    return Box(
        name,
        min(b.x0 for b in boxes),
        min(b.y0 for b in boxes),
        max(b.x1 for b in boxes),
        max(b.y1 for b in boxes),
        boxes[0].kind,
    )


def review_bands(warped: np.ndarray, placements: Mapping[str, Placement]) -> dict[str, np.ndarray]:
    """Полосы строк дат и кроп номера с рамками боксов подполей (для просмотра, Q5).

    Полоса строки — объединение её четырёх боксов, то есть только строка дат бланка; рамки
    показывают, где именно резались кропы (диагностика этапа «кроп»).
    """
    gray = warped if warped.ndim == 2 else cv2.cvtColor(warped, cv2.COLOR_BGR2GRAY)
    groups: dict[str, list[Box]] = {VOUCHER_NUMBER: [placements[VOUCHER_NUMBER].box]}
    for row in ROWS:
        groups[row] = [placements[subfield(row, p)].box for p in ("day", "month", "hour", "minute")]
    out: dict[str, np.ndarray] = {}
    for name, boxes in groups.items():
        band_box = _union(boxes, name)
        band = cv2.cvtColor(crop_box(gray, band_box), cv2.COLOR_GRAY2BGR)
        for b in boxes:
            cv2.rectangle(
                band,
                (b.x0 - band_box.x0, b.y0 - band_box.y0),
                (b.x1 - band_box.x0 - 1, b.y1 - band_box.y0 - 1),
                (230, 150, 40),
                1,
            )
        out[name] = band
    return out


def recognize_scan(
    path: Path, resources: Resources, *, tug_hint: str | None = None
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    """Один скан через `app/ocr/*`: запись для `<split>_scans.jsonl` и полосы для просмотра.

    Статусы: `ok`, `align_failed` (ни один вариант не прошёл пороги), `error` (сбой чтения
    или модели; в причине — только тип исключения).
    """
    timing: dict[str, float] = {}
    rec: dict[str, Any] = {"status": "ok", "reason": "", "timing": timing}
    t_all = t0 = time.perf_counter()
    page = normalize_width(load_scan(path))
    timing["load"] = time.perf_counter() - t0

    t0 = time.perf_counter()
    cv2.setRNGSeed(CV_SEED)
    match = detect_variant(page, resources.variants, prefer=tug_hint or None)
    timing["align"] = time.perf_counter() - t0
    res = match.result
    rec["variant"] = match.variant
    rec["align"] = {
        "ok": bool(match.ok),
        "score": round(float(res.score), 4),
        "inliers": int(res.inliers),
        "reproj_err": round(float(res.reproj_err), 3),
        "rotated180": bool(res.rotated180),
        "method": res.method,
        "margin": round(float(match.margin), 4),
        "checked": sorted(match.results),
    }
    if not match.ok or res.warped is None:
        rec["status"] = "align_failed"
        rec["reason"] = res.reason or "выравнивание не прошло пороги"
        timing["total"] = time.perf_counter() - t_all
        return rec, {}

    t0 = time.perf_counter()
    layout = resources.boxes[match.variant]
    placements = locate_boxes(res.warped, layout)
    crops = {name: crop_box(res.warped, pl.box) for name, pl in placements.items()}
    timing["crops"] = time.perf_counter() - t0
    rec["sources"] = {name: pl.source for name, pl in placements.items()}

    t0 = time.perf_counter()
    dists = resources.recognizer.digits({n: crops[n] for n in DIGIT_SUBFIELDS})
    timing["digits"] = time.perf_counter() - t0
    rec["inputs"] = {n: [[int(v), float(p)] for v, p in dists[n]] for n in DIGIT_SUBFIELDS}

    t0 = time.perf_counter()
    output, kind, temperature = resources.recognizer.number(crops[VOUCHER_NUMBER])
    timing["number"] = time.perf_counter() - t0
    rec["number_output"] = np.asarray(output, dtype=float).round(6).tolist()
    rec["number_kind"] = kind
    rec["number_temperature"] = temperature
    timing["total"] = time.perf_counter() - t_all
    return rec, review_bands(res.warped, placements)


def tug_hint_of(row: Mapping[str, str]) -> str | None:
    """Код буксира из имени файла (T01) — так подсказку получит приложение."""
    hint = filename_hint(row.get("voucher_file"))
    return hint.tug_code if hint is not None else None


def run_split(
    split: str,
    resources: Resources,
    *,
    manifest: Path = paths.MANIFEST,
    name: str = NAME,
    limit: int | None = None,
    max_minutes: float | None = None,
    force: bool = False,
    log: Callable[[str], None] | None = None,
) -> Counter[str]:
    """Прогон сплита; готовые сканы пропускаются (дозапись в JSONL)."""
    rows = [r for r in read_manifest(manifest) if r.get("split") == split]
    out = scans_path(split, name)
    paths.ensure_dir(out.parent)
    if force and out.exists():
        out.unlink()
    done = {r["scan_id"] for r in read_jsonl(out)}
    todo = [r for r in rows if r["scan_id"] not in done]
    if limit is not None:
        todo = todo[:limit]
    stats: Counter[str] = Counter(skipped=len(rows) - len(todo))
    deadline = time.monotonic() + 60 * max_minutes if max_minutes is not None else None
    with out.open("a", encoding="utf-8") as fh:
        for i, row in enumerate(todo):
            if deadline is not None and time.monotonic() >= deadline:
                stats["stopped"] = len(todo) - i
                break
            sid = row["scan_id"]
            try:
                rec, bands = recognize_scan(
                    Path(row["scan_path"].strip()), resources, tug_hint=tug_hint_of(row)
                )
            except Exception as exc:  # сбой одного скана не роняет прогон
                rec, bands = {"status": "error", "reason": type(exc).__name__, "timing": {}}, {}
            for band_name, img in bands.items():
                _write_png(bands_dir(name) / sid / f"{band_name}.png", img)
            fh.write(json.dumps({"scan_id": sid, "split": split, **rec}, ensure_ascii=False))
            fh.write("\n")
            fh.flush()
            stats[rec["status"]] += 1
            if log and (i + 1) % 10 == 0:
                log(f"{split}: {i + 1}/{len(todo)}")
    return stats


# ---------------------------------------------------------------------------
# decode: ядро + номер + пары
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class DecodeAssets:
    model: DecoderModel
    number_priors: NumberPriors
    pair_priors: PairPriors


def load_assets(decoder_json: Path, number_json: Path, pair_json: Path) -> DecodeAssets:
    return DecodeAssets(
        DecoderModel.load(decoder_json),
        NumberPriors.from_json(json.loads(number_json.read_text("utf-8"))),
        PairPriors.from_json(json.loads(pair_json.read_text("utf-8"))),
    )


def _inputs_of(rec: Mapping[str, Any]) -> dict[str, list[tuple[int, float]]]:
    return {n: [(int(v), float(p)) for v, p in d] for n, d in rec.get("inputs", {}).items()}


def _forms_json(forms: Mapping[str, FormRow] | None) -> dict[str, list[Any]] | None:
    if forms is None:
        return None
    return {r: [forms[r][0].isoformat(), forms[r][1], forms[r][2]] for r in ROWS}


def forms_from_json(data: Mapping[str, Sequence[Any]] | None) -> dict[str, FormRow] | None:
    if not data:
        return None
    return {r: (date.fromisoformat(v[0]), int(v[1]), int(v[2])) for r, v in data.items()}


def _top_fields(dist: Sequence[tuple[int, float]], k: int = FIELDS_K) -> list[list[float]]:
    ranked = sorted(dist, key=lambda vp: (-vp[1], vp[0]))[:k]
    return [[int(v), float(p)] for v, p in ranked]


def decode_split(
    split: str,
    assets: DecodeAssets,
    *,
    manifest: Path = paths.MANIFEST,
    name: str = NAME,
) -> dict[str, Any]:
    """Декодер по распознанным сканам сплита → JSONL T13 и подробности."""
    all_rows = read_manifest(manifest)
    rows = [r for r in all_rows if r.get("split") == split]
    scans = {r["scan_id"]: r for r in read_jsonl(scans_path(split, name))}
    history = history_from_manifest(all_rows)
    model = replace(assets.model, search=replace(assets.model.search, top_n=TOP_N))
    ok = {r["scan_id"]: r for r in rows if scans.get(r["scan_id"], {}).get("status") == "ok"}
    inputs = {sid: _inputs_of(scans[sid]) for sid in ok}
    contexts: dict[str, DecodeContext] = {sid: context_for(row, None) for sid, row in ok.items()}

    core: dict[str, DecodeResult] = {}
    t_dec: dict[str, float] = {}
    for sid in ok:
        t0 = time.perf_counter()
        core[sid] = decode(inputs[sid], contexts[sid], model)
        t_dec[sid] = time.perf_counter() - t0

    numbers: dict[str, NumberResult] = {}
    for sid, row in ok.items():
        rec = scans[sid]
        top = core[sid].top
        before = top.rows[CHAIN[0]].dt if top is not None else contexts[sid].app_dt
        t0 = time.perf_counter()
        ctx = number_context(row, history, before, use_filename=False)
        output = np.asarray(rec["number_output"], dtype=float)
        numbers[sid] = decode_number_output(
            output,
            ctx,
            assets.number_priors,
            kind=rec.get("number_kind", "ctc"),
            temperature=rec.get("number_temperature", 1.0),
        )
        t_dec[sid] += time.perf_counter() - t0

    final: dict[str, DecodeResult] = dict(core)
    partner: dict[str, str] = {}
    p_same: dict[str, float] = {}
    for a, b in manifest_pairs(rows):
        if a not in ok or b not in ok:
            continue
        t0 = time.perf_counter()
        pres = decode_pair(
            inputs[a],
            inputs[b],
            PairContext(a=contexts[a], b=contexts[b]),
            model,
            assets.pair_priors,
        )
        dt = (time.perf_counter() - t0) / 2
        assert pres.b is not None
        final[a], final[b] = pres.a, pres.b
        partner[a], partner[b] = b, a
        p_same[a] = p_same[b] = pres.p_same
        t_dec[a] += dt
        t_dec[b] += dt

    pred_lines: list[str] = []
    detail_lines: list[str] = []
    for row in rows:
        sid = row["scan_id"]
        rec = scans.get(sid, {"status": "missing", "timing": {}})
        detail: dict[str, Any] = {
            "scan_id": sid,
            "split": split,
            "status": rec.get("status"),
            "reason": rec.get("reason", ""),
            "variant": rec.get("variant"),
            "align": rec.get("align"),
            "sources": rec.get("sources", {}),
            "timing": {**rec.get("timing", {})},
        }
        item: dict[str, Any] = {"scan_id": sid, "source": name, "fields": {}}
        if sid in ok:
            res, cres, nres = final[sid], core[sid], numbers[sid]
            item["fields"] = {n: _top_fields(d) for n, d in inputs[sid].items() if d}
            if nres.candidates:
                item["fields"][VOUCHER_NUMBER] = nres.to_json()
            item.update(res.to_json())
            if not res.records:
                item.pop("confidence")
                item.pop("margin")
            detail["timing"]["decode"] = t_dec[sid]
            detail["timing"]["total"] = rec["timing"].get("total", 0.0) + t_dec[sid]
            top_forms = res.top.forms() if res.top else None
            # Вероятность итоговой записи у ядра (без партнёра); вне top-N ядра — 0.
            p_core_final = next(
                (r.p for r in cres.records if top_forms is not None and r.forms() == top_forms),
                0.0,
            )
            detail.update(
                {
                    "p_core": cres.confidence if cres.records else None,
                    "p_final": res.confidence if res.records else None,
                    "p_core_final": p_core_final if res.records else None,
                    "partner": partner.get(sid),
                    "p_same": p_same.get(sid),
                    "pair_changed": FLAG_PAIR_CHANGED in res.flags,
                    "flags": list(res.flags),
                    "missing": list(res.missing),
                    "overridden": list(res.overridden),
                    "forms": _forms_json(res.top.forms() if res.top else None),
                    "core_forms": _forms_json(cres.top.forms() if cres.top else None),
                    "number": nres.value,
                    "p_number": nres.confidence if nres.candidates else None,
                    "number_flags": list(nres.flags),
                    "marginals": {
                        r: [[o.value.form_date.isoformat(), o.value.hour, o.value.minute, o.p]
                            for o in opts]
                        for r, opts in res.marginals.items()
                    },
                }
            )
        pred_lines.append(json.dumps(item, ensure_ascii=False))
        detail_lines.append(json.dumps(detail, ensure_ascii=False))
    for path, lines in ((predictions_path(split, name), pred_lines),
                        (details_path(split, name), detail_lines)):
        paths.ensure_dir(path.parent)
        path.write_text("".join(line + "\n" for line in lines), encoding="utf-8")
    statuses = Counter(json.loads(line)["status"] for line in detail_lines)
    return {"n": len(rows), "statuses": dict(statuses), "pairs": len(partner) // 2}


# ---------------------------------------------------------------------------
# Оценка по ваучерам
# ---------------------------------------------------------------------------

#: Какая уверенность решает автоприём:
#: - `core` — ядро T19 без партнёра;
#: - `final` — итог (с парой k/p, если партнёр в сплите);
#: - `guarded` — min(итог, вероятность той же записи у ядра): запись должна быть уверенной и с
#:   партнёром, и без него (предложение T20: пара закрепляет ошибку картинки, совпавшую с
#:   партнёром, `2026_258k_2`).
CONF_KINDS: dict[str, str] = {
    "core": "ядро T19 (без пары)",
    "final": "итог (ядро + пара k/p)",
    "guarded": "min(итог, p той же записи у ядра)",
}


@dataclass
class VoucherOutcome:
    """Итог одного ваучера: верность по критериям T13, уверенности, деньги."""

    scan_id: str
    split: str
    status: str
    rows_ok: bool
    number_ok: bool
    wrong_rows: tuple[str, ...]
    conf: dict[str, float]
    p_number: float
    money: dict[str, bool] = field(default_factory=dict)


def form_text(form: FormRow) -> str:
    """`ДД.ММ.ГГГГ ЧЧ:ММ` — формат поправок `truth_corrections.csv` (T02)."""
    d, h, mi = form
    return f"{d.day:02d}.{d.month:02d}.{d.year} {h:02d}:{mi:02d}"


def money_changes(
    truth: Mapping[str, datetime | None], pred: Mapping[str, datetime | None]
) -> dict[str, bool]:
    """Что ошибка меняет в деньгах (метрики T13): минуты работ и занятости, ночь, дату курса."""
    ts, tf, tl, ta = (truth.get(r) for r in ("started_work", "finished_work", "left_base",
                                             "arrived_base"))
    ps, pf, pl, pa = (pred.get(r) for r in ("started_work", "finished_work", "left_base",
                                            "arrived_base"))
    out = {"work_minutes": False, "busy_minutes": False, "night": False, "fx_date": False}
    if ts and tf:
        out["work_minutes"] = not (ps and pf) or minutes_between(ps, pf) != minutes_between(ts, tf)
    if tl and ta:
        out["busy_minutes"] = not (pl and pa) or minutes_between(pl, pa) != minutes_between(tl, ta)
    if ts:
        out["night"] = ps is None or _is_night(ts) != _is_night(ps)
    if tf:
        out["fx_date"] = pf is None or tf.date() != pf.date()
    return out


def outcomes(
    split: str, *, name: str = NAME, manifest: Path = paths.MANIFEST
) -> tuple[list[VoucherOutcome], dict[str, Any]]:
    """Итоги ваучеров сплита и метрики `ocr_lab.evaluate` по JSONL прогона."""
    from ocr_lab.evaluate import evaluate, load_printed_flags, load_truth
    from ocr_lab.predictions import best_record, read_predictions

    truth = load_truth(manifest, split)
    preds = {p.scan_id: p for p in read_predictions(predictions_path(split, name))}
    details = {d["scan_id"]: d for d in read_jsonl(details_path(split, name))}
    metrics = evaluate(
        truth, list(preds.values()), printed_flags=load_printed_flags(paths.PRINTED_FLAGS_CSV)
    )
    out: list[VoucherOutcome] = []
    for sid, t in truth.items():
        d = details.get(sid, {})
        pred = preds.get(sid)
        rec = best_record(pred, t.row_years) if pred is not None else None
        prow = rec.rows if rec is not None else {}
        wrong = tuple(r for r in ROWS if t.rows.get(r) is not None and prow.get(r) != t.rows[r])
        number_ok = rec is not None and rec.voucher_number == t.voucher_number
        p_final = float(d.get("p_final") or 0.0)
        conf = {
            "core": float(d.get("p_core") or 0.0),
            "final": p_final,
            "guarded": min(p_final, float(d.get("p_core_final") or 0.0)),
        }
        out.append(
            VoucherOutcome(
                scan_id=sid,
                split=split,
                status=str(d.get("status") or "missing"),
                rows_ok=not wrong,
                number_ok=bool(number_ok),
                wrong_rows=wrong,
                conf=conf,
                p_number=float(d.get("p_number") or 0.0),
                money=money_changes(t.rows, prow) if wrong else {},
            )
        )
    return out, metrics


# ---------------------------------------------------------------------------
# Порог
# ---------------------------------------------------------------------------


def accept_stats(items: Sequence[VoucherOutcome], kind: str, threshold: float | None) -> dict:
    """Покрытие, точность и интервал Уилсона при пороге (критерий — 4 строки)."""
    from ocr_lab.evaluate import wilson_interval

    n = len(items)
    acc = [] if threshold is None else [v for v in items if v.conf[kind] >= threshold]
    k = sum(v.rows_ok for v in acc)
    low, high = wilson_interval(k, len(acc)) if acc else (None, None)
    return {
        "threshold": threshold,
        "n_total": n,
        "n_accepted": len(acc),
        "n_errors": len(acc) - k,
        "coverage": len(acc) / n if n else None,
        "precision": k / len(acc) if acc else None,
        "ci_low": low,
        "ci_high": high,
        "errors": sorted(v.scan_id for v in acc if not v.rows_ok),
    }


def val_threshold(items: Sequence[VoucherOutcome], kind: str, target: float) -> dict[str, Any]:
    """Порог на `val` по `ocr_lab.evaluate.threshold_for_precision` (точечная оценка/Уилсон)."""
    from dataclasses import asdict

    from ocr_lab.evaluate import threshold_for_precision

    thr = threshold_for_precision(
        [v.conf[kind] for v in items], [v.rows_ok for v in items], target, n_total=len(items)
    )
    return asdict(thr)


def conservative_threshold(target: float, t_val: float | None, safety: float) -> float:
    """Консервативный порог: модель сама оценивает ошибку ≤ (1 − цель) / `safety`.

    Запас `safety` раз — на самоуверенность модели, которую `val` из ~50 ваучеров не
    измеряет; порог не ниже подобранного на `val`.
    """
    base = 1.0 - (1.0 - target) / safety
    return max(base, t_val) if t_val is not None else base


#: Запас консервативного порога: модель может ошибаться в 10 раз чаще, чем думает.
CONSERVATIVE_SAFETY = 10.0
FIXED_THRESHOLDS: tuple[float, ...] = (
    0.5, 0.8, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999, 0.99999,
)


def choose_kind(val_items: Sequence[VoucherOutcome], target: float) -> tuple[str, dict]:
    """Вид уверенности с наибольшим покрытием на `val` при цели; при равенстве — `guarded`,
    затем `final` (меньше риск закрепить ошибку парой)."""
    order = ("guarded", "final", "core")
    per_kind = {k: val_threshold(val_items, k, target) for k in order}
    best = max(order, key=lambda k: (per_kind[k]["n_accepted"], -order.index(k)))
    return best, per_kind


def select_thresholds(
    val_items: Sequence[VoucherOutcome], target: float, fixed: float | None = None
) -> dict[str, Any]:
    """Выбор вида уверенности и порогов на `val`; `fixed` — порог извне (повтор в T22)."""
    kind, per_kind = choose_kind(val_items, target)
    t_val = per_kind[kind]["threshold"]
    return {
        "target": target,
        "kind": kind,
        "per_kind_val": per_kind,
        "t_val": fixed if fixed is not None else t_val,
        "t_conservative": conservative_threshold(target, t_val, CONSERVATIVE_SAFETY),
        "fixed_from_cli": fixed is not None,
    }


# ---------------------------------------------------------------------------
# Разбор ошибок по этапам
# ---------------------------------------------------------------------------

STAGES: dict[str, str] = {
    "alignment": "выравнивание",
    "crop": "кроп",
    "recognition": "распознавание",
    "decoder": "декодер выбрал не то",
    "truth_noise": "шум истины",
}
VERDICTS_CSV = paths.REVIEW_DIR / "e2e_error_verdicts.csv"
NOTES_MD = paths.REVIEW_DIR / "e2e_notes.md"


def wrong_subfields(
    detail: Mapping[str, Any], truth_row: Mapping[str, str]
) -> list[dict[str, Any]]:
    """Подполя, где итоговая запись расходится с истиной: ранг истины в картинке и бокс."""
    forms = forms_from_json(detail.get("forms"))
    if forms is None:
        return []
    scans_inputs = detail.get("_inputs", {})
    out: list[dict[str, Any]] = []
    for r in ROWS:
        d, h, mi = forms[r]
        pred = {"day": d.day, "month": d.month, "hour": h, "minute": mi}
        for part, value in pred.items():
            truth = _int(truth_row.get(f"{r}_{part}"))
            if truth is None or truth == value:
                continue
            name = subfield(r, part)
            dist = sorted(scans_inputs.get(name, []), key=lambda vp: -vp[1])
            ranks = [i for i, (v, _) in enumerate(dist) if int(v) == truth]
            out.append(
                {
                    "subfield": name,
                    "truth": truth,
                    "pred": value,
                    "truth_rank": ranks[0] + 1 if ranks else None,
                    "truth_p": float(dist[ranks[0]][1]) if ranks else 0.0,
                    "image_top1": int(dist[0][0]) if dist else None,
                    "source": detail.get("sources", {}).get(name),
                }
            )
    return out


def auto_stage(detail: Mapping[str, Any], wrong: Sequence[Mapping[str, Any]]) -> str:
    """Черновая классификация ошибки (уточняется просмотром кропов, `VERDICTS_CSV`)."""
    if detail.get("status") != "ok":
        return "alignment"
    if any(w["source"] in FALLBACK_SOURCES for w in wrong):
        return "crop"
    if wrong and all(w["truth_rank"] == 1 for w in wrong):
        return "decoder"
    return "recognition"


def read_verdicts(path: Path = VERDICTS_CSV) -> dict[tuple[str, str], dict[str, str]]:
    if not path.exists():
        return {}
    with path.open(encoding="utf-8-sig", newline="") as fh:
        return {(r["split"].strip(), r["scan_id"].strip()): r for r in csv.DictReader(fh)}


def error_analysis(
    split: str, items: Sequence[VoucherOutcome], *, name: str, manifest: Path
) -> list[dict[str, Any]]:
    """Все ошибочные (по 4 строкам) ваучеры сплита: этап (авто и после просмотра), деньги."""
    rows = {r["scan_id"]: r for r in read_manifest(manifest)}
    details = {d["scan_id"]: d for d in read_jsonl(details_path(split, name))}
    scans = {s["scan_id"]: s for s in read_jsonl(scans_path(split, name))}
    verdicts = read_verdicts()
    out: list[dict[str, Any]] = []
    for v in items:
        if v.rows_ok:
            continue
        detail = dict(details.get(v.scan_id, {"status": "missing"}))
        detail["_inputs"] = scans.get(v.scan_id, {}).get("inputs", {})
        wrong = wrong_subfields(detail, rows[v.scan_id])
        auto = auto_stage(detail, wrong)
        verdict = verdicts.get((split, v.scan_id), {})
        out.append(
            {
                "scan_id": v.scan_id,
                "conf": v.conf,
                "wrong_rows": list(v.wrong_rows),
                "wrong_subfields": wrong,
                "auto_stage": auto,
                "stage": (verdict.get("stage") or auto).strip(),
                "note": (verdict.get("note") or "").strip(),
                "money": v.money,
            }
        )
    return out


# ---------------------------------------------------------------------------
# Геометрия и время
# ---------------------------------------------------------------------------


def geometry_summary(split: str, *, name: str) -> dict[str, Any]:
    """Выравнивание, вариант против T07, запасные боксы, паритет с кропами T10/T19/T20."""
    from ocr_lab.predictions import read_predictions

    scans = read_jsonl(scans_path(split, name))
    assign = {}
    assign_csv = paths.WORK_DIR / "layout_assignment.csv"
    if assign_csv.exists():
        with assign_csv.open(encoding="utf-8-sig", newline="") as fh:
            assign = {r["scan_id"].strip(): r for r in csv.DictReader(fh)}
    statuses = Counter(s["status"] for s in scans)
    ok = [s for s in scans if s["status"] == "ok"]
    variant_mismatch = sorted(
        s["scan_id"] for s in ok
        if s["scan_id"] in assign and assign[s["scan_id"]].get("variant") != s.get("variant")
    )
    fallback = Counter(
        src for s in ok for src in s.get("sources", {}).values() if src in FALLBACK_SOURCES
    )
    fallback_scans = sorted(
        s["scan_id"] for s in ok
        if any(src in FALLBACK_SOURCES for src in s.get("sources", {}).values())
    )
    # Паритет с распределениями по кропам T10 (вход декодера T19) и с итогом T20.
    max_dp = None
    t19 = paths.MODELS_DIR / "decoder_v0" / f"{split}_inputs.jsonl"
    if t19.exists():
        ref = {r["scan_id"].strip(): r["fields"] for r in read_jsonl(t19)}
        diffs = []
        for s in ok:
            fields = ref.get(s["scan_id"])
            if fields is None:
                continue
            for n, dist in s.get("inputs", {}).items():
                a = {int(v): p for v, p in dist}
                b = {int(v): p for v, p in fields.get(n, [])}
                diffs.append(max(abs(a.get(k, 0.0) - b.get(k, 0.0)) for k in set(a) | set(b)))
        max_dp = max(diffs) if diffs else None
    same_record = n_compared = 0
    t20 = paths.MODELS_DIR / "decoder_v1" / f"{split}_predictions.jsonl"
    differ: list[str] = []
    if t20.exists() and predictions_path(split, name).exists():
        ref_p = {p.scan_id.strip(): p for p in read_predictions(t20)}
        for p in read_predictions(predictions_path(split, name)):
            q = ref_p.get(p.scan_id)
            if q is None or not p.records or not q.records:
                continue
            n_compared += 1
            if p.records[0].rows == q.records[0].rows:
                same_record += 1
            else:
                differ.append(p.scan_id)
    return {
        "n": len(scans),
        "statuses": dict(statuses),
        "variant_mismatch_t07": variant_mismatch,
        "fallback_boxes": dict(fallback),
        "fallback_scans": fallback_scans,
        "max_dp_vs_t19_inputs": max_dp,
        "t20_compared": n_compared,
        "t20_same_record": same_record,
        "t20_differ": differ,
    }


def timing_summary(splits: Sequence[str], *, name: str) -> dict[str, Any]:
    """Время по этапам на ваучер (CPU): медиана, p95, максимум, среднее."""
    values: dict[str, list[float]] = {}
    for split in splits:
        for d in read_jsonl(details_path(split, name)):
            if d.get("status") != "ok":
                continue
            for k, v in d.get("timing", {}).items():
                values.setdefault(k, []).append(1000.0 * float(v))
    out: dict[str, Any] = {"cpu_count": os.cpu_count(), "ort_threads": os.environ.get(
        "OCR_ORT_THREADS") or "по умолчанию"}
    for k, arr in values.items():
        a = np.asarray(arr)
        out[k] = {
            "n": int(a.size),
            "median": float(np.median(a)),
            "p95": float(np.percentile(a, 95)),
            "max": float(a.max()),
            "mean": float(a.mean()),
        }
    return out


# ---------------------------------------------------------------------------
# Отчёт
# ---------------------------------------------------------------------------

REPORT_MD = paths.REPORTS_DIR / "e2e_report.md"
METRICS_JSON = paths.REPORTS_DIR / "e2e_metrics.json"


def _pct(v: Any) -> str:
    return "—" if v is None else f"{100 * float(v):.2f} %"


def _thr(v: Any) -> str:
    return "—" if v is None else f"{float(v):.6g}"


def build_metrics(
    *, name: str, manifest: Path, target: float, fixed: float | None = None
) -> dict[str, Any]:
    """Всё для отчёта: уровни, порог (val → test), кривые, ошибки, деньги, геометрия, время."""
    from ocr_lab.evaluate import coverage_precision_curve, min_n_to_confirm

    res: dict[str, Any] = {"name": name, "target": target, "splits": {}}
    items: dict[str, list[VoucherOutcome]] = {}
    for split in ("val", "test"):
        items[split], metrics = outcomes(split, name=name, manifest=manifest)
        res["splits"][split] = {"evaluate": metrics, "geometry": geometry_summary(split, name=name)}
    sel = select_thresholds(items["val"], target, fixed)
    res["selection"] = sel
    kind = sel["kind"]
    res["min_n_to_confirm"] = min_n_to_confirm(target)
    for split, its in items.items():
        cell = res["splits"][split]
        cell["accept"] = {
            "t_val": accept_stats(its, kind, sel["t_val"]),
            "t_conservative": accept_stats(its, kind, sel["t_conservative"]),
            "fixed": {f"{t:g}": accept_stats(its, kind, t) for t in FIXED_THRESHOLDS},
            "by_kind_at_t_val": {
                k: accept_stats(its, k, sel["per_kind_val"][k]["threshold"]) for k in CONF_KINDS
            },
        }
        cell["curve"] = coverage_precision_curve(
            [v.conf[kind] for v in its], [v.rows_ok for v in its], len(its)
        )
        cell["errors"] = error_analysis(split, its, name=name, manifest=manifest)
        cell["n"] = len(its)
        cell["rows_ok"] = sum(v.rows_ok for v in its)
        cell["rows_number_ok"] = sum(v.rows_ok and v.number_ok for v in its)
        cell["number_ok"] = sum(v.number_ok for v in its)
        cell["calibration"] = {
            "mean_conf": float(np.mean([v.conf[kind] for v in its])) if its else None,
            "accuracy": cell["rows_ok"] / len(its) if its else None,
        }
    res["timing"] = timing_summary(("val", "test"), name=name)
    return res


def _levels_table(res: Mapping[str, Any]) -> list[str]:
    lines = [
        "| Уровень | val | test |",
        "|---|---:|---:|",
    ]
    ev = {s: res["splits"][s]["evaluate"] for s in ("val", "test")}

    def sub(s: str, dim: str, key: str, metric: str = "top1") -> Any:
        return ev[s]["subfields"].get(dim, {}).get(key, {}).get(metric)

    for key, title in (("day", "день"), ("month", "месяц"), ("hour", "час"),
                       ("minute", "минуты"), ("number", "номер ваучера")):
        lines.append(
            f"| Подполе: {title}, top-1 | {_pct(sub('val', 'part', key))} | "
            f"{_pct(sub('test', 'part', key))} |"
        )
    lines.append(
        f"| Подполя дат и номер, все, top-1 | {_pct(sub('val', 'overall', 'all'))} | "
        f"{_pct(sub('test', 'overall', 'all'))} |"
    )
    ts = {s: ev[s]["timestamps"].get("overall", {}).get("all", {}) for s in ev}
    lines.append(
        f"| Метка времени (строка целиком) | {_pct(ts['val'].get('accuracy'))} | "
        f"{_pct(ts['test'].get('accuracy'))} |"
    )
    for crit, title in (("rows", "Ваучер: 4 строки"), ("rows_number", "Ваучер: 4 строки + номер")):
        cells = [ev[s]["vouchers"].get(crit, {}).get("all", {}) for s in ("val", "test")]
        lines.append(
            f"| {title} | {_pct(cells[0].get('accuracy'))} (N = {cells[0].get('n')}) | "
            f"{_pct(cells[1].get('accuracy'))} (N = {cells[1].get('n')}) |"
        )
    for key, title in (("night_flag_changed", "Сменился ночной флаг начала"),
                       ("finish_date_changed", "Сменилась дата курса (окончание)")):
        lines.append(
            f"| {title} | {_pct(ev['val']['money'][key]['share'])} | "
            f"{_pct(ev['test']['money'][key]['share'])} |"
        )
    for key, title in (("abs_delta_work_minutes", "|Δ минут работ| > 0"),
                       ("abs_delta_busy_minutes", "|Δ минут занятости| > 0")):
        lines.append(
            f"| {title} | {_pct(ev['val']['money'][key]['share_nonzero'])} | "
            f"{_pct(ev['test']['money'][key]['share_nonzero'])} |"
        )
    return lines


def _accept_row(title: str, a: Mapping[str, Any]) -> str:
    return (
        f"| {title} | {_thr(a['threshold'])} | {a['n_accepted']} из {a['n_total']} | "
        f"{_pct(a['coverage'])} | {_pct(a['precision'])} | "
        f"[{_pct(a['ci_low'])}; {_pct(a['ci_high'])}] | {', '.join(a['errors']) or '—'} |"
    )


def render_report(res: Mapping[str, Any], notes: str = "") -> str:
    sel = res["selection"]
    kind = sel["kind"]
    val, test = res["splits"]["val"], res["splits"]["test"]
    lines = [
        f"# T21: сквозная оценка цепочки распознавания (`{res['name']}`)",
        "",
        "Цепочка: файл скана → `app.ocr.io` → `variants.detect_variant` (подсказка — буксир из "
        "имени файла) → `crops.locate_boxes` → `runtime` (onnxruntime, `digits_v0`, `number_v0`) "
        "→ декодер (ядро T19, номер T20 по истории без имени файла, пары k/p T20). Контекст "
        "(год, заявка, вид работ, история номеров, пары) — из манифеста, как в T19/T20.",
        "",
        f"- Ваучеров: val {val['n']}, test {test['n']}. Запись верна, если все 4 строки совпали "
        "с истиной манифеста (критерий T13; номер — отдельной строкой таблиц).",
        f"- Цель точности среди автопринятых — {_pct(res['target'])} (`DECISIONS.md`, Q4). "
        f"Подтвердить её нижней границей Уилсона можно, только если принято не меньше "
        f"{res['min_n_to_confirm']} ваучеров **без единой ошибки**.",
        "",
        "## Геометрия и паритет с кропами T10",
        "",
        "| | val | test |",
        "|---|---:|---:|",
    ]
    for key, title in (("statuses", "Статусы"), ):
        lines.append(f"| {title} | {val['geometry'][key]} | {test['geometry'][key]} |")
    g = {s: res["splits"][s]["geometry"] for s in ("val", "test")}
    lines += [
        f"| Вариант ≠ назначению T07 | {', '.join(g['val']['variant_mismatch_t07']) or '0'} | "
        f"{', '.join(g['test']['variant_mismatch_t07']) or '0'} |",
        f"| Боксы на запасном пути (column/row/static) | {g['val']['fallback_boxes'] or 0} | "
        f"{g['test']['fallback_boxes'] or 0} |",
        f"| Сканы с запасными боксами | {', '.join(g['val']['fallback_scans']) or '—'} | "
        f"{', '.join(g['test']['fallback_scans']) or '—'} |",
        f"| max \\|Δp\\| распределений против входов T19 (кропы T10) | "
        f"{_thr(g['val']['max_dp_vs_t19_inputs'])} | {_thr(g['test']['max_dp_vs_t19_inputs'])} |",
        f"| Лучшая запись совпала с T20 (`decoder_v1`) | {g['val']['t20_same_record']} из "
        f"{g['val']['t20_compared']} | {g['test']['t20_same_record']} из "
        f"{g['test']['t20_compared']} |",
        "",
        "## Время на ваучер (CPU)",
        "",
        f"Ядер CPU: {res['timing']['cpu_count']}, потоков onnxruntime: "
        f"{res['timing']['ort_threads']}. Один процесс, ваучеры по очереди; первый ваучер "
        "включает создание сессий onnxruntime.",
        "",
        "| Этап | медиана, мс | p95, мс | макс., мс | среднее, мс |",
        "|---|---:|---:|---:|---:|",
    ]
    stage_titles = {
        "load": "чтение PDF и рендер 200 dpi", "align": "вариант + выравнивание",
        "crops": "подгонка боксов и кропы", "digits": "модель цифр (16 кропов)",
        "number": "модель номера", "decode": "декодер (ядро + номер + пара)",
        "total": "всего",
    }
    for key, title in stage_titles.items():
        t = res["timing"].get(key)
        if t:
            lines.append(
                f"| {title} | {t['median']:.0f} | {t['p95']:.0f} | {t['max']:.0f} | "
                f"{t['mean']:.0f} |"
            )
    lines += ["", "## Точность по уровням", "", *_levels_table(res), ""]
    lines += [
        "## Порог автоприёма",
        "",
        "### Подбор на val",
        "",
        "| Вид уверенности | Порог (точечная оценка ≥ цели) | Принято | Покрытие | Точность | "
        "Нижняя граница Уилсона |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for k, title in CONF_KINDS.items():
        t = sel["per_kind_val"][k]
        lines.append(
            f"| `{k}` — {title} | {_thr(t['threshold'])} | {t['n_accepted']} | "
            f"{_pct(t['coverage'])} | {_pct(t['precision'])} | {_pct(t['ci_low'])} |"
        )
    lines += [
        "",
        f"Выбран вид `{kind}` (наибольшее покрытие на val; при равенстве — `guarded`, затем "
        f"`final`). Порог val: **{_thr(sel['t_val'])}**"
        + (" (задан извне, `--threshold`)" if sel["fixed_from_cli"] else "")
        + f"; консервативный порог: **{_thr(sel['t_conservative'])}** = max(порог val, "
        f"1 − (1 − цель) / {CONSERVATIVE_SAFETY:g}).",
        "",
        "### Применение (критерий — 4 строки)",
        "",
        "| Сплит и порог | Порог | Принято | Покрытие | Точность | 95 % ДИ Уилсона | Ошибки среди "
        "принятых |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for split in ("val", "test"):
        acc = res["splits"][split]["accept"]
        lines.append(_accept_row(f"{split}: порог val", acc["t_val"]))
        lines.append(_accept_row(f"{split}: консервативный", acc["t_conservative"]))
    lines += [
        "",
        "### Кривая «покрытие — точность» (вид `" + kind + "`, фиксированные пороги)",
        "",
        "| Порог | val: принято | val: покрытие | val: точность | test: принято | "
        "test: покрытие | test: точность | test: ошибки |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for t in FIXED_THRESHOLDS:
        a, b = val["accept"]["fixed"][f"{t:g}"], test["accept"]["fixed"][f"{t:g}"]
        lines.append(
            f"| {t:g} | {a['n_accepted']} | {_pct(a['coverage'])} | {_pct(a['precision'])} | "
            f"{b['n_accepted']} | {_pct(b['coverage'])} | {_pct(b['precision'])} | "
            f"{', '.join(b['errors']) or '—'} |"
        )
    lines += [
        "",
        "Полные кривые (каждое уникальное значение уверенности) — в `e2e_metrics.json`, "
        "ключ `splits.<split>.curve`.",
        "",
        "### Другие виды уверенности на test при их пороге val",
        "",
        "| Вид | Порог | Принято | Покрытие | Точность | 95 % ДИ Уилсона | Ошибки |",
        "|---|---:|---:|---:|---:|---|---|",
    ]
    for k in CONF_KINDS:
        lines.append(_accept_row(f"`{k}`", test["accept"]["by_kind_at_t_val"][k]))
    lines += [
        "",
        f"Калибровка записи (вид `{kind}`): средняя уверенность / точность — val "
        f"{_pct(val['calibration']['mean_conf'])} / {_pct(val['calibration']['accuracy'])}, "
        f"test {_pct(test['calibration']['mean_conf'])} / {_pct(test['calibration']['accuracy'])}.",
        "",
        "## Ошибки по этапам",
        "",
    ]
    for split in ("val", "test"):
        errs = res["splits"][split]["errors"]
        n = res["splits"][split]["n"]
        counts = Counter(e["stage"] for e in errs)
        lines += [
            f"### {split}: ошибочных ваучеров {len(errs)} из {n}",
            "",
            "| Этап | Ваучеров | Доля ошибок |",
            "|---|---:|---:|",
        ]
        for key, title in STAGES.items():
            c = counts.get(key, 0)
            lines.append(f"| {title} | {c} | {_pct(c / len(errs)) if errs else '—'} |")
        money = Counter(k for e in errs for k, v in e["money"].items() if v)
        lines += [
            "",
            f"Меняют деньги (из {len(errs)}): минуты работ — {money.get('work_minutes', 0)}, "
            f"минуты занятости — {money.get('busy_minutes', 0)}, ночной флаг — "
            f"{money.get('night', 0)}, дата курса — {money.get('fx_date', 0)}; хоть что-то из "
            f"этого — {sum(any(e['money'].values()) for e in errs)}.",
            "",
            "| scan_id | Уверенность (core / final / guarded) | Строки | Подполя: истина → "
            "предсказание (ранг истины в картинке, бокс) | Этап (авто → итог) | Заметка |",
            "|---|---|---|---|---|---|",
        ]
        for e in errs:
            cf = e["conf"]
            subs = "; ".join(
                f"{w['subfield']}: {w['truth']} → {w['pred']} "
                f"(ранг {w['truth_rank'] or '>60'}, {w['source']})"
                for w in e["wrong_subfields"]
            )
            lines.append(
                f"| {e['scan_id']} | {cf['core']:.4f} / {cf['final']:.4f} / "
                f"{cf['guarded']:.4f} | "
                f"{', '.join(e['wrong_rows'])} | {subs or '—'} | "
                f"{STAGES.get(e['auto_stage'], e['auto_stage'])} → "
                f"{STAGES.get(e['stage'], e['stage'])} | {e['note'] or '—'} |"
            )
        lines.append("")
    if notes:
        lines += [notes.strip(), ""]
    return "\n".join(lines)


def _json_default(obj: Any) -> Any:
    if isinstance(obj, datetime | date):
        return obj.isoformat()
    raise TypeError(type(obj).__name__)


# ---------------------------------------------------------------------------
# «Уверенные расхождения»
# ---------------------------------------------------------------------------

REVIEW_OUT = paths.REVIEW_DIR / "confident_disagreements"
REVIEW_VERDICTS = REVIEW_OUT / "verdicts.csv"


def disagreements(
    items_by_split: Mapping[str, Sequence[VoucherOutcome]],
    kind: str,
    threshold: float,
    *,
    limit: int = MAX_REVIEW,
) -> list[VoucherOutcome]:
    """Ваучеры с уверенностью ≥ порога, где истина говорит другое (строки или номер)."""
    picked = [
        v
        for items in items_by_split.values()
        for v in items
        if (not v.rows_ok and v.conf[kind] >= threshold)
        or (v.rows_ok and not v.number_ok and v.p_number >= threshold)
    ]
    picked.sort(key=lambda v: (-v.conf[kind], v.split, v.scan_id))
    return picked[:limit]


def _truth_form(row: Mapping[str, str], r: str) -> FormRow | None:
    parts = [_int(row.get(f"{r}_{p}")) for p in ("year", "month", "day", "hour", "minute")]
    y, m, d, h, mi = parts
    if y is None:
        y = _int(row.get("year"))
    if None in (y, m, d, h, mi):
        return None
    assert y is not None and m is not None and d is not None and h is not None and mi is not None
    return (date(y, m, d), h, mi)


def _short(form: FormRow | None) -> str:
    if form is None:
        return "—"
    d, h, mi = form
    return f"{d.day:02d}.{d.month:02d} {h:02d}:{mi:02d}"


def review_image(
    v: VoucherOutcome, detail: Mapping[str, Any], row: Mapping[str, str], kind: str, name: str
) -> Any:
    """Склейка «номер + 4 строки дат» с истиной, предсказанием и уверенностью (PIL)."""
    from ocr_lab.sheets import make_strip

    forms = forms_from_json(detail.get("forms")) or {}
    strips = []
    number_truth = _int(row.get("voucher_number"))
    mark = "" if v.number_ok else "  ≠"
    title = f"Номер: истина {number_truth} · модель {detail.get('number')}{mark}"
    band_dir = bands_dir(name) / v.scan_id
    items = [(VOUCHER_NUMBER, title)] + [
        (
            r,
            f"{ROW_TITLES[r]}: истина {_short(_truth_form(row, r))} · модель "
            f"{_short(forms.get(r))}{'  ≠' if r in v.wrong_rows else ''}",
        )
        for r in ROWS
    ]
    bands = {
        b: _read_png(band_dir / f"{b}.png") for b, _ in items if (band_dir / f"{b}.png").exists()
    }
    width = max((img.shape[1] for img in bands.values()), default=0)
    for band, caption in items:
        if band not in bands:
            continue
        img = bands[band]
        # Узкий кроп (номер) не растягиваем, а дополняем белым до общей ширины.
        padded = cv2.copyMakeBorder(
            img, 0, 0, 0, width - img.shape[1], cv2.BORDER_CONSTANT, value=(255, 255, 255)
        )
        strips.append(make_strip([padded], caption, font_size=24))
    p_number = detail.get("p_number")
    header = (
        f"{v.scan_id} ({v.split}) — уверенность {kind} {v.conf[kind]:.6f}; ядро "
        f"{v.conf['core']:.4f}, итог {v.conf['final']:.4f}"
        + (f", пара с {detail['partner']}" if detail.get("partner") else "")
        + (f"; номер p = {p_number:.4f}" if p_number is not None else "")
    )
    return make_strip(strips, header, font_size=26, gap=10)


def write_review(
    *, name: str, manifest: Path, target: float, threshold: float | None, kind: str | None,
    splits: Sequence[str] = ("val", "test"), limit: int = MAX_REVIEW, out: Path = REVIEW_OUT,
) -> list[dict[str, Any]]:
    """Картинки «уверенных расхождений» и `index.csv` (индекс → scan_id).

    Без `threshold`/`kind` берутся вид уверенности и меньший из порогов (val и
    консервативного), подобранные на `val`. `threshold=0` — все ошибочные ваучеры (разбор).
    """
    items = {s: outcomes(s, name=name, manifest=manifest)[0] for s in splits}
    if threshold is None or kind is None:
        val_items = items.get("val") or outcomes("val", name=name, manifest=manifest)[0]
        sel = select_thresholds(val_items, target)
        kind = kind or sel["kind"]
        if threshold is None:
            threshold = min(t for t in (sel["t_val"], sel["t_conservative"]) if t is not None)
    assert threshold is not None and kind is not None
    rows = {r["scan_id"]: r for r in read_manifest(manifest)}
    picked = disagreements(items, kind, threshold, limit=limit)
    details = {s: {d["scan_id"]: d for d in read_jsonl(details_path(s, name))} for s in splits}
    paths.ensure_dir(out)
    index: list[dict[str, Any]] = []
    for i, v in enumerate(picked):
        img = review_image(v, details[v.split][v.scan_id], rows[v.scan_id], kind, name)
        img.thumbnail((MAX_SIDE, MAX_SIDE))  # длинная сторона ≤ 2000 px (`_common.md`, § 3)
        img.save(out / f"{v.scan_id}.png")
        index.append(
            {
                "index": i,
                "scan_id": v.scan_id,
                "split": v.split,
                "kind": kind,
                "confidence": f"{v.conf[kind]:.6f}",
                "wrong_rows": " ".join(v.wrong_rows),
                "number_wrong": not v.number_ok,
                "image": f"{v.scan_id}.png",
            }
        )
    with (out / "index.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(
            fh, fieldnames=["index", "scan_id", "split", "kind", "confidence", "wrong_rows",
                            "number_wrong", "image"]
        )
        writer.writeheader()
        writer.writerows(index)
    return index


# ---------------------------------------------------------------------------
# Поправки истины по вердиктам просмотра
# ---------------------------------------------------------------------------


def correction_rows(
    verdicts: Sequence[Mapping[str, str]],
    details: Mapping[str, Mapping[str, Any]],
    manifest_rows: Mapping[str, Mapping[str, str]],
) -> list[dict[str, str]]:
    """Строки `truth_corrections.csv` для вердиктов `model`: расходящиеся строки бланка.

    `rows` вердикта (через пробел или запятую) ограничивает строки и `voucher_number`; пусто —
    все расходящиеся. Поправку номера (`field=voucher_number`) манифест T02 пока не применяет —
    по `DECISIONS.md` это делает T22.
    """
    out: list[dict[str, str]] = []
    for v in verdicts:
        if (v.get("verdict") or "").strip() != "model":
            continue
        sid = v["scan_id"].strip()
        detail = details[sid]
        forms = forms_from_json(detail.get("forms"))
        row = manifest_rows[sid]
        if forms is None:
            continue
        only = {s for s in (v.get("rows") or "").replace(",", " ").split() if s}
        note = (v.get("note") or "").strip()
        values: list[tuple[str, str, str]] = []
        for r in ROWS:
            truth = _truth_form(row, r)
            if truth is not None and truth != forms[r]:
                values.append((r, form_text(truth), form_text(forms[r])))
        number, truth_number = detail.get("number"), _int(row.get("voucher_number"))
        if number is not None and truth_number is not None and int(number) != truth_number:
            values.append((VOUCHER_NUMBER, str(truth_number), str(int(number))))
        for fld, export_value, corrected in values:
            if only and fld not in only:
                continue
            out.append(
                {
                    "scan_id": sid,
                    "field": fld,
                    "export_value": export_value,
                    "corrected_value": corrected,
                    "status": "proposed",
                    "source": "T21",
                    "note": note,
                }
            )
    return out


def append_corrections(path: Path, rows: Sequence[Mapping[str, str]]) -> int:
    """Дописать поправки в конец файла, не переписывая существующие строки.

    Пропускаются строки, у которых уже есть та же `(scan_id, field, corrected_value)`.
    """
    from ocr_lab.manifest import CORRECTIONS_COLUMNS

    existing: set[tuple[str, str, str]] = set()
    if path.exists():
        with path.open(encoding="utf-8-sig", newline="") as fh:
            existing = {
                (r["scan_id"].strip(), r["field"].strip(), r["corrected_value"].strip())
                for r in csv.DictReader(fh)
            }
    new = [
        r for r in rows if (r["scan_id"], r["field"], r["corrected_value"]) not in existing
    ]
    if not new:
        return 0
    write_header = not path.exists()
    if path.exists():
        data = path.read_bytes()
        if data and not data.endswith(b"\n"):
            with path.open("ab") as fh:
                fh.write(b"\r\n")
    with path.open("a", encoding="utf-8-sig" if write_header else "utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(CORRECTIONS_COLUMNS))
        if write_header:
            writer.writeheader()
        writer.writerows(new)
    return len(new)


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def cmd_run(args: argparse.Namespace) -> int:
    resources = load_resources(
        args.layouts_dir, args.models_dir, digits=args.digits, number=args.number
    )
    for split in args.split:
        stats = run_split(
            split,
            resources,
            manifest=args.manifest,
            name=args.name,
            limit=args.limit,
            max_minutes=args.max_minutes,
            force=args.force,
            log=print,
        )
        print(f"{split}: {dict(stats)} → {scans_path(split, args.name)}")
    return 0


def cmd_decode(args: argparse.Namespace) -> int:
    assets = load_assets(args.decoder_priors, args.number_priors, args.pair_priors)
    for split in args.split:
        res = decode_split(split, assets, manifest=args.manifest, name=args.name)
        print(f"{split}: {res} → {predictions_path(split, args.name)}")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    res = build_metrics(
        name=args.name, manifest=args.manifest, target=args.target, fixed=args.threshold
    )
    notes = args.notes.read_text(encoding="utf-8") if args.notes.exists() else ""
    out = args.out
    paths.ensure_dir(out.parent)
    out.write_text(render_report(res, notes), encoding="utf-8")
    out.with_name(METRICS_JSON.name if out == REPORT_MD else f"{out.stem}.json").write_text(
        json.dumps(res, ensure_ascii=False, indent=1, default=_json_default), encoding="utf-8"
    )
    sel = res["selection"]
    for split in ("val", "test"):
        cell = res["splits"][split]
        a, c = cell["accept"]["t_val"], cell["accept"]["t_conservative"]
        print(
            f"{split}: 4 строки {cell['rows_ok']}/{cell['n']}; порог val {_thr(sel['t_val'])} "
            f"({sel['kind']}): принято {a['n_accepted']}, ошибок {a['n_errors']}; "
            f"консервативный {_thr(sel['t_conservative'])}: принято {c['n_accepted']}, "
            f"ошибок {c['n_errors']}"
        )
    print(f"отчёт: {out}")
    return 0


def cmd_review(args: argparse.Namespace) -> int:
    index = write_review(
        name=args.name,
        manifest=args.manifest,
        target=args.target,
        threshold=args.threshold,
        kind=args.kind,
        splits=args.split,
        limit=args.limit,
        out=args.out,
    )
    print(f"расхождений: {len(index)} → {args.out}")
    for item in index:
        print(f"  {item['scan_id']} ({item['split']}): {item['confidence']} {item['wrong_rows']}")
    return 0


def cmd_propose(args: argparse.Namespace) -> int:
    if not args.verdicts.exists():
        print(f"нет файла вердиктов: {args.verdicts}", file=sys.stderr)
        return 2
    with args.verdicts.open(encoding="utf-8-sig", newline="") as fh:
        verdicts = list(csv.DictReader(fh))
    details: dict[str, Mapping[str, Any]] = {}
    for split in SPLITS:
        details.update({d["scan_id"]: d for d in read_jsonl(details_path(split, args.name))})
    manifest_rows = {r["scan_id"]: r for r in read_manifest(args.manifest)}
    rows = correction_rows(verdicts, details, manifest_rows)
    n = append_corrections(args.corrections, rows)
    print(f"предложено поправок: {len(rows)}, дописано новых: {n} → {args.corrections}")
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocr_lab.eval_pipeline", description=__doc__.splitlines()[0]
    )
    parser.add_argument("--name", default=NAME, help="имя прогона: каталог e2e/<name>, source")
    parser.add_argument("--manifest", type=Path, default=paths.MANIFEST)
    sub = parser.add_subparsers(dest="cmd", required=True)

    def split_list(value: str) -> list[str]:
        items = [s.strip() for s in value.split(",") if s.strip()]
        bad = set(items) - set(SPLITS)
        if bad:
            raise argparse.ArgumentTypeError(f"неизвестный сплит: {', '.join(sorted(bad))}")
        return items

    p_run = sub.add_parser("run", help="скан → распределения подполей (app/ocr/*)")
    p_run.add_argument("--split", type=split_list, default=["val", "test"])
    p_run.add_argument("--layouts-dir", type=Path, default=paths.LAYOUTS_DIR)
    p_run.add_argument("--models-dir", type=Path, default=paths.MODELS_DIR)
    p_run.add_argument("--digits", default="digits_v0")
    p_run.add_argument("--number", default="number_v0")
    p_run.add_argument("--limit", type=int, default=None)
    p_run.add_argument("--max-minutes", type=float, default=None)
    p_run.add_argument("--force", action="store_true", help="начать сплит заново")
    p_run.set_defaults(func=cmd_run)

    p_dec = sub.add_parser("decode", help="ядро + номер + пары → JSONL T13")
    p_dec.add_argument("--split", type=split_list, default=["val", "test"])
    p_dec.add_argument(
        "--decoder-priors", type=Path, default=paths.MODELS_DIR / "decoder_priors_v0.json"
    )
    p_dec.add_argument(
        "--number-priors", type=Path, default=paths.MODELS_DIR / "number_priors_v0.json"
    )
    p_dec.add_argument("--pair-priors", type=Path, default=paths.MODELS_DIR / "pair_priors_v0.json")
    p_dec.set_defaults(func=cmd_decode)

    p_rep = sub.add_parser("report", help="уровни, порог val → test, ошибки по этапам")
    p_rep.add_argument("--target", type=float, default=DEFAULT_TARGET)
    p_rep.add_argument("--threshold", type=float, default=None, help="порог извне (T22)")
    p_rep.add_argument("--notes", type=Path, default=NOTES_MD)
    p_rep.add_argument("--out", type=Path, default=REPORT_MD)
    p_rep.set_defaults(func=cmd_report)

    p_rev = sub.add_parser("review", help="картинки «уверенных расхождений» для H3")
    p_rev.add_argument("--split", type=split_list, default=["val", "test"])
    p_rev.add_argument("--target", type=float, default=DEFAULT_TARGET)
    p_rev.add_argument("--threshold", type=float, default=None)
    p_rev.add_argument("--kind", choices=sorted(CONF_KINDS), default=None)
    p_rev.add_argument("--limit", type=int, default=MAX_REVIEW)
    p_rev.add_argument("--out", type=Path, default=REVIEW_OUT)
    p_rev.set_defaults(func=cmd_review)

    p_pro = sub.add_parser("propose", help="вердикты просмотра → поправки status=proposed")
    p_pro.add_argument("--verdicts", type=Path, default=REVIEW_VERDICTS)
    p_pro.add_argument("--corrections", type=Path, default=paths.CORRECTIONS_CSV)
    p_pro.set_defaults(func=cmd_propose)

    args = parser.parse_args(argv)
    return int(args.func(args))


if __name__ == "__main__":
    raise SystemExit(main())
