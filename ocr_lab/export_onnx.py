"""T18: экспорт моделей `digits` и `number` в ONNX для рантайма на CPU (`app/ocr/runtime.py`).

Подкоманды (`python -m ocr_lab.export_onnx <команда> …`):

- `export digits|number` — `model.onnx` и `meta.json` рядом с `best.pt` (`models/<имя>_v0/`):
  opset 17, динамическая ось батча, затем проверка паритета с torch на кропах `val`;
- `predict digits|number --split val|test` — откалиброванные предсказания через onnxruntime
  (`<split>_predictions.jsonl`, для номера ещё и `<split>_outputs.npz` с сырыми логитами);
- `bench` — время одного ваучера (16 кропов цифр и номер) на CPU.

**Температуры лежат в `meta.json`, а не в графе.** Граф отдаёт сырые логиты, поэтому паритет с
torch проверяется напрямую, а декодер T19/T20 может перекалибровать температуры (или заново
подобрать веса) без повторного экспорта. Рантайм делит логиты на температуру сам
(`app.ocr.runtime`). Нормализация входа `(x − mean) / std` остаётся внутри графа (так её
считает и torch-модель), `meta.json` описывает её для справки (`normalization.in_graph`).

Адаптивный пулинг `(1, 4)` на ширину карты признаков 5 (канва 64×160) legacy-экспортёр ONNX не
умеет (ширина не кратна 4), поэтому в экспортируемой копии он заменён точным аналогом на срезах
(:class:`StaticAdaptivePool`): окна те же, что у PyTorch (`floor(i·W/n) … ceil((i+1)·W/n)`).
"""

from __future__ import annotations

import argparse
import copy
import datetime as dt
import hashlib
import json
import math
import statistics
import subprocess
import sys
import time
from collections.abc import Sequence
from pathlib import Path
from typing import Any

import numpy as np
import onnx
import torch
from torch import nn

from app.ocr import runtime as rt
from app.ocr.digit_values import PART_RANGES, TOP_K
from app.ocr.number_scores import HEADS, top_numbers
from ocr_lab import paths
from ocr_lab.calibrate import CALIBRATION_JSON
from ocr_lab.cut_crops import read_gray
from ocr_lab.dataset import PART_ID, SUBFIELDS, build_samples
from ocr_lab.models import GRAY_MEAN, GRAY_STD, DigitNet
from ocr_lab.number_models import NumberNet
from ocr_lab.predictions import VOUCHER_NUMBER, ScanPrediction, write_predictions
from ocr_lab.train_digits import (
    V0_DIR as DIGITS_V0_DIR,
)
from ocr_lab.train_digits import (
    eval_tensors,
    infer_logits,
    split_crops,
)
from ocr_lab.train_digits import (
    load_model as load_digits,
)
from ocr_lab.train_number import (
    V0_DIR as NUMBER_V0_DIR,
)
from ocr_lab.train_number import (
    crop_tensors,
    infer_outputs,
    number_crops,
)
from ocr_lab.train_number import (
    load_model as load_number,
)

OPSET = 17
ONNX_NAME = "model.onnx"
#: Допуск паритета (по вероятностям) и размер выборки из задачи T18.
PARITY_TOL = 1e-4
PARITY_N = 200
#: Целевое время одного ваучера на CPU, мс.
BENCH_TARGET_MS = 150.0

MODEL_DIRS = {"digits": DIGITS_V0_DIR, "number": NUMBER_V0_DIR}


class StaticAdaptivePool(nn.Module):
    """`AdaptiveAvgPool2d((1, out_w))` для известной ширины входа — на срезах (для ONNX).

    Окно `i` — столбцы `floor(i·W/out_w) … ceil((i+1)·W/out_w)` (как в PyTorch), среднее по ним
    и по высоте. Эквивалентность проверяется тестом и паритетом экспорта.
    """

    def __init__(self, in_w: int, out_w: int) -> None:
        super().__init__()
        self.windows = [
            ((i * in_w) // out_w, -((-(i + 1) * in_w) // out_w)) for i in range(out_w)
        ]

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        cols = [x[:, :, :, a:b].mean(dim=(2, 3), keepdim=True) for a, b in self.windows]
        return torch.cat(cols, dim=3)


def _feature_width(features: nn.Module, size: tuple[int, int]) -> int:
    with torch.no_grad():
        return int(features(torch.zeros(1, 1, *size)).shape[-1])


def exportable(model: nn.Module, size: tuple[int, int]) -> nn.Module:
    """Копия модели в режиме `eval`, пригодная для экспорта (пулинг заменён на срезы)."""
    clone = copy.deepcopy(model).eval()
    pool = getattr(clone, "pool", None)
    if isinstance(pool, nn.AdaptiveAvgPool2d):
        out = pool.output_size
        assert isinstance(out, tuple) and out[0] == 1 and out[1], "ожидался пулинг (1, n)"
        width = _feature_width(clone.features, size)  # type: ignore[arg-type]
        clone.pool = StaticAdaptivePool(width, int(out[1]))  # type: ignore[assignment]
    return clone


def export_model(model: nn.Module, kind: str, size: tuple[int, int], out: Path) -> None:
    """Записать ONNX: `digits` — входы `image`, `part_id`; `number` — вход `image`."""
    net = exportable(model, size)
    image = torch.zeros(2, 1, *size)
    if kind == "digits":
        args: tuple[torch.Tensor, ...] = (image, torch.tensor([0, 3], dtype=torch.long))
        names = ["image", "part_id"]
        outputs = ["tens_logits", "units_logits"]
    else:
        args = (image,)
        names = ["image"]
        outputs = ["logits"]
    dynamic = {n: {0: "batch"} for n in names + outputs}
    paths.ensure_dir(out.parent)
    with torch.no_grad():
        torch.onnx.export(
            net, args, str(out), opset_version=OPSET, input_names=names, output_names=outputs,
            dynamic_axes=dynamic, do_constant_folding=True, dynamo=False,
        )
    onnx.checker.check_model(onnx.load(str(out)))


# --- meta.json ----------------------------------------------------------------------------


def _git(*args: str) -> str:
    try:
        out = subprocess.run(["git", *args], cwd=paths.REPO_ROOT, capture_output=True, text=True,
                             check=False)
        return out.stdout.strip()
    except OSError:
        return ""


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _config_json(model_dir: Path) -> dict[str, Any]:
    path = model_dir / "config.json"
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def _calibration(model_dir: Path) -> dict[str, Any]:
    path = model_dir / CALIBRATION_JSON
    if not path.exists():
        raise FileNotFoundError(
            f"нет {CALIBRATION_JSON} в {model_dir.name}: сначала `python -m ocr_lab.calibrate`"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def _brief(calibration: dict[str, Any]) -> dict[str, Any]:
    """Метрики калибровки без корзин диаграммы — для `meta.json`."""
    out: dict[str, Any] = {}
    for split, stages in calibration["metrics"].items():
        out[split] = {
            stage: {k: v for k, v in m.items() if k not in ("bins", "by_part")}
            for stage, m in stages.items()
        }
    return out


def build_meta(
    kind: str, model_dir: Path, *, size: tuple[int, int], trained_at: str,
    parity: dict[str, Any] | None, number_kind: str | None = None,
) -> dict[str, Any]:
    """Контракт рантайма (`meta.json`) для `digits` или `number`."""
    cfg = _config_json(model_dir)
    calibration = _calibration(model_dir)
    meta: dict[str, Any] = {
        "format_version": rt.FORMAT_VERSION,
        "name": model_dir.name,
        "task": kind,
        "onnx": ONNX_NAME,
        "opset": OPSET,
        "input": {
            "name": "image", "dtype": "float32", "layout": "NCHW", "channels": 1,
            "size": list(size), "range": [0.0, 1.0], "background": 255,
            "fit": "keep_aspect_center_pad",
            "interpolation_down": "INTER_AREA", "interpolation_up": "INTER_LINEAR",
            "note": "серый uint8 → канва size на белом фоне → /255; батч — динамический",
        },
        "normalization": {"mean": GRAY_MEAN, "std": GRAY_STD, "in_graph": True},
        "temperatures": calibration["temperatures"],
        "calibration": {
            "fitted_on": calibration["fitted_on"], "n_fit": calibration["n_fit"],
            "metrics": _brief(calibration),
        },
        "train": {
            "run": cfg.get("run"), "git_commit": cfg.get("git_commit"),
            "manifest_sha256": cfg.get("manifest_sha256"),
            "crops_index_sha256": cfg.get("crops_index_sha256"),
            "checkpoint_sha256": file_sha256(model_dir / "best.pt"),
            "trained_at": trained_at, "torch": cfg.get("torch"),
        },
        "export": {
            "git_commit": _git("rev-parse", "--short", "HEAD"),
            "exported_at": dt.datetime.now(dt.UTC).strftime("%Y-%m-%d"),
            "torch": torch.__version__, "onnx": onnx.__version__,
            "onnx_sha256": file_sha256(model_dir / ONNX_NAME),
        },
        "parity": parity,
    }
    if kind == "digits":
        meta["part_input"] = {"name": "part_id", "dtype": "int64", "shape": ["batch"]}
        meta["part_ids"] = dict(PART_ID)
        meta["value_ranges"] = {p: list(r) for p, r in PART_RANGES.items()}
        meta["outputs"] = [
            {"name": "tens_logits", "classes": 11,
             "labels": [str(d) for d in range(10)] + ["empty"], "head": "tens"},
            {"name": "units_logits", "classes": 10,
             "labels": [str(d) for d in range(10)], "head": "units"},
        ]
        meta["subfields"] = list(SUBFIELDS)
        meta["decoding"] = (
            "P(v) ∝ P(tens(v))·P(units(v)); для v<10 десятки = «0» или «empty»; нормировка по "
            "value_ranges; логиты голов делятся на temperatures.tens / temperatures.units"
        )
    else:
        assert number_kind is not None
        meta["kind"] = number_kind
        meta["heads"] = list(HEADS)
        if number_kind == "ctc":
            meta["outputs"] = [{"name": "logits", "shape": ["batch", "T", 11],
                                "classes": 11, "blank": 0, "label_of_digit": "digit + 1"}]
            meta["decoding"] = (
                "CTC: log P(номер) — forward-алгоритм по кадрам; логиты делятся на "
                "temperatures.ctc; ведущие нули отбрасываются; номера 1..999"
            )
        else:
            meta["outputs"] = [{"name": "logits", "shape": ["batch", 3, 11], "classes": 11,
                                "heads": list(HEADS), "empty": 10}]
            meta["decoding"] = (
                "три головы по 11 классов (10 = пусто), выравнивание по правому краю; логиты "
                "каждой головы делятся на её температуру"
            )
    return meta


def trained_at_of(model_dir: Path, override: str | None) -> str:
    """Дата обучения: `--trained-at`, иначе `saved_at` из чекпоинта, иначе `unknown`."""
    if override:
        return override
    state = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=False)
    for key in ("saved_at", "trained_at", "timestamp"):
        if state.get(key):
            return str(state[key])
    return "unknown"


# --- паритет ------------------------------------------------------------------------------


def _evenly(n_total: int, n: int) -> list[int]:
    """`n` равномерно распределённых индексов из `n_total` (детерминированно)."""
    if n_total <= n:
        return list(range(n_total))
    return sorted({int(round(i * (n_total - 1) / (n - 1))) for i in range(n)})


def softmax_np(x: np.ndarray, axis: int = -1) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64)
    z = z - z.max(axis=axis, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=axis, keepdims=True)


def parity_digits(model: DigitNet, onnx_model: rt.OnnxModel, n: int = PARITY_N) -> dict[str, Any]:
    """torch против onnxruntime на `n` кропах `val`: разница вероятностей и совпадение top-1.

    Torch берёт кроп через `ocr_lab.dataset.fit_with_aspect`, onnxruntime — через numpy-
    предобработку рантайма (`rt.OnnxModel.prepare`): сравнивается весь путь.
    """
    samples = build_samples(split="val")
    pick = [samples[i] for i in _evenly(len(samples), n)]
    size = onnx_model.meta.size
    images, part_ids = eval_tensors(pick, size)
    t_tens, t_units = infer_logits(model, images, part_ids, torch.device("cpu"))
    parts = [s.subfield.split(".")[1] for s in pick]
    o_tens, o_units = rt.digit_logits(onnx_model, [read_gray(s.path) for s in pick], parts)
    temps = onnx_model.meta.temperatures
    t_top = rt.digit_top_values(t_tens, t_units, parts, temps, TOP_K)
    o_top = rt.digit_top_values(o_tens, o_units, parts, temps, TOP_K)
    p_diff = max(
        float(np.abs(softmax_np(t_tens) - softmax_np(o_tens)).max()),
        float(np.abs(softmax_np(t_units) - softmax_np(o_units)).max()),
    )
    v_diff = max(
        abs(a[1] - b[1]) for ta, tb in zip(t_top, o_top, strict=True)
        for a, b in zip(ta, tb, strict=True) if a[0] == b[0]
    )
    same = sum(int(a[0][0] == b[0][0]) for a, b in zip(t_top, o_top, strict=True))
    return {
        "n": len(pick), "max_logit_diff": float(max(np.abs(t_tens - o_tens).max(),
                                                    np.abs(t_units - o_units).max())),
        "max_head_prob_diff": p_diff, "max_value_prob_diff": float(v_diff),
        "top1_match": same / len(pick), "tolerance": PARITY_TOL,
        "ok": bool(max(p_diff, v_diff) < PARITY_TOL and same == len(pick)),
    }


def parity_number(model: NumberNet, onnx_model: rt.OnnxModel, kind: str,
                  n: int = PARITY_N) -> dict[str, Any]:
    """То же для номера на кропах `val` (их всего ~50, берутся все, не больше `n`)."""
    crops = number_crops("val", with_truth=True)
    pick = [crops[i] for i in _evenly(len(crops), n)]
    size = onnx_model.meta.size
    t_out = infer_outputs(model, crop_tensors(pick, size), torch.device("cpu"))
    o_out = rt.number_logits(onnx_model, [read_gray(c.path) for c in pick])
    temperature = rt.number_temperature(onnx_model)
    t_top = [top_numbers(o, kind=kind, temperature=temperature) for o in t_out]
    o_top = [top_numbers(o, kind=kind, temperature=temperature) for o in o_out]
    axis = -1
    p_diff = float(np.abs(softmax_np(t_out, axis) - softmax_np(o_out, axis)).max())
    v_diff = max(
        abs(a[1] - b[1]) for ta, tb in zip(t_top, o_top, strict=True)
        for a, b in zip(ta, tb, strict=True) if a[0] == b[0]
    )
    same = sum(int(a[0][0] == b[0][0]) for a, b in zip(t_top, o_top, strict=True))
    return {
        "n": len(pick), "max_logit_diff": float(np.abs(t_out - o_out).max()),
        "max_frame_prob_diff": p_diff, "max_value_prob_diff": float(v_diff),
        "top1_match": same / len(pick), "tolerance": PARITY_TOL,
        "ok": bool(max(p_diff, v_diff) < PARITY_TOL and same == len(pick)),
    }


# --- export -------------------------------------------------------------------------------


def export(kind: str, model_dir: Path | None = None, *, trained_at: str | None = None) -> Path:
    """Экспортировать модель, записать `meta.json`, проверить паритет. Вернуть путь к ONNX."""
    model_dir = model_dir or MODEL_DIRS[kind]
    model: nn.Module
    number_kind: str | None = None
    if kind == "digits":
        model, digits_cfg, _ = load_digits(model_dir)
        target = digits_cfg.target_size
    else:
        model, number_cfg, _ = load_number(model_dir)
        target = number_cfg.target_size
        number_kind = number_cfg.model.kind
    size = (int(target[0]), int(target[1]))
    out = model_dir / ONNX_NAME
    export_model(model, kind, size, out)
    when = trained_at_of(model_dir, trained_at)
    # meta.json нужен рантайму для паритета (сначала без паритета, затем с ним)
    meta = build_meta(kind, model_dir, size=size, trained_at=when, parity=None,
                      number_kind=number_kind)
    write_meta(model_dir, meta)
    onnx_model = rt.OnnxModel(model_dir)
    if kind == "digits":
        assert isinstance(model, DigitNet)
        parity = parity_digits(model, onnx_model)
    else:
        assert number_kind is not None
        parity = parity_number(model, onnx_model, number_kind)  # type: ignore[arg-type]
    meta["parity"] = parity
    write_meta(model_dir, meta)
    print(f"{model_dir.name}: {out.name} {out.stat().st_size / 1e6:.1f} МБ, паритет {parity}")
    if not parity["ok"]:
        print("ВНИМАНИЕ: паритет вне допуска", file=sys.stderr)
    return out


def write_meta(model_dir: Path, meta: dict[str, Any]) -> None:
    (model_dir / rt.META_NAME).write_text(
        json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )


# --- откалиброванные предсказания через onnxruntime -------------------------------------


def predict_digits_split(split: str, model_dir: Path | None = None, *,
                         batch: int = 256) -> Path:
    """`<split>_predictions.jsonl` модели цифр: onnxruntime + температуры из `meta.json`."""
    model_dir = model_dir or DIGITS_V0_DIR
    model = rt.OnnxModel(model_dir)
    crops = split_crops(split)
    parts = [c.subfield.split(".")[1] for c in crops]
    tens_all, units_all = [], []
    for s in range(0, len(crops), batch):
        chunk = crops[s : s + batch]
        t, u = rt.digit_logits(model, [read_gray(c.path) for c in chunk], parts[s : s + batch])
        tens_all.append(t)
        units_all.append(u)
    tens, units = np.concatenate(tens_all), np.concatenate(units_all)
    tops = rt.digit_top_values(tens, units, parts, model.meta.temperatures, TOP_K)
    by_scan: dict[str, ScanPrediction] = {}
    for crop, cands in zip(crops, tops, strict=True):
        pred = by_scan.setdefault(crop.scan_id, ScanPrediction(scan_id=crop.scan_id,
                                                               source=model_dir.name))
        pred.fields[crop.subfield] = list(cands)
    out = model_dir / f"{split}_predictions.jsonl"
    n = write_predictions(out, [by_scan[k] for k in sorted(by_scan)])
    print(f"{model_dir.name} {split}: {len(crops)} кропов, {n} сканов → {out}")
    return out


def predict_number_split(split: str, model_dir: Path | None = None, *,
                         batch: int = 64) -> Path:
    """`<split>_predictions.jsonl` и `<split>_outputs.npz` модели номера через onnxruntime."""
    model_dir = model_dir or NUMBER_V0_DIR
    model = rt.OnnxModel(model_dir)
    kind = rt.number_kind(model)
    temperature = rt.number_temperature(model)
    crops = number_crops(split, with_truth=False)
    outs = [rt.number_logits(model, [read_gray(c.path) for c in crops[s : s + batch]])
            for s in range(0, len(crops), batch)]
    outputs = np.concatenate(outs) if outs else np.zeros((0, 0, 0), dtype=np.float32)
    preds = [
        ScanPrediction(scan_id=c.scan_id, source=model_dir.name, fields={
            VOUCHER_NUMBER: [(v, p) for v, p in top_numbers(o, kind=kind,
                                                            temperature=temperature)]
        })
        for c, o in zip(crops, outputs, strict=True)
    ]
    out = model_dir / f"{split}_predictions.jsonl"
    n = write_predictions(out, preds)
    np.savez_compressed(
        model_dir / f"{split}_outputs.npz", scan_id=np.array([c.scan_id for c in crops]),
        output=outputs, kind=np.array(kind),
        temperature=np.atleast_1d(np.asarray(temperature, dtype=np.float64)),
    )
    print(f"{model_dir.name} {split}: {n} сканов → {out}")
    return out


# --- скорость -----------------------------------------------------------------------------


def bench(reps: int = 50, warmup: int = 5) -> dict[str, Any]:
    """Время одного ваучера на CPU: 16 кропов цифр (батч) и номер, медиана и p95 в мс.

    Берётся первый скан `val`, у которого есть все 16 кропов цифр и кроп номера; замеряется
    всё, что делает рантайм: предобработка, onnxruntime и распределение значений/номеров.
    """
    digits = rt.OnnxModel(DIGITS_V0_DIR)
    number = rt.OnnxModel(NUMBER_V0_DIR)
    crops = split_crops("val")
    by_scan: dict[str, dict[str, np.ndarray]] = {}
    for c in crops:
        by_scan.setdefault(c.scan_id, {})[c.subfield] = read_gray(c.path)
    num_crops = {c.scan_id: read_gray(c.path) for c in number_crops("val", with_truth=False)}
    scan = next(s for s, d in sorted(by_scan.items())
                if len(d) == len(SUBFIELDS) and s in num_crops)
    digit_crops, number_crop = by_scan[scan], num_crops[scan]

    def one() -> tuple[float, float]:
        t0 = time.perf_counter()
        rt.predict_digits(digit_crops, model=digits)
        t1 = time.perf_counter()
        rt.predict_number(number_crop, model=number)
        t2 = time.perf_counter()
        return (t1 - t0) * 1000, (t2 - t1) * 1000

    for _ in range(warmup):
        one()
    runs = [one() for _ in range(reps)]
    total = sorted(a + b for a, b in runs)
    result: dict[str, Any] = {
        "scan": scan, "reps": reps, "n_digit_crops": len(digit_crops),
        "digits_ms_median": statistics.median(a for a, _ in runs),
        "number_ms_median": statistics.median(b for _, b in runs),
        "total_ms_median": statistics.median(total),
        "total_ms_p95": total[min(len(total) - 1, math.ceil(0.95 * len(total)) - 1)],
        "target_ms": BENCH_TARGET_MS,
    }
    result["ok"] = bool(result["total_ms_median"] < BENCH_TARGET_MS)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ocr_lab.export_onnx", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)
    p_exp = sub.add_parser("export", help="ONNX + meta.json + паритет")
    p_exp.add_argument("kind", choices=("digits", "number"))
    p_exp.add_argument("--model-dir", type=Path, default=None)
    p_exp.add_argument("--trained-at", default=None, help="дата обучения (ГГГГ-ММ-ДД)")
    p_pred = sub.add_parser("predict", help="откалиброванные предсказания через onnxruntime")
    p_pred.add_argument("kind", choices=("digits", "number"))
    p_pred.add_argument("--split", choices=("val", "test"), required=True)
    p_pred.add_argument("--model-dir", type=Path, default=None)
    p_bench = sub.add_parser("bench", help="время одного ваучера на CPU")
    p_bench.add_argument("--reps", type=int, default=50)
    args = parser.parse_args(argv)
    if args.cmd == "export":
        export(args.kind, args.model_dir, trained_at=args.trained_at)
    elif args.cmd == "predict":
        if args.kind == "digits":
            predict_digits_split(args.split, args.model_dir)
        else:
            predict_number_split(args.split, args.model_dir)
    else:
        bench(args.reps)
    return 0


if __name__ == "__main__":
    sys.exit(main())
