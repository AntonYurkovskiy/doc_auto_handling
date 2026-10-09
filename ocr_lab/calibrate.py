"""T18: калибровка вероятностей температурой (temperature scaling) на `val`.

Декодер T19 складывает логарифмы вероятностей моделей с приорами. Если модель самоуверенна,
приоры не сработают, поэтому логиты делятся на температуру `T`: `softmax(logits / T)`.
`T > 1` смягчает, `T < 1` заостряет распределение; `argmax` и ранжирование не меняются.

Температура подбирается **на `val`** по NLL одним скаляром на голову (LBFGS по `log T`):

- `digits` — голова десятков (`tens`: для значений < 10 истина не знает «0» это или «пусто»,
  поэтому NLL маргинальный, как в обучении) и голова единиц (`units`);
- `number`, подход `ctc` — одна температура на все кадры (NLL — CTC-правдоподобие истины,
  это ровно та вероятность строки, которую выдаёт `score_candidates`); подход `heads` —
  по температуре на голову (сотни, десятки, единицы).

`test` для подбора не используется: его метрики до и после считаются только для отчёта и
ни на что не влияют. Результат — `calibration.json` в каталоге модели (его читает
`ocr_lab.export_onnx` и кладёт температуры в `meta.json`) и диаграмма надёжности
`<reports>/calibration_<digits|number>.png`.

Запуск: `python -m ocr_lab.calibrate digits|number [--model-dir <каталог>]`.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F

from app.ocr.digit_values import TENS_EMPTY, value_log_probs
from app.ocr.number_scores import HEADS, encode_heads, number_log_probs
from ocr_lab import paths
from ocr_lab.dataset import PART_ID, TARGET_SIZE, build_samples
from ocr_lab.number_models import ctc_targets
from ocr_lab.train_digits import (
    V0_DIR as DIGITS_V0_DIR,
)
from ocr_lab.train_digits import (
    eval_tensors,
    infer_logits,
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

CALIBRATION_JSON = "calibration.json"
#: Корзины ECE: равной ширины по уверенности top-1.
ECE_BINS = 15
#: Допустимый диапазон температуры (защита от вырожденных решений на малой выборке).
T_MIN, T_MAX = 0.25, 8.0


# --- метрики ------------------------------------------------------------------------------


def reliability_bins(
    conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS
) -> list[dict[str, float]]:
    """Корзины диаграммы надёжности: число, средняя уверенность и точность (пустые пропущены)."""
    conf = np.asarray(conf, dtype=np.float64)
    correct = np.asarray(correct, dtype=np.float64)
    idx = np.minimum((conf * n_bins).astype(int), n_bins - 1)
    out = []
    for b in range(n_bins):
        mask = idx == b
        if mask.any():
            out.append({
                "lo": b / n_bins, "hi": (b + 1) / n_bins, "n": int(mask.sum()),
                "conf": float(conf[mask].mean()), "acc": float(correct[mask].mean()),
            })
    return out


def ece(conf: np.ndarray, correct: np.ndarray, n_bins: int = ECE_BINS) -> float:
    """Expected Calibration Error: взвешенное среднее |точность − уверенность| по корзинам."""
    n = len(conf)
    if n == 0:
        return 0.0
    return float(sum(
        b["n"] / n * abs(b["acc"] - b["conf"]) for b in reliability_bins(conf, correct, n_bins)
    ))


def _metrics(conf: np.ndarray, correct: np.ndarray, nll: Sequence[float]) -> dict[str, Any]:
    return {
        "n": int(len(conf)),
        "acc": float(np.mean(correct)) if len(conf) else 0.0,
        "mean_conf": float(np.mean(conf)) if len(conf) else 0.0,
        "ece": ece(conf, correct),
        "nll": float(np.mean(nll)) if len(nll) else 0.0,
        "bins": reliability_bins(conf, correct),
    }


# --- подбор температуры -------------------------------------------------------------------


def fit_temperature(
    nll_fn: Callable[[torch.Tensor], torch.Tensor], *, t_min: float = T_MIN, t_max: float = T_MAX
) -> float:
    """Температура, минимизирующая `nll_fn(T)`: LBFGS по одному скаляру `log T`.

    `nll_fn` получает положительный скаляр-тензор. Параметр ограничен `[t_min, t_max]`
    (клип `log T`), старт — `T = 1`.
    """
    log_t = torch.zeros(1, dtype=torch.float64, requires_grad=True)
    lo, hi = math.log(t_min), math.log(t_max)
    opt = torch.optim.LBFGS([log_t], lr=0.5, max_iter=200, line_search_fn="strong_wolfe")

    def closure() -> torch.Tensor:
        opt.zero_grad()
        loss = nll_fn(torch.exp(log_t.clamp(lo, hi))[0])
        loss.backward()
        return loss

    opt.step(closure)
    return float(torch.exp(log_t.detach().clamp(lo, hi))[0])


def fit_guarded(
    name: str, nll_fn: Callable[[torch.Tensor], torch.Tensor], notes: dict[str, str]
) -> float:
    """Температура с защитой от вырожденной выборки.

    Если оптимум упёрся в границу `[T_MIN, T_MAX]` (на `val` нет ошибок, NLL монотонно падает
    при `T → 0`, либо наоборот), температура по этой выборке не определена: возвращается `1.0`,
    причина — в `notes[name]` (попадает в `calibration.json`).
    """
    t = fit_temperature(nll_fn)
    if t <= T_MIN * 1.001 or t >= T_MAX / 1.001:
        notes[name] = (
            f"подбор упёрся в границу ({t:.3f}): на val температура не определена "
            "(нет ошибок или их слишком мало), оставлена T=1"
        )
        return 1.0
    return t


# --- цифры --------------------------------------------------------------------------------


def digit_targets(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Значения подполей → `(tens, units, ambiguous)` как в обучении (`ambiguous = v < 10`)."""
    v = np.asarray(values, dtype=np.int64)
    return v // 10, v % 10, v < 10


def tens_nll(logits: torch.Tensor, tens: torch.Tensor, ambiguous: torch.Tensor,
             temperature: torch.Tensor) -> torch.Tensor:
    """Средний NLL головы десятков при температуре (маргинал «0 или пусто» для `v < 10`)."""
    lt = F.log_softmax(logits / temperature, dim=1)
    exact = -lt.gather(1, tens.long().view(-1, 1)).squeeze(1)
    marginal = -torch.logsumexp(lt[:, [0, TENS_EMPTY]], dim=1)
    return torch.where(ambiguous.bool(), marginal, exact).mean()


def units_nll(logits: torch.Tensor, units: torch.Tensor, temperature: torch.Tensor
              ) -> torch.Tensor:
    """Средний NLL головы единиц при температуре."""
    lu = F.log_softmax(logits / temperature, dim=1)
    return -lu.gather(1, units.long().view(-1, 1)).squeeze(1).mean()


def fit_digit_temperatures(
    tens_logits: np.ndarray, units_logits: np.ndarray, values: np.ndarray,
    notes: dict[str, str] | None = None,
) -> dict[str, float]:
    """Температуры голов `tens` и `units` по логитам и истине (подбирается на `val`)."""
    tens, units, ambiguous = (torch.from_numpy(a) for a in digit_targets(values))
    lt = torch.from_numpy(np.asarray(tens_logits, dtype=np.float64))
    lu = torch.from_numpy(np.asarray(units_logits, dtype=np.float64))
    notes = notes if notes is not None else {}
    return {
        "tens": fit_guarded("tens", lambda t: tens_nll(lt, tens, ambiguous, t), notes),
        "units": fit_guarded("units", lambda t: units_nll(lu, units, t), notes),
    }


def digit_metrics(
    tens_logits: np.ndarray, units_logits: np.ndarray, parts: Sequence[str],
    values: Sequence[int], temperatures: dict[str, float] | None = None,
) -> dict[str, Any]:
    """Метрики значения подполя (то, что видит декодер): ECE и NLL — общие и по частям."""
    temperatures = temperatures or {}
    t_tens, t_units = temperatures.get("tens", 1.0), temperatures.get("units", 1.0)
    conf, correct, nll = [], [], []
    for i, (part, value) in enumerate(zip(parts, values, strict=True)):
        vals, logp = value_log_probs(
            tens_logits[i], units_logits[i], part, t_tens=t_tens, t_units=t_units
        )
        top = int(np.argmax(logp))
        conf.append(float(np.exp(logp[top])))
        correct.append(float(vals[top] == value))
        hit = np.nonzero(vals == value)[0]
        nll.append(-float(logp[hit[0]]) if hit.size else -math.log(1e-6))
    conf_a, correct_a = np.array(conf), np.array(correct)
    out = _metrics(conf_a, correct_a, nll)
    by_part = {}
    for part in PART_ID:
        mask = np.array([p == part for p in parts])
        if mask.any():
            sub = _metrics(conf_a[mask], correct_a[mask], np.array(nll)[mask])
            sub.pop("bins")
            by_part[part] = sub
    out["by_part"] = by_part
    return out


def _digit_logits_for(split: str, model: Any, size: tuple[int, int]
                      ) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Логиты `digits_v0` на кропах сплита, как при отборе модели T16 (`build_samples`)."""
    samples = build_samples(split=split)
    images, part_ids = eval_tensors(samples, size)
    tens, units = infer_logits(model, images, part_ids, torch.device("cpu"))
    parts = [s.subfield.split(".")[1] for s in samples]
    return tens, units, parts, np.array([s.value for s in samples])


def calibrate_digits(model_dir: Path = DIGITS_V0_DIR, *, test: bool = True) -> dict[str, Any]:
    """Подобрать температуры на `val`, замерить до/после на `val` (и, справочно, на `test`)."""
    model, cfg, _ = load_digits(model_dir)
    size = tuple(cfg.target_size) or TARGET_SIZE
    splits = ("val", "test") if test else ("val",)
    data = {s: _digit_logits_for(s, model, size) for s in splits}  # type: ignore[arg-type]
    tens_v, units_v, parts_v, values_v = data["val"]
    notes: dict[str, str] = {}
    temps = fit_digit_temperatures(tens_v, units_v, values_v, notes)
    result: dict[str, Any] = {
        "model": model_dir.name, "kind": "digits", "fitted_on": "val",
        "temperatures": temps, "n_fit": len(values_v), "notes": notes, "metrics": {},
    }
    for split, (tens, units, parts, values) in data.items():
        result["metrics"][split] = {
            "before": digit_metrics(tens, units, parts, values.tolist()),
            "after": digit_metrics(tens, units, parts, values.tolist(), temps),
        }
    return result


# --- номер --------------------------------------------------------------------------------


def number_nll_ctc(logits: torch.Tensor, values: Sequence[int], temperature: torch.Tensor
                   ) -> torch.Tensor:
    """Средний NLL истинной строки по CTC при температуре (`logits`: `(N, T, 11)`)."""
    log_probs = F.log_softmax(logits / temperature, dim=2).transpose(0, 1)
    flat, lengths = ctc_targets(values)
    n_frames = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
    nll = F.ctc_loss(log_probs, flat, n_frames, lengths, blank=0, reduction="none",
                     zero_infinity=True)
    return nll.mean()


def number_nll_heads(logits: torch.Tensor, values: Sequence[int], head: int,
                     temperature: torch.Tensor) -> torch.Tensor:
    """Средний NLL одной головы (`heads`: `(N, 3, 11)`) при температуре."""
    targets = torch.tensor([encode_heads(int(v))[head] for v in values], dtype=torch.long)
    lp = F.log_softmax(logits[:, head, :] / temperature, dim=1)
    return -lp.gather(1, targets.view(-1, 1)).squeeze(1).mean()


def fit_number_temperatures(outputs: np.ndarray, values: Sequence[int], kind: str,
                            notes: dict[str, str] | None = None) -> dict[str, float]:
    """Температуры модели номера: `{"ctc": T}` либо `{"hundreds"|"tens"|"units": T}`."""
    logits = torch.from_numpy(np.asarray(outputs, dtype=np.float64))
    notes = notes if notes is not None else {}
    if kind == "ctc":
        return {"ctc": fit_guarded("ctc", lambda t: number_nll_ctc(logits, values, t), notes)}
    if kind == "heads":
        temps: dict[str, float] = {}
        for h, name in enumerate(HEADS):
            def head_nll(t: torch.Tensor, h: int = h) -> torch.Tensor:
                return number_nll_heads(logits, values, h, t)

            temps[name] = fit_guarded(name, head_nll, notes)
        return temps
    raise ValueError(f"неизвестный вид модели номера: {kind}")


def temperature_arg(temps: dict[str, float], kind: str) -> float | np.ndarray:
    """Словарь температур → аргумент `temperature` функций `app.ocr.number_scores`."""
    if kind == "ctc":
        return float(temps.get("ctc", 1.0))
    return np.array([temps.get(h, 1.0) for h in HEADS], dtype=np.float64)


def number_metrics(outputs: np.ndarray, values: Sequence[int], kind: str,
                   temps: dict[str, float] | None = None) -> dict[str, Any]:
    """ECE и NLL распределения номеров `1..999` (то, что отдаётся как top-k формата T13)."""
    temperature = temperature_arg(temps or {}, kind)
    conf, correct, nll = [], [], []
    for out, value in zip(outputs, values, strict=True):
        nums, logp = number_log_probs(out, kind=kind, temperature=temperature)
        top = int(np.argmax(logp))
        conf.append(float(np.exp(logp[top])))
        correct.append(float(nums[top] == value))
        hit = np.nonzero(nums == value)[0]
        nll.append(-float(logp[hit[0]]) if hit.size else -math.log(1e-9))
    return _metrics(np.array(conf), np.array(correct), nll)


def calibrate_number(model_dir: Path = NUMBER_V0_DIR, *, test: bool = True) -> dict[str, Any]:
    """Подобрать температуру номера на `val`, замерить до/после на `val` и справочно на `test`."""
    model, cfg, _ = load_number(model_dir)
    kind = cfg.model.kind
    device = torch.device("cpu")
    data = {}
    for split in ("val", "test") if test else ("val",):
        crops = number_crops(split, with_truth=True)
        outputs = infer_outputs(model, crop_tensors(crops, cfg.target_size), device)
        data[split] = (outputs, [int(c.value) for c in crops if c.value is not None])
    outputs_v, values_v = data["val"]
    notes: dict[str, str] = {}
    temps = fit_number_temperatures(outputs_v, values_v, kind, notes)
    result: dict[str, Any] = {
        "model": model_dir.name, "kind": "number", "number_kind": kind, "fitted_on": "val",
        "temperatures": temps, "n_fit": len(values_v), "notes": notes, "metrics": {},
    }
    for split, (outputs, values) in data.items():
        result["metrics"][split] = {
            "before": number_metrics(outputs, values, kind),
            "after": number_metrics(outputs, values, kind, temps),
        }
    return result


# --- вывод --------------------------------------------------------------------------------


def plot_reliability(result: dict[str, Any], out: Path) -> Path:
    """Диаграммы надёжности: строки — `val` и `test`, столбцы — до и после калибровки."""
    import matplotlib  # noqa: PLC0415 — только для отчёта

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt  # noqa: PLC0415

    splits = list(result["metrics"])
    fig, axes = plt.subplots(len(splits), 2, figsize=(9, 4.2 * len(splits)), squeeze=False)
    for r, split in enumerate(splits):
        for c, stage in enumerate(("before", "after")):
            ax = axes[r][c]
            m = result["metrics"][split][stage]
            bins = m["bins"]
            ax.plot([0, 1], [0, 1], "--", color="gray", lw=1)
            ax.bar([(b["lo"] + b["hi"]) / 2 for b in bins], [b["acc"] for b in bins],
                   width=1 / ECE_BINS * 0.9, color="#4c72b0", alpha=0.85)
            ax.plot([b["conf"] for b in bins], [b["acc"] for b in bins], "o", color="#c44e52",
                    ms=3)
            for b in bins:
                ax.annotate(str(b["n"]), ((b["lo"] + b["hi"]) / 2, 0.02), ha="center",
                            fontsize=6, color="white" if b["acc"] > 0.08 else "black",
                            rotation=90)
            title = "до" if stage == "before" else "после"
            ax.set_title(f"{split}, {title} калибровки: ECE {m['ece']:.4f}, NLL {m['nll']:.3f}, "
                         f"n={m['n']}", fontsize=9)
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.set_xlabel("уверенность top-1")
            ax.set_ylabel("доля верных")
    temps = ", ".join(f"{k}={v:.3f}" for k, v in result["temperatures"].items())
    fig.suptitle(f"{result['model']}: диаграмма надёжности (T: {temps}; подбор — val)",
                 fontsize=10)
    fig.tight_layout()
    paths.ensure_dir(out.parent)
    fig.savefig(out, dpi=110)
    plt.close(fig)
    return out


def summary_lines(result: dict[str, Any]) -> list[str]:
    """Короткая сводка для консоли и журнала."""
    lines = [f"{result['model']}: температуры " + ", ".join(
        f"{k}={v:.4f}" for k, v in result["temperatures"].items()
    ) + f" (подбор на val, n={result['n_fit']})"]
    lines += [f"  ! {name}: {text}" for name, text in result.get("notes", {}).items()]
    for split, stages in result["metrics"].items():
        b, a = stages["before"], stages["after"]
        lines.append(
            f"  {split}: ECE {b['ece']:.4f} → {a['ece']:.4f}, NLL {b['nll']:.4f} → "
            f"{a['nll']:.4f}, acc {b['acc']:.4f} (n={b['n']})"
        )
    return lines


def run(kind: str, model_dir: Path | None = None, *, test: bool = True,
        reports_dir: Path | None = None) -> dict[str, Any]:
    """Откалибровать модель, записать `calibration.json` и PNG; вернуть результат."""
    if kind == "digits":
        result = calibrate_digits(model_dir or DIGITS_V0_DIR, test=test)
    elif kind == "number":
        result = calibrate_number(model_dir or NUMBER_V0_DIR, test=test)
    else:
        raise ValueError(f"неизвестная модель: {kind}")
    target = model_dir or (DIGITS_V0_DIR if kind == "digits" else NUMBER_V0_DIR)
    (target / CALIBRATION_JSON).write_text(
        json.dumps(result, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    png = plot_reliability(
        result, (reports_dir or paths.REPORTS_DIR) / f"calibration_{kind}.png"
    )
    print("\n".join(summary_lines(result)))
    print(f"диаграмма: {png}")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ocr_lab.calibrate", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("kind", choices=("digits", "number"))
    parser.add_argument("--model-dir", type=Path, default=None)
    parser.add_argument("--no-test", action="store_true",
                        help="не считать справочные метрики на test")
    args = parser.parse_args(argv)
    run(args.kind, args.model_dir, test=not args.no_test)
    return 0


if __name__ == "__main__":
    sys.exit(main())
