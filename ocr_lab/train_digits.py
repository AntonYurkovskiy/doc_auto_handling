"""T16: обучение CNN для двузначных подполей (день, месяц, часы, минуты всех строк).

CLI (`.venv-train/Scripts/python -m ocr_lab.train_digits <команда>`):

- `train --run NAME [--backbone resnet18|mobilenet_v3_small] [--no-part-embed]
  [--mnist N] [--epochs 40] [--batch-size 128] [--max-minutes 9] [--resume]` — обучение в
  `data/ocr/models/digits_runs/NAME/`: `best.pt` (лучшая эпоха по средней точности подполей
  на `val`), `last.pt` (состояние для `--resume`, пишется каждую эпоху), `config.json`,
  `history.csv`, `train.log`. По `--max-minutes` обучение останавливается между эпохами и
  продолжается с `--resume`; ранняя остановка — по той же метрике на `val`;
- `predict --run-dir DIR --split val|test [--source NAME]` — предсказания в формате T13
  (`DIR/<split>_predictions.jsonl`, top-5 значений на подполе);
- `errors --run-dir DIR [--n 60]` — лист худших ошибок на `val`
  (`data/ocr/sheets/digits_errors_val.png` + CSV «индекс → scan_id»);
- `promote --run NAME` — скопировать `best.pt`, `config.json`, `history.csv` выбранного
  запуска в `data/ocr/models/digits_v0/`.

`test` в обучении и выборе модели не участвует: `train` читает только `train` и `val`.

Оптимизация: AdamW, lr голов 1e-3, бэкбона 3e-4, линейный прогрев 1 эпоха и косинус по
шагам; FP32 без AMP (GTX 1050 — Pascal). Сэмплер — баланс частей строки из T15
(`ocr_lab.dataset.part_weights`); аугментации — `ocr_lab.augment` (T15). Сид аугментации
кропа зависит от «виртуального индекса» `эпоха × N + i`, поэтому эпохи различаются, а
результат не зависит от числа процессов загрузчика и от `--resume`.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import shutil
import subprocess
import sys
import time
from collections import defaultdict
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ocr_lab import paths
from ocr_lab.augment import augment as augment_crop
from ocr_lab.cut_crops import read_csv_rows, read_gray
from ocr_lab.dataset import (
    INDEX_CSV,
    PART_ID,
    SUBFIELDS,
    TARGET_SIZE,
    Sample,
    SyntheticDigitsDataset,
    build_samples,
    encode_target,
    fit_with_aspect,
    load_mnist_digit_bank,
    part_weights,
)
from ocr_lab.digit_values import predict_values, top_values, value_log_probs
from ocr_lab.evaluate import load_truth
from ocr_lab.models import DigitNet, ModelConfig, digit_loss
from ocr_lab.predictions import ScanPrediction, write_predictions

RUNS_DIR = paths.MODELS_DIR / "digits_runs"
V0_DIR = paths.MODELS_DIR / "digits_v0"
MNIST_ROOT = paths.CACHE_DIR / "mnist"
PARTS_ORDER: tuple[str, ...] = tuple(PART_ID)
#: Сколько ошибок `val` показывать на листе разбора.
ERRORS_N = 60


@dataclass
class TrainConfig:
    """Всё, что определяет запуск, — пишется в `config.json` и в чекпоинт."""

    run: str
    model: ModelConfig = field(default_factory=ModelConfig)
    epochs: int = 40
    batch_size: int = 128
    lr_head: float = 1e-3
    lr_backbone: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    patience: int = 12
    seed: int = 0
    augment: bool = True
    mnist: int = 0
    num_workers: int = 2
    target_size: tuple[int, int] = TARGET_SIZE

    def to_dict(self) -> dict[str, Any]:
        out = asdict(self)
        out["model"] = self.model.to_dict()
        out["target_size"] = list(self.target_size)
        return out

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> TrainConfig:
        data = dict(data)
        data["model"] = ModelConfig.from_dict(data.get("model", {}))
        data["target_size"] = tuple(data.get("target_size", TARGET_SIZE))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# --- данные -------------------------------------------------------------------------------


def to_tensor(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Серый кроп → канва `size` (без искажения пропорций) → `1 x H x W` в `0..1`."""
    canvas = fit_with_aspect(image, size)
    return torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)


def _part_of(subfield: str) -> str:
    return subfield.split(".")[1]


class TrainSet(Dataset):
    """Реальные кропы (в памяти) плюс, по желанию, синтетика MNIST.

    Индекс — «виртуальный»: `v = эпоха × len + i`. Сам элемент — `i = v % len`, а сид
    аугментации (и синтетики) — `v`, поэтому каждая эпоха видит новые искажения. Элемент —
    кортеж тензоров `(image, tens, units, ambiguous, part_id)`.
    """

    def __init__(
        self,
        images: Sequence[np.ndarray],
        samples: Sequence[Sample],
        *,
        augment: bool,
        seed: int,
        size: tuple[int, int],
        synthetic: SyntheticDigitsDataset | None = None,
        n_synthetic: int = 0,
    ) -> None:
        self.images = list(images)
        self.samples = list(samples)
        self.augment = augment
        self.seed = seed
        self.size = size
        self.synthetic = synthetic
        self.n_synthetic = n_synthetic if synthetic is not None else 0

    def __len__(self) -> int:
        return len(self.samples) + self.n_synthetic

    def __getitem__(self, vidx: int) -> tuple[torch.Tensor, ...]:
        i = vidx % len(self)
        if i >= len(self.samples):
            assert self.synthetic is not None
            image, t = self.synthetic[vidx]
            return (
                image,
                torch.tensor(int(str(t["tens"]))),
                torch.tensor(int(str(t["units"]))),
                torch.tensor(bool(t["leading_zero_ambiguous"])),
                torch.tensor(-1),
            )
        sample = self.samples[i]
        image_np = self.images[i]
        if self.augment:
            image_np = augment_crop(image_np, (self.seed * 1_000_003 + vidx) % (2**31 - 1))
        target = encode_target(sample.value)
        return (
            to_tensor(image_np, self.size),
            torch.tensor(target.tens),
            torch.tensor(target.units),
            torch.tensor(target.leading_zero_ambiguous),
            torch.tensor(PART_ID[_part_of(sample.subfield)]),
        )


class EpochSampler(Sampler[int]):
    """Взвешенная выборка с возвращением, детерминированная по `(seed, epoch)`.

    Выдаёт виртуальные индексы `epoch × N + i` (см. :class:`TrainSet`).
    """

    def __init__(self, weights: Sequence[float], *, seed: int) -> None:
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return len(self.weights)

    def __iter__(self) -> Iterator[int]:
        gen = torch.Generator().manual_seed(self.seed * 10_007 + self.epoch)
        n = len(self.weights)
        idx = torch.multinomial(self.weights, n, replacement=True, generator=gen)
        offset = self.epoch * n
        return iter((idx + offset).tolist())


def train_weights(samples: Sequence[Sample], n_synthetic: int) -> list[float]:
    """Веса сэмплера: баланс частей строки (T15) у реальных кропов, синтетика — по 1.

    Сумма весов реальных кропов равна их числу, поэтому доля синтетики в эпохе —
    `n_synthetic / (N + n_synthetic)`.
    """
    real = part_weights([PART_ID[_part_of(s.subfield)] for s in samples])
    scale = len(real) / sum(real) if real else 0.0
    return [w * scale for w in real] + [1.0] * n_synthetic


def eval_tensors(
    samples: Sequence[Sample], size: tuple[int, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    """Все кропы выборки без аугментации: `(images (N,1,H,W), part_id (N,))`."""
    images = torch.stack([to_tensor(read_gray(s.path), size) for s in samples])
    parts = torch.tensor([PART_ID[_part_of(s.subfield)] for s in samples])
    return images, parts


@torch.no_grad()
def infer_logits(
    model: DigitNet, images: torch.Tensor, parts: torch.Tensor, device: torch.device,
    batch_size: int = 256,
) -> tuple[np.ndarray, np.ndarray]:
    """Логиты голов на всей выборке (`model.eval()`): `(tens (N,11), units (N,10))`."""
    model.eval()
    tens_out, units_out = [], []
    for start in range(0, len(images), batch_size):
        x = images[start : start + batch_size].to(device)
        p = parts[start : start + batch_size].to(device)
        t, u = model(x, p)
        tens_out.append(t.float().cpu().numpy())
        units_out.append(u.float().cpu().numpy())
    if not tens_out:
        return np.zeros((0, 11)), np.zeros((0, 10))
    return np.concatenate(tens_out), np.concatenate(units_out)


def score_logits(
    tens: np.ndarray, units: np.ndarray, samples: Sequence[Sample]
) -> dict[str, Any]:
    """Метрики на выборке с истиной: точность top-1 значения по подполям и частям, NLL.

    `mean_subfield_acc` — среднее точностей 16 подполей; это метрика выбора модели.
    """
    hits: dict[str, list[int]] = defaultdict(lambda: [0, 0])
    nll: list[float] = []
    for i, s in enumerate(samples):
        part = _part_of(s.subfield)
        values, logp = value_log_probs(tens[i], units[i], part)
        pred = int(values[int(np.argmax(logp))])
        hit = np.nonzero(values == s.value)[0]
        nll.append(-float(logp[hit[0]]) if hit.size else float(-math.log(1e-6)))
        ok = int(pred == s.value)
        for key in (s.subfield, f"part:{part}", "all"):
            hits[key][0] += ok
            hits[key][1] += 1
    sub_acc = {k: v[0] / v[1] for k, v in hits.items() if k in SUBFIELDS}
    return {
        "mean_subfield_acc": float(np.mean(list(sub_acc.values()))) if sub_acc else 0.0,
        "micro_acc": hits["all"][0] / hits["all"][1] if hits["all"][1] else 0.0,
        "part_acc": {p: hits[f"part:{p}"][0] / hits[f"part:{p}"][1]
                     for p in PARTS_ORDER if hits[f"part:{p}"][1]},
        "subfield_acc": dict(sorted(sub_acc.items())),
        "nll": float(np.mean(nll)) if nll else 0.0,
        "n": len(samples),
    }


# --- служебное -----------------------------------------------------------------------------


def _file_sha256(path: Path) -> str:
    if not path.exists():
        return ""
    return hashlib.sha256(path.read_bytes()).hexdigest()[:16]


def _git_commit() -> str:
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"], cwd=paths.REPO_ROOT,
            capture_output=True, text=True, check=False,
        )
        return out.stdout.strip()
    except OSError:
        return ""


def set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def lr_lambda(step: int, *, warmup: int, total: int) -> float:
    """Множитель lr: линейный прогрев `warmup` шагов, затем косинус до 0 к шагу `total`."""
    if warmup > 0 and step < warmup:
        return (step + 1) / warmup
    progress = min(1.0, (step - warmup) / max(1, total - warmup))
    return 0.5 * (1.0 + math.cos(math.pi * progress))


class Logger:
    """Печать в консоль и дозапись в `train.log`."""

    def __init__(self, path: Path) -> None:
        self.path = path

    def __call__(self, msg: str) -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        with self.path.open("a", encoding="utf-8") as fh:
            fh.write(line + "\n")


HISTORY_COLUMNS = (
    "epoch", "train_loss", "train_tens", "train_units", "val_mean_subfield_acc",
    "val_micro_acc", "val_day", "val_month", "val_hour", "val_minute", "val_nll",
    "lr_head", "seconds",
)


def write_history(path: Path, history: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in HISTORY_COLUMNS})


def build_model(config: ModelConfig, *, pretrained: bool) -> DigitNet:
    if pretrained:
        paths.configure_model_caches()
    return DigitNet(config, pretrained=pretrained)


def load_model(run_dir: Path, device: torch.device | None = None,
               checkpoint: str = "best.pt") -> tuple[DigitNet, TrainConfig, dict[str, Any]]:
    """Загрузить модель из каталога запуска (для предсказаний, T17 и T18)."""
    device = device or torch.device("cpu")
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig.from_dict(state["config"])
    model = DigitNet(cfg.model, pretrained=False)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, cfg, state


# --- обучение ------------------------------------------------------------------------------


def train(cfg: TrainConfig, *, runs_dir: Path = RUNS_DIR, max_minutes: float | None = None,
          resume: bool = False) -> int:
    """Обучить (или продолжить) запуск. Код возврата: 0 — готово, 3 — прервано по времени."""
    t_start = time.time()
    run_dir = paths.ensure_dir(runs_dir / cfg.run)
    log = Logger(run_dir / "train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    last_path = run_dir / "last.pt"
    state: dict[str, Any] | None = None
    if resume:
        if not last_path.exists():
            log("нет last.pt — начинаю заново")
        else:
            state = torch.load(last_path, map_location="cpu", weights_only=False)
            assert state is not None
            cfg = TrainConfig.from_dict(state["config"])
            if state.get("done"):
                log("запуск уже завершён")
                return 0
    elif last_path.exists():
        raise SystemExit(f"{run_dir.name}: запуск уже есть — используйте --resume")

    set_seed(cfg.seed)
    train_samples = build_samples(split="train")
    val_samples = build_samples(split="val")
    log(f"запуск {cfg.run}: train {len(train_samples)}, val {len(val_samples)}, "
        f"устройство {device}")
    t0 = time.time()
    train_images = [read_gray(s.path) for s in train_samples]
    val_x, val_p = eval_tensors(val_samples, cfg.target_size)
    log(f"кропы загружены за {time.time() - t0:.1f} с")

    synthetic = None
    if cfg.mnist > 0:
        bank = load_mnist_digit_bank(MNIST_ROOT)
        synthetic = SyntheticDigitsDataset(bank, n=cfg.mnist, seed=cfg.seed,
                                           target_size=cfg.target_size)
    train_set = TrainSet(train_images, train_samples, augment=cfg.augment, seed=cfg.seed,
                         size=cfg.target_size, synthetic=synthetic, n_synthetic=cfg.mnist)
    sampler = EpochSampler(train_weights(train_samples, train_set.n_synthetic), seed=cfg.seed)
    loader = DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0, pin_memory=device.type == "cuda",
        drop_last=True,
    )

    model = build_model(cfg.model, pretrained=state is None).to(device)
    head_ids = {id(p) for p in model.head_parameters()}
    backbone_params = [p for p in model.parameters() if id(p) not in head_ids]
    optimizer = torch.optim.AdamW(
        [{"params": backbone_params, "lr": cfg.lr_backbone},
         {"params": model.head_parameters(), "lr": cfg.lr_head}],
        weight_decay=cfg.weight_decay,
    )
    steps_per_epoch = len(loader)
    total = steps_per_epoch * cfg.epochs
    warmup = steps_per_epoch * cfg.warmup_epochs
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lambda s: lr_lambda(s, warmup=warmup, total=total)
    )

    history: list[dict[str, Any]] = []
    best_key: tuple[float, float] = (-1.0, 0.0)
    best_epoch = -1
    start_epoch = 0
    if state is not None:
        model.load_state_dict(state["model"])
        optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"])
        history = state["history"]
        best_key = (float(state["best_key"][0]), float(state["best_key"][1]))
        best_epoch = state["best_epoch"]
        start_epoch = state["epoch"] + 1
        torch.set_rng_state(state["rng_cpu"])
        if device.type == "cuda" and state.get("rng_cuda") is not None:
            torch.cuda.set_rng_state(state["rng_cuda"])
        log(f"продолжаю с эпохи {start_epoch}, лучшая {best_epoch} ({best_key[0]:.4f})")
    else:
        meta = cfg.to_dict() | {
            "git_commit": _git_commit(),
            "manifest_sha256": _file_sha256(paths.MANIFEST),
            "crops_index_sha256": _file_sha256(INDEX_CSV),
            "n_train": len(train_samples),
            "n_val": len(val_samples),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        }
        (run_dir / "config.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    done = False
    last_epoch_s = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        if max_minutes is not None and epoch > start_epoch:
            if time.time() - t_start + last_epoch_s * 1.1 > max_minutes * 60:
                log(f"остановка по --max-minutes перед эпохой {epoch}; продолжить: --resume")
                return 3
        te = time.time()
        model.train()
        sampler.set_epoch(epoch)
        sums = np.zeros(3)
        n_batches = 0
        for x, tens, units, amb, part in loader:
            x = x.to(device, non_blocking=True)
            part = part.to(device)
            t_logits, u_logits = model(x, part)
            loss, parts_loss = digit_loss(t_logits, u_logits, tens.to(device),
                                          units.to(device), amb.to(device))
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            sums += (float(loss.detach()), parts_loss["tens"], parts_loss["units"])
            n_batches += 1
        tl, ul = infer_logits(model, val_x, val_p, device)
        metrics = score_logits(tl, ul, val_samples)
        last_epoch_s = time.time() - te
        row = {
            "epoch": epoch,
            "train_loss": round(sums[0] / max(1, n_batches), 5),
            "train_tens": round(sums[1] / max(1, n_batches), 5),
            "train_units": round(sums[2] / max(1, n_batches), 5),
            "val_mean_subfield_acc": round(metrics["mean_subfield_acc"], 5),
            "val_micro_acc": round(metrics["micro_acc"], 5),
            **{f"val_{p}": round(metrics["part_acc"].get(p, 0.0), 5) for p in PARTS_ORDER},
            "val_nll": round(metrics["nll"], 5),
            "lr_head": f"{optimizer.param_groups[1]['lr']:.2e}",
            "seconds": round(last_epoch_s, 1),
        }
        history.append(row)
        key = (metrics["mean_subfield_acc"], -metrics["nll"])
        improved = key > best_key
        if improved:
            best_key, best_epoch = key, epoch
        log(f"эпоха {epoch}: loss {row['train_loss']:.4f}, val acc "
            f"{row['val_mean_subfield_acc']:.4f} (д {row['val_day']:.3f} м {row['val_month']:.3f}"
            f" ч {row['val_hour']:.3f} мин {row['val_minute']:.3f}), nll {row['val_nll']:.4f},"
            f" {last_epoch_s:.0f} с{' *' if improved else ''}")
        if epoch == start_epoch and device.type == "cuda":
            log(f"пик VRAM: {torch.cuda.max_memory_allocated() / 2**20:.0f} МБ "
                f"(батч {cfg.batch_size})")
        done = epoch + 1 >= cfg.epochs or epoch - best_epoch >= cfg.patience
        ckpt = {
            "config": cfg.to_dict(),
            "model": model.state_dict(),
            "epoch": epoch,
            "val_metrics": metrics,
        }
        if improved:
            torch.save(ckpt, run_dir / "best.pt")
        torch.save(ckpt | {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_key": list(best_key),
            "best_epoch": best_epoch,
            "rng_cpu": torch.get_rng_state(),
            "rng_cuda": torch.cuda.get_rng_state() if device.type == "cuda" else None,
            "done": done,
        }, last_path)
        write_history(run_dir / "history.csv", history)
        if done:
            if epoch + 1 < cfg.epochs:
                log(f"ранняя остановка: {cfg.patience} эпох без улучшения")
            break
    log(f"готово: лучшая эпоха {best_epoch}, val mean_subfield_acc {best_key[0]:.4f}")
    return 0


# --- предсказания и разбор ошибок ---------------------------------------------------------


@dataclass(frozen=True)
class CropRef:
    scan_id: str
    subfield: str
    path: Path


def split_crops(split: str, *, index_csv: Path = INDEX_CSV,
                work_dir: Path = paths.WORK_DIR) -> list[CropRef]:
    """Все кропы двузначных подполей сканов сплита (без фильтров по истине и `crops_qc`).

    Предсказание делается для каждого кропа, даже плохого: честная оценка видит и их.
    """
    split_ids = set(load_truth(paths.MANIFEST, split))
    out = []
    for row in read_csv_rows(index_csv):
        if row["subfield"] not in SUBFIELDS or row["scan_id"] not in split_ids:
            continue
        if (row.get("align_ok") or "True") != "True":
            continue
        out.append(CropRef(row["scan_id"], row["subfield"], work_dir / row["path"]))
    return out


def crop_logits(model: DigitNet, crops: Sequence[CropRef], size: tuple[int, int],
                device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    images = torch.stack([to_tensor(read_gray(c.path), size) for c in crops])
    parts = torch.tensor([PART_ID[_part_of(c.subfield)] for c in crops])
    return infer_logits(model, images, parts, device)


def build_predictions(crops: Sequence[CropRef], tens: np.ndarray, units: np.ndarray,
                      source: str) -> list[ScanPrediction]:
    """Сгруппировать top-5 значений кропов по сканам в строки формата T13."""
    tops = predict_values(tens, units, [_part_of(c.subfield) for c in crops])
    by_scan: dict[str, ScanPrediction] = {}
    for crop, cands in zip(crops, tops, strict=True):
        pred = by_scan.setdefault(crop.scan_id, ScanPrediction(scan_id=crop.scan_id,
                                                               source=source))
        pred.fields[crop.subfield] = [(v, p) for v, p in cands]
    return [by_scan[k] for k in sorted(by_scan)]


def predict(run_dir: Path, split: str, *, source: str | None = None,
            out: Path | None = None) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_model(run_dir, device)
    crops = split_crops(split)
    tens, units = crop_logits(model, crops, cfg.target_size, device)
    preds = build_predictions(crops, tens, units, source or run_dir.name)
    out = out or run_dir / f"{split}_predictions.jsonl"
    n = write_predictions(out, preds)
    print(f"{split}: {len(crops)} кропов, {n} сканов → {out}")
    return out


def errors_sheet(run_dir: Path, *, n: int = ERRORS_N, out: Path | None = None) -> Path:
    """Лист `n` худших ошибок на `val`: кроп, истина, top-3 с вероятностями.

    «Худшие» — с наименьшей вероятностью истинного значения. CSV рядом с листом:
    индекс, `scan_id`, подполе, истина, top-3, `p_truth`, `qc` (пометка T10, если есть).
    """
    from ocr_lab.sheets import make_sheet  # noqa: PLC0415 — Pillow-шрифты нужны только здесь

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_model(run_dir, device)
    truth = load_truth(paths.MANIFEST, "val")
    crops = [c for c in split_crops("val") if truth[c.scan_id].parts.get(c.subfield) is not None]
    tens, units = crop_logits(model, crops, cfg.target_size, device)
    qc = {(r["subfield"], r["scan_id"]): r["problem"]
          for r in read_csv_rows(paths.REVIEW_DIR / "crops_qc.csv")}
    rows = []
    for i, c in enumerate(crops):
        value = truth[c.scan_id].parts[c.subfield]
        values, logp = value_log_probs(tens[i], units[i], _part_of(c.subfield))
        top = top_values(values, logp, 3)
        if top[0][0] == value:
            continue
        hit = np.nonzero(values == value)[0]
        p_truth = float(np.exp(logp[hit[0]])) if hit.size else 0.0
        rows.append((p_truth, c, value, top))
    rows.sort(key=lambda r: r[0])
    rows = rows[:n]
    out = out or paths.SHEETS_DIR / "digits_errors_val.png"
    short = {"left_base": "L", "arrived_base": "A", "started_work": "S", "finished_work": "F"}
    images, captions = [], []
    for _p, c, value, top in rows:
        r, part = c.subfield.split(".")
        images.append(read_gray(c.path))
        tops = " ".join(f"{v}:{p:.2f}" for v, p in top)
        captions.append(f"{short[r]}.{part} ист={value} | {tops}")
    if images:
        make_sheet(images, captions, cols=6, cell_w=220, out_path=out, font_size=13)
    with out.with_name(out.stem + "_index.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["index", "scan_id", "subfield", "truth", "top3", "p_truth", "qc"])
        for i, (p, c, value, top) in enumerate(rows):
            writer.writerow([i, c.scan_id, c.subfield, value,
                             " ".join(f"{v}:{q:.3f}" for v, q in top), f"{p:.4f}",
                             qc.get((c.subfield, c.scan_id), "")])
    print(f"ошибок на val: {len(rows)} показано → {out}")
    return out


def promote(run: str, *, runs_dir: Path = RUNS_DIR, dest: Path = V0_DIR) -> None:
    src = runs_dir / run
    paths.ensure_dir(dest)
    for name in ("best.pt", "config.json", "history.csv", "train.log"):
        shutil.copy2(src / name, dest / name)
    print(f"{run} → {dest}")


# --- CLI -----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ocr_lab.train_digits", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="обучить запуск")
    p_train.add_argument("--run", required=True)
    p_train.add_argument("--backbone", default="resnet18",
                         choices=("resnet18", "mobilenet_v3_small"))
    p_train.add_argument("--no-part-embed", action="store_true")
    p_train.add_argument("--mnist", type=int, default=0, help="синтетических сэмплов на эпоху")
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--batch-size", type=int, default=128)
    p_train.add_argument("--lr-head", type=float, default=1e-3)
    p_train.add_argument("--lr-backbone", type=float, default=3e-4)
    p_train.add_argument("--patience", type=int, default=12)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--workers", type=int, default=2)
    p_train.add_argument("--no-augment", action="store_true")
    p_train.add_argument("--max-minutes", type=float, default=None)
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--runs-dir", type=Path, default=RUNS_DIR)

    p_pred = sub.add_parser("predict", help="предсказания в формате T13")
    p_pred.add_argument("--run-dir", type=Path, default=V0_DIR)
    p_pred.add_argument("--split", choices=("val", "test", "train"), required=True)
    p_pred.add_argument("--source", default=None)
    p_pred.add_argument("--out", type=Path, default=None)

    p_err = sub.add_parser("errors", help="лист худших ошибок на val")
    p_err.add_argument("--run-dir", type=Path, default=V0_DIR)
    p_err.add_argument("--n", type=int, default=ERRORS_N)
    p_err.add_argument("--out", type=Path, default=None)

    p_prom = sub.add_parser("promote", help="скопировать запуск в digits_v0")
    p_prom.add_argument("--run", required=True)
    p_prom.add_argument("--runs-dir", type=Path, default=RUNS_DIR)

    args = parser.parse_args(argv)
    if args.cmd == "train":
        cfg = TrainConfig(
            run=args.run,
            model=ModelConfig(backbone=args.backbone, part_embed=not args.no_part_embed),
            epochs=args.epochs, batch_size=args.batch_size, lr_head=args.lr_head,
            lr_backbone=args.lr_backbone, patience=args.patience, seed=args.seed,
            augment=not args.no_augment, mnist=args.mnist, num_workers=args.workers,
        )
        return train(cfg, runs_dir=args.runs_dir, max_minutes=args.max_minutes,
                     resume=args.resume)
    if args.cmd == "predict":
        predict(args.run_dir, args.split, source=args.source, out=args.out)
    elif args.cmd == "errors":
        errors_sheet(args.run_dir, n=args.n, out=args.out)
    elif args.cmd == "promote":
        promote(args.run, runs_dir=args.runs_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
