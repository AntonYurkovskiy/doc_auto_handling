"""T17: обучение модели номера ваучера (подход А — три головы, подход Б — CRNN + CTC).

CLI (`python -m ocr_lab.train_number <команда>`):

- `train --run NAME --kind heads|ctc [--synth N] [--real-repeats R] [--epochs 40]
  [--batch-size 32] [--max-minutes 9] [--resume] [--no-init]` — обучение в
  `data/ocr/models/number_runs/NAME/`: `best.pt` (лучшая эпоха по точности top-1 на `val`,
  при равенстве — меньший NLL), `last.pt` (состояние для `--resume`, пишется каждую эпоху),
  `config.json`, `history.csv`, `train.log`. По `--max-minutes` обучение останавливается
  между эпохами и продолжается с `--resume`; ранняя остановка — по той же метрике;
- `predict --run-dir DIR --split val|test [--source NAME]` — предсказания в формате T13
  (`DIR/<split>_predictions.jsonl`, поле `voucher_number`, top-5) и сырые выходы модели
  (`DIR/<split>_outputs.npz`: `scan_id`, `output`, `kind`) — вход `score_candidates` для T20
  без torch;
- `compare --run-dirs A B … [--split val] [--stress N]` — таблица «подход против подхода»:
  top-1, top-5, средний ранг истины среди кандидатов ±15, NLL, срезы «печатный /
  рукописный» (метки T12), с `--stress N` — top-1 и ранг на N аугментированных копиях
  каждого кропа;
- `promote --run NAME` — скопировать выбранный запуск в `data/ocr/models/number_v0/`.

`test` в обучении и выборе модели не участвует: `train` читает только `train` и `val`.

Данные: кропы `voucher_number` из `crops_index.csv` (T10, `align_ok`), истина — номер из
манифеста (через `ocr_lab.evaluate.load_truth`). Кроп вписывается без искажения пропорций
в канву :data:`TARGET_SIZE` = 64 × 176 (белый фон, `ocr_lab.dataset.fit_with_aspect`):
кропы номера — 105 или 110 px по высоте и 162–275 px по ширине; при высоте 64 (масштаб
цифр тот же, что у `digits_v0`: его кропы — 92–112 px по высоте) все 769 кропов
помещаются по высоте целиком, а в канву 64 × 160 модели цифр не влезли бы 17 широких
кропов. Аугментации — `ocr_lab.augment` (T15) на исходном кропе.

Синтетика (`--synth N`, N сэмплов на эпоху): номера из 1–3 цифр MNIST (кэш T16,
`OCR_CACHE_DIR/mnist`), склеенные в масштабе реального кропа (высота цифры 46–76 px,
случайный зазор и вертикальный сдвиг, линия подчёркивания), затем те же аугментации.
Номера синтетики — 1..399, с перевесом трёхзначных: у реальных номеров сотни бывают
только 1–3, синтетика закрывает остальные сочетания цифр по позициям.

Оптимизация — как в T16: AdamW, lr новых слоёв 1e-3, бэкбона 3e-4, прогрев 1 эпоха и
косинус по шагам, FP32. Сид аугментации — «виртуальный индекс» `эпоха × N + i`.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import sys
import time
from collections.abc import Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch
from torch.utils.data import DataLoader, Dataset, Sampler

from ocr_lab import paths
from ocr_lab.augment import augment as augment_crop
from ocr_lab.cut_crops import read_csv_rows, read_gray
from ocr_lab.dataset import INDEX_CSV, fit_with_aspect, load_mnist_digit_bank
from ocr_lab.evaluate import load_printed_flags, load_truth
from ocr_lab.number_models import (
    NumberModelConfig,
    NumberNet,
    build_number_model,
    ctc_loss,
    heads_loss,
    load_digits_backbone,
)
from ocr_lab.number_scores import (
    KINDS,
    encode_heads,
    number_log_probs,
    score_candidates,
    truth_rank,
    window_candidates,
)
from ocr_lab.predictions import VOUCHER_NUMBER, ScanPrediction, write_predictions
from ocr_lab.train_digits import (
    Logger,
    _file_sha256,
    _git_commit,
    lr_lambda,
    set_seed,
)

RUNS_DIR = paths.MODELS_DIR / "number_runs"
V0_DIR = paths.MODELS_DIR / "number_v0"
DIGITS_V0 = paths.MODELS_DIR / "digits_v0" / "best.pt"
MNIST_ROOT = paths.CACHE_DIR / "mnist"
#: Канва (H, W) для кропа номера — обоснование в докстринге модуля.
TARGET_SIZE: tuple[int, int] = (64, 176)
#: Радиус окна кандидатов для метрики ранжирования (истина ± 15).
RANK_RADIUS = 15
TOP_K = 5


@dataclass
class TrainConfig:
    """Всё, что определяет запуск, — пишется в `config.json` и в чекпоинт."""

    run: str
    model: NumberModelConfig = field(default_factory=NumberModelConfig)
    epochs: int = 40
    batch_size: int = 32
    lr_head: float = 1e-3
    lr_backbone: float = 3e-4
    weight_decay: float = 1e-4
    warmup_epochs: int = 1
    patience: int = 12
    seed: int = 0
    augment: bool = True
    real_repeats: int = 2
    synth: int = 0
    init_digits: bool = True
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
        data["model"] = NumberModelConfig.from_dict(data.get("model", {}))
        data["target_size"] = tuple(data.get("target_size", TARGET_SIZE))
        known = set(cls.__dataclass_fields__)
        return cls(**{k: v for k, v in data.items() if k in known})


# --- данные -------------------------------------------------------------------------------


@dataclass(frozen=True)
class NumberCrop:
    """Кроп номера ваучера; `value` — истина (`None`, если номера в манифесте нет)."""

    scan_id: str
    path: Path
    value: int | None


def number_crops(
    split: str, *, manifest: Path | None = None, index_csv: Path = INDEX_CSV,
    work_dir: Path | None = None, with_truth: bool = True,
) -> list[NumberCrop]:
    """Кропы `voucher_number` сканов сплита (`align_ok`), по `scan_id`.

    `with_truth=True` — только кропы с номером в истине (обучение и метрики); `False` —
    все кропы сплита (предсказания делаются для каждого кропа).
    """
    manifest = manifest or paths.MANIFEST
    work_dir = work_dir or paths.WORK_DIR
    truth = load_truth(manifest, split)
    out: list[NumberCrop] = []
    for row in read_csv_rows(index_csv):
        # `load_truth` срезает пробелы в `scan_id`, индекс кропов T10 — нет (скан
        # `2026_223k ` из файла «223k .pdf»), поэтому сверяем по срезанному `scan_id`.
        scan_id = row["scan_id"].strip()
        if row["subfield"] != VOUCHER_NUMBER or scan_id not in truth:
            continue
        if (row.get("align_ok") or "True") != "True":
            continue
        value = truth[scan_id].voucher_number
        if with_truth and value is None:
            continue
        out.append(NumberCrop(scan_id, work_dir / row["path"], value))
    return sorted(out, key=lambda c: c.scan_id)


def to_tensor(image: np.ndarray, size: tuple[int, int]) -> torch.Tensor:
    """Серый кроп → канва `size` (без искажения пропорций) → `1 x H x W` в `0..1`."""
    canvas = fit_with_aspect(image, size)
    return torch.from_numpy(canvas.astype(np.float32) / 255.0).unsqueeze(0)


def _trim_ink(glyph: np.ndarray, thr: int = 200) -> np.ndarray:
    """Срезать белые поля глифа со всех сторон (у MNIST цифра окружена полями ~4 px)."""
    ink = glyph < thr
    rows = np.nonzero(ink.any(axis=1))[0]
    cols = np.nonzero(ink.any(axis=0))[0]
    if rows.size == 0 or cols.size == 0:
        return glyph
    return glyph[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]


def synth_value(rng: np.random.Generator) -> int:
    """Номер синтетики: 60 % трёхзначных 100..399, 30 % двузначных, 10 % однозначных."""
    r = rng.random()
    if r < 0.6:
        return int(rng.integers(100, 400))
    if r < 0.9:
        return int(rng.integers(10, 100))
    return int(rng.integers(1, 10))


def render_synthetic_number(
    value: int, bank: dict[int, list[np.ndarray]], rng: np.random.Generator
) -> np.ndarray:
    """Склеить номер из глифов MNIST в масштабе реального кропа номера (серый, белый фон).

    Высота цифры (по чернилам) 46–76 px, ширина — по пропорциям глифа с разбросом ±15 %,
    зазор между цифрами от лёгкого нахлёста до ~¼ высоты, вертикальный сдвиг каждой
    цифры, линия подчёркивания под номером (80 %).
    """
    digit_h = int(rng.integers(46, 77))
    glyphs = []
    for ch in str(value):
        pool = bank[int(ch)]
        g = _trim_ink(pool[int(rng.integers(0, len(pool)))])
        ratio = g.shape[1] / g.shape[0] * float(rng.uniform(0.85, 1.15))
        width = max(4, round(digit_h * ratio))
        glyphs.append(cv2.resize(g, (width, digit_h), interpolation=cv2.INTER_LINEAR))
    canvas_h = int(rng.integers(100, 117))
    canvas_w = int(rng.integers(190, 276))
    canvas = np.full((canvas_h, canvas_w), 255, dtype=np.uint8)
    gaps = [int(rng.integers(-digit_h // 10, digit_h // 4 + 1)) for _ in glyphs[1:]]
    total_w = sum(g.shape[1] for g in glyphs) + sum(gaps)
    if total_w > canvas_w - 10:
        scale = (canvas_w - 10) / total_w
        glyphs = [cv2.resize(g, (max(1, round(g.shape[1] * scale)), max(1, round(digit_h * scale))))
                  for g in glyphs]
        gaps = [round(gp * scale) for gp in gaps]
        total_w = sum(g.shape[1] for g in glyphs) + sum(gaps)
    x = int(rng.integers(10, max(11, canvas_w - total_w - 8)))
    base_y = canvas_h - int(rng.integers(14, 26))
    for i, g in enumerate(glyphs):
        gh, gw = g.shape
        y = int(np.clip(base_y - gh + rng.integers(-4, 5), 0, canvas_h - gh))
        x0 = int(np.clip(x, 0, canvas_w - gw))
        region = canvas[y : y + gh, x0 : x0 + gw]
        np.minimum(region, g, out=region)
        x += gw + (gaps[i] if i < len(gaps) else 0)
    if rng.random() < 0.8:
        line_y = min(canvas_h - 2, base_y + int(rng.integers(2, 10)))
        x_lo = int(rng.integers(0, 30))
        x_hi = canvas_w - int(rng.integers(0, 40))
        thick = int(rng.integers(1, 4))
        canvas[line_y : line_y + thick, x_lo:x_hi] = int(rng.integers(0, 90))
    return canvas


class NumberTrainSet(Dataset):
    """Реальные кропы (в памяти, `real_repeats` раз за эпоху) плюс синтетика MNIST.

    Индекс — «виртуальный»: `v = эпоха × len + i`, элемент — `i = v % len`, сид
    аугментации и синтетики — `v`. Элемент — `(image (1, H, W), номер)`.
    """

    def __init__(
        self, images: Sequence[np.ndarray], values: Sequence[int], *, augment: bool, seed: int,
        size: tuple[int, int], real_repeats: int = 1,
        bank: dict[int, list[np.ndarray]] | None = None, n_synth: int = 0,
    ) -> None:
        self.images = list(images)
        self.values = list(values)
        self.augment = augment
        self.seed = seed
        self.size = size
        self.real_repeats = real_repeats
        self.bank = bank
        self.n_synth = n_synth if bank is not None else 0

    @property
    def n_real(self) -> int:
        return len(self.images) * self.real_repeats

    def __len__(self) -> int:
        return self.n_real + self.n_synth

    def __getitem__(self, vidx: int) -> tuple[torch.Tensor, int]:
        i = vidx % len(self)
        aug_seed = (self.seed * 1_000_003 + vidx) % (2**31 - 1)
        if i >= self.n_real:
            assert self.bank is not None
            rng = np.random.default_rng(aug_seed)
            value = synth_value(rng)
            image = render_synthetic_number(value, self.bank, rng)
        else:
            j = i % len(self.images)
            image, value = self.images[j], self.values[j]
        if self.augment:
            image = augment_crop(image, aug_seed)
        return to_tensor(image, self.size), value


class EpochPermutation(Sampler[int]):
    """Перестановка всех элементов, детерминированная по `(seed, epoch)`.

    Выдаёт виртуальные индексы `epoch × N + i` (см. :class:`NumberTrainSet`).
    """

    def __init__(self, n: int, *, seed: int) -> None:
        self.n = n
        self.seed = seed
        self.epoch = 0

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch

    def __len__(self) -> int:
        return self.n

    def __iter__(self) -> Iterator[int]:
        gen = torch.Generator().manual_seed(self.seed * 10_007 + self.epoch)
        perm = torch.randperm(self.n, generator=gen)
        return iter((perm + self.epoch * self.n).tolist())


def collate(batch: Sequence[tuple[torch.Tensor, int]]) -> tuple[torch.Tensor, list[int]]:
    images = torch.stack([b[0] for b in batch])
    return images, [int(b[1]) for b in batch]


# --- модель, выходы и метрики -------------------------------------------------------------


def number_loss(model: NumberNet, x: torch.Tensor, values: Sequence[int]
                ) -> tuple[torch.Tensor, float]:
    out = model(x)
    if model.config.kind == "heads":
        targets = torch.tensor([encode_heads(v) for v in values], device=out.device)
        return heads_loss(out, targets)
    return ctc_loss(out, values)


def build_model(cfg: TrainConfig, *, init_from: Path | None) -> NumberNet:
    model = build_number_model(cfg.model)
    if init_from is not None:
        load_digits_backbone(model, init_from)
    return model


def load_model(run_dir: Path, device: torch.device | None = None,
               checkpoint: str = "best.pt") -> tuple[NumberNet, TrainConfig, dict[str, Any]]:
    """Загрузить модель номера из каталога запуска (для предсказаний, T18 и T20)."""
    device = device or torch.device("cpu")
    state = torch.load(run_dir / checkpoint, map_location=device, weights_only=False)
    cfg = TrainConfig.from_dict(state["config"])
    model = build_number_model(cfg.model)
    model.load_state_dict(state["model"])
    model.to(device).eval()
    return model, cfg, state


def crop_tensors(crops: Sequence[NumberCrop], size: tuple[int, int]) -> torch.Tensor:
    return torch.stack([to_tensor(read_gray(c.path), size) for c in crops])


@torch.no_grad()
def infer_outputs(model: NumberNet, images: torch.Tensor, device: torch.device,
                  batch_size: int = 64) -> np.ndarray:
    """Сырые логиты модели на выборке: `(N, 3, 11)` (heads) или `(N, T, 11)` (ctc)."""
    model.eval()
    outs = [model(images[s : s + batch_size].to(device)).float().cpu().numpy()
            for s in range(0, len(images), batch_size)]
    return np.concatenate(outs) if outs else np.zeros((0, 0, 0))


def score_outputs(
    outputs: np.ndarray, values: Sequence[int], kind: str, *, radius: int = RANK_RADIUS,
) -> dict[str, Any]:
    """Метрики на выборке с истиной.

    - `top1`, `top5` — точность по распределению номеров 1..999 (:func:`number_log_probs`);
    - `nll` — `-log P(истина)` по тому же распределению;
    - `mean_rank` — средний ранг истины среди кандидатов `истина ± radius` (1 — лучший,
      при равенстве — пессимистично), `rank1` — доля ранга 1 в окне.
    """
    n = len(values)
    if n == 0:
        return {"n": 0, "top1": 0.0, "top5": 0.0, "nll": 0.0, "mean_rank": 0.0, "rank1": 0.0}
    top1 = top5 = 0
    nll: list[float] = []
    ranks: list[int] = []
    for out, value in zip(outputs, values, strict=True):
        nums, logp = number_log_probs(out, kind=kind)
        order = np.argsort(-logp, kind="stable")[:TOP_K]
        top = [int(nums[i]) for i in order]
        top1 += int(top[0] == value)
        top5 += int(value in top)
        hit = np.nonzero(nums == value)[0]
        nll.append(-float(logp[hit[0]]) if hit.size else -math.log(1e-9))
        scores = score_candidates(out, window_candidates(value, radius), kind=kind)
        ranks.append(truth_rank(scores, str(value)))
    return {
        "n": n,
        "top1": top1 / n,
        "top5": top5 / n,
        "nll": float(np.mean(nll)),
        "mean_rank": float(np.mean(ranks)),
        "rank1": sum(1 for r in ranks if r == 1) / n,
    }


# --- обучение ------------------------------------------------------------------------------

HISTORY_COLUMNS = (
    "epoch", "train_loss", "val_top1", "val_top5", "val_nll", "val_mean_rank", "lr_head",
    "seconds",
)


def write_history(path: Path, history: Sequence[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=HISTORY_COLUMNS)
        writer.writeheader()
        for row in history:
            writer.writerow({k: row.get(k, "") for k in HISTORY_COLUMNS})


def train(cfg: TrainConfig, *, runs_dir: Path = RUNS_DIR, max_minutes: float | None = None,
          resume: bool = False, digits_checkpoint: Path = DIGITS_V0) -> int:
    """Обучить (или продолжить) запуск. Код возврата: 0 — готово, 3 — прервано по времени."""
    t_start = time.time()
    run_dir = paths.ensure_dir(runs_dir / cfg.run)
    log = Logger(run_dir / "train.log")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    last_path = run_dir / "last.pt"
    state: dict[str, Any] | None = None
    if resume and last_path.exists():
        state = torch.load(last_path, map_location="cpu", weights_only=False)
        assert state is not None
        cfg = TrainConfig.from_dict(state["config"])
        if state.get("done"):
            log("запуск уже завершён")
            return 0
    elif resume:
        log("нет last.pt — начинаю заново")
    elif last_path.exists():
        raise SystemExit(f"{run_dir.name}: запуск уже есть — используйте --resume")

    set_seed(cfg.seed)
    train_crops = number_crops("train")
    val_crops = number_crops("val")
    log(f"запуск {cfg.run} ({cfg.model.kind}): train {len(train_crops)}, "
        f"val {len(val_crops)}, устройство {device}, потоков {torch.get_num_threads()}")
    train_images = [read_gray(c.path) for c in train_crops]
    val_x = crop_tensors(val_crops, cfg.target_size)
    val_values = [int(c.value) for c in val_crops if c.value is not None]

    bank = load_mnist_digit_bank(MNIST_ROOT) if cfg.synth > 0 else None
    train_set = NumberTrainSet(
        train_images, [int(c.value) for c in train_crops if c.value is not None],
        augment=cfg.augment, seed=cfg.seed, size=cfg.target_size,
        real_repeats=cfg.real_repeats, bank=bank, n_synth=cfg.synth,
    )
    sampler = EpochPermutation(len(train_set), seed=cfg.seed)
    loader = DataLoader(
        train_set, batch_size=cfg.batch_size, sampler=sampler, num_workers=cfg.num_workers,
        persistent_workers=cfg.num_workers > 0, collate_fn=collate, drop_last=True,
    )

    init_from = digits_checkpoint if (state is None and cfg.init_digits) else None
    model = build_model(cfg, init_from=init_from).to(device)
    if init_from is not None:
        log(f"бэкбон инициализирован из {init_from.parent.name}")
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
        log(f"продолжаю с эпохи {start_epoch}, лучшая {best_epoch} ({best_key[0]:.4f})")
    else:
        meta = cfg.to_dict() | {
            "git_commit": _git_commit(),
            "manifest_sha256": _file_sha256(paths.MANIFEST),
            "crops_index_sha256": _file_sha256(INDEX_CSV),
            "digits_checkpoint_sha256": _file_sha256(digits_checkpoint)
            if cfg.init_digits else "",
            "n_train": len(train_crops),
            "n_val": len(val_crops),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "torch": torch.__version__,
        }
        (run_dir / "config.json").write_text(
            json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    last_epoch_s = 0.0
    for epoch in range(start_epoch, cfg.epochs):
        if max_minutes is not None and epoch > start_epoch:
            if time.time() - t_start + last_epoch_s * 1.1 > max_minutes * 60:
                log(f"остановка по --max-minutes перед эпохой {epoch}; продолжить: --resume")
                return 3
        te = time.time()
        model.train()
        sampler.set_epoch(epoch)
        loss_sum, n_batches = 0.0, 0
        for x, values in loader:
            loss, loss_value = number_loss(model, x.to(device), values)
            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()
            scheduler.step()
            loss_sum += loss_value
            n_batches += 1
        metrics = score_outputs(infer_outputs(model, val_x, device), val_values,
                                cfg.model.kind)
        last_epoch_s = time.time() - te
        row = {
            "epoch": epoch,
            "train_loss": round(loss_sum / max(1, n_batches), 5),
            "val_top1": round(metrics["top1"], 5),
            "val_top5": round(metrics["top5"], 5),
            "val_nll": round(metrics["nll"], 5),
            "val_mean_rank": round(metrics["mean_rank"], 4),
            "lr_head": f"{optimizer.param_groups[1]['lr']:.2e}",
            "seconds": round(last_epoch_s, 1),
        }
        history.append(row)
        key = (metrics["top1"], -metrics["nll"])
        improved = key > best_key
        if improved:
            best_key, best_epoch = key, epoch
        log(f"эпоха {epoch}: loss {row['train_loss']:.4f}, val top1 {row['val_top1']:.4f} "
            f"top5 {row['val_top5']:.4f} ранг {row['val_mean_rank']:.3f} "
            f"nll {row['val_nll']:.4f}, {last_epoch_s:.0f} с{' *' if improved else ''}")
        done = epoch + 1 >= cfg.epochs or epoch - best_epoch >= cfg.patience
        ckpt = {"config": cfg.to_dict(), "model": model.state_dict(), "epoch": epoch,
                "val_metrics": metrics}
        if improved:
            torch.save(ckpt, run_dir / "best.pt")
        torch.save(ckpt | {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "history": history,
            "best_key": list(best_key),
            "best_epoch": best_epoch,
            "rng_cpu": torch.get_rng_state(),
            "done": done,
        }, last_path)
        write_history(run_dir / "history.csv", history)
        if done:
            if epoch + 1 < cfg.epochs:
                log(f"ранняя остановка: {cfg.patience} эпох без улучшения")
            break
    log(f"готово: лучшая эпоха {best_epoch}, val top1 {best_key[0]:.4f}")
    return 0


# --- предсказания, сравнение, promote ------------------------------------------------------


def build_predictions(crops: Sequence[NumberCrop], outputs: np.ndarray, kind: str,
                      source: str) -> list[ScanPrediction]:
    """Строки формата T13: поле `voucher_number` с top-5 номеров на скан."""
    preds = []
    for crop, out in zip(crops, outputs, strict=True):
        nums, logp = number_log_probs(out, kind=kind)
        order = np.argsort(-logp, kind="stable")[:TOP_K]
        cands: list[tuple[int, float | None]] = [
            (int(nums[i]), float(min(1.0, np.exp(logp[i])))) for i in order
        ]
        preds.append(ScanPrediction(scan_id=crop.scan_id, source=source,
                                    fields={VOUCHER_NUMBER: cands}))
    return preds


def predict(run_dir: Path, split: str, *, source: str | None = None) -> Path:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg, _ = load_model(run_dir, device)
    crops = number_crops(split, with_truth=False)
    outputs = infer_outputs(model, crop_tensors(crops, cfg.target_size), device)
    kind = cfg.model.kind
    out = run_dir / f"{split}_predictions.jsonl"
    n = write_predictions(out, build_predictions(crops, outputs, kind, source or run_dir.name))
    np.savez_compressed(run_dir / f"{split}_outputs.npz",
                        scan_id=np.array([c.scan_id for c in crops]), output=outputs,
                        kind=np.array(kind))
    print(f"{split}: {n} сканов → {out}")
    return out


def stress_tensors(crops: Sequence[NumberCrop], size: tuple[int, int], copies: int,
                   *, seed: int = 12_345) -> tuple[torch.Tensor, list[int]]:
    """`copies` аугментированных копий каждого кропа (аугментации T15) и их истина.

    Стресс-тест устойчивости на `val`: 51 кроп — слишком мало, чтобы различить модели
    по точности, а копии с искажениями (сдвиг, поворот, толщина штриха, шум) показывают,
    насколько модель держит ошибки выравнивания и качество скана.
    """
    images, values = [], []
    for i, crop in enumerate(crops):
        base = read_gray(crop.path)
        for k in range(copies):
            images.append(to_tensor(augment_crop(base, seed + i * 1_009 + k), size))
            values.append(int(crop.value or 0))
    return torch.stack(images), values


def compare(run_dirs: Sequence[Path], *, split: str = "val", stress: int = 0) -> str:
    """Таблица метрик запусков на сплите (markdown), со срезами «печатный / рукописный».

    `stress > 0` — дополнительно top-1 и средний ранг на `stress` аугментированных копиях
    каждого кропа (:func:`stress_tensors`).
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    crops = number_crops(split)
    flags = load_printed_flags(paths.PRINTED_FLAGS_CSV)
    kinds = ["printed" if flags.get((c.scan_id, VOUCHER_NUMBER)) else "handwritten"
             if (c.scan_id, VOUCHER_NUMBER) in flags else "unknown" for c in crops]
    stress_head = f" Стресс ×{stress}: top-1 | Стресс: ранг |" if stress else ""
    lines = [
        f"| Запуск | Подход | {split} n | top-1 | top-5 | Средний ранг (±{RANK_RADIUS}) | "
        f"Ранг 1 | NLL | top-1 печ. | top-1 рук. |{stress_head}",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|" + ("---:|---:|" if stress else ""),
    ]
    for run_dir in run_dirs:
        model, cfg, _ = load_model(run_dir, device)
        outputs = infer_outputs(model, crop_tensors(crops, cfg.target_size), device)
        values = [int(c.value) for c in crops if c.value is not None]
        m = score_outputs(outputs, values, cfg.model.kind)
        sliced = {}
        for kind in ("printed", "handwritten"):
            idx = [i for i, k in enumerate(kinds) if k == kind]
            s = score_outputs(outputs[idx], [values[i] for i in idx], cfg.model.kind)
            sliced[kind] = f"{s['top1']:.2%} ({round(s['top1'] * s['n'])}/{s['n']})"
        stress_cells = ""
        if stress:
            sx, sv = stress_tensors(crops, cfg.target_size, stress)
            sm = score_outputs(infer_outputs(model, sx, device), sv, cfg.model.kind)
            stress_cells = f" {sm['top1']:.2%} | {sm['mean_rank']:.3f} |"
        lines.append(
            f"| {run_dir.name} | {cfg.model.kind} | {m['n']} | {m['top1']:.2%} | "
            f"{m['top5']:.2%} | {m['mean_rank']:.3f} | {m['rank1']:.2%} | {m['nll']:.3f} | "
            f"{sliced['printed']} | {sliced['handwritten']} |{stress_cells}"
        )
    return "\n".join(lines)


def promote(run: str, *, runs_dir: Path = RUNS_DIR, dest: Path = V0_DIR) -> None:
    src = runs_dir / run
    paths.ensure_dir(dest)
    for name in ("best.pt", "config.json", "history.csv", "train.log"):
        shutil.copy2(src / name, dest / name)
    print(f"{run} → {dest}")


# --- CLI -----------------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m ocr_lab.train_number", description=__doc__,
                                     formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="cmd", required=True)

    p_train = sub.add_parser("train", help="обучить запуск")
    p_train.add_argument("--run", required=True)
    p_train.add_argument("--kind", choices=KINDS, required=True)
    p_train.add_argument("--synth", type=int, default=0, help="синтетических номеров на эпоху")
    p_train.add_argument("--real-repeats", type=int, default=2,
                         help="сколько раз за эпоху показывать реальные кропы")
    p_train.add_argument("--epochs", type=int, default=40)
    p_train.add_argument("--batch-size", type=int, default=32)
    p_train.add_argument("--lr-head", type=float, default=1e-3)
    p_train.add_argument("--lr-backbone", type=float, default=3e-4)
    p_train.add_argument("--patience", type=int, default=12)
    p_train.add_argument("--seed", type=int, default=0)
    p_train.add_argument("--workers", type=int, default=2)
    p_train.add_argument("--no-augment", action="store_true")
    p_train.add_argument("--no-init", action="store_true",
                         help="не брать бэкбон из digits_v0 (случайная инициализация)")
    p_train.add_argument("--max-minutes", type=float, default=None)
    p_train.add_argument("--resume", action="store_true")
    p_train.add_argument("--runs-dir", type=Path, default=RUNS_DIR)

    p_pred = sub.add_parser("predict", help="предсказания в формате T13 и сырые выходы")
    p_pred.add_argument("--run-dir", type=Path, default=V0_DIR)
    p_pred.add_argument("--split", choices=("val", "test", "train"), required=True)
    p_pred.add_argument("--source", default=None)

    p_cmp = sub.add_parser("compare", help="таблица метрик запусков")
    p_cmp.add_argument("--run-dirs", type=Path, nargs="+", required=True)
    p_cmp.add_argument("--split", choices=("val", "test"), default="val")
    p_cmp.add_argument("--stress", type=int, default=0,
                       help="аугментированных копий каждого кропа для стресс-теста")
    p_cmp.add_argument("--out", type=Path, default=None)

    p_prom = sub.add_parser("promote", help="скопировать запуск в number_v0")
    p_prom.add_argument("--run", required=True)
    p_prom.add_argument("--runs-dir", type=Path, default=RUNS_DIR)

    args = parser.parse_args(argv)
    if args.cmd == "train":
        cfg = TrainConfig(
            run=args.run, model=NumberModelConfig(kind=args.kind), epochs=args.epochs,
            batch_size=args.batch_size, lr_head=args.lr_head, lr_backbone=args.lr_backbone,
            patience=args.patience, seed=args.seed, augment=not args.no_augment,
            real_repeats=args.real_repeats, synth=args.synth, init_digits=not args.no_init,
            num_workers=args.workers,
        )
        return train(cfg, runs_dir=args.runs_dir, max_minutes=args.max_minutes,
                     resume=args.resume)
    if args.cmd == "predict":
        predict(args.run_dir, args.split, source=args.source)
    elif args.cmd == "compare":
        table = compare(args.run_dirs, split=args.split, stress=args.stress)
        print(table)
        if args.out is not None:
            paths.ensure_dir(args.out.parent)
            args.out.write_text(table + "\n", encoding="utf-8")
    elif args.cmd == "promote":
        promote(args.run, runs_dir=args.runs_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
