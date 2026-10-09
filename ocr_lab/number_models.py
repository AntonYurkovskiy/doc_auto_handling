"""T17: модели номера ваучера — подход А (три головы) и подход Б (CRNN + CTC).

Обе модели берут свёрточную часть у модели цифр T16 (`digits_v0`, ResNet18 с серым
входом, :func:`ocr_lab.models.build_backbone`) и дообучают её целиком: номеров всего
~620 в `train`, поэтому начинать с признаков, уже обученных на ~10 тыс. рукописных цифр
бланка, выгоднее, чем с ImageNet. Нормализация входа — как у `DigitNet` (внутри модели),
вход — канва `0..1` `(B, 1, H, W)`.

- :class:`NumberHeadsNet` (А) — пулинг `(1, pool_w)` как у `DigitNet` (положение слева/справа
  сохраняется), затем одна линейная голова на `3 × 11` логитов: сотни, десятки, единицы,
  у каждой класс «пусто» (:data:`ocr_lab.number_scores.EMPTY`). Выход — `(B, 3, 11)`.
- :class:`NumberCRNN` (Б) — у `layer3` и `layer4` ResNet18 шаг по ширине убран (`(2, 1)`
  вместо `(2, 2)`), поэтому по ширине остаётся `W / 8` кадров (22 при ширине канвы 176):
  для CTC нужно `T ≥ 2L − 1` кадров, и на одну цифру номера приходится ~5–7 кадров. Карта
  признаков усредняется по высоте, дальше BiLSTM и линейный слой на 11 классов (blank +
  10 цифр). Выход — логиты `(B, T, 11)` (log_softmax — в функции потерь и в
  :mod:`ocr_lab.number_scores`).

Функции потерь — :func:`heads_loss` (сумма кросс-энтропий трёх голов) и :func:`ctc_loss`.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ocr_lab.models import GRAY_MEAN, GRAY_STD, build_backbone
from ocr_lab.number_scores import CTC_BLANK, HEADS, KINDS, NUM_CLASSES, encode_ctc

#: Каналы на выходе `layer4` ResNet18.
RESNET18_CHANNELS = 512


@dataclass(frozen=True)
class NumberModelConfig:
    """Гиперпараметры архитектуры номера — сохраняются в чекпоинт и `config.json`."""

    kind: str = "heads"
    backbone: str = "resnet18"
    pool_w: int = 4
    dropout: float = 0.2
    lstm_hidden: int = 128
    lstm_layers: int = 2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> NumberModelConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _horizontal_stride_one(layer: nn.Module) -> None:
    """Убрать шаг по ширине у первого блока слоя ResNet (`(2, 2)` → `(2, 1)`)."""
    block = layer[0]  # type: ignore[index]
    block.conv1.stride = (2, 1)
    if block.downsample is not None:
        block.downsample[0].stride = (2, 1)


def build_features(config: NumberModelConfig) -> tuple[nn.Module, int]:
    """Свёрточная часть без весов (веса грузятся из `digits_v0`) и число её каналов."""
    if config.backbone != "resnet18":
        raise ValueError("модель номера поддерживает только resnet18 (бэкбон digits_v0)")
    features, channels = build_backbone(config.backbone, pretrained=False)
    if config.kind == "ctc":
        assert isinstance(features, nn.Sequential)
        _horizontal_stride_one(features[6])  # layer3
        _horizontal_stride_one(features[7])  # layer4
    return features, channels


class NumberHeadsNet(nn.Module):
    """Подход А: три головы (сотни, десятки, единицы) по 11 классов. Выход `(B, 3, 11)`."""

    def __init__(
        self, config: NumberModelConfig, *, features: nn.Module | None = None,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if features is None:
            features, channels = build_features(config)
        assert channels is not None
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d((1, config.pool_w))
        self.dropout = nn.Dropout(config.dropout)
        self.head = nn.Linear(channels * config.pool_w, len(HEADS) * NUM_CLASSES)

    def head_parameters(self) -> list[nn.Parameter]:
        """Параметры новых слоёв — у них свой, больший lr."""
        return list(self.head.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - GRAY_MEAN) / GRAY_STD
        h = self.dropout(self.pool(self.features(x)).flatten(1))
        return self.head(h).view(-1, len(HEADS), NUM_CLASSES)


class NumberCRNN(nn.Module):
    """Подход Б: свёрточная часть → кадры по ширине → BiLSTM → 11 классов CTC.

    Выход — логиты `(B, T, 11)`, класс 0 — blank.
    """

    def __init__(
        self, config: NumberModelConfig, *, features: nn.Module | None = None,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if features is None:
            features, channels = build_features(config)
        assert channels is not None
        self.features = features
        self.dropout = nn.Dropout(config.dropout)
        self.rnn = nn.LSTM(
            channels, config.lstm_hidden, num_layers=config.lstm_layers, bidirectional=True,
            batch_first=True, dropout=config.dropout if config.lstm_layers > 1 else 0.0,
        )
        self.head = nn.Linear(2 * config.lstm_hidden, NUM_CLASSES)

    def head_parameters(self) -> list[nn.Parameter]:
        return list(self.rnn.parameters()) + list(self.head.parameters())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = (x - GRAY_MEAN) / GRAY_STD
        f = self.features(x).mean(dim=2)  # (B, C, T)
        seq = self.dropout(f.transpose(1, 2))  # (B, T, C)
        out, _ = self.rnn(seq)
        return self.head(self.dropout(out))


NumberNet = NumberHeadsNet | NumberCRNN


def build_number_model(config: NumberModelConfig) -> NumberNet:
    if config.kind not in KINDS:
        raise ValueError(f"неизвестный вид модели номера: {config.kind}")
    return NumberHeadsNet(config) if config.kind == "heads" else NumberCRNN(config)


def load_digits_backbone(model: NumberNet, digits_checkpoint: Path) -> int:
    """Перенести веса свёрточной части из чекпоинта T16 (`digits_v0/best.pt`).

    Шаг свёрток на веса не влияет, поэтому веса подходят и для CRNN с изменённым шагом.
    Возвращает число перенесённых тензоров.
    """
    state = torch.load(digits_checkpoint, map_location="cpu", weights_only=False)
    prefix = "features."
    feats = {k[len(prefix):]: v for k, v in state["model"].items() if k.startswith(prefix)}
    model.features.load_state_dict(feats)
    return len(feats)


def heads_loss(logits: torch.Tensor, targets: torch.Tensor) -> tuple[torch.Tensor, float]:
    """Сумма кросс-энтропий трёх голов, среднее по батчу. `targets` — `(B, 3)`."""
    nll = F.cross_entropy(logits.reshape(-1, NUM_CLASSES), targets.reshape(-1).long(),
                          reduction="none").view(-1, len(HEADS)).sum(dim=1)
    loss = nll.mean()
    return loss, float(loss.detach())


def ctc_targets(values: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """Номера → склеенные метки CTC и их длины (вход `F.ctc_loss`)."""
    labels = [encode_ctc(int(v)) for v in values]
    flat = torch.tensor([c for seq in labels for c in seq], dtype=torch.long)
    lengths = torch.tensor([len(seq) for seq in labels], dtype=torch.long)
    return flat, lengths


def ctc_loss(logits: torch.Tensor, values: Sequence[int]) -> tuple[torch.Tensor, float]:
    """CTC-потеря (среднее по батчу NLL строки; не делится на длину строки)."""
    log_probs = F.log_softmax(logits, dim=2).transpose(0, 1)  # (T, B, C)
    flat, lengths = ctc_targets(values)
    n_frames = torch.full((logits.shape[0],), logits.shape[1], dtype=torch.long)
    nll = F.ctc_loss(log_probs, flat.to(logits.device), n_frames, lengths, blank=CTC_BLANK,
                     reduction="none", zero_infinity=True)
    loss = nll.mean()
    return loss, float(loss.detach())
