"""T16: CNN для двузначных подполей — одна модель на день, месяц, часы и минуты.

Архитектура (`DigitNet`):

- **бэкбон** — torchvision `resnet18` или `mobilenet_v3_small` с весами ImageNet. Вход серый
  (1 канал): веса первой свёртки суммируются по трём каналам RGB, поэтому на сером
  изображении первая свёртка даёт тот же отклик, что на RGB-картинке с тремя одинаковыми
  каналами. Нормализация (среднее/СКО ImageNet, усреднённые по каналам) — внутри модели,
  на вход подаётся канва `0..1` из `ocr_lab.dataset.fit_with_aspect`. Веса скачиваются в
  `TORCH_HOME` (`ocr_lab.paths.configure_model_caches`, на ПК — E:);
- **пулинг** — `AdaptiveAvgPool2d((1, POOL_W))`, а не глобальный: десятки и единицы
  различаются положением (слева/справа), глобальное усреднение это положение стирает;
- **головы** — `tens` (11 классов, 10 = «пусто») и `units` (10 классов). Опционально перед
  головами приклеивается embedding `part_id` (день/месяц/час/минута + «неизвестно» для
  синтетики MNIST).

Функция потерь — :func:`digit_loss`: кросс-энтропия единиц плюс для десятков либо
кросс-энтропия (значение ≥ 10), либо маргинальное правдоподобие
`-logsumexp(logp[0], logp[empty])` (значение < 10 — истина не знает, «09» это или «9»).

Распределение по значениям — :mod:`ocr_lab.digit_values` (numpy, без torch).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, fields
from typing import Any

import torch
import torch.nn.functional as F
from torch import nn

from ocr_lab.digit_values import NUM_TENS_CLASSES, NUM_UNITS_CLASSES, TENS_EMPTY

BACKBONES: tuple[str, ...] = ("resnet18", "mobilenet_v3_small")
#: Число частей строки (день/месяц/час/минута) и индекс «неизвестной» части (синтетика).
NUM_PARTS = 4
UNKNOWN_PART = NUM_PARTS
#: Ширина карты признаков после пулинга: позиционная информация «слева/справа».
POOL_W = 4
#: Среднее и СКО ImageNet, усреднённые по каналам RGB (для серого входа).
GRAY_MEAN = 0.449
GRAY_STD = 0.226


@dataclass(frozen=True)
class ModelConfig:
    """Гиперпараметры архитектуры — сохраняются в чекпоинт и `config.json`."""

    backbone: str = "resnet18"
    part_embed: bool = True
    embed_dim: int = 16
    pool_w: int = POOL_W
    dropout: float = 0.2

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelConfig:
        names = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in data.items() if k in names})


def _gray_first_conv(conv: nn.Conv2d) -> nn.Conv2d:
    """Свёртка 3→C → 1→C с весами, просуммированными по входным каналам."""
    geometry: dict[str, Any] = {
        "kernel_size": conv.kernel_size, "stride": conv.stride, "padding": conv.padding,
        "dilation": conv.dilation,
    }
    new = nn.Conv2d(1, conv.out_channels, groups=conv.groups, bias=conv.bias is not None,
                    **geometry)
    with torch.no_grad():
        new.weight.copy_(conv.weight.sum(dim=1, keepdim=True))
        if conv.bias is not None and new.bias is not None:
            new.bias.copy_(conv.bias)
    return new


def build_backbone(name: str, *, pretrained: bool = True) -> tuple[nn.Module, int]:
    """Свёрточная часть бэкбона (без пулинга и классификатора) и число её каналов.

    `torchvision` импортируется лениво: он есть только в `.venv-train`, а модуль
    импортируют и тесты из `.venv`.
    """
    from torchvision import models as tvm  # noqa: PLC0415 — только для обучения

    if name == "resnet18":
        net = tvm.resnet18(weights=tvm.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None)
        net.conv1 = _gray_first_conv(net.conv1)
        features = nn.Sequential(
            net.conv1, net.bn1, net.relu, net.maxpool,
            net.layer1, net.layer2, net.layer3, net.layer4,
        )
        return features, 512
    if name == "mobilenet_v3_small":
        mnet = tvm.mobilenet_v3_small(
            weights=tvm.MobileNet_V3_Small_Weights.IMAGENET1K_V1 if pretrained else None
        )
        first = mnet.features[0][0]
        assert isinstance(first, nn.Conv2d)
        mnet.features[0][0] = _gray_first_conv(first)
        return mnet.features, 576
    raise ValueError(f"неизвестный бэкбон: {name} (есть {', '.join(BACKBONES)})")


class DigitNet(nn.Module):
    """Общая CNN для двузначных подполей: вход `(B, 1, H, W)` в `0..1`, `part_id` `(B,)`.

    `forward` возвращает сырые логиты `(tens (B, 11), units (B, 10))`.
    """

    def __init__(
        self,
        config: ModelConfig,
        *,
        pretrained: bool = True,
        features: nn.Module | None = None,
        channels: int | None = None,
    ) -> None:
        super().__init__()
        self.config = config
        if features is None:
            features, channels = build_backbone(config.backbone, pretrained=pretrained)
        assert channels is not None
        self.features = features
        self.pool = nn.AdaptiveAvgPool2d((1, config.pool_w))
        feat_dim = channels * config.pool_w
        self.part_embedding: nn.Embedding | None = None
        if config.part_embed:
            self.part_embedding = nn.Embedding(NUM_PARTS + 1, config.embed_dim)
            feat_dim += config.embed_dim
        self.dropout = nn.Dropout(config.dropout)
        self.head_tens = nn.Linear(feat_dim, NUM_TENS_CLASSES)
        self.head_units = nn.Linear(feat_dim, NUM_UNITS_CLASSES)

    def head_parameters(self) -> list[nn.Parameter]:
        """Параметры «новых» слоёв (головы и embedding) — у них свой, больший lr."""
        params = list(self.head_tens.parameters()) + list(self.head_units.parameters())
        if self.part_embedding is not None:
            params += list(self.part_embedding.parameters())
        return params

    def forward(
        self, x: torch.Tensor, part_id: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        x = (x - GRAY_MEAN) / GRAY_STD
        h = self.pool(self.features(x)).flatten(1)
        if self.part_embedding is not None:
            if part_id is None:
                part_id = torch.full((x.shape[0],), UNKNOWN_PART, device=x.device)
            pid = torch.where(part_id < 0, torch.full_like(part_id, UNKNOWN_PART), part_id)
            h = torch.cat([h, self.part_embedding(pid.long())], dim=1)
        h = self.dropout(h)
        return self.head_tens(h), self.head_units(h)


def digit_loss(
    tens_logits: torch.Tensor,
    units_logits: torch.Tensor,
    tens: torch.Tensor,
    units: torch.Tensor,
    ambiguous: torch.Tensor,
) -> tuple[torch.Tensor, dict[str, float]]:
    """Средняя по батчу потеря и её части (`tens`, `units`) для журнала.

    - `units` — кросс-энтропия;
    - `tens` при `ambiguous=False` (значение ≥ 10 или явно известные десятки) —
      кросс-энтропия, при `ambiguous=True` (значение < 10) — маргинальное правдоподобие
      `-logsumexp(logp[0], logp[empty])`: штрафа нет ни за «0», ни за «пусто», есть только
      за любую другую цифру десятков.
    """
    lt = F.log_softmax(tens_logits, dim=1)
    lu = F.log_softmax(units_logits, dim=1)
    units_nll = -lu.gather(1, units.long().view(-1, 1)).squeeze(1)
    tens_exact = -lt.gather(1, tens.long().view(-1, 1)).squeeze(1)
    tens_marg = -torch.logsumexp(lt[:, [0, TENS_EMPTY]], dim=1)
    tens_nll = torch.where(ambiguous.bool(), tens_marg, tens_exact)
    loss = (tens_nll + units_nll).mean()
    parts = {"tens": float(tens_nll.mean().detach()), "units": float(units_nll.mean().detach())}
    return loss, parts
