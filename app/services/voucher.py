"""Распознавание ваучеров-сканов (заглушка под будущие фазы).

Ваучеры приходят как скан-картинки (CamScanner). Печатные поля (буксир, судно,
агент, вид работ) впечатаны в фиксированный бланк; рукописные (№, даты/времена,
remarks) вписаны от руки.

План:
  * Фаза 3: резать бланк по фиксированным зонам и распознавать печатные поля
    (OCR: rapidocr/tesseract). Рукописные даты/времена — ручной ввод + проверка.
  * Фаза 6 (опц., локально): локальная vision-модель (Qwen2.5-VL через Ollama)
    для предзаполнения рукописных полей; оператор подтверждает (human-in-the-loop).

Пока функция возвращает пустой черновик — поля ваучера вводятся вручную в UI.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime


@dataclass
class ParsedVoucher:
    number: str | None = None
    tug_name: str | None = None
    vessel_name: str | None = None
    agent: str | None = None
    work_type: str | None = None
    left_base_dt: datetime | None = None
    arrived_base_dt: datetime | None = None
    started_dt: datetime | None = None
    finished_dt: datetime | None = None
    remarks: str | None = None
    joint_with: str | None = None


def parse_voucher(_path: str) -> ParsedVoucher:
    """Заглушка: авто-распознавание ваучеров появится в Фазе 3/6.

    Возвращает пустой черновик, чтобы оператор заполнил поля вручную.
    """
    return ParsedVoucher()
