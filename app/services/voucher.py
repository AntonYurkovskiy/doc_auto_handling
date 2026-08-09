"""Разбор ваучеров-сканов (заглушка под классификацию регионов).

Ваучеры приходят как скан-картинки (CamScanner). Печатные поля (буксир, судно,
агент, вид работ) впечатаны в фиксированный бланк; рукописные (№, даты/времена,
remarks) вписаны от руки.

План:
  * применить фиксированный шаблон регионов;
  * получить кандидатов из заявок и истории;
  * классифицировать изображение каждого региона по малому множеству вариантов;
  * сохранить предсказания и ручное подтверждение отдельно.

Свободный OCR не является основным способом распознавания рукописных полей.
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
