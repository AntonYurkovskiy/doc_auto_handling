"""Курс валют ЦБ РФ.

Источник: https://www.cbr.ru/scripts/XML_daily.asp?date_req=DD/MM/YYYY
Бесплатно, без ключа. Возвращает курс валюты к рублю на заданную дату.
"""

from __future__ import annotations

import xml.etree.ElementTree as ET
from datetime import date

import requests

CBR_URL = "https://www.cbr.ru/scripts/XML_daily.asp"
_CURRENCY_CODES = {"USD": "R01235", "EUR": "R01239"}
_cache: dict[tuple[str, str], float] = {}


def get_cbr_rate(currency: str, on_date: date, timeout: float = 10.0) -> float:
    """Курс `currency` к рублю на дату `on_date`.

    Для RUB возвращает 1.0. Результат = Value / Nominal (курс за единицу валюты).
    Кидает исключение, если данные недоступны — вызывающий код решает, что делать.
    """
    if currency.upper() in ("RUB", "РУБ", "РУБЛЬ"):
        return 1.0

    code = _CURRENCY_CODES.get(currency.upper())
    if code is None:
        raise ValueError(f"Неизвестный код валюты для ЦБ: {currency}")

    key = (code, on_date.isoformat())
    if key in _cache:
        return _cache[key]

    resp = requests.get(
        CBR_URL,
        params={"date_req": on_date.strftime("%d/%m/%Y")},
        timeout=timeout,
    )
    resp.raise_for_status()
    root = ET.fromstring(resp.content)

    for valute in root.findall("Valute"):
        if valute.findtext("CharCode") == currency.upper():
            nominal = int((valute.findtext("Nominal") or "1").replace(" ", ""))
            value = float((valute.findtext("Value") or "0").replace(",", ".").replace(" ", ""))
            rate = value / nominal
            _cache[key] = rate
            return rate

    raise LookupError(f"Курс {currency} на {on_date} не найден в ответе ЦБ")
