"""Производственный календарь РФ: рабочий день или выходной/праздник.

Основной источник — https://isdayoff.ru/YYYYMMDD (0 — рабочий, 1 — выходной/праздник).
При недоступности сети — откат на простое правило (суббота/воскресенье = выходной).
"""

from __future__ import annotations

from datetime import date

import requests

ISDAYOFF_URL = "https://isdayoff.ru/{ymd}"
_cache: dict[str, bool] = {}


def is_day_off(day: date, timeout: float = 5.0) -> bool:
    """True, если `day` — выходной или праздник."""
    key = day.isoformat()
    if key in _cache:
        return _cache[key]

    try:
        resp = requests.get(ISDAYOFF_URL.format(ymd=day.strftime("%Y%m%d")), timeout=timeout)
        if resp.status_code == 200 and resp.text.strip() in ("0", "1"):
            result = resp.text.strip() == "1"
            _cache[key] = result
            return result
    except requests.RequestException:
        pass

    # Откат: выходные — суббота (5) и воскресенье (6).
    result = day.weekday() >= 5
    _cache[key] = result
    return result
