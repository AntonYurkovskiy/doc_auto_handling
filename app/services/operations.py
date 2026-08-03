"""Правила, относящиеся к операциям судозахода."""

from __future__ import annotations


def escort_likely(draft_m: float | None) -> bool:
    """Вернуть подсказку сопровождения для осадки 9.3–9.6 м включительно."""
    return draft_m is not None and 9.3 <= draft_m <= 9.6


def recommended_tug_count(loa_m: float | None) -> int | None:
    """Рекомендация буксиров по LOA; пороги настраиваемы в будущем."""
    if loa_m is None:
        return None
    if loa_m < 120:
        return 1
    if loa_m <= 160:
        return 2
    return 3
