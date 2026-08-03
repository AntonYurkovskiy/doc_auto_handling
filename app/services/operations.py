"""Правила, относящиеся к операциям судозахода."""

from __future__ import annotations


def escort_likely(draft_m: float | None) -> bool:
    """Вернуть подсказку сопровождения для осадки 9.3–9.6 м включительно."""
    return draft_m is not None and 9.3 <= draft_m <= 9.6
