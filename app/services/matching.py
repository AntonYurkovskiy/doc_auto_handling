"""Сопоставление заявки и ваучера.

Ключи (по требованию заказчика): судно + вид работ + дата.
Одна заявка может соответствовать НЕСКОЛЬКИМ ваучерам (несколько буксиров).
На старте — подсказка кандидатов со score; финальное решение подтверждает оператор.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

from app.models import Application, Voucher
from app.services.calculation import normalize_work_type


def _norm_name(name: str | None) -> str:
    if not name:
        return ""
    return "".join(ch for ch in name.lower() if ch.isalnum())


def _same_day(a: datetime | None, b: datetime | None, tolerance_days: int = 1) -> bool:
    if a is None or b is None:
        return False
    return abs((a.date() - b.date()).days) <= tolerance_days


@dataclass
class MatchCandidate:
    voucher: Voucher
    score: float
    reasons: list[str]


def score_match(application: Application, voucher: Voucher) -> MatchCandidate:
    """Оценить, насколько заявка и ваучер относятся к одной работе (0..1)."""
    score = 0.0
    reasons: list[str] = []

    if _norm_name(application.vessel_name) and _norm_name(
        application.vessel_name
    ) == _norm_name(voucher.vessel_name):
        score += 0.5
        reasons.append("совпадает судно")

    app_wt = normalize_work_type(_direction_to_work_type(application.direction))
    vch_wt = normalize_work_type(voucher.work_type)
    if app_wt and vch_wt and app_wt == vch_wt:
        score += 0.3
        reasons.append("совпадает вид работ")

    app_dt = application.entry_datetime or application.exit_datetime
    vch_dt = voucher.started_dt or voucher.finished_dt
    if _same_day(app_dt, vch_dt):
        score += 0.2
        reasons.append("совпадает дата")

    return MatchCandidate(voucher=voucher, score=round(score, 3), reasons=reasons)


def _direction_to_work_type(direction: str | None) -> str | None:
    if direction == "вход":
        return "швартовка"
    if direction == "выход":
        return "отшвартовка"
    return direction


def find_candidates(
    application: Application, vouchers: list[Voucher], min_score: float = 0.5
) -> list[MatchCandidate]:
    """Вернуть отсортированные по score кандидаты-ваучеры для заявки."""
    candidates = [score_match(application, v) for v in vouchers]
    candidates = [c for c in candidates if c.score >= min_score]
    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates
