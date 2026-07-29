"""Тесты предсказания номера ваучера."""

from __future__ import annotations

from datetime import datetime

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.database import Base
from app.models import Tug, Voucher
from app.services.voucher_number import parse_voucher_number, predict_next


def test_parse_voucher_number():
    assert parse_voucher_number("262k(2).pdf") == (262, "k")
    assert parse_voucher_number("100p.pdf") == (100, "p")
    assert parse_voucher_number("123.pdf") == (123, None)
    assert parse_voucher_number("garbage.pdf") == (None, None)


def test_predict_next_is_independent_by_tug_and_year():
    engine = create_engine("sqlite:///:memory:")
    Base.metadata.create_all(engine)
    session_factory = sessionmaker(bind=engine)

    with session_factory() as db:
        tug_k = Tug(name="Туг K", code="k")
        tug_p = Tug(name="Туг P", code="p")
        db.add_all([tug_k, tug_p])
        db.flush()
        db.add_all(
            [
                Voucher(
                    tug_id=tug_k.id,
                    number="100k",
                    started_dt=datetime(2026, 1, 10),
                ),
                Voucher(
                    tug_id=tug_k.id,
                    number="262k(2).pdf",
                    started_dt=datetime(2026, 7, 10),
                ),
                Voucher(
                    tug_id=tug_k.id,
                    number="999k",
                    left_base_dt=datetime(2025, 12, 31),
                ),
                Voucher(
                    tug_id=tug_p.id,
                    number="700p",
                    started_dt=datetime(2026, 7, 10),
                ),
                Voucher(
                    tug_id=tug_k.id,
                    number="garbage",
                    started_dt=datetime(2026, 8, 10),
                ),
            ]
        )
        db.commit()

        assert predict_next(db, tug_k.id, 2026) == 263
        assert predict_next(db, tug_k.id, 2025) == 1000
        assert predict_next(db, tug_p.id, 2026) == 701
        assert predict_next(db, tug_k.id, 2024) == 1
