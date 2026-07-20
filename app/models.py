"""Модели БД.

Схема отражает итоговую таблицу корп-программы (matching.xls) и промежуточные
сущности пайплайна: заявка -> ваучер -> сопоставленная работа.
"""

from __future__ import annotations

import enum
from datetime import datetime

from sqlalchemy import DateTime, Enum, Float, ForeignKey, Integer, String, Text
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


class DocStatus(str, enum.Enum):
    """Статусы жизненного цикла документа/записи."""

    new = "new"                # только загружен
    parsed = "parsed"          # поля извлечены
    needs_review = "needs_review"  # требует ручной проверки
    confirmed = "confirmed"    # подтверждён оператором
    matched = "matched"        # сопоставлен (заявка<->ваучер)
    calculated = "calculated"  # выполнен расчёт
    done = "done"              # финальная запись
    error = "error"
    cancelled = "cancelled"


class Direction(str, enum.Enum):
    entry = "вход"
    exit = "выход"
    other = "прочее"


class Tug(Base):
    """Буксир."""

    __tablename__ = "tugs"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(100), unique=True)
    code: Mapped[str | None] = mapped_column(String(10), nullable=True)  # k / p


class Agent(Base):
    __tablename__ = "agents"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)


class Ship(Base):
    __tablename__ = "ships"

    id: Mapped[int] = mapped_column(primary_key=True)
    name: Mapped[str] = mapped_column(String(200), unique=True)
    imo: Mapped[str | None] = mapped_column(String(20), nullable=True)


class Application(Base):
    """Заявка на выполнение работ (из письма/PDF)."""

    __tablename__ = "applications"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[DocStatus] = mapped_column(Enum(DocStatus), default=DocStatus.new)

    source: Mapped[str | None] = mapped_column(String(20), nullable=True)  # email / pdf / manual
    sender: Mapped[str | None] = mapped_column(String(200), nullable=True)
    subject: Mapped[str | None] = mapped_column(String(500), nullable=True)
    received_at: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    direction: Mapped[Direction] = mapped_column(Enum(Direction), default=Direction.other)
    vessel_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    imo: Mapped[str | None] = mapped_column(String(20), nullable=True)
    agent: Mapped[str | None] = mapped_column(String(200), nullable=True)
    gross_tonnage: Mapped[int | None] = mapped_column(Integer, nullable=True)
    net_tonnage: Mapped[int | None] = mapped_column(Integer, nullable=True)

    entry_datetime: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    exit_datetime: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    destination: Mapped[str | None] = mapped_column(String(300), nullable=True)
    tugs_text: Mapped[str | None] = mapped_column(String(300), nullable=True)

    raw_text: Mapped[str | None] = mapped_column(Text, nullable=True)
    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    works: Mapped[list[Work]] = relationship(back_populates="application")


class Voucher(Base):
    """Ваучер (наряд), подтверждающий фактическое выполнение работ."""

    __tablename__ = "vouchers"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[DocStatus] = mapped_column(Enum(DocStatus), default=DocStatus.new)

    number: Mapped[str | None] = mapped_column(String(50), nullable=True)  # № ваучера
    tug_id: Mapped[int | None] = mapped_column(ForeignKey("tugs.id"), nullable=True)
    vessel_name: Mapped[str | None] = mapped_column(String(200), nullable=True)
    agent: Mapped[str | None] = mapped_column(String(200), nullable=True)
    work_type: Mapped[str | None] = mapped_column(String(200), nullable=True)  # Order / вид работ

    left_base_dt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    arrived_base_dt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    started_dt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_dt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    remarks: Mapped[str | None] = mapped_column(Text, nullable=True)
    joint_with: Mapped[str | None] = mapped_column(String(300), nullable=True)

    is_ice: Mapped[bool] = mapped_column(default=False)  # ледовые условия (ручной флажок)
    escort_hours: Mapped[float | None] = mapped_column(Float, nullable=True)  # часы сопровождения

    file_path: Mapped[str | None] = mapped_column(String(500), nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    tug: Mapped[Tug | None] = relationship()
    works: Mapped[list[Work]] = relationship(back_populates="voucher")


class Work(Base):
    """Итоговая запись о выполненной работе (заявка + ваучер + расчёт).

    Соответствует строке таблицы matching.xls.
    """

    __tablename__ = "works"

    id: Mapped[int] = mapped_column(primary_key=True)
    status: Mapped[DocStatus] = mapped_column(Enum(DocStatus), default=DocStatus.matched)

    application_id: Mapped[int | None] = mapped_column(ForeignKey("applications.id"), nullable=True)
    voucher_id: Mapped[int | None] = mapped_column(ForeignKey("vouchers.id"), nullable=True)

    tug_id: Mapped[int | None] = mapped_column(ForeignKey("tugs.id"), nullable=True)
    object_name: Mapped[str | None] = mapped_column(String(200), nullable=True)  # судно/причал
    work_type: Mapped[str | None] = mapped_column(String(200), nullable=True)
    agent: Mapped[str | None] = mapped_column(String(200), nullable=True)

    left_base_dt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    arrived_base_dt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    started_dt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)
    finished_dt: Mapped[datetime | None] = mapped_column(DateTime, nullable=True)

    gross_tonnage: Mapped[int | None] = mapped_column(Integer, nullable=True)

    amount: Mapped[float | None] = mapped_column(Float, nullable=True)      # Сумма
    currency: Mapped[str | None] = mapped_column(String(20), nullable=True)
    cbr_rate: Mapped[float | None] = mapped_column(Float, nullable=True)    # Курс
    revenue_rub: Mapped[float | None] = mapped_column(Float, nullable=True)  # Выручка, руб
    calc_note: Mapped[str | None] = mapped_column(Text, nullable=True)      # Примечания (формула)

    is_ice: Mapped[bool] = mapped_column(default=False)
    escort_hours: Mapped[float | None] = mapped_column(Float, nullable=True)

    created_at: Mapped[datetime] = mapped_column(DateTime, default=datetime.utcnow)

    application: Mapped[Application | None] = relationship(back_populates="works")
    voucher: Mapped[Voucher | None] = relationship(back_populates="works")
    tug: Mapped[Tug | None] = relationship()
