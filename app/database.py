"""Настройка подключения к БД (SQLite с WAL) и сессий SQLAlchemy."""

from __future__ import annotations

from collections.abc import Iterator

from sqlalchemy import create_engine, event, inspect, text
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from app.config import DATA_DIR, settings

DATA_DIR.mkdir(parents=True, exist_ok=True)

_connect_args = {"check_same_thread": False} if settings.database_url.startswith("sqlite") else {}
engine = create_engine(settings.database_url, connect_args=_connect_args)


if settings.database_url.startswith("sqlite"):

    @event.listens_for(engine, "connect")
    def _set_sqlite_pragma(dbapi_connection, _connection_record):  # noqa: ANN001
        cursor = dbapi_connection.cursor()
        cursor.execute("PRAGMA journal_mode=WAL")
        cursor.execute("PRAGMA foreign_keys=ON")
        cursor.close()


SessionLocal = sessionmaker(bind=engine, autoflush=False, autocommit=False)


class Base(DeclarativeBase):
    pass


def get_db() -> Iterator[Session]:
    """Зависимость FastAPI: выдаёт сессию БД и гарантированно закрывает её."""
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


def _add_missing_columns() -> None:
    """Идемпотентно добавить недостающие колонки в существующие таблицы.

    Проект без Alembic: create_all создаёт новые таблицы, но не изменяет старые.
    Для SQLite добавляем недостающие колонки через ALTER TABLE.
    """
    inspector = inspect(engine)
    existing = set(inspector.get_table_names())
    wanted = {"applications": [("portcall_id", "INTEGER"), ("message_id", "VARCHAR(500)")]}
    with engine.begin() as conn:
        for table, columns in wanted.items():
            if table not in existing:
                continue
            present = {col["name"] for col in inspector.get_columns(table)}
            for name, ddl_type in columns:
                if name not in present:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {ddl_type}'))


def init_db() -> None:
    """Создать таблицы, применить лёгкую миграцию и заполнить справочники."""
    from app import models  # noqa: F401  (регистрация моделей)

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
