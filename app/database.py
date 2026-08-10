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
    wanted = {
        "applications": [
            ("portcall_id", "INTEGER"),
            ("message_id", "VARCHAR(500)"),
            ("loa_m", "FLOAT"),
            ("draft_m", "FLOAT"),
            ("raw_html", "TEXT"),
        ],
        "operations": [
            ("work_start", "DATETIME"),
            ("work_end", "DATETIME"),
            ("is_ice", "BOOLEAN"),
            ("amount", "FLOAT"),
            ("currency", "VARCHAR(10)"),
            ("cbr_rate", "FLOAT"),
            ("revenue_rub", "FLOAT"),
            ("calc_note", "TEXT"),
            ("calculated_at", "DATETIME"),
        ],
        "operation_tugs": [
            ("is_external", "BOOLEAN DEFAULT 0"),
            ("display_name", "VARCHAR(200)"),
            ("work_start", "DATETIME"),
            ("work_end", "DATETIME"),
            ("voucher_id", "INTEGER"),
        ],
        "vouchers": [
            ("original_filename", "VARCHAR(255)"),
            ("content_type", "VARCHAR(100)"),
            ("sha256", "VARCHAR(64)"),
            ("template_id", "INTEGER"),
            ("application_id", "INTEGER"),
            ("operation_id", "INTEGER"),
            ("predicted_at", "DATETIME"),
            ("reviewed_at", "DATETIME"),
        ],
    }
    with engine.begin() as conn:
        for table, columns in wanted.items():
            if table not in existing:
                continue
            present = {col["name"] for col in inspector.get_columns(table)}
            for name, ddl_type in columns:
                if name not in present:
                    conn.execute(text(f'ALTER TABLE {table} ADD COLUMN {name} {ddl_type}'))


def _relax_operation_tug_id() -> None:
    """Сделать operation_tugs.tug_id NULL-able: у сторонних участников буксира нет.

    SQLite не умеет ALTER COLUMN, поэтому таблица пересоздаётся с переносом данных.
    """
    inspector = inspect(engine)
    if "operation_tugs" not in set(inspector.get_table_names()):
        return
    columns = inspector.get_columns("operation_tugs")
    tug_id = next((col for col in columns if col["name"] == "tug_id"), None)
    if tug_id is None or tug_id["nullable"]:
        return
    names = [col["name"] for col in columns if col["name"] != "id"]
    column_list = ", ".join(names)
    with engine.begin() as conn:
        conn.execute(text("ALTER TABLE operation_tugs RENAME TO operation_tugs_old"))
    Base.metadata.tables["operation_tugs"].create(bind=engine)
    with engine.begin() as conn:
        conn.execute(
            text(
                f"INSERT INTO operation_tugs (id, {column_list}) "
                f"SELECT id, {column_list} FROM operation_tugs_old"
            )
        )
        conn.execute(text("DROP TABLE operation_tugs_old"))


def init_db() -> None:
    """Создать таблицы, применить лёгкую миграцию и заполнить справочники."""
    from app import models  # noqa: F401  (регистрация моделей)
    from app.services.voucher_template import ensure_default_template

    Base.metadata.create_all(bind=engine)
    _add_missing_columns()
    _relax_operation_tug_id()
    with SessionLocal() as db:
        ensure_default_template(db)
