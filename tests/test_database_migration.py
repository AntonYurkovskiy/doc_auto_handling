"""Тесты миграции БД (T26): новые колонки `vouchers.ocr_*` и `voucher_field_predictions.topk`.

`_add_missing_columns`/`init_db` работают с глобальным `app.database.engine` (в проекте
нет Alembic, см. `_common.md`), поэтому тесты подменяют его на временный файловый SQLite
со «старой» схемой (без колонок T26) и проверяют, что миграция добавляет недостающее и
второй прогон ничего не меняет (идемпотентность).
"""

from __future__ import annotations

from pathlib import Path

from sqlalchemy import create_engine, inspect, text
from sqlalchemy.orm import sessionmaker

import app.database as app_database

# Схема `vouchers`/`voucher_field_predictions` до T26 — без ocr_status/ocr_confidence/topk.
_LEGACY_VOUCHERS_SQL = """
CREATE TABLE vouchers (
    id INTEGER PRIMARY KEY,
    status VARCHAR(20),
    number VARCHAR(50),
    tug_id INTEGER,
    vessel_name VARCHAR(200),
    agent VARCHAR(200),
    work_type VARCHAR(200),
    left_base_dt DATETIME,
    arrived_base_dt DATETIME,
    started_dt DATETIME,
    finished_dt DATETIME,
    remarks TEXT,
    joint_with VARCHAR(300),
    is_ice BOOLEAN,
    escort_hours FLOAT,
    file_path VARCHAR(500),
    original_filename VARCHAR(255),
    content_type VARCHAR(100),
    sha256 VARCHAR(64),
    template_id INTEGER,
    application_id INTEGER,
    operation_id INTEGER,
    predicted_at DATETIME,
    reviewed_at DATETIME,
    created_at DATETIME
)
"""
_LEGACY_PREDICTIONS_SQL = """
CREATE TABLE voucher_field_predictions (
    id INTEGER PRIMARY KEY,
    voucher_id INTEGER NOT NULL,
    region_id INTEGER,
    field_name VARCHAR(50),
    predicted_value TEXT,
    predicted_normalized_value TEXT,
    confirmed_value TEXT,
    confidence FLOAT,
    source VARCHAR(30),
    confirmed_at DATETIME,
    created_at DATETIME
)
"""


def _legacy_engine(tmp_path: Path):
    engine = create_engine(f"sqlite:///{tmp_path / 'legacy.db'}")
    with engine.begin() as conn:
        conn.execute(text(_LEGACY_VOUCHERS_SQL))
        conn.execute(text(_LEGACY_PREDICTIONS_SQL))
    return engine


def _columns(engine, table: str) -> set[str]:
    return {col["name"] for col in inspect(engine).get_columns(table)}


def test_add_missing_columns_adds_t26_fields_on_old_schema(tmp_path, monkeypatch):
    engine = _legacy_engine(tmp_path)
    monkeypatch.setattr(app_database, "engine", engine)
    assert "ocr_status" not in _columns(engine, "vouchers")
    assert "topk" not in _columns(engine, "voucher_field_predictions")

    app_database._add_missing_columns()

    assert {"ocr_status", "ocr_confidence"} <= _columns(engine, "vouchers")
    assert "topk" in _columns(engine, "voucher_field_predictions")


def test_add_missing_columns_is_idempotent(tmp_path, monkeypatch):
    engine = _legacy_engine(tmp_path)
    monkeypatch.setattr(app_database, "engine", engine)

    app_database._add_missing_columns()
    before = {
        "vouchers": sorted(_columns(engine, "vouchers")),
        "voucher_field_predictions": sorted(_columns(engine, "voucher_field_predictions")),
    }

    app_database._add_missing_columns()  # второй раз не должен падать и ничего не меняет

    after = {
        "vouchers": sorted(_columns(engine, "vouchers")),
        "voucher_field_predictions": sorted(_columns(engine, "voucher_field_predictions")),
    }
    assert before == after


def test_init_db_on_legacy_schema_adds_columns_and_is_idempotent(tmp_path, monkeypatch):
    engine = _legacy_engine(tmp_path)
    session_factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    monkeypatch.setattr(app_database, "engine", engine)
    monkeypatch.setattr(app_database, "SessionLocal", session_factory)

    app_database.init_db()

    voucher_cols = _columns(engine, "vouchers")
    prediction_cols = _columns(engine, "voucher_field_predictions")
    assert {"ocr_status", "ocr_confidence"} <= voucher_cols
    assert "topk" in prediction_cols
    # Побочный эффект init_db — эталонный шаблон создан один раз.
    with session_factory() as db:
        from app.models import VoucherTemplate

        templates = db.query(VoucherTemplate).filter_by(name="baltiyskie_buksiry").all()
        assert len(templates) == 1

    snapshot_before = {
        "vouchers": sorted(voucher_cols),
        "voucher_field_predictions": sorted(prediction_cols),
    }

    app_database.init_db()  # второй прогон ничего не меняет

    snapshot_after = {
        "vouchers": sorted(_columns(engine, "vouchers")),
        "voucher_field_predictions": sorted(_columns(engine, "voucher_field_predictions")),
    }
    assert snapshot_before == snapshot_after
    with session_factory() as db:
        from app.models import VoucherTemplate

        templates = db.query(VoucherTemplate).filter_by(name="baltiyskie_buksiry").all()
        assert len(templates) == 1  # не задублировался
