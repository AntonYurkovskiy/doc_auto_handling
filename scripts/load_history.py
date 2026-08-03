"""Командная строка для загрузки исторического CSV."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import (  # noqa: E402
    Base,
    SessionLocal,
    init_db,
)
from app.database import engine as default_engine  # noqa: E402
from app.services.history_loader import load_history  # noqa: E402


def main() -> None:
    """Прочитать CSV, загрузить его в БД и напечатать сводку."""
    parser = argparse.ArgumentParser(description="Загрузка истории судозаходов")
    parser.add_argument("csv_path", type=Path)
    parser.add_argument("--db-url", default=None)
    args = parser.parse_args()

    if args.db_url:
        connect_args = {"check_same_thread": False} if args.db_url.startswith("sqlite") else {}
        selected_engine = create_engine(args.db_url, connect_args=connect_args)
        Base.metadata.create_all(bind=selected_engine)
        session_factory = sessionmaker(bind=selected_engine, autoflush=False, autocommit=False)
    else:
        init_db()
        selected_engine = default_engine
        session_factory = SessionLocal

    with args.csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        rows = csv.DictReader(file)
        with session_factory() as db:
            summary = load_history(db, rows)

    print(
        "Сводка загрузки: "
        f"создано судов — {summary['vessels_created']}, "
        f"обновлено судов — {summary['vessels_updated']}, "
        f"создано судозаходов — {summary['portcalls_created']}, "
        f"пропущено — {summary['skipped']}"
    )


if __name__ == "__main__":
    main()
