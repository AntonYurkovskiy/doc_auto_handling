"""Командная строка для приёма заявок по IMAP.

Читает настройки из окружения (`.env`, префикс `APP_IMAP_*`), подключается к
почте, забирает непрочитанные письма-заявки и сохраняет их в БД.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal, init_db  # noqa: E402
from app.services.imap_ingest import fetch_new_applications  # noqa: E402


def main() -> None:
    init_db()
    with SessionLocal() as db:
        summary = fetch_new_applications(db)

    print(
        "Приём почты: "
        f"получено — {summary.fetched}, "
        f"создано заявок — {summary.created}, "
        f"пропущено дублей — {summary.skipped_duplicates}, "
        f"сохранено вложений — {summary.attachments_saved}"
    )
    for err in summary.errors:
        print(f"  ошибка: {err}")


if __name__ == "__main__":
    main()
