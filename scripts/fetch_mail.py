"""Командная строка для приёма заявок и ваучеров по IMAP.

Читает настройки из окружения (`.env`, префикс `APP_IMAP_*`), подключается к
почте, забирает непрочитанные письма-заявки из папки `APP_IMAP_APPLICATION_FOLDER`
и ваучеры/сканы из `APP_IMAP_VOUCHER_FOLDER`, сохраняет их в БД.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from app.database import SessionLocal, init_db  # noqa: E402
from app.services.imap_ingest import fetch_new_applications, fetch_new_vouchers  # noqa: E402


def _print_summary(label: str, summary: object) -> None:
    print(
        f"{label}: "
        f"получено — {summary.fetched}, "
        f"создано — {summary.created}, "
        f"пропущено дублей — {summary.skipped_duplicates}, "
        f"сохранено вложений — {summary.attachments_saved}"
    )
    for err in summary.errors:
        print(f"  ошибка: {err}")


def main() -> None:
    init_db()
    with SessionLocal() as db:
        app_summary = fetch_new_applications(db)
        voucher_summary = fetch_new_vouchers(db)

    _print_summary("Заявки", app_summary)
    _print_summary("Ваучеры", voucher_summary)


if __name__ == "__main__":
    main()
