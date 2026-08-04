"""Тесты приёма заявок по IMAP (без сети)."""

from __future__ import annotations

from email.message import EmailMessage
from pathlib import Path

from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

from app.config import Settings
from app.database import Base
from app.models import Application, Direction
from app.services.imap_ingest import fetch_new_applications, ingest_messages


def _session():
    engine = create_engine("sqlite://", connect_args={"check_same_thread": False})
    Base.metadata.create_all(engine)
    return sessionmaker(bind=engine, autoflush=False, autocommit=False)()


_HTML = """
<html><body>
<table>
  <tr><td>Название судна</td><td>AURORA</td></tr>
  <tr><td>№ ИМО</td><td>1234567</td></tr>
  <tr><td>Брутто/нетто</td><td>2500 / 1200</td></tr>
  <tr><td>Дата/время входа</td><td>20.07.2026 в 04:00</td></tr>
</table>
<p>Заявка на вход судна</p>
</body></html>
"""


def _build_eml(message_id: str = "<msg-1@yandex.ru>", *, with_pdf: bool = True) -> bytes:
    msg = EmailMessage()
    msg["Message-ID"] = message_id
    msg["From"] = "agency@yandex.ru"
    msg["Subject"] = "Заявка на вход"
    msg["Date"] = "Mon, 20 Jul 2026 04:00:00 +0000"
    msg.set_content("Заявка на вход")
    msg.add_alternative(_HTML, subtype="html")
    if with_pdf:
        msg.add_attachment(
            b"%PDF-1.4 fake voucher",
            maintype="application",
            subtype="pdf",
            filename="voucher_1.pdf",
        )
    return msg.as_bytes()


def test_ingest_creates_application_and_saves_attachment(tmp_path: Path):
    db = _session()
    apps_dir = tmp_path / "apps"
    vouchers_dir = tmp_path / "vouchers"

    summary = ingest_messages(
        db, [_build_eml()], applications_dir=apps_dir, vouchers_dir=vouchers_dir
    )

    assert summary.fetched == 1
    assert summary.created == 1
    assert summary.attachments_saved == 1

    rows = db.query(Application).all()
    assert len(rows) == 1
    app_row = rows[0]
    assert app_row.vessel_name == "AURORA"
    assert app_row.imo == "1234567"
    assert app_row.gross_tonnage == 2500
    assert app_row.net_tonnage == 1200
    assert app_row.direction == Direction.entry
    assert app_row.source == "imap"
    assert app_row.message_id == "<msg-1@yandex.ru>"
    assert list(apps_dir.glob("*.eml"))
    assert list(vouchers_dir.glob("*.pdf"))


def test_ingest_is_idempotent_by_message_id(tmp_path: Path):
    db = _session()
    raw = _build_eml()
    kwargs = {"applications_dir": tmp_path / "a", "vouchers_dir": tmp_path / "v"}

    ingest_messages(db, [raw], **kwargs)
    summary = ingest_messages(db, [raw], **kwargs)

    assert summary.created == 0
    assert summary.skipped_duplicates == 1
    assert db.query(Application).count() == 1


class _FakeIMAP:
    """Минимальный fake IMAP-клиент для теста склейки без сети."""

    def __init__(self, raw_messages: list[bytes]):
        self._raw = raw_messages
        self.stored: list[bytes] = []

    def select(self, folder):  # noqa: ANN001
        return ("OK", [b"1"])

    def search(self, charset, *criteria):  # noqa: ANN001
        return ("OK", [b" ".join(str(i).encode() for i in range(1, len(self._raw) + 1))])

    def fetch(self, uid, parts):  # noqa: ANN001
        idx = int(uid) - 1
        return ("OK", [(b"1 (RFC822 {})", self._raw[idx])])

    def store(self, uid, flags, value):  # noqa: ANN001
        self.stored.append(uid)
        return ("OK", [b""])

    def logout(self):
        return ("BYE", [b""])


def test_fetch_new_applications_with_fake_connection(tmp_path: Path):
    db = _session()
    cfg = Settings(
        imap_host="imap.yandex.ru",
        imap_user="user@yandex.ru",
        imap_password="app-password",
        incoming_applications_dir=tmp_path / "apps",
        incoming_vouchers_dir=tmp_path / "vouchers",
    )
    fake = _FakeIMAP([_build_eml(with_pdf=False)])

    summary = fetch_new_applications(db, settings=cfg, connection=fake)

    assert summary.created == 1
    assert fake.stored == [b"1"]  # письмо помечено прочитанным
    assert db.query(Application).count() == 1
