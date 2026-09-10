"""Приём заявок и ваучеров по IMAP.

Забирает непрочитанные письма (по умолчанию с Яндекса, `imap.yandex.ru:993`,
SSL), сохраняет оригинал `.eml`, разбирает заявку тем же парсером, что и
ручная загрузка, создаёт предварительную запись `Application` (статус
`needs_review`) и сохраняет вложения (PDF/скан) в каталог ваучеров.
Также поддерживает отдельную IMAP-папку для ваучеров/сканов.

Логика приёма отделена от IMAP-транспорта: чистая функция `ingest_messages`
работает над «сырыми» письмами (bytes) и полностью тестируется без сети.
"""

from __future__ import annotations

import base64
import email
import hashlib
import imaplib
import io
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from datetime import datetime
from email import policy
from email.message import EmailMessage
from pathlib import Path
from typing import cast

from sqlalchemy.orm import Session

from app.config import Settings
from app.config import settings as default_settings
from app.models import Agent, Application, Direction, DocStatus, Voucher
from app.services.application_parser import parse_eml
from app.services.vessels import ensure_vessel
from app.services.voucher import predict_and_store
from app.services.voucher_files import ALLOWED_SUFFIXES, store_upload
from app.services.voucher_template import ensure_default_template

_ATTACHMENT_SUFFIXES = {".pdf", ".jpg", ".jpeg", ".png", ".tif", ".tiff"}
_NON_FILENAME = re.compile(r"[^\w.\- ]+")


@dataclass
class IngestSummary:
    """Итог приёма почты."""

    fetched: int = 0
    created: int = 0
    skipped_duplicates: int = 0
    attachments_saved: int = 0
    errors: list[str] = field(default_factory=list)

    def as_dict(self) -> dict[str, int | list[str]]:
        return {
            "fetched": self.fetched,
            "created": self.created,
            "skipped_duplicates": self.skipped_duplicates,
            "attachments_saved": self.attachments_saved,
            "errors": self.errors,
        }


def _load_message(raw: bytes) -> EmailMessage:
    return cast(
        EmailMessage,
        email.message_from_bytes(raw, policy=policy.default),  # type: ignore[arg-type]
    )


def _safe_filename(name: str, fallback: str) -> str:
    cleaned = _NON_FILENAME.sub("_", name).strip("._ ")
    return cleaned or fallback


def _save_attachments(msg: EmailMessage, voucher_dir: Path, stem: str) -> int:
    saved = 0
    for idx, part in enumerate(msg.walk()):
        if part.get_content_maintype() == "multipart":
            continue
        if part.get_content_disposition() != "attachment":
            continue
        filename = part.get_filename()
        suffix = Path(filename).suffix.lower() if filename else ""
        if suffix not in _ATTACHMENT_SUFFIXES:
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue
        target_name = _safe_filename(filename or f"{stem}_{idx}{suffix}", f"{stem}_{idx}{suffix}")
        target = voucher_dir / f"{stem}__{target_name}"
        target.write_bytes(payload)
        saved += 1
    return saved


def ingest_messages(
    db: Session,
    raw_messages: Iterable[bytes],
    *,
    applications_dir: Path,
    vouchers_dir: Path,
) -> IngestSummary:
    """Разобрать и сохранить письма-заявки. Идемпотентно по Message-ID.

    Возвращает сводку. Письма без Message-ID сохраняются всегда (без дедупа).
    """
    applications_dir.mkdir(parents=True, exist_ok=True)
    vouchers_dir.mkdir(parents=True, exist_ok=True)

    summary = IngestSummary()
    known_agents = [row.name for row in db.query(Agent).all()]
    for raw in raw_messages:
        summary.fetched += 1
        msg = _load_message(raw)
        message_id = msg.get("message-id")

        if message_id:
            exists = (
                db.query(Application).filter(Application.message_id == message_id).first()
                is not None
            )
            if exists:
                summary.skipped_duplicates += 1
                continue

        stem = _safe_filename(
            message_id or datetime.utcnow().strftime("%Y%m%d%H%M%S%f"),
            "message",
        )
        eml_path = applications_dir / f"{stem}.eml"
        eml_path.write_bytes(raw)

        parsed = parse_eml(eml_path, known_agents=known_agents)
        direction = (
            Direction(parsed.direction)
            if parsed.direction in Direction._value2member_map_
            else Direction.other
        )
        app_row = Application(
            status=DocStatus.needs_review,
            source="imap",
            message_id=message_id,
            sender=parsed.sender,
            subject=parsed.subject,
            received_at=parsed.received_at,
            direction=direction,
            vessel_name=parsed.vessel_name,
            imo=parsed.imo,
            gross_tonnage=parsed.gross_tonnage,
            net_tonnage=parsed.net_tonnage,
            loa_m=parsed.loa_m,
            draft_m=parsed.draft_m,
            entry_datetime=parsed.entry_datetime,
            exit_datetime=parsed.exit_datetime,
            destination=parsed.destination,
            agent=parsed.agent,
            tugs_text=parsed.tugs_text,
            raw_text=parsed.raw_text,
            raw_html=parsed.raw_html,
            file_path=str(eml_path),
        )
        db.add(app_row)
        summary.created += 1
        summary.attachments_saved += _save_attachments(msg, vouchers_dir, stem)
        ensure_vessel(db, parsed.vessel_name, parsed.imo, loa_m=parsed.loa_m)

    db.commit()
    return summary


def _connect(settings: Settings) -> imaplib.IMAP4:
    if settings.imap_ssl:
        return imaplib.IMAP4_SSL(settings.imap_host, settings.imap_port)
    return imaplib.IMAP4(settings.imap_host, settings.imap_port)


def _imap_utf7_encode(s: str) -> bytes:
    """Кодирование имени IMAP-папки по RFC 3501 (modified UTF-7)."""
    out = bytearray()
    buf: list[str] = []
    for c in s:
        o = ord(c)
        if 0x20 <= o <= 0x7E:
            if buf:
                out.extend(b"&" + _b64_utf7(buf) + b"-")
                buf.clear()
            if o == 0x26:  # "&"
                out.extend(b"&-")
            else:
                out.append(o)
        else:
            buf.append(c)
    if buf:
        out.extend(b"&" + _b64_utf7(buf) + b"-")
    return bytes(out)


def _b64_utf7(chars: list[str]) -> bytes:
    return base64.b64encode("".join(chars).encode("utf-16be")).rstrip(b"=").replace(b"/", b",")


def _select_folder(conn: imaplib.IMAP4, name: str) -> None:
    conn.select(_imap_utf7_encode(name))  # type: ignore[arg-type]


def _fetch_unseen(
    conn: imaplib.IMAP4,
    folder: str,
    sender: str = "",
) -> list[tuple[bytes, bytes]]:
    """Выбрать папку и вернуть непрочитанные письма (seq, raw). Не ставит Seen."""
    _select_folder(conn, folder)
    criteria = ["UNSEEN"]
    if sender:
        criteria = ["UNSEEN", "FROM", sender]
    typ, data = conn.search(None, *criteria)
    if typ != "OK" or not data or not data[0]:
        return []
    messages: list[tuple[bytes, bytes]] = []
    for seq in data[0].split():
        typ, msg_data = conn.fetch(seq, "(RFC822)")
        if typ != "OK" or not msg_data:
            continue
        for item in msg_data:
            if isinstance(item, tuple) and isinstance(item[1], bytes | bytearray):
                messages.append((seq, bytes(item[1])))
                break
    return messages


def _mark_seen(conn: imaplib.IMAP4, seqs: Iterable[bytes]) -> None:
    for seq in seqs:
        try:
            conn.store(seq, "+FLAGS", "\\Seen")
        except OSError:
            pass


def _ingest_voucher_message(
    db: Session,
    msg: EmailMessage,
    vouchers_dir: Path,
    summary: IngestSummary,
) -> None:
    for part in msg.walk():
        if part.get_content_maintype() == "multipart":
            continue
        if part.get_content_disposition() != "attachment":
            continue
        filename = part.get_filename()
        suffix = Path(filename).suffix.lower() if filename else ""
        if suffix not in ALLOWED_SUFFIXES:
            continue
        payload = part.get_payload(decode=True)
        if not isinstance(payload, bytes):
            continue

        sha256 = hashlib.sha256(payload).hexdigest()
        existing = db.query(Voucher).filter_by(sha256=sha256).first()
        if existing:
            summary.skipped_duplicates += 1
            continue

        content_type = part.get_content_type()
        stored = store_upload(
            io.BytesIO(payload),
            filename,
            content_type,
            vouchers_dir,
        )

        voucher = Voucher(
            status=DocStatus.needs_review,
            template=ensure_default_template(db),
            file_path=str(stored.path),
            original_filename=stored.original_filename,
            content_type=stored.content_type,
            sha256=stored.sha256,
        )
        db.add(voucher)
        db.flush()
        predict_and_store(db, voucher)
        summary.created += 1
        summary.attachments_saved += 1


def fetch_new_applications(
    db: Session,
    *,
    settings: Settings | None = None,
    connection: imaplib.IMAP4 | None = None,
) -> IngestSummary:
    """Подключиться к IMAP, забрать непрочитанные заявки и сохранить их.

    `connection` можно передать для тестов (fake-объект с тем же интерфейсом).
    """
    cfg = settings or default_settings
    if not cfg.imap_host or not cfg.imap_user or not cfg.imap_password:
        raise RuntimeError(
            "IMAP не настроен: задай APP_IMAP_HOST/APP_IMAP_USER/APP_IMAP_PASSWORD в .env"
        )

    folder = cfg.imap_application_folder or cfg.imap_folder or "INBOX"

    conn = connection or _connect(cfg)
    owns_conn = connection is None
    try:
        if owns_conn:
            conn.login(cfg.imap_user, cfg.imap_password)
        fetched = _fetch_unseen(conn, folder, cfg.application_sender)
        raw_messages = [raw for _, raw in fetched]
        summary = ingest_messages(
            db,
            raw_messages,
            applications_dir=cfg.incoming_applications_dir,
            vouchers_dir=cfg.incoming_vouchers_dir,
        )
        _mark_seen(conn, (seq for seq, _ in fetched))
    finally:
        if owns_conn:
            try:
                conn.logout()
            except OSError:
                pass

    return summary


def fetch_new_vouchers(
    db: Session,
    *,
    settings: Settings | None = None,
    connection: imaplib.IMAP4 | None = None,
) -> IngestSummary:
    """Подключиться к IMAP, забрать непрочитанные ваучеры из отдельной папки.

    `connection` можно передать для тестов.
    """
    cfg = settings or default_settings
    if not cfg.imap_voucher_folder:
        return IngestSummary()

    if not cfg.imap_host or not cfg.imap_user or not cfg.imap_password:
        raise RuntimeError(
            "IMAP не настроен: задай APP_IMAP_HOST/APP_IMAP_USER/APP_IMAP_PASSWORD в .env"
        )

    conn = connection or _connect(cfg)
    owns_conn = connection is None
    summary = IngestSummary()
    try:
        if owns_conn:
            conn.login(cfg.imap_user, cfg.imap_password)
        fetched = _fetch_unseen(conn, cfg.imap_voucher_folder)
        for seq, raw in fetched:
            try:
                msg = _load_message(raw)
                _ingest_voucher_message(db, msg, cfg.incoming_vouchers_dir, summary)
                _mark_seen(conn, [seq])
            except Exception as exc:  # noqa: BLE001
                summary.errors.append(f"seq {seq.decode()}: {exc}")
    finally:
        if owns_conn:
            try:
                conn.logout()
            except OSError:
                pass

    return summary
