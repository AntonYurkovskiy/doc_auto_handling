"""Тесты приёма ваучеров: безопасный upload, выдача файла и подтверждение."""

from __future__ import annotations

import hashlib
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker
from sqlalchemy.pool import StaticPool

from app.config import settings
from app.database import Base, get_db
from app.main import app
from app.models import DocStatus, Tug, Voucher, VoucherFieldPrediction
from app.services.voucher_files import resolve_stored_file, safe_basename, store_upload

PDF_BYTES = b"%PDF-1.4\ntest voucher scan\n%%EOF\n"


@pytest.fixture
def vouchers_dir(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    folder = tmp_path / "incoming" / "vouchers"
    folder.mkdir(parents=True)
    monkeypatch.setattr(settings, "incoming_vouchers_dir", folder)
    return folder


@pytest.fixture
def session_factory() -> Iterator[sessionmaker[Session]]:
    engine = create_engine(
        "sqlite://",
        connect_args={"check_same_thread": False},
        poolclass=StaticPool,
    )
    Base.metadata.create_all(engine)
    factory = sessionmaker(bind=engine, autoflush=False, autocommit=False)
    try:
        yield factory
    finally:
        Base.metadata.drop_all(engine)
        engine.dispose()


@pytest.fixture
def client(session_factory: sessionmaker[Session]) -> Iterator[TestClient]:
    def override_get_db() -> Iterator[Session]:
        db = session_factory()
        try:
            yield db
        finally:
            db.close()

    app.dependency_overrides[get_db] = override_get_db
    try:
        yield TestClient(app)
    finally:
        app.dependency_overrides.clear()


def _upload(client: TestClient, filename: str = "voucher 262k.pdf") -> int:
    response = client.post(
        "/vouchers/upload",
        files={"file": (filename, PDF_BYTES, "application/pdf")},
        follow_redirects=False,
    )
    assert response.status_code == 303
    return int(response.headers["location"].rsplit("/", 1)[-1])


def test_safe_basename_strips_paths_and_unsafe_chars() -> None:
    assert safe_basename("../../etc/passwd.pdf") == "passwd.pdf"
    assert safe_basename(r"C:\scans\ваучер 262k(2).pdf") == "262k_2.pdf"
    assert safe_basename("script.sh") == "script.bin"
    assert safe_basename(None) == "voucher.bin"


def test_store_upload_keeps_file_inside_folder(vouchers_dir: Path) -> None:
    import io

    stored = store_upload(
        io.BytesIO(PDF_BYTES), "../../evil.pdf", "application/pdf", vouchers_dir
    )

    assert stored.path.parent == vouchers_dir
    assert stored.sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
    assert stored.path.read_bytes() == PDF_BYTES
    assert list(vouchers_dir.iterdir()) == [stored.path]


def test_resolve_stored_file_rejects_traversal(vouchers_dir: Path) -> None:
    secret = vouchers_dir.parent / "secret.txt"
    secret.write_text("секрет", encoding="utf-8")

    with pytest.raises(ValueError):
        resolve_stored_file("../secret.txt", vouchers_dir)
    with pytest.raises(ValueError):
        resolve_stored_file(str(secret), vouchers_dir)
    with pytest.raises(ValueError):
        resolve_stored_file("", vouchers_dir)


def test_upload_fills_file_metadata_and_serves_original(
    client: TestClient, session_factory: sessionmaker[Session], vouchers_dir: Path
) -> None:
    voucher_id = _upload(client, "../../ваучер 262k.pdf")

    db = session_factory()
    try:
        voucher = db.get(Voucher, voucher_id)
        assert voucher is not None
        assert voucher.status is DocStatus.needs_review
        assert voucher.original_filename == "262k.pdf"
        assert voucher.content_type == "application/pdf"
        assert voucher.sha256 == hashlib.sha256(PDF_BYTES).hexdigest()
        stored = Path(voucher.file_path or "")
        assert stored.parent == vouchers_dir
    finally:
        db.close()

    assert client.get(f"/vouchers/{voucher_id}").status_code == 200

    file_response = client.get(f"/vouchers/{voucher_id}/file")
    assert file_response.status_code == 200
    assert file_response.headers["content-type"] == "application/pdf"
    assert file_response.content == PDF_BYTES

    by_name = client.get(f"/files/vouchers/{stored.name}")
    assert by_name.status_code == 200
    assert by_name.content == PDF_BYTES


def test_file_endpoints_reject_paths_outside_storage(
    client: TestClient, session_factory: sessionmaker[Session], vouchers_dir: Path
) -> None:
    secret = vouchers_dir.parent / "secret.txt"
    secret.write_text("секрет", encoding="utf-8")

    assert client.get("/files/vouchers/%2e%2e%2fsecret.txt").status_code == 404
    assert client.get(f"/files/vouchers/{secret}").status_code == 404
    assert client.get("/files/vouchers/missing.pdf").status_code == 404

    db = session_factory()
    try:
        voucher = Voucher(status=DocStatus.needs_review, file_path=str(secret))
        db.add(voucher)
        db.commit()
        voucher_id = voucher.id
    finally:
        db.close()

    assert client.get(f"/vouchers/{voucher_id}/file").status_code == 404
    assert client.get(f"/vouchers/{voucher_id}").status_code == 200


def test_manual_save_keeps_needs_review(
    client: TestClient, session_factory: sessionmaker[Session], vouchers_dir: Path
) -> None:
    voucher_id = _upload(client)

    response = client.post(
        "/vouchers",
        data={"voucher_id": str(voucher_id), "number": "262", "vessel_name": "Судно"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    db = session_factory()
    try:
        voucher = db.get(Voucher, voucher_id)
        assert voucher is not None
        assert voucher.number == "262"
        assert voucher.vessel_name == "Судно"
        assert voucher.status is DocStatus.needs_review
        assert voucher.reviewed_at is None
        assert voucher.predictions == []
    finally:
        db.close()


def test_confirm_writes_confirmed_values_and_status(
    client: TestClient, session_factory: sessionmaker[Session], vouchers_dir: Path
) -> None:
    db = session_factory()
    try:
        tug = Tug(name="БК Коммунар", code="k")
        db.add(tug)
        db.commit()
        tug_id = tug.id
    finally:
        db.close()

    voucher_id = _upload(client)

    response = client.post(
        f"/vouchers/{voucher_id}/confirm",
        data={
            "number": "262",
            "tug_id": str(tug_id),
            "vessel_name": "Судно",
            "work_type": "швартовка",
            "started_dt": "2025-01-15T12:30",
            "escort_hours": "1,5",
            "is_ice": "1",
        },
        follow_redirects=False,
    )
    assert response.status_code == 303
    assert response.headers["location"] == f"/vouchers/{voucher_id}"

    db = session_factory()
    try:
        voucher = db.get(Voucher, voucher_id)
        assert voucher is not None
        assert voucher.status is DocStatus.confirmed
        assert voucher.reviewed_at is not None
        assert voucher.escort_hours == 1.5
        assert voucher.is_ice is True

        confirmed = {
            row.field_name: row.confirmed_value
            for row in db.scalars(select(VoucherFieldPrediction)).all()
        }
        assert confirmed["voucher_number"] == "262"
        assert confirmed["tugboat"] == "БК Коммунар"
        assert confirmed["vessel"] == "Судно"
        assert confirmed["work_type"] == "швартовка"
        assert confirmed["started_work"] == "2025-01-15 12:30"
        assert confirmed["arrived_base"] is None
        assert all(row.confirmed_at is not None for row in voucher.predictions)
    finally:
        db.close()

    assert client.get(f"/vouchers/{voucher_id}").status_code == 200


def test_confirm_updates_existing_prediction(
    client: TestClient, session_factory: sessionmaker[Session], vouchers_dir: Path
) -> None:
    voucher_id = _upload(client)

    db = session_factory()
    try:
        db.add(
            VoucherFieldPrediction(
                voucher_id=voucher_id,
                field_name="voucher_number",
                predicted_value="261",
                confidence=0.7,
            )
        )
        db.commit()
    finally:
        db.close()

    response = client.post(
        f"/vouchers/{voucher_id}/confirm",
        data={"number": "262"},
        follow_redirects=False,
    )
    assert response.status_code == 303

    db = session_factory()
    try:
        rows = db.scalars(
            select(VoucherFieldPrediction).where(
                VoucherFieldPrediction.field_name == "voucher_number"
            )
        ).all()
        assert len(rows) == 1
        assert rows[0].predicted_value == "261"
        assert rows[0].confirmed_value == "262"
    finally:
        db.close()
