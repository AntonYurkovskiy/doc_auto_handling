from __future__ import annotations

import csv
import zipfile
from email.message import EmailMessage
from pathlib import Path

import pytest

from scripts.package_review_sample import build_sample


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
        file,
        fieldnames=["voucher_file", "application_file", "vessel", "email_path"],
        )
        writer.writeheader()
        writer.writerows(rows)


def test_build_sample_deduplicates_vouchers_and_writes_manifest(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    vouchers = tmp_path / "vouchers"
    applications = tmp_path / "applications"
    vouchers.mkdir()
    applications.mkdir()
    (vouchers / "243k.pdf").write_bytes(b"voucher")
    (applications / "application.pdf").write_bytes(b"application")
    _write_csv(
        csv_path,
        [
            {
                "voucher_file": "243k.pdf",
                "application_file": "application.pdf",
                "vessel": "VESSEL",
            },
            {
                "voucher_file": "243k.pdf",
                "application_file": "application.pdf",
                "vessel": "VESSEL",
            },
        ],
    )

    output = tmp_path / "nested" / "sample.zip"
    assert build_sample(csv_path, vouchers, applications, output, count=1) == 1

    with zipfile.ZipFile(output) as archive:
        assert sorted(archive.namelist()) == [
            "README.txt",
            "applications/application.pdf",
            "manifest.csv",
            "skipped_missing.csv",
            "vouchers/243k.pdf",
        ]
        manifest = archive.read("manifest.csv").decode("utf-8")
        assert "243k.pdf" in manifest
        assert "application.pdf" in manifest
        assert archive.read("skipped_missing.csv").decode("utf-8").splitlines() == [
            "source_row_number,voucher_file,application_file,missing"
        ]


def test_build_sample_reports_missing_files(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    vouchers = tmp_path / "vouchers"
    applications = tmp_path / "applications"
    vouchers.mkdir()
    applications.mkdir()
    _write_csv(
        csv_path,
        [{"voucher_file": "missing.pdf", "application_file": "application.pdf", "vessel": ""}],
    )

    with pytest.raises(FileNotFoundError, match="missing.pdf"):
        build_sample(csv_path, vouchers, applications, tmp_path / "sample.zip", count=1)


def test_build_sample_skips_missing_rows_until_count_is_reached(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    vouchers = tmp_path / "vouchers"
    applications = tmp_path / "applications"
    vouchers.mkdir()
    applications.mkdir()
    (vouchers / "complete.pdf").write_bytes(b"voucher")
    (applications / "complete.pdf").write_bytes(b"application")
    _write_csv(
        csv_path,
        [
            {
                "voucher_file": "missing.pdf",
                "application_file": "missing.pdf",
                "vessel": "",
            },
            {
                "voucher_file": "complete.pdf",
                "application_file": "complete.pdf",
                "vessel": "",
            },
        ],
    )

    output = tmp_path / "sample.zip"
    assert build_sample(csv_path, vouchers, applications, output, count=1) == 1
    with zipfile.ZipFile(output) as archive:
        skipped = archive.read("skipped_missing.csv").decode("utf-8")
        assert "missing.pdf" in skipped


def test_build_sample_uses_eml_subject_for_nameless_exports(tmp_path: Path) -> None:
    csv_path = tmp_path / "history.csv"
    vouchers = tmp_path / "vouchers"
    applications = tmp_path / "orders"
    vouchers.mkdir()
    applications.mkdir()
    (applications / "2025").mkdir()
    (vouchers / "243k.pdf").write_bytes(b"voucher")
    message = EmailMessage()
    message["Subject"] = "Вход тн VESSEL 20.07 в 15:00 / ТСС №9"
    (applications / "2025" / "NoName-24").write_bytes(message.as_bytes())
    _write_csv(
        csv_path,
        [
            {
                "voucher_file": "243k.pdf",
                "application_file": "Вход тн VESSEL 20.07 в 1500 ТСС №9.pdf",
                "vessel": "VESSEL",
                "email_path": "",
            }
        ],
    )

    output = tmp_path / "sample.zip"
    assert build_sample(csv_path, vouchers, applications, output, count=1) == 1
    with zipfile.ZipFile(output) as archive:
        manifest = archive.read("manifest.csv").decode("utf-8")
        assert "заявка_найдена_по_теме" in manifest
        assert archive.read("applications/NoName-24") == message.as_bytes()
