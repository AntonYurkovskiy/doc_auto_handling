from __future__ import annotations

import csv
import zipfile
from pathlib import Path

import pytest

from scripts.package_review_sample import build_sample


def _write_csv(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=["voucher_file", "application_file", "vessel"],
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
            "vouchers/243k.pdf",
        ]
        manifest = archive.read("manifest.csv").decode("utf-8")
        assert "243k.pdf" in manifest
        assert "application.pdf" in manifest


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
