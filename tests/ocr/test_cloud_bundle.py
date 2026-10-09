"""Тесты обезличенного пакета: запрещённые колонки и значения не попадают в пакет."""

from __future__ import annotations

import csv
from pathlib import Path

from ocr_lab import cloud_bundle
from ocr_lab.cloud_bundle import MANIFEST_COLUMNS, anonymize_manifest, check_manifest


def _row(**extra: str) -> dict[str, str]:
    row = {col: "" for col in MANIFEST_COLUMNS}
    row.update(scan_id="2026_1k", tug_code="k", voucher_file="1k.pdf", work_type="швартовка")
    row.update(extra)
    return row


def test_anonymize_drops_forbidden_columns() -> None:
    raw = _row()
    raw.update(
        application_file="Вход тн TESTSHIP 31.12.pdf",
        scan_path=r"E:\data\1k.pdf",
        agent_group="A",
        work_type_raw="Швартовка",
        ext="pdf",
        scan_exists="True",
    )
    out = anonymize_manifest([raw])[0]
    assert set(out) == set(MANIFEST_COLUMNS)
    assert not set(out) & cloud_bundle.FORBIDDEN_COLUMNS
    assert "TESTSHIP" not in "".join(out.values())


def test_check_manifest_flags_extra_column_and_free_text() -> None:
    fields = [*MANIFEST_COLUMNS, "application_file"]
    problems = check_manifest(fields, [_row(work_type="Вход тн TESTSHIP")])
    assert any("application_file" in p for p in problems)
    assert any("work_type" in p for p in problems)


def test_check_manifest_clean() -> None:
    assert check_manifest(MANIFEST_COLUMNS, [_row()]) == []


def test_build_and_verify_roundtrip(tmp_path: Path) -> None:
    work = tmp_path / "work"
    (work / "crops" / "voucher_number").mkdir(parents=True)
    (work / "crops" / "voucher_number" / "2026_1k.png").write_bytes(b"png")
    raw = _row(application_file="secret.pdf", scan_path="E:/x.pdf")
    with (work / "manifest.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=[*MANIFEST_COLUMNS, "application_file", "scan_path"])
        writer.writeheader()
        writer.writerow(raw)
    with (work / "crops_index.csv").open("w", encoding="utf-8", newline="") as fh:
        writer = csv.writer(fh)
        writer.writerow(["scan_id", "variant", "subfield", "path", "align_ok"])
        writer.writerow(["2026_1k", "kommunar_v1", "voucher_number",
                         "crops/voucher_number/2026_1k.png", "True"])

    out = tmp_path / "bundle"
    stats = cloud_bundle.build(out, work=work, mnist=tmp_path / "no_mnist")
    assert stats["crops_copied"] == 1
    assert cloud_bundle.verify(out) == []
    text = (out / "manifest.csv").read_text(encoding="utf-8")
    assert "secret.pdf" not in text and "E:/x.pdf" not in text
