"""Тесты сборки манифеста на синтетической выгрузке (`tests/ocr/test_manifest.py`).

Покрывает дубли (с конфликтом истины и без), третий буксир, «24:00», переход через
полночь без «24:00», применение поправки истины, границы сплита и формирование пар.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import pandas as pd
import pytest

from ocr_lab.manifest import (
    CORRECTIONS_COLUMNS,
    build,
    ensure_seed_corrections,
    extract_voucher_number,
    load_corrections,
    make_scan_id,
    parse_dt,
)

DATASET_COLUMNS = [
    "row_number",
    "tug",
    "voucher_number",
    "voucher_file",
    "voucher_scan_path",
    "base_departure",
    "base_arrival",
    "work_start",
    "work_end",
    "work_type",
    "agent",
    "application_file",
    "date_raw",
    "time_raw",
]


def _row(row_number: int, **overrides: Any) -> dict[str, Any]:
    base = {
        "row_number": row_number,
        "tug": "БК Коммунар",
        "voucher_number": row_number,
        "voucher_file": f"{row_number}k.pdf",
        "voucher_scan_path": f"/scans/{row_number}k.pdf",
        "base_departure": "05.01.2025 08:00",
        "base_arrival": "05.01.2025 09:00",
        "work_start": "05.01.2025 08:10",
        "work_end": "05.01.2025 08:50",
        "work_type": "Швартовка",
        "agent": "Транс-Агро",
        "application_file": f"app_{row_number}.pdf",
        "date_raw": "",
        "time_raw": "",
    }
    base.update(overrides)
    return base


def _write_dataset(path: Path, rows: list[dict[str, Any]]) -> None:
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=DATASET_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


ROWS: list[dict[str, Any]] = [
    # Пара (оба буксира, одна работа), времена не идентичны — не "полное совпадение".
    _row(
        1,
        voucher_number=10,
        voucher_file="10k.pdf",
        application_file="app_pair_a.pdf",
        date_raw="04.01",
        time_raw="20:00",
    ),
    _row(
        2,
        tug="БК Пионер",
        voucher_number=11,
        voucher_file="11p.pdf",
        base_departure="05.01.2025 08:30",
        base_arrival="05.01.2025 09:40",
        work_start="05.01.2025 08:40",
        work_end="05.01.2025 09:20",
        application_file="app_pair_a.pdf",
        date_raw="04.01",
        time_raw="20:00",
    ),
    # Пара с полным совпадением всех четырёх времён.
    _row(
        3,
        voucher_number=20,
        voucher_file="20k.pdf",
        base_departure="10.01.2025 10:00",
        base_arrival="10.01.2025 11:00",
        work_start="10.01.2025 10:10",
        work_end="10.01.2025 10:50",
        work_type="Отшвартовка",
        application_file="app_pair_b.pdf",
    ),
    _row(
        4,
        tug="БК Пионер",
        voucher_number=21,
        voucher_file="21p.pdf",
        base_departure="10.01.2025 10:00",
        base_arrival="10.01.2025 11:00",
        work_start="10.01.2025 10:10",
        work_end="10.01.2025 10:50",
        work_type="Отшвартовка",
        application_file="app_pair_b.pdf",
    ),
    # Общий файл заявки, но разный вид работ — пары быть не должно.
    _row(
        5,
        voucher_number=30,
        voucher_file="30k.pdf",
        base_departure="15.01.2025 10:00",
        work_type="Швартовка",
        application_file="app_no_pair.pdf",
    ),
    _row(
        6,
        tug="БК Пионер",
        voucher_number=31,
        voucher_file="31p.pdf",
        base_departure="15.01.2025 10:10",
        work_type="Перестановка",
        application_file="app_no_pair.pdf",
    ),
    # Третий буксир — не наш, исключается.
    _row(
        7,
        tug="МБ Лигер",
        voucher_number=40,
        voucher_file="40l.pdf",
        base_departure="20.01.2025 09:00",
        application_file="app_third_tug.pdf",
    ),
    # Дубль без конфликта истины (один и тот же скан дважды, значения совпадают).
    _row(
        8,
        voucher_number=50,
        voucher_file="50k.pdf",
        base_departure="25.01.2025 12:00",
        base_arrival="25.01.2025 13:00",
        work_start="25.01.2025 12:10",
        work_end="25.01.2025 12:50",
        application_file="app_dup_ok.pdf",
    ),
    _row(
        9,
        voucher_number=50,
        voucher_file="50k.pdf",
        base_departure="25.01.2025 12:00",
        base_arrival="25.01.2025 13:00",
        work_start="25.01.2025 12:10",
        work_end="25.01.2025 12:50",
        application_file="app_dup_ok.pdf",
    ),
    # Дубль с конфликтом истины — должна остаться первая по row_number строка.
    _row(
        10,
        voucher_number=60,
        voucher_file="60k.pdf",
        base_departure="26.01.2025 08:00",
        base_arrival="26.01.2025 09:00",
        work_start="26.01.2025 08:10",
        work_end="26.01.2025 08:50",
        application_file="app_dup_conflict.pdf",
    ),
    _row(
        11,
        voucher_number=60,
        voucher_file="60k.pdf",
        base_departure="26.01.2025 08:00",
        base_arrival="26.01.2025 09:00",
        work_start="26.01.2025 09:10",  # расходится со строкой 10
        work_end="26.01.2025 08:50",
        application_file="app_dup_conflict.pdf",
    ),
    # «24:00» на бланке — конец суток, не начало следующих.
    _row(
        12,
        voucher_number=70,
        voucher_file="70k.pdf",
        base_departure="25.02.2025 22:00",
        base_arrival="26.02.2025 00:30",
        work_start="25.02.2025 22:10",
        work_end="25.02.2025 24:00",
        application_file="app_hour24.pdf",
    ),
    # Переход через полночь обычными цифрами, без «24:00».
    _row(
        13,
        voucher_number=80,
        voucher_file="80k.pdf",
        base_departure="01.03.2025 23:30",
        base_arrival="02.03.2025 00:40",
        work_start="01.03.2025 23:40",
        work_end="02.03.2025 00:20",
        application_file="app_midnight.pdf",
    ),
    # Ошибка выгрузки: started_work раньше left_base — поправка её чинит.
    _row(
        14,
        voucher_number=90,
        voucher_file="90k.pdf",
        base_departure="05.03.2025 08:00",
        base_arrival="05.03.2025 09:00",
        work_start="04.03.2025 08:10",
        work_end="05.03.2025 08:50",
        application_file="app_correction.pdf",
    ),
    # Границы сплита (Q7 DECISIONS.md).
    _row(
        15,
        voucher_number=100,
        voucher_file="100k.pdf",
        base_departure="31.03.2026 10:00",
        base_arrival="31.03.2026 11:00",
        work_start="31.03.2026 10:10",
        work_end="31.03.2026 10:50",
        application_file="app_split_train.pdf",
    ),
    _row(
        16,
        voucher_number=101,
        voucher_file="101k.pdf",
        base_departure="01.04.2026 10:00",
        base_arrival="01.04.2026 11:00",
        work_start="01.04.2026 10:10",
        work_end="01.04.2026 10:50",
        application_file="app_split_val_start.pdf",
    ),
    _row(
        17,
        voucher_number=102,
        voucher_file="102k.pdf",
        base_departure="31.05.2026 10:00",
        base_arrival="31.05.2026 11:00",
        work_start="31.05.2026 10:10",
        work_end="31.05.2026 10:50",
        application_file="app_split_val_end.pdf",
    ),
    _row(
        18,
        voucher_number=103,
        voucher_file="103k.pdf",
        base_departure="01.06.2026 10:00",
        base_arrival="01.06.2026 11:00",
        work_start="01.06.2026 10:10",
        work_end="01.06.2026 10:50",
        application_file="app_split_test.pdf",
    ),
]


@pytest.fixture
def dataset_csv(tmp_path: Path) -> Path:
    path = tmp_path / "reconciled_dataset.csv"
    _write_dataset(path, ROWS)
    return path


@pytest.fixture
def corrections_csv(tmp_path: Path) -> Path:
    path = tmp_path / "truth_corrections.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CORRECTIONS_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "scan_id": "2025_90k",
                "field": "started_work",
                "export_value": "04.03.2025 08:10",
                "corrected_value": "05.03.2025 08:10",
                "status": "confirmed",
                "source": "test",
                "note": "тестовая поправка",
            }
        )
    return path


@pytest.fixture
def built(dataset_csv: Path, corrections_csv: Path) -> tuple[pd.DataFrame, Any]:
    return build(dataset_csv, corrections_csv)


def _get(df: pd.DataFrame, scan_id: str) -> pd.Series:
    rows = df[df["scan_id"] == scan_id]
    assert len(rows) == 1, f"ожидалась ровно одна строка для {scan_id}, нашлось {len(rows)}"
    return rows.iloc[0]


# --- parse_dt / make_scan_id / extract_voucher_number --------------------------------------


def test_parse_dt_with_and_without_time() -> None:
    full = parse_dt("05.01.2025 08:30")
    assert full is not None
    assert (full.day, full.month, full.year, full.hour, full.minute) == (5, 1, 2025, 8, 30)
    date_only = parse_dt("05.01.2025")
    assert date_only is not None
    assert date_only.hour is None and date_only.dt is None
    assert parse_dt("") is None
    assert parse_dt(None) is None


def test_parse_dt_hour24_shifts_dt_to_next_day() -> None:
    parsed = parse_dt("25.02.2025 24:00")
    assert parsed is not None
    assert parsed.hour == 24
    assert parsed.dt is not None
    assert parsed.dt.isoformat() == "2025-02-26T00:00:00"


def test_make_scan_id_replaces_copy_marker() -> None:
    assert make_scan_id("156k(2).pdf", 2025) == "2025_156k_2"
    assert make_scan_id("243k.pdf", 2025) == "2025_243k"


def test_extract_voucher_number_strips_letter_suffix() -> None:
    assert extract_voucher_number("16a") == 16
    assert extract_voucher_number("068") == 68
    assert extract_voucher_number("243") == 243
    assert extract_voucher_number("abc") is None


# --- Поправки истины -------------------------------------------------------------------------


def test_ensure_seed_corrections_creates_and_is_idempotent(tmp_path: Path) -> None:
    path = tmp_path / "truth_corrections.csv"
    added_first = ensure_seed_corrections(path)
    assert added_first == 7
    rows_after_first = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert len(rows_after_first) == 7

    added_second = ensure_seed_corrections(path)
    assert added_second == 0
    rows_after_second = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert rows_after_second == rows_after_first


def test_ensure_seed_corrections_preserves_existing_rows(tmp_path: Path) -> None:
    path = tmp_path / "truth_corrections.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CORRECTIONS_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "scan_id": "2099_1k",
                "field": "arrived_base",
                "export_value": "01.01.2099 00:00",
                "corrected_value": "01.01.2099 01:00",
                "status": "proposed",
                "source": "t04",
                "note": "своя строка другой задачи",
            }
        )
    ensure_seed_corrections(path)
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig")))
    assert any(r["scan_id"] == "2099_1k" for r in rows)
    assert len(rows) == 8  # своя строка + 7 из приложения плана


def test_load_corrections_ignores_non_confirmed(tmp_path: Path) -> None:
    path = tmp_path / "truth_corrections.csv"
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CORRECTIONS_COLUMNS)
        writer.writeheader()
        writer.writerow(
            {
                "scan_id": "2025_1k",
                "field": "arrived_base",
                "export_value": "a",
                "corrected_value": "b",
                "status": "proposed",
                "source": "x",
                "note": "",
            }
        )
        writer.writerow(
            {
                "scan_id": "2025_2k",
                "field": "arrived_base",
                "export_value": "a",
                "corrected_value": "b",
                "status": "rejected",
                "source": "x",
                "note": "",
            }
        )
    assert load_corrections(path) == {}


def test_correction_applied_and_fixes_chain(built: tuple[pd.DataFrame, Any]) -> None:
    df, report = built
    row = _get(df, "2025_90k")
    assert row["truth_corrected"] == "started_work"
    assert row["started_work_dt"] == "2025-03-05T08:10:00"
    assert ("2025_90k", "left_base>started_work") in report.chain_violations_before
    assert row["chain_ok"]
    assert not any(scan_id == "2025_90k" for scan_id, _ in report.chain_violations_after)


# --- Дубли -----------------------------------------------------------------------------------


def test_duplicate_without_conflict_is_collapsed(built: tuple[pd.DataFrame, Any]) -> None:
    df, report = built
    assert len(df[df["scan_id"] == "2025_50k"]) == 1
    assert ("2025_50k", 1) in report.dedup_groups
    assert "2025_50k" not in report.dedup_conflicts


def test_duplicate_with_conflict_keeps_first_row_number(built: tuple[pd.DataFrame, Any]) -> None:
    df, report = built
    assert ("2025_60k", 1) in report.dedup_groups
    assert "2025_60k" in report.dedup_conflicts
    row = _get(df, "2025_60k")
    assert row["dedup_conflict"]
    # Строка 10 (row_number меньше) должна победить — started_work "08:10", а не "09:10".
    assert row["started_work_dt"] == "2025-01-26T08:10:00"


# --- Третий буксир -----------------------------------------------------------------------------


def test_third_tug_excluded(built: tuple[pd.DataFrame, Any]) -> None:
    df, report = built
    assert "2025_40l" not in set(df["scan_id"])
    assert "2025_40l" in report.excluded_third_tug


# --- 24:00 и переход через полночь --------------------------------------------------------------


def test_hour24_is_preserved_as_end_of_day(built: tuple[pd.DataFrame, Any]) -> None:
    df, _ = built
    row = _get(df, "2025_70k")
    assert bool(row["has_hour24"])
    assert row["finished_work_hour"] == 24
    assert row["finished_work_dt"] == "2025-02-26T00:00:00"
    assert bool(row["crosses_midnight"])


def test_ordinary_midnight_crossing_without_hour24(built: tuple[pd.DataFrame, Any]) -> None:
    df, _ = built
    row = _get(df, "2025_80k")
    assert not bool(row["has_hour24"])
    assert bool(row["crosses_midnight"])


def test_same_day_voucher_does_not_cross_midnight(built: tuple[pd.DataFrame, Any]) -> None:
    df, _ = built
    row = _get(df, "2025_10k")
    assert not bool(row["crosses_midnight"])


# --- Сплит -----------------------------------------------------------------------------------


@pytest.mark.parametrize(
    "scan_id,expected_split",
    [
        ("2026_100k", "train"),
        ("2026_101k", "val"),
        ("2026_102k", "val"),
        ("2026_103k", "test"),
    ],
)
def test_split_boundaries(
    built: tuple[pd.DataFrame, Any], scan_id: str, expected_split: str
) -> None:
    df, _ = built
    row = _get(df, scan_id)
    assert row["split"] == expected_split


# --- Пары ------------------------------------------------------------------------------------


def test_pair_formed_across_tugs_same_application_and_worktype(
    built: tuple[pd.DataFrame, Any],
) -> None:
    df, _ = built
    a, b = _get(df, "2025_10k"), _get(df, "2025_11p")
    assert a["pair_id"] != "" and a["pair_id"] == b["pair_id"]


def test_pair_full_time_match_detected(built: tuple[pd.DataFrame, Any]) -> None:
    df, report = built
    a, b = _get(df, "2025_20k"), _get(df, "2025_21p")
    assert a["pair_id"] == b["pair_id"] != ""
    assert report.pair_full_match_count >= 1


def test_different_work_type_blocks_pairing(built: tuple[pd.DataFrame, Any]) -> None:
    df, _ = built
    a, b = _get(df, "2025_30k"), _get(df, "2025_31p")
    assert a["pair_id"] == "" and b["pair_id"] == ""


# --- Идемпотентность и число строк ------------------------------------------------------------


def test_build_is_idempotent(dataset_csv: Path, corrections_csv: Path) -> None:
    df1, _ = build(dataset_csv, corrections_csv)
    df2, _ = build(dataset_csv, corrections_csv)
    pd.testing.assert_frame_equal(df1, df2)


def test_final_row_count_excludes_third_tug_and_duplicates(
    built: tuple[pd.DataFrame, Any],
) -> None:
    df, report = built
    # 18 строк выгрузки: -1 третий буксир, -1 дубль без конфликта, -1 дубль с конфликтом.
    assert report.total_rows_with_scan == len(ROWS)
    assert len(df) == len(ROWS) - 1 - 1 - 1
    assert report.n_final == len(df)
