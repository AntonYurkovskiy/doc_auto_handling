"""Собрать ZIP с выборкой связанных ваучеров и заявок для ручной проверки."""

from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
import zipfile
from collections.abc import Iterable
from email.header import decode_header
from email.parser import BytesParser
from pathlib import Path

REQUIRED_COLUMNS = {"voucher_file", "application_file"}
MANIFEST_COLUMNS = (
    "source_row_number",
    "voucher_file",
    "application_file",
    "voucher_source",
    "application_source",
    "voucher_in_archive",
    "application_in_archive",
    "lookup_status",
)
SKIPPED_COLUMNS = (
    "source_row_number",
    "voucher_file",
    "application_file",
    "missing",
)


def _normalise_name(value: str | None) -> str:
    return (value or "").replace("\\", "/").rsplit("/", 1)[-1].strip()


def _normalise_subject(value: str | None) -> str:
    subject = (value or "").casefold().replace("ё", "е")
    subject = subject.replace("_", " ")
    subject = re.sub(r"\.(pdf|eml)$", "", subject)
    subject = re.sub(r"^(?:(?:re|fw|fwd)\s*[:_ -]+\s*)+", "", subject)
    subject = re.sub(r"\b([01]?\d|2[0-3]):([0-5]\d)\b", r"\1 \2", subject)
    subject = re.sub(r"\b([01]\d|2[0-3])([0-5]\d)\b", r"\1 \2", subject)
    subject = re.sub(r"[/\\|]+", " ", subject)
    subject = re.sub(r"[^\w№.-]+", " ", subject, flags=re.UNICODE)
    return " ".join(subject.split())


def _decode_subject(value: str | None) -> str:
    if not value:
        return ""
    parts = []
    for part, encoding in decode_header(value):
        if isinstance(part, bytes):
            parts.append(part.decode(encoding or "utf-8", errors="replace"))
        else:
            parts.append(part)
    return "".join(parts)


def _build_file_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        index.setdefault(path.name.casefold(), []).append(path)
    return index


def _build_subject_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        try:
            message = BytesParser().parsebytes(path.read_bytes())
            subject = _normalise_subject(_decode_subject(message.get("subject")))
        except (OSError, UnicodeError, ValueError):
            continue
        if subject:
            index.setdefault(subject, []).append(path)
    return index


def _find_source_by_subject(
    value: str,
    index: dict[str, list[Path]],
) -> tuple[Path | None, bool]:
    matches = index.get(_normalise_subject(value), [])
    if not matches:
        return None, False
    return matches[0], len(matches) > 1


def _find_source(
    filename: str,
    index: dict[str, list[Path]],
) -> tuple[Path | None, bool]:
    matches = index.get(_normalise_name(filename).casefold(), [])
    if not matches:
        return None, False
    return matches[0], len(matches) > 1


def _find_source_by_year(
    filename: str,
    year: str | None,
    root: Path,
    index: dict[str, list[Path]],
) -> tuple[Path | None, bool]:
    if not year:
        return None, False
    matches = index.get(_normalise_name(filename).casefold(), [])
    year_matches = [
        path
        for path in matches
        if year.casefold() in {part.casefold() for part in path.relative_to(root).parts}
    ]
    if not year_matches:
        return None, False
    return year_matches[0], len(year_matches) > 1


def _build_relative_file_index(root: Path) -> dict[str, list[Path]]:
    index: dict[str, list[Path]] = {}
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = str(path.relative_to(root)).replace("\\", "/").casefold()
        index.setdefault(relative, []).append(path)
    return index


def _find_source_by_path(
    value: str | None,
    root: Path,
    relative_index: dict[str, list[Path]],
) -> Path | None:
    if not value:
        return None
    normalised = value.replace("\\", "/").strip("/")
    marker = f"/{root.name.casefold()}/"
    lowered = f"/{normalised.casefold()}"
    marker_position = lowered.rfind(marker)
    if marker_position >= 0:
        relative = lowered[marker_position + len(marker) :]
        matches = relative_index.get(relative.casefold(), [])
        if matches:
            return matches[0]
    return None


def _voucher_key(row: dict[str, str], voucher_name: str) -> str:
    source_path = row.get("voucher_scan_path", "").strip()
    if source_path:
        return source_path.replace("\\", "/").casefold()
    return f"{voucher_name.casefold()}|{row.get('base_year', '').strip()}"


def _unique_pairs(rows: Iterable[dict[str, str]]) -> list[dict[str, str]]:
    selected: list[dict[str, str]] = []
    seen_vouchers: set[str] = set()
    for row in rows:
        voucher_file = _normalise_name(row.get("voucher_file"))
        application_file = _normalise_name(row.get("application_file"))
        if not voucher_file or not application_file:
            continue
        key = _voucher_key(row, voucher_file)
        if key in seen_vouchers:
            continue
        seen_vouchers.add(key)
        selected.append(row)
    return selected


def _copy_with_unique_name(source: Path, destination_dir: Path) -> str:
    candidate = source.name
    stem, suffix = source.stem, source.suffix
    counter = 2
    while (destination_dir / candidate).exists():
        candidate = f"{stem}_{counter}{suffix}"
        counter += 1
    shutil.copy2(source, destination_dir / candidate)
    return candidate


def _write_manifest(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=MANIFEST_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_skipped(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=SKIPPED_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)


def _write_readme(path: Path, count: int) -> None:
    path.write_text(
        "Выборка для проверки doc_auto_handling\n\n"
        f"Полных комплектов: {count}\n"
        "Для каждого комплекта совпадают voucher_file и application_file "
        "из исторической выгрузки.\n"
        "Исходные имена и пути указаны в manifest.csv.\n"
        "Пропущенные неполные пары указаны в skipped_missing.csv.\n",
        encoding="utf-8",
    )


def build_sample(
    csv_path: Path,
    vouchers_dir: Path,
    applications_dir: Path,
    output_zip: Path,
    count: int,
    offset: int = 0,
) -> int:
    if count < 1:
        raise ValueError("--count должен быть положительным")
    if offset < 0:
        raise ValueError("--offset не может быть отрицательным")
    if not csv_path.is_file():
        raise FileNotFoundError(f"CSV не найден: {csv_path}")
    if not vouchers_dir.is_dir():
        raise NotADirectoryError(f"Каталог ваучеров не найден: {vouchers_dir}")
    if not applications_dir.is_dir():
        raise NotADirectoryError(f"Каталог заявок не найден: {applications_dir}")
    if output_zip.exists():
        raise FileExistsError(
            f"Файл уже существует: {output_zip}; выбери другое имя или удали его вручную"
        )

    with csv_path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        columns = set(reader.fieldnames or [])
        missing = REQUIRED_COLUMNS - columns
        if missing:
            raise ValueError(f"В CSV отсутствуют колонки: {', '.join(sorted(missing))}")
        rows = _unique_pairs(reader)

    voucher_index = _build_file_index(vouchers_dir)
    voucher_relative_index = _build_relative_file_index(vouchers_dir)
    application_index = _build_file_index(applications_dir)
    application_subject_index = _build_subject_index(applications_dir)
    application_relative_index = _build_relative_file_index(applications_dir)
    staging = output_zip.with_name(f".{output_zip.stem}.staging")
    if staging.exists():
        raise FileExistsError(f"Временный каталог уже существует: {staging}")

    staging.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()
    (staging / "vouchers").mkdir()
    (staging / "applications").mkdir()
    manifest_rows: list[dict[str, str]] = []
    skipped_rows: list[dict[str, str]] = []
    try:
        for index, row in enumerate(rows[offset:], start=offset + 1):
            if len(manifest_rows) >= count:
                break
            voucher_name = _normalise_name(row["voucher_file"])
            application_name = _normalise_name(row["application_file"])
            voucher_by_path = _find_source_by_path(
                row.get("voucher_scan_path"),
                vouchers_dir,
                voucher_relative_index,
            )
            voucher_by_year = False
            if voucher_by_path is not None:
                voucher_source = voucher_by_path
                voucher_ambiguous = False
            else:
                voucher_source, voucher_ambiguous = _find_source_by_year(
                    voucher_name,
                    row.get("base_year"),
                    vouchers_dir,
                    voucher_index,
                )
                if voucher_source is None:
                    voucher_source, voucher_ambiguous = _find_source(
                        voucher_name, voucher_index
                    )
                else:
                    voucher_by_year = True
            application_source, application_ambiguous = _find_source(
                application_name, application_index
            )
            application_by_subject, subject_ambiguous = _find_source_by_subject(
                application_name,
                application_subject_index,
            )
            application_by_email_path = _find_source_by_path(
                row.get("email_path"),
                applications_dir,
                application_relative_index,
            )
            if application_by_email_path is not None:
                application_source = application_by_email_path
                application_ambiguous = False
            elif application_by_subject is not None and not subject_ambiguous:
                application_source = application_by_subject
                application_ambiguous = False
            if voucher_source is None or application_source is None:
                missing_names: list[str] = []
                if voucher_source is None:
                    missing_names.append(f"ваучер: {voucher_name}")
                if application_source is None:
                    missing_names.append(f"заявка: {application_name}")
                skipped_rows.append(
                    {
                        "source_row_number": str(index),
                        "voucher_file": voucher_name,
                        "application_file": application_name,
                        "missing": "; ".join(missing_names),
                    }
                )
                continue

            voucher_archive_name = _copy_with_unique_name(
                voucher_source, staging / "vouchers"
            )
            application_archive_name = _copy_with_unique_name(
                application_source, staging / "applications"
            )
            statuses = []
            if voucher_ambiguous:
                statuses.append("несколько_ваучеров_выбран_первый")
            if voucher_by_path is not None:
                statuses.append("ваучер_найден_по_voucher_scan_path")
            elif voucher_by_year:
                statuses.append("ваучер_найден_по_base_year")
            if application_ambiguous:
                statuses.append("несколько_заявок_выбрана_первая")
            if application_by_email_path is not None:
                statuses.append("заявка_найдена_по_email_path")
            elif application_by_subject is not None:
                statuses.append("заявка_найдена_по_теме")
            manifest_rows.append(
                {
                    "source_row_number": str(index),
                    "voucher_file": voucher_name,
                    "application_file": application_name,
                    "voucher_source": str(voucher_source),
                    "application_source": str(application_source),
                    "voucher_in_archive": f"vouchers/{voucher_archive_name}",
                    "application_in_archive": f"applications/{application_archive_name}",
                    "lookup_status": ";".join(statuses) or "ok",
                }
            )

        if len(manifest_rows) < count:
            details = "\n  - " + "\n  - ".join(
                row["missing"] for row in skipped_rows
            )
            raise FileNotFoundError(
                f"Удалось собрать только {len(manifest_rows)} полных комплектов "
                f"из требуемых {count}; пропущено пар: {len(skipped_rows)}"
                f"{details}"
            )

        _write_manifest(staging / "manifest.csv", manifest_rows)
        _write_skipped(staging / "skipped_missing.csv", skipped_rows)
        _write_readme(staging / "README.txt", len(manifest_rows))
        output_zip.parent.mkdir(parents=True, exist_ok=True)
        with zipfile.ZipFile(output_zip, "w", compression=zipfile.ZIP_DEFLATED) as archive:
            for path in sorted(staging.rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(staging))
    except Exception:
        if output_zip.exists():
            output_zip.unlink()
        raise
    finally:
        shutil.rmtree(staging, ignore_errors=True)

    return len(manifest_rows)


def main() -> None:
    """Разобрать аргументы, собрать архив и вывести результат."""
    parser = argparse.ArgumentParser(
        description="Собрать выборку связанных сканов ваучеров и заявок"
    )
    parser.add_argument("--csv", dest="csv_path", type=Path, required=True)
    parser.add_argument("--vouchers-dir", type=Path, required=True)
    parser.add_argument("--applications-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("voucher_review_sample.zip"))
    parser.add_argument("--count", type=int, default=10)
    parser.add_argument("--offset", type=int, default=0)
    args = parser.parse_args()

    try:
        count = build_sample(
            args.csv_path,
            args.vouchers_dir,
            args.applications_dir,
            args.output,
            args.count,
            args.offset,
        )
    except (FileExistsError, FileNotFoundError, NotADirectoryError, ValueError) as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        raise SystemExit(2) from exc
    print(f"Готово: {args.output} ({count} комплектов)")


if __name__ == "__main__":
    main()
