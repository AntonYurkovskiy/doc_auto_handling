"""Сборка манифеста OCR-датасета — «замороженной истины».

Источник — сырая сверка `reconciled_dataset.csv` (см. CONTEXT.md): в ней есть дубли строк
на один скан, два ваучера третьего буксира (МБ Лигер) и несколько ошибок цифровизации
дат, уже найденных вручную (`docs/ocr_dates_plan.md`, приложение). Поправки хранятся
отдельно в `data/ocr/truth_corrections.csv` и применяются явно, а не молча правят CSV.

Запуск: `python -m ocr_lab.manifest build`.

Колонки `scan_id, split, tug_code, year, voucher_number, has_hour24` и для каждой строки
`r` (`left_base`, `arrived_base`, `started_work`, `finished_work`) колонки
`r_dt, r_day, r_month, r_year, r_hour, r_minute` — контракт, которым уже пользуется
`ocr_lab.evaluate.load_truth` (написан в T13 заранее). Остальные колонки — служебные,
для последующих задач плана.
"""

from __future__ import annotations

import argparse
import csv
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import date, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from app.config import agent_group
from app.services.calculation import normalize_work_type
from ocr_lab.paths import CORRECTIONS_CSV, DATASET_CSV, MANIFEST, REPORTS_DIR, ensure_dir
from ocr_lab.predictions import PARTS, ROWS, row_datetime

# --- Домен --------------------------------------------------------------------------------

# Источник каждой строки бланка в сырой выгрузке.
ROW_SOURCE_COLUMN: dict[str, str] = {
    "left_base": "base_departure",
    "arrived_base": "base_arrival",
    "started_work": "work_start",
    "finished_work": "work_end",
}

# Буксиры, попадающие в OCR-датасет (Q-decision T01: код буксира берём из колонки `tug`,
# а не из имени файла). Третий буксир (МБ Лигер) — не наш, исключается (см. план).
TUG_CODE_BY_NAME: dict[str, str] = {"БК Коммунар": "k", "БК Пионер": "p"}

# Границы сплита по дате `left_base` (DECISIONS.md, Q7 — подтверждено человеком).
TRAIN_END = date(2026, 3, 31)
VAL_END = date(2026, 5, 31)

# Пара = тот же файл заявки + тот же нормализованный вид работ + выход из базы
# в пределах этого окна.
PAIR_WINDOW_HOURS = 6

CORRECTIONS_COLUMNS: tuple[str, ...] = (
    "scan_id",
    "field",
    "export_value",
    "corrected_value",
    "status",
    "source",
    "note",
)

# Четыре проверенных случая из приложения `docs/ocr_dates_plan.md` (ошибки выгрузки,
# не бланка). Значения — в формате сырой выгрузки «ДД.ММ.ГГГГ ЧЧ:ММ».
_SEED_CORRECTIONS: tuple[dict[str, str], ...] = (
    {
        "scan_id": "2025_1k",
        "field": "arrived_base",
        "export_value": "05.01.2025 12:50",
        "corrected_value": "05.01.2025 13:40",
        "status": "confirmed",
        "source": "ocr_dates_plan",
        "note": "ошибка выгрузки, приложение плана",
    },
    {
        "scan_id": "2025_179k",
        "field": "started_work",
        "export_value": "27.07.2025 07:30",
        "corrected_value": "24.07.2025 07:30",
        "status": "confirmed",
        "source": "ocr_dates_plan",
        "note": "ошибка выгрузки, приложение плана",
    },
    {
        "scan_id": "2025_179k",
        "field": "finished_work",
        "export_value": "27.07.2025 08:10",
        "corrected_value": "24.07.2025 08:10",
        "status": "confirmed",
        "source": "ocr_dates_plan",
        "note": "ошибка выгрузки, приложение плана",
    },
    {
        "scan_id": "2025_326p",
        "field": "started_work",
        "export_value": "21.12.2025 21:40",
        "corrected_value": "20.12.2025 21:40",
        "status": "confirmed",
        "source": "ocr_dates_plan",
        "note": "ошибка выгрузки, приложение плана",
    },
    {
        "scan_id": "2025_326p",
        "field": "finished_work",
        "export_value": "21.12.2025 21:50",
        "corrected_value": "20.12.2025 21:50",
        "status": "confirmed",
        "source": "ocr_dates_plan",
        "note": "ошибка выгрузки, приложение плана",
    },
    {
        "scan_id": "2026_64p",
        "field": "arrived_base",
        "export_value": "11.02.2026 15:30",
        "corrected_value": "11.02.2026 16:30",
        "status": "confirmed",
        "source": "ocr_dates_plan",
        "note": "перепутаны местами arrived_base и finished_work, приложение плана",
    },
    {
        "scan_id": "2026_64p",
        "field": "finished_work",
        "export_value": "11.02.2026 16:30",
        "corrected_value": "11.02.2026 15:30",
        "status": "confirmed",
        "source": "ocr_dates_plan",
        "note": "перепутаны местами arrived_base и finished_work, приложение плана",
    },
)


# --- Разбор даты/времени --------------------------------------------------------------------


@dataclass(frozen=True)
class ParsedDT:
    """Дата/время строки бланка, разобранные из «ДД.ММ.ГГГГ[ ЧЧ:ММ]».

    `hour` может быть 24 (конец суток на бланке, не начало следующих — Q2 DECISIONS.md).
    `dt` в этом случае — 00:00 следующих суток (см. `ocr_lab.predictions.row_datetime`).
    """

    day: int
    month: int
    year: int
    hour: int | None
    minute: int | None

    @property
    def dt(self) -> datetime | None:
        if self.hour is None or self.minute is None:
            return None
        return row_datetime(self.year, self.month, self.day, self.hour, self.minute)

    @property
    def date_only(self) -> date:
        return date(self.year, self.month, self.day)


def parse_dt(raw: Any) -> ParsedDT | None:
    """Разобрать «ДД.ММ.ГГГГ ЧЧ:ММ» или «ДД.ММ.ГГГГ» (без времени — часть строк выгрузки)."""
    if raw is None or (isinstance(raw, float) and pd.isna(raw)):
        return None
    text = str(raw).strip()
    if not text:
        return None
    date_part, _, time_part = text.partition(" ")
    try:
        day_s, month_s, year_s = date_part.split(".")
        day, month, year = int(day_s), int(month_s), int(year_s)
    except ValueError:
        return None
    if not time_part:
        return ParsedDT(day, month, year, None, None)
    try:
        hour_s, minute_s = time_part.split(":")
        hour, minute = int(hour_s), int(minute_s)
    except ValueError:
        return None
    return ParsedDT(day, month, year, hour, minute)


def format_dt(parsed: ParsedDT) -> str:
    """Обратное преобразование — для сверки с `export_value` поправок."""
    base = f"{parsed.day:02d}.{parsed.month:02d}.{parsed.year:04d}"
    if parsed.hour is None or parsed.minute is None:
        return base
    return f"{base} {parsed.hour:02d}:{parsed.minute:02d}"


# --- Поправки истины ------------------------------------------------------------------------


def ensure_seed_corrections(path: Path = CORRECTIONS_CSV) -> int:
    """Создать `truth_corrections.csv`, если его нет, и дописать недостающие проверенные
    случаи из приложения плана. Существующие строки не трогает и не переупорядочивает —
    T04 и T21 тоже пишут в этот файл. Возвращает число добавленных строк.
    """
    ensure_dir(path.parent)
    existing: list[dict[str, str]] = []
    if path.exists():
        with path.open("r", encoding="utf-8-sig", newline="") as fh:
            existing = list(csv.DictReader(fh))
    existing_keys = {(row["scan_id"], row["field"]) for row in existing}
    missing = [
        row for row in _SEED_CORRECTIONS if (row["scan_id"], row["field"]) not in existing_keys
    ]
    if not missing:
        return 0
    rows = [*existing, *missing]
    with path.open("w", encoding="utf-8-sig", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=CORRECTIONS_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    return len(missing)


@dataclass(frozen=True)
class Correction:
    export_value: str
    corrected_value: str


def load_corrections(path: Path = CORRECTIONS_CSV) -> dict[tuple[str, str], Correction]:
    """Подтверждённые (`status=confirmed`) поправки истины: `(scan_id, field) -> Correction`.

    `proposed`/`rejected` не применяются (ждут решения человека, H1).
    """
    if not path.exists():
        return {}
    out: dict[tuple[str, str], Correction] = {}
    with path.open("r", encoding="utf-8-sig", newline="") as fh:
        for row in csv.DictReader(fh):
            if row.get("status") != "confirmed":
                continue
            out[(row["scan_id"].strip(), row["field"].strip())] = Correction(
                export_value=row["export_value"].strip(),
                corrected_value=row["corrected_value"].strip(),
            )
    return out


# --- Вспомогательное --------------------------------------------------------------------------


def make_scan_id(voucher_file: str, year: int) -> str:
    """`<год>_<имя файла без расширения>`, «(2)» заменено на «_2»."""
    stem = Path(str(voucher_file).strip()).stem
    stem = stem.replace("(", "_").replace(")", "")
    return f"{year}_{stem}"


def extract_voucher_number(raw: Any) -> int | None:
    """Номер ваучера как целое: убирает буквенный хвост (`16a` -> 16, `068` -> 68)."""
    text = str(raw).strip()
    digits = ""
    for ch in text:
        if ch.isdigit():
            digits += ch
        else:
            break
    return int(digits) if digits else None


class _UnionFind:
    """Структура «система непересекающихся множеств» для кластеризации пар ваучеров."""

    def __init__(self, n: int) -> None:
        self.parent = list(range(n))

    def find(self, x: int) -> int:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: int, b: int) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.parent[rb] = ra


# --- Отчёт -------------------------------------------------------------------------------------


@dataclass
class BuildReport:
    """Агрегаты сборки манифеста — без имён судов/агентов и текстов писем."""

    total_rows_with_scan: int = 0
    excluded_third_tug: list[str] = field(default_factory=list)
    dedup_groups: list[tuple[str, int]] = field(default_factory=list)  # (scan_id, лишних строк)
    dedup_conflicts: list[str] = field(default_factory=list)
    missing_scans: list[str] = field(default_factory=list)
    unparsed_voucher_number: list[str] = field(default_factory=list)
    correction_mismatches: list[tuple[str, str]] = field(default_factory=list)
    chain_violations_before: list[tuple[str, str]] = field(default_factory=list)
    chain_violations_after: list[tuple[str, str]] = field(default_factory=list)
    nonmult10: list[tuple[str, str, int]] = field(default_factory=list)
    hour_histogram: dict[str, Counter[int]] = field(default_factory=dict)
    busy_hours: list[float] = field(default_factory=list)
    pair_count: int = 0
    pair_full_match_count: int = 0
    counts_by_tug_year_split: Counter[tuple[str, int, str]] = field(default_factory=Counter)
    crosses_midnight_count: int = 0
    minutes_mult10_count: int = 0
    n_final: int = 0


# --- Основная сборка -----------------------------------------------------------------------


def _row_chain_ok(dts: dict[str, datetime | None]) -> tuple[bool, str]:
    """Проверить `left_base <= started_work <= finished_work <= arrived_base`.

    Пропуск хотя бы одного значения не даёт подтвердить цепочку — это тоже нарушение,
    отдельно помеченное как `missing:<row>`.
    """
    order = ["left_base", "started_work", "finished_work", "arrived_base"]
    missing = [r for r in order if dts[r] is None]
    if missing:
        return False, ";".join(f"missing:{r}" for r in missing)
    violations = []
    for a, b in zip(order, order[1:], strict=False):
        va, vb = dts[a], dts[b]
        assert va is not None and vb is not None
        if va > vb:
            violations.append(f"{a}>{b}")
    return not violations, ";".join(violations)


def build(
    dataset_csv: Path = DATASET_CSV,
    corrections_csv: Path = CORRECTIONS_CSV,
) -> tuple[pd.DataFrame, BuildReport]:
    """Собрать манифест из сырой выгрузки. Детерминирована при неизменных входах."""
    report = BuildReport()
    ensure_seed_corrections(corrections_csv)
    corrections = load_corrections(corrections_csv)

    df = pd.read_csv(dataset_csv, dtype=str, keep_default_na=True)
    has_scan = df["voucher_scan_path"].notna() & (df["voucher_scan_path"].str.strip() != "")
    df = df[has_scan].copy()
    report.total_rows_with_scan = len(df)

    df["_row_number"] = df["row_number"].astype(int)

    def _departure_year(raw: str) -> int | None:
        parsed = parse_dt(raw)
        return parsed.year if parsed is not None else None

    df["_year"] = df["base_departure"].apply(_departure_year)
    df["scan_id"] = [
        make_scan_id(vf, int(y)) for vf, y in zip(df["voucher_file"], df["_year"], strict=False)
    ]

    # Третий буксир (МБ Лигер) — не наш, исключаем и перечисляем сканы.
    is_known_tug = df["tug"].isin(TUG_CODE_BY_NAME)
    report.excluded_third_tug = sorted(set(df.loc[~is_known_tug, "scan_id"]))
    df = df[is_known_tug].copy()

    # Дедупликация по scan_id: одна строка на уникальный скан (первая по row_number),
    # лишние — в отчёт.
    kept_rows: list[pd.Series] = []
    for scan_id, group in df.groupby("scan_id", sort=False):
        group = group.sort_values("_row_number")
        if len(group) > 1:
            report.dedup_groups.append((scan_id, len(group) - 1))
            cols = ["tug", "voucher_number", *ROW_SOURCE_COLUMN.values()]
            if group[cols].nunique().gt(1).any():
                report.dedup_conflicts.append(scan_id)
        kept_rows.append(group.iloc[0])
    deduped = pd.DataFrame(kept_rows).reset_index(drop=True)

    records: list[dict[str, Any]] = []
    for row_d in deduped.to_dict("records"):
        scan_id = row_d["scan_id"]
        year = int(row_d["_year"])
        tug_code = TUG_CODE_BY_NAME[row_d["tug"]]
        voucher_number = extract_voucher_number(row_d["voucher_number"])
        if voucher_number is None:
            report.unparsed_voucher_number.append(scan_id)

        scan_path = row_d["voucher_scan_path"]
        scan_exists = Path(scan_path).exists()
        if not scan_exists:
            report.missing_scans.append(scan_id)

        work_type_raw = row_d.get("work_type")
        work_type = normalize_work_type(work_type_raw)
        group = agent_group(row_d.get("agent"))
        application_file = row_d.get("application_file") or ""

        pre: dict[str, ParsedDT | None] = {}
        final: dict[str, ParsedDT | None] = {}
        corrected_fields: list[str] = []
        for r, src_col in ROW_SOURCE_COLUMN.items():
            parsed = parse_dt(row_d.get(src_col))
            pre[r] = parsed
            correction = corrections.get((scan_id, r))
            if correction is not None:
                corrected_fields.append(r)
                if parsed is not None and format_dt(parsed) != correction.export_value:
                    report.correction_mismatches.append((scan_id, r))
                final[r] = parse_dt(correction.corrected_value)
            else:
                final[r] = parsed

        dts_before: dict[str, datetime | None] = {}
        dts_after: dict[str, datetime | None] = {}
        for r in ROWS:
            p_pre, p_final = pre[r], final[r]
            dts_before[r] = p_pre.dt if p_pre is not None else None
            dts_after[r] = p_final.dt if p_final is not None else None
        chain_ok_before, violation_before = _row_chain_ok(dts_before)
        chain_ok_after, violation_after = _row_chain_ok(dts_after)
        if not chain_ok_before:
            report.chain_violations_before.append((scan_id, violation_before))
        if not chain_ok_after:
            report.chain_violations_after.append((scan_id, violation_after))

        # Реальная (со сдвигом «24:00» → 00:00 следующих суток) календарная дата, а не
        # буквально написанная на бланке — иначе переход через полночь, записанный как
        # «24:00», не отличался бы от обычного дня.
        dates_present: set[date] = set()
        for r in ROWS:
            dt_after = dts_after[r]
            if dt_after is not None:
                dates_present.add(dt_after.date())
        crosses_midnight = len(dates_present) > 1
        if crosses_midnight:
            report.crosses_midnight_count += 1

        minutes_present: dict[str, int] = {}
        for r in ROWS:
            p_final = final[r]
            if p_final is not None and p_final.minute is not None:
                minutes_present[r] = p_final.minute
        minutes_mult10 = all(m % 10 == 0 for m in minutes_present.values())
        if minutes_mult10:
            report.minutes_mult10_count += 1
        for r, m in minutes_present.items():
            if m % 10 != 0:
                report.nonmult10.append((scan_id, r, m))

        has_hour24 = False
        for r in ROWS:
            p_final = final[r]
            if p_final is None or p_final.hour is None:
                continue
            if p_final.hour == 24:
                has_hour24 = True
            report.hour_histogram.setdefault(r, Counter())[p_final.hour] += 1

        left_dt = dts_after["left_base"]
        arrived_dt = dts_after["arrived_base"]
        if left_dt is not None and arrived_dt is not None:
            report.busy_hours.append((arrived_dt - left_dt).total_seconds() / 3600.0)

        left_date = final["left_base"].date_only if final["left_base"] else None
        split = "train"
        if left_date is not None:
            if left_date <= TRAIN_END:
                split = "train"
            elif left_date <= VAL_END:
                split = "val"
            else:
                split = "test"

        rec: dict[str, Any] = {
            "scan_id": scan_id,
            "tug_code": tug_code,
            "year": year,
            "voucher_number": voucher_number,
            "voucher_file": row_d["voucher_file"],
            "scan_path": scan_path,
            "scan_exists": scan_exists,
            "ext": Path(scan_path).suffix.lstrip(".").lower(),
            "application_file": application_file,
            "work_type_raw": work_type_raw,
            "work_type": work_type,
            "agent_group": group,
            "app_dt": "",
            "truth_corrected": ";".join(corrected_fields),
            "dedup_conflict": scan_id in report.dedup_conflicts,
            "chain_ok": chain_ok_after,
            "chain_violation": violation_after,
            "crosses_midnight": crosses_midnight,
            "minutes_mult10": minutes_mult10,
            "has_hour24": has_hour24,
            "pair_id": "",
            "split": split,
            "_left_base_dt_for_pairing": left_dt
            or (datetime(left_date.year, left_date.month, left_date.day) if left_date else None),
        }
        # Время из заявки (date_raw + time_raw + base_year) — нет единой готовой колонки
        # app_dt в выгрузке (решение без человека, см. журнал). `date_raw` без года — год
        # подбирается ближайшим к `left_base`, чтобы не промахнуться через Новый год
        # (заявка «31.12» перед работой «05.01» — это прошлый год, не текущий).
        left_ref = left_dt or (
            datetime(left_date.year, left_date.month, left_date.day) if left_date else None
        )
        app_dt = _combine_app_dt(row_d.get("date_raw"), row_d.get("time_raw"), year, left_ref)
        rec["app_dt"] = app_dt.isoformat() if app_dt else ""

        for r in ROWS:
            parsed = final[r]
            dt = dts_after[r]
            rec[f"{r}_dt"] = dt.isoformat() if dt is not None else ""
            for p in PARTS:
                value = getattr(parsed, p) if parsed is not None else None
                rec[f"{r}_{p}"] = value if value is not None else ""
            rec[f"{r}_year"] = parsed.year if parsed is not None else ""

        records.append(rec)

    pairs_meta = _assign_pairs(records)
    report.pair_count = pairs_meta[0]
    report.pair_full_match_count = pairs_meta[1]

    for rec in records:
        rec.pop("_left_base_dt_for_pairing", None)
        report.counts_by_tug_year_split[(rec["tug_code"], rec["year"], rec["split"])] += 1

    report.n_final = len(records)
    columns = _column_order()
    manifest_df = pd.DataFrame(records, columns=columns)
    manifest_df = manifest_df.sort_values(["year", "tug_code", "voucher_number"]).reset_index(
        drop=True
    )
    return manifest_df, report


def _combine_app_dt(
    date_raw: Any, time_raw: Any, year: int, reference: datetime | None
) -> datetime | None:
    """Время заявки — приор декодера, не истина. `date_raw` («ДД.ММ»/«Д.ММ», без года) +
    `time_raw` («ЧЧ:ММ»). Года в `date_raw` нет, а `base_year` иногда не тот: заявку могли
    подать в декабре на работу в начале следующего года. Поэтому год подбирается из
    {`year`-1, `year`, `year`+1} — тот, что даёт дату ближе всего к `reference`
    (обычно `left_base`; приор заявки — максимум несколько суток до работы, T19).
    """
    if date_raw is None or time_raw is None:
        return None
    if isinstance(date_raw, float) and pd.isna(date_raw):
        return None
    if isinstance(time_raw, float) and pd.isna(time_raw):
        return None
    date_text, time_text = str(date_raw).strip(), str(time_raw).strip()
    if not date_text or not time_text:
        return None
    try:
        day_s, month_s = date_text.split(".")
        hour_s, minute_s = time_text.split(":")
        day, month, hour, minute = int(day_s), int(month_s), int(hour_s), int(minute_s)
    except ValueError:
        return None
    candidates = []
    for y in (year - 1, year, year + 1):
        try:
            candidates.append(datetime(y, month, day, hour, minute))
        except ValueError:
            continue
    if not candidates:
        return None
    if reference is None:
        return next((c for c in candidates if c.year == year), candidates[0])
    return min(candidates, key=lambda c: abs((c - reference).total_seconds()))


def _assign_pairs(records: list[dict[str, Any]]) -> tuple[int, int]:
    """Назначить `pair_id` и выровнять `split` внутри пары по более раннему `left_base`.

    Пара = тот же `application_file` + тот же нормализованный вид работ + выход из базы
    в пределах `PAIR_WINDOW_HOURS`. Возвращает (число пар, число пар с полным совпадением
    всех четырёх времён между участниками).
    """
    n = len(records)
    uf = _UnionFind(n)
    by_app: dict[str, list[int]] = defaultdict(list)
    for i, rec in enumerate(records):
        if rec["application_file"]:
            by_app[rec["application_file"]].append(i)
    for idxs in by_app.values():
        for a_pos in range(len(idxs)):
            for b_pos in range(a_pos + 1, len(idxs)):
                i, j = idxs[a_pos], idxs[b_pos]
                if records[i]["work_type"] != records[j]["work_type"]:
                    continue
                dt_i = records[i]["_left_base_dt_for_pairing"]
                dt_j = records[j]["_left_base_dt_for_pairing"]
                if dt_i is None or dt_j is None:
                    continue
                if abs((dt_i - dt_j).total_seconds()) <= PAIR_WINDOW_HOURS * 3600:
                    uf.union(i, j)

    groups: dict[int, list[int]] = defaultdict(list)
    for i in range(n):
        groups[uf.find(i)].append(i)
    components = [g for g in groups.values() if len(g) > 1]
    components.sort(key=lambda g: min(records[i]["scan_id"] for i in g))

    full_match = 0
    for k, comp in enumerate(components, start=1):
        pair_id = f"P{k:04d}"
        earliest = min(
            comp,
            key=lambda i: records[i]["_left_base_dt_for_pairing"] or datetime.max,
        )
        split = records[earliest]["split"]
        row_values = []
        for i in comp:
            records[i]["pair_id"] = pair_id
            records[i]["split"] = split
            row_values.append(tuple(records[i][f"{r}_dt"] for r in ROWS))
        if len(set(row_values)) == 1:
            full_match += 1
    return len(components), full_match


def _column_order() -> list[str]:
    cols = [
        "scan_id",
        "tug_code",
        "year",
        "voucher_number",
        "voucher_file",
        "scan_path",
        "scan_exists",
        "ext",
        "application_file",
        "work_type_raw",
        "work_type",
        "agent_group",
        "app_dt",
        "truth_corrected",
        "dedup_conflict",
        "chain_ok",
        "chain_violation",
        "crosses_midnight",
        "minutes_mult10",
        "has_hour24",
        "pair_id",
        "split",
    ]
    for r in ROWS:
        cols.append(f"{r}_dt")
        for p in PARTS:
            cols.append(f"{r}_{p}")
        cols.append(f"{r}_year")
    return cols


# --- Отчёт в markdown ------------------------------------------------------------------------


def _percentile(values: list[float], q: float) -> float | None:
    if not values:
        return None
    s = sorted(values)
    idx = min(len(s) - 1, max(0, round(q * (len(s) - 1))))
    return s[idx]


def render_report(report: BuildReport) -> str:
    """Отчёт только из агрегатов: счётчики и `scan_id`, без имён судов/агентов."""
    lines: list[str] = ["# Отчёт по манифесту OCR-датасета", ""]

    lines.append("## Источник")
    lines.append(f"- строк со сканом в выгрузке: {report.total_rows_with_scan}")
    lines.append(f"- исключено (третий буксир, МБ Лигер): {len(report.excluded_third_tug)}")
    if report.excluded_third_tug:
        lines.append(f"  - scan_id: {', '.join(report.excluded_third_tug)}")
    lines.append(f"- итоговых строк манифеста (уникальных сканов): {report.n_final}")
    lines.append("")

    lines.append("## Дубли по scan_id")
    extra = sum(n for _, n in report.dedup_groups)
    lines.append(f"- групп с дублями: {len(report.dedup_groups)}, удалено лишних строк: {extra}")
    for scan_id, n in report.dedup_groups:
        conflict = " (конфликт истины)" if scan_id in report.dedup_conflicts else ""
        lines.append(f"  - {scan_id}: +{n}{conflict}")
    lines.append("")

    lines.append("## Отсутствующие файлы сканов")
    lines.append(f"- {len(report.missing_scans)}")
    if report.missing_scans:
        lines.append(f"  - scan_id: {', '.join(report.missing_scans)}")
    lines.append("")

    if report.unparsed_voucher_number:
        lines.append("## Номер ваучера не разобран")
        lines.append(f"- scan_id: {', '.join(report.unparsed_voucher_number)}")
        lines.append("")

    if report.correction_mismatches:
        lines.append("## Поправки с расхождением export_value")
        for scan_id, r in report.correction_mismatches:
            lines.append(f"  - {scan_id} {r}: значение в выгрузке не совпало с export_value")
        lines.append("")

    lines.append("## Нарушения цепочки left ≤ start ≤ end ≤ arrived")
    lines.append(f"- до поправок: {len(report.chain_violations_before)}")
    for scan_id, violation in report.chain_violations_before:
        lines.append(f"  - {scan_id}: {violation}")
    lines.append(f"- после поправок: {len(report.chain_violations_after)}")
    for scan_id, violation in report.chain_violations_after:
        lines.append(f"  - {scan_id}: {violation}")
    lines.append("")

    lines.append("## Полночь и минуты")
    pct_midnight = 100.0 * report.crosses_midnight_count / report.n_final if report.n_final else 0.0
    pct_mult10 = 100.0 * report.minutes_mult10_count / report.n_final if report.n_final else 0.0
    lines.append(f"- переходов через полночь: {pct_midnight:.1f} % (план: 2,2 %)")
    lines.append(f"- минуты кратны 10 (все 4 строки): {pct_mult10:.1f} % (план: 99,8 %)")
    lines.append(f"- некратных значений минут: {len(report.nonmult10)}")
    for scan_id, r, minute in report.nonmult10:
        lines.append(f"  - {scan_id} {r}: {minute}")
    lines.append("")

    lines.append("## Гистограмма часов по строкам")
    for r in ROWS:
        counter = report.hour_histogram.get(r, Counter())
        parts = ", ".join(f"{h}:{counter[h]}" for h in sorted(counter))
        lines.append(f"- {r}: {parts}")
    lines.append("")

    lines.append("## Занятость (arrived − left), часы")
    if report.busy_hours:
        p50 = _percentile(report.busy_hours, 0.5)
        p90 = _percentile(report.busy_hours, 0.9)
        p99 = _percentile(report.busy_hours, 0.99)
        share_le_53 = sum(1 for v in report.busy_hours if v <= 5.3) / len(report.busy_hours)
        lines.append(
            f"- n={len(report.busy_hours)}, медиана={p50:.2f}, p90={p90:.2f}, p99={p99:.2f}"
        )
        lines.append(f"- доля ≤ 5,3 ч: {100 * share_le_53:.1f} % (план: 99 %)")
    else:
        lines.append("- нет данных")
    lines.append("")

    lines.append("## Пары ваучеров (оба буксира, одна работа)")
    pct_full = (
        100.0 * report.pair_full_match_count / report.pair_count if report.pair_count else 0.0
    )
    lines.append(f"- найдено пар: {report.pair_count} (план: ≈195)")
    lines.append(f"- все 4 времени совпадают: {pct_full:.1f} % (план: ≈67 %)")
    lines.append("")

    lines.append("## Буксир × год × сплит")
    for (tug_code, year, split), n in sorted(report.counts_by_tug_year_split.items()):
        lines.append(f"- {tug_code} {year} {split}: {n}")
    lines.append("")

    return "\n".join(lines)


# --- CLI -----------------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Сборка манифеста OCR-датасета ваучеров")
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("build", help="собрать data/ocr/manifest.csv и отчёт")
    args = parser.parse_args(argv)

    if args.command == "build":
        manifest_df, report = build()
        ensure_dir(MANIFEST.parent)
        manifest_df.to_csv(MANIFEST, index=False, encoding="utf-8-sig")
        ensure_dir(REPORTS_DIR)
        report_path = REPORTS_DIR / "manifest_report.md"
        report_path.write_text(render_report(report), encoding="utf-8")
        print(f"manifest: {len(manifest_df)} строк -> {MANIFEST}")
        print(f"отчёт -> {report_path}")
        return 0
    return 1


if __name__ == "__main__":
    sys.exit(main())
