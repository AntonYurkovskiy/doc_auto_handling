"""Reconcile the clean export with voucher and order indexes."""

from __future__ import annotations

import argparse
import re
from pathlib import Path

import pandas as pd
from bs4 import BeautifulSoup
from eml_parser import FIELDS as EML_FIELDS
from eml_parser import parse_message

EXPORT_COLUMNS = [
    "row_number",
    "tug",
    "vessel",
    "work_type",
    "agent",
    "voucher_number",
    "base_departure",
    "base_arrival",
    "occupied_raw",
    "work_start",
    "work_end",
    "work_duration_raw",
    "calculation_note",
    "application_file",
    "voucher_file",
    "grt_raw",
    "amount",
    "currency",
    "exchange_rate",
    "revenue_rub",
    "report",
    "edit",
    "delete",
]


def read_export(path: Path) -> pd.DataFrame:
    soup = BeautifulSoup(path.read_bytes().decode("cp1251"), "html.parser")
    rows = []
    for row in soup.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        if len(cells) >= len(EXPORT_COLUMNS):
            rows.append(cells[: len(EXPORT_COLUMNS)])
    return pd.DataFrame(rows, columns=EXPORT_COLUMNS)


def normalize_key(value: object) -> str:
    text = str(value or "").lower()
    text = re.sub(r"\.pdf$", "", text)
    text = re.sub(r"^(?:re|fw|fwd)[\s:_-]*", "", text)
    text = re.sub(r"[№#]", "", text)
    return re.sub(r"[^0-9a-zа-яё]", "", text)


def extract_year(value: object) -> str:
    match = re.search(r"(?<!\d)(\d{4})(?!\d)", str(value or ""))
    return match.group(1) if match else ""


def read_indexes(vouchers_path: Path, orders_path: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    vouchers = pd.read_csv(vouchers_path, encoding="utf-8-sig", dtype=str).fillna("")
    orders = pd.read_csv(orders_path, encoding="utf-8-sig", dtype=str).fillna("")
    vouchers["voucher_name_key"] = vouchers["File"].map(normalize_key)
    vouchers["voucher_year"] = vouchers["Folder"].map(extract_year)
    vouchers["voucher_key"] = vouchers.apply(
        lambda row: make_voucher_key(row["voucher_name_key"], row["voucher_year"]),
        axis=1,
    )
    orders["order_key"] = orders["Subject"].map(normalize_key)
    return vouchers, orders


def email_records(orders: pd.DataFrame) -> pd.DataFrame:
    records = []
    for path in orders["Path"]:
        file_path = Path(path)
        if file_path.exists():
            records.append(parse_message(file_path))
    if not records:
        return pd.DataFrame(columns=EML_FIELDS)
    return pd.DataFrame(records).drop_duplicates(subset=["path"])


def agent_group(value: object) -> str:
    agent = str(value or "").strip()
    if agent == "Транс-Агро":
        return "A=Транс-Агро"
    if agent in {"Терминал", "Содружество - Соя"}:
        return "B={Терминал, Содружество - Соя}"
    if not agent or agent == "-":
        return "unknown"
    return "C=прочие"


def make_voucher_key(name: object, year: object) -> str:
    name_key = str(name or "")
    year_key = str(year or "")
    return f"{name_key}|{year_key}" if name_key and year_key else ""


def markdown_counts(series: pd.Series) -> str:
    lines = ["| Группа | Строк |", "|---|---:|"]
    lines.extend(f"| {index} | {value} |" for index, value in series.items())
    return "\n".join(lines)


def save_report(
    report_path: Path,
    export: pd.DataFrame,
    vouchers: pd.DataFrame,
    orders: pd.DataFrame,
    voucher_matches: pd.Series,
    order_matches: pd.Series,
) -> None:
    orphan_vouchers = vouchers.loc[
        ~vouchers["voucher_key"].isin(export["voucher_key"]), "File"
    ]
    orphan_export_vouchers = export.loc[
        ~export["voucher_key"].isin(vouchers["voucher_key"]), "voucher_file"
    ]
    orphan_orders = orders.loc[~orders["order_key"].isin(export["order_key"]), "Subject"]
    orphan_export_orders = export.loc[
        ~export["order_key"].isin(orders["order_key"]), "application_file"
    ]
    unmatched_agents = export.loc[~order_matches, "agent"].map(agent_group).value_counts()
    voucher_keys = set(vouchers.loc[vouchers["voucher_key"] != "", "voucher_key"])
    export_voucher_keys = set(export.loc[export["voucher_key"] != "", "voucher_key"])
    voucher_name_keys = set(
        vouchers.loc[vouchers["voucher_name_key"] != "", "voucher_name_key"]
    )
    export_voucher_name_keys = set(
        export.loc[export["voucher_name_key"] != "", "voucher_name_key"]
    )
    order_keys = set(orders.loc[orders["order_key"] != "", "order_key"])
    export_order_keys = set(export.loc[export["order_key"] != "", "order_key"])
    voucher_intersection = voucher_keys & export_voucher_keys
    voucher_name_intersection = voucher_name_keys & export_voucher_name_keys
    voucher_name_matches = export["voucher_name_key"].isin(voucher_name_keys)
    voucher_year_collisions = voucher_name_matches & ~voucher_matches
    voucher_collision_keys = set(
        export.loc[
            voucher_year_collisions & export["voucher_key"].ne(""), "voucher_key"
        ]
    )
    order_intersection = order_keys & export_order_keys
    order_row_intersection = int(orders["order_key"].isin(export_order_keys).sum())
    lines = [
        "# Сверка исторических заявок, ваучеров и выгрузки",
        "",
        f"- Строк выгрузки: **{len(export):,}**.",
        (
            f"- Ваучеров в индексе: **{len(voucher_keys):,}**; связано по имени+году: "
            f"**{len(voucher_intersection):,}**; уникальных ваучеров выгрузки найдено "
            f"на диске: **{len(voucher_intersection):,}**."
        ),
        (
            f"- При сверке только по имени совпало **{len(voucher_name_intersection):,}** "
            f"уникальных имён; строк, где имя есть, но год не совпал: "
            f"**{int(voucher_year_collisions.sum()):,}**; уникальных пар имя+год "
            f"с такой коллизией: **{len(voucher_collision_keys):,}**."
        ),
        f"- Строк выгрузки без ваучера в индексе: **{int((~voucher_matches).sum()):,}**.",
        (
            f"- Писем в индексе: **{len(order_keys):,}**; найдено в выгрузке: "
            f"**{len(order_intersection):,}**; уникальных заявок выгрузки найдено "
            f"в индексе: **{len(order_intersection):,}**; строк индекса с совпадением: "
            f"**{order_row_intersection:,}**."
        ),
        f"- Строк выгрузки без заявки в индексе: **{int((~order_matches).sum()):,}**.",
        "",
        "## Несвязанные строки по группам агентов",
        "",
        markdown_counts(unmatched_agents),
        "",
        "## Файлы-сироты (примеры)",
        "",
        f"- Ваучеры на диске без строки выгрузки: {list(orphan_vouchers.head(20))}",
        f"- Ваучеры выгрузки без файла в индексе: {list(orphan_export_vouchers.head(20))}",
        f"- Письма без строки выгрузки: {list(orphan_orders.head(20))}",
        f"- Заявки выгрузки без письма: {list(orphan_export_orders.head(20))}",
    ]
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


def build_combined(
    export: pd.DataFrame,
    vouchers: pd.DataFrame,
    orders: pd.DataFrame,
    emls: pd.DataFrame,
    output: Path,
) -> None:
    voucher_columns = vouchers[["voucher_key", "Path"]].rename(
        columns={"Path": "voucher_scan_path"}
    ).drop_duplicates(subset=["voucher_key"])
    order_columns = orders[["order_key", "Path"]].rename(
        columns={"Path": "email_path"}
    ).drop_duplicates(subset=["order_key"])
    combined = export.merge(voucher_columns, on="voucher_key", how="left")
    combined = combined.merge(order_columns, on="order_key", how="left")
    if not emls.empty:
        eml_columns = emls.rename(columns={"path": "email_path"})
        combined = combined.merge(eml_columns, on="email_path", how="left", suffixes=("", "_eml"))
    output.parent.mkdir(parents=True, exist_ok=True)
    combined.to_csv(output, index=False, encoding="utf-8-sig")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("extraction", type=Path)
    parser.add_argument("vouchers", type=Path)
    parser.add_argument("orders", type=Path)
    parser.add_argument("--report", type=Path, default=Path("analysis/reconcile_report.md"))
    parser.add_argument(
        "--combined-output", type=Path, default=Path("analysis/reconciled_dataset.csv")
    )
    args = parser.parse_args()

    export = read_export(args.extraction)
    vouchers, orders = read_indexes(args.vouchers, args.orders)
    export["voucher_name_key"] = export["voucher_file"].map(normalize_key)
    export["base_year"] = export["base_departure"].map(extract_year)
    work_year = export["work_start"].map(extract_year)
    export.loc[export["base_year"] == "", "base_year"] = work_year
    export["voucher_key"] = export.apply(
        lambda row: make_voucher_key(row["voucher_name_key"], row["base_year"]),
        axis=1,
    )
    export["order_key"] = export["application_file"].map(normalize_key)
    voucher_matches = export["voucher_key"].isin(vouchers["voucher_key"])
    order_matches = export["order_key"].isin(orders["order_key"])
    emls = email_records(orders)
    save_report(args.report, export, vouchers, orders, voucher_matches, order_matches)
    build_combined(export, vouchers, orders, emls, args.combined_output)
    print(f"\nCombined dataset: {args.combined_output}")


if __name__ == "__main__":
    main()
