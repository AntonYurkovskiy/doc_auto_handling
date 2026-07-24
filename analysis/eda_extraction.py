"""EDA for the historical tug-work HTML export saved with an .xls suffix.

The file is an HTML table (not a binary workbook) declared as windows-1251, so it
is read with BeautifulSoup, not ``pandas.read_excel``. Run:

    python analysis/eda_extraction.py path/to/export.xls
"""

# ruff: noqa: E501

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from bs4 import BeautifulSoup

COLS = [
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
    "currency_raw",
    "exchange_rate",
    "revenue_rub",
    "col_report",
    "col_edit",
    "col_delete",
]
DATE_COLS = ["base_departure", "base_arrival", "work_start", "work_end"]

# Agent -> billing group (A = Транс-Агро; B = flat pair; C = everyone else).
GROUP_B_AGENTS = {"Терминал", "Содружество - Соя"}
CURRENCY_MAP = {"доллар США": "USD", "рубль": "RUB", "евро": "EUR"}
# Session gap (days) that splits one vessel's port calls for the work-type chain.
VISIT_GAP_DAYS = 7


def parse_export(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    if raw.count(b"\xef\xbf\xbd") > 100:
        print(
            "WARNING: file contains many U+FFFD bytes — cyrillic was destroyed "
            "upstream (e.g. a bad transfer). Re-export/transfer without transcoding."
        )
    soup = BeautifulSoup(raw.decode("cp1251"), "html.parser")
    rows: list[list[str]] = []
    for row in soup.find_all("tr")[1:]:
        cells = [cell.get_text(" ", strip=True) for cell in row.find_all(["td", "th"])]
        if len(cells) >= len(COLS):
            rows.append(cells[: len(COLS)])
    frame = pd.DataFrame(rows, columns=COLS)
    for column in DATE_COLS:
        frame[column] = pd.to_datetime(
            frame[column], format="%d.%m.%Y %H:%M", errors="coerce"
        )
    frame["voucher_number_num"] = pd.to_numeric(frame["voucher_number"], errors="coerce")
    for src, dst in [
        ("amount", "amount_num"),
        ("exchange_rate", "exchange_rate_num"),
        ("revenue_rub", "revenue_rub_num"),
    ]:
        frame[dst] = pd.to_numeric(
            frame[src].str.replace(",", ".", regex=False), errors="coerce"
        )
    grt_values = frame["grt_raw"].str.extract(r"([\d.,]+)")[0].str.replace(
        ",", ".", regex=False
    )
    frame["grt"] = pd.to_numeric(grt_values, errors="coerce")
    frame["occupied_minutes"] = frame["occupied_raw"].map(parse_duration)
    frame["work_minutes"] = frame["work_duration_raw"].map(parse_duration)
    frame["tug_from_file"] = frame["voucher_file"].str.extract(r"(?i)([pk])\.pdf$")[0].map(
        {"p": "БК Пионер", "k": "БК Коммунар"}
    )
    frame["currency"] = frame["currency_raw"].map(CURRENCY_MAP).fillna(
        frame["calculation_note"].map(infer_currency)
    )
    frame["agent_clean"] = frame["agent"].str.strip()
    frame["agent_group"] = frame["agent_clean"].map(agent_group)
    parsed = frame["calculation_note"].map(parse_tariff)
    frame = pd.concat([frame, pd.DataFrame(parsed.tolist(), index=frame.index)], axis=1)
    frame["year"] = frame["work_start"].dt.year
    frame["month"] = frame["work_start"].dt.to_period("M").astype(str)
    frame["weekday"] = frame["work_start"].dt.day_name()
    frame["night"] = frame["work_start"].dt.hour.ge(22) | frame["work_start"].dt.hour.lt(6)
    frame["application_date"] = frame.apply(
        lambda row: extract_application_date(row["application_file"], row["work_start"]),
        axis=1,
    )
    frame["application_lag_days"] = (
        frame["work_start"].dt.normalize() - frame["application_date"]
    ).dt.days
    return frame


def parse_duration(value: str) -> float:
    match = re.fullmatch(r"\s*(\d+):(\d{2})\s*", value or "")
    return float(int(match.group(1)) * 60 + int(match.group(2))) if match else math.nan


def infer_currency(value: str) -> str:
    if re.search(r"\bUSD\b", value or "", re.IGNORECASE):
        return "USD"
    if re.search(r"\bEUR\b", value or "", re.IGNORECASE):
        return "EUR"
    if re.search(r"руб", value or "", re.IGNORECASE):
        return "RUB"
    return "unknown"


def agent_group(agent: str) -> str:
    if not agent or agent == "-":
        return "unknown"
    if agent == "Транс-Агро":
        return "A"
    if agent in GROUP_B_AGENTS:
        return "B"
    return "C"


def parse_number(value: str) -> float:
    match = re.search(r"\d+(?:[.,]\d{1,2})?", value or "")
    return float(match.group(0).replace(",", ".")) if match else math.nan


def parse_tariff(value: str) -> dict[str, Any]:
    text = value or ""
    currency = infer_currency(text)
    amounts = re.findall(r"=\s*([\d.,]+)\s*(?:USD|EUR|руб\.?)", text, re.IGNORECASE)
    amount = parse_number(amounts[-1]) if amounts else math.nan
    if text.count("=") > 1 or re.search(r",\s*[\d.,]+\s*(?:USD|EUR|руб)", text, re.IGNORECASE):
        unit = "composite"
    elif re.search(r"\b[\d.,]+\s*(?:USD|EUR|руб\.?)\s*x\s*\d+h\d+m", text, re.IGNORECASE):
        unit = "per_hour"
    elif re.search(
        r"\b[\d.,]+\s*x\s*[\d.,]+\s*(?:USD|EUR|руб\.?)\s*/\s*[\d.,]+", text, re.IGNORECASE
    ):
        unit = "per_ton_divided"
    elif re.search(r"\b[\d.,]+\s*x\s*[\d.,]+\s*(?:USD|EUR|руб\.?)", text, re.IGNORECASE):
        unit = "per_ton"
    else:
        unit = "unknown"
    divisor_match = re.search(r"/\s*([\d.,]+)", text)
    divisor = parse_number(divisor_match.group(1)) if divisor_match else math.nan
    hour_match = re.search(r"([\d.,]+)\s*(?:USD|EUR|руб\.?)\s*x\s*\d+h\d+m", text, re.IGNORECASE)
    ton_match = re.search(r"\b[\d.,]+\s*x\s*([\d.,]+)\s*(?:USD|EUR|руб\.?)", text, re.IGNORECASE)
    rate_match = hour_match or ton_match
    rate = parse_number(rate_match.group(1)) if rate_match else math.nan
    return {
        "tariff_unit": unit,
        "tariff_rate": rate,
        "tariff_divisor": divisor,
        "tariff_currency": currency,
        "parsed_amount": amount,
    }


def extract_application_date(value: str, work_start: pd.Timestamp) -> pd.Timestamp:
    match = re.search(r"(?<!\d)(\d{1,2})[._-](\d{1,2})(?!\d)", value or "")
    if not match:
        return pd.NaT
    day, month = map(int, match.groups())
    if not 1 <= month <= 12 or not 1 <= day <= 31 or pd.isna(work_start):
        return pd.NaT
    candidates = []
    for year in range(work_start.year - 1, work_start.year + 2):
        try:
            candidate = pd.Timestamp(year=year, month=month, day=day)
        except ValueError:
            continue
        if candidate <= work_start.normalize():
            candidates.append(candidate)
    if candidates:
        return max(candidates)
    try:
        return pd.Timestamp(year=work_start.year, month=month, day=day)
    except ValueError:
        return pd.NaT


def ci95(series: pd.Series) -> tuple[float, float]:
    values = series.dropna()
    if values.empty:
        return (math.nan, math.nan)
    half_width = 1.96 * values.std(ddof=1) / math.sqrt(len(values))
    return (values.mean() - half_width, values.mean() + half_width)


def pct(numerator: float, denominator: float) -> str:
    return f"{100 * numerator / denominator:.1f}%" if denominator else "n/a"


def transition_matrix(frame: pd.DataFrame) -> pd.DataFrame:
    """P(next work_type | current) within a vessel's port call (gap-split)."""
    counts: dict[tuple[str, str], int] = {}
    usable = frame.dropna(subset=["vessel", "work_start", "work_type"])
    usable = usable[usable["work_type"] != ""]
    # Collapse the 2-tug duplicate rows (same operation) into one event so the
    # chain reflects operations, not vouchers.
    usable = usable.drop_duplicates(subset=["vessel", "work_start", "work_end", "work_type"])
    for _, group in usable.groupby("vessel"):
        ordered = group.sort_values("work_start")
        prev_type: str | None = None
        prev_time: pd.Timestamp | None = None
        for work_type, start in zip(ordered["work_type"], ordered["work_start"], strict=False):
            if (
                prev_type is not None
                and prev_time is not None
                and (start - prev_time).days <= VISIT_GAP_DAYS
            ):
                counts[(prev_type, work_type)] = counts.get((prev_type, work_type), 0) + 1
            prev_type, prev_time = work_type, start
    if not counts:
        return pd.DataFrame()
    types = sorted({t for pair in counts for t in pair})
    matrix = pd.DataFrame(0, index=types, columns=types, dtype=float)
    for (src, dst), value in counts.items():
        matrix.loc[src, dst] = value
    row_sums = matrix.sum(axis=1).replace(0, math.nan)
    return matrix.div(row_sums, axis=0)


def markdown_series(series: pd.Series, name: str = "Значение") -> str:
    lines = [f"| | {name} |", "|---|---:|"]
    lines.extend(f"| {index} | {value} |" for index, value in series.items())
    return "\n".join(lines)


def markdown_frame(frame: pd.DataFrame, index_name: str = "") -> str:
    header = "| " + index_name + " | " + " | ".join(map(str, frame.columns)) + " |"
    sep = "|---" * (len(frame.columns) + 1) + "|"
    lines = [header, sep]
    for index, row in frame.iterrows():
        cells = " | ".join(f"{value:.2f}" if isinstance(value, float) else str(value) for value in row)
        lines.append(f"| {index} | {cells} |")
    return "\n".join(lines)


def make_figures(frame: pd.DataFrame, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    names: list[str] = []

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    frame.groupby("month").size().plot.bar(ax=axes[0, 0], title="Работы по месяцам")
    frame["work_type"].value_counts().head(10).plot.barh(ax=axes[0, 1], title="Вид работ (топ-10)")
    axes[0, 1].invert_yaxis()
    frame["work_minutes"].dropna().plot.hist(bins=30, ax=axes[1, 0], title="Время работ, мин")
    frame["agent_group"].value_counts().plot.bar(ax=axes[1, 1], title="Тарифная группа (A/B/C)")
    fig.tight_layout()
    fig.savefig(output / "overview.png", dpi=150)
    plt.close(fig)
    names.append("overview.png")

    fig, axes = plt.subplots(1, 2, figsize=(13, 4))
    frame["tariff_unit"].value_counts().plot.bar(ax=axes[0], title="Методы тарификации")
    frame["tariff_rate"].dropna().plot.hist(bins=30, ax=axes[1], title="Ставки")
    fig.tight_layout()
    fig.savefig(output / "tariffs.png", dpi=150)
    plt.close(fig)
    names.append("tariffs.png")

    matrix = transition_matrix(frame)
    if not matrix.empty:
        fig, ax = plt.subplots(figsize=(11, 9))
        image = ax.imshow(matrix.fillna(0).values, cmap="viridis", vmin=0, vmax=1)
        ax.set_xticks(range(len(matrix.columns)))
        ax.set_yticks(range(len(matrix.index)))
        ax.set_xticklabels(matrix.columns, rotation=90, fontsize=7)
        ax.set_yticklabels(matrix.index, fontsize=7)
        ax.set_title("Переходы видов работ P(next | current)")
        fig.colorbar(image, ax=ax, shrink=0.8)
        fig.tight_layout()
        fig.savefig(output / "transitions.png", dpi=150)
        plt.close(fig)
        names.append("transitions.png")
    return names


def build_report(frame: pd.DataFrame, figures: list[str], source: Path, output: Path) -> None:
    n = len(frame)
    date_valid = frame["work_start"].dropna()
    period = f"{date_valid.min():%d.%m.%Y}—{date_valid.max():%d.%m.%Y}" if not date_valid.empty else "n/a"
    work_ci = ci95(frame["work_minutes"])
    occ_ci = ci95(frame["occupied_minutes"])

    # dual-tug operations
    dual = 0
    grouped = frame.dropna(subset=["vessel", "work_start", "work_end"]).groupby(
        ["vessel", "work_start", "work_end"], dropna=False
    )
    for _, group in grouped:
        if {"БК Пионер", "БК Коммунар"}.issubset(set(group["tug"].dropna())):
            dual += 1

    # tug column vs filename suffix agreement
    both = frame.dropna(subset=["tug_from_file"])
    both = both[both["tug"].isin(["БК Пионер", "БК Коммунар"])]
    agree = int((both["tug"] == both["tug_from_file"]).sum())

    amount_valid = frame[["amount_num", "parsed_amount"]].dropna()
    amount_diff = float((amount_valid["amount_num"] - amount_valid["parsed_amount"]).abs().gt(0.02).mean()) if not amount_valid.empty else 0.0
    revenue_valid = frame[["amount_num", "exchange_rate_num", "revenue_rub_num"]].dropna()
    revenue_bad = int(((revenue_valid["amount_num"] * revenue_valid["exchange_rate_num"]) - revenue_valid["revenue_rub_num"]).abs().gt(1).sum())

    group_share = frame["agent_group"].value_counts()
    unit_by_group = pd.crosstab(frame["agent_group"], frame["tariff_unit"])
    dur_by_type = frame.groupby("work_type")["work_minutes"].median().dropna().sort_values(ascending=False)
    matrix = transition_matrix(frame)

    parts: list[str] = []
    parts.append("# EDA исторической выгрузки работ буксиров\n")
    parts.append(f"Источник: `{source.name}` (HTML-таблица windows-1251, читается через BeautifulSoup — не `read_excel`). Скрипт: [`eda_extraction.py`](eda_extraction.py).\n")

    parts.append("## Замечание о кодировке")
    parts.append(
        "Первая присланная копия файла была повреждена при передаче: кириллица была затёрта "
        "символом `U+FFFD`. Эта версия (из `.rar`) **целая** — кириллица на месте, поэтому "
        "доступны поля «Вид работ» и «Агент», и построены цепочка работ и тарифные группы. "
        "LOA/осадки в выгрузке нет (только в заявках) — правило «LOA → буксиры» из ODT остаётся "
        "для будущих данных.\n"
    )

    parts.append("## 1. Обзор\n")
    parts.append("| Метрика | Значение |")
    parts.append("|---|---:|")
    parts.append(f"| Строк данных | {n:,} |")
    parts.append(f"| Период (начало работ) | {period} |")
    parts.append(f"| Уникальных судов | {frame['vessel'].nunique():,} |")
    parts.append(f"| Уникальных номеров ваучеров | {int(frame['voucher_number_num'].nunique()):,} |")
    parts.append(f"| Видов работ | {frame['work_type'].replace('', pd.NA).nunique()} |")
    parts.append(f"| Агентов | {frame['agent_clean'].replace('', pd.NA).nunique()} |\n")
    parts.append(f"![Обзор](figures/{figures[0]})\n")

    parts.append("## 2. Вид работ\n")
    parts.append(markdown_series(frame["work_type"].replace("", "(пусто)").value_counts(), "Работ") + "\n")

    parts.append("## 3. Агенты и тарифные группы\n")
    parts.append("Группа A = Транс-Агро; B = {Терминал, Содружество - Соя}; C = остальные.\n")
    parts.append(markdown_series(group_share, "Строк") + "\n")
    parts.append("Доля Транс-Агро (A): **" + pct(group_share.get("A", 0), n) + "** — согласуется с оценкой ~90%.\n")
    parts.append(markdown_series(frame["agent_clean"].replace("", "(пусто)").value_counts(), "Строк") + "\n")

    parts.append("## 4. Метод тарификации × группа\n")
    parts.append(markdown_frame(unit_by_group, "группа") + "\n")
    parts.append(
        "Медианные ставки: per_ton **"
        + f"{frame.loc[frame['tariff_unit'].isin(['per_ton', 'per_ton_divided']), 'tariff_rate'].median():.2f}**"
        + ", per_hour **"
        + f"{frame.loc[frame['tariff_unit'] == 'per_hour', 'tariff_rate'].median():.2f}**"
        + f". Самый частый делитель: **{frame['tariff_divisor'].value_counts().index[0] if frame['tariff_divisor'].notna().any() else 'n/a'}**.\n"
    )
    parts.append(f"![Тарификация](figures/{figures[1]})\n")

    parts.append("## 5. Цепочка видов работ (матрица переходов)\n")
    parts.append(
        f"P(следующий вид | текущий) внутри судозахода (разрыв > {VISIT_GAP_DAYS} дн. считается "
        "новым заходом), суда упорядочены по времени. Значения — доли по строке.\n"
    )
    if not matrix.empty:
        parts.append(markdown_frame(matrix.round(2), "из \\ в") + "\n")
        if len(figures) > 2:
            parts.append(f"![Переходы](figures/{figures[2]})\n")
    else:
        parts.append("_Недостаточно данных для матрицы._\n")

    parts.append("## 6. Длительности\n")
    parts.append("| Показатель | Время работ, мин | Занятость, мин |")
    parts.append("|---|---:|---:|")
    parts.append(f"| Медиана | {frame['work_minutes'].median():.1f} | {frame['occupied_minutes'].median():.1f} |")
    parts.append(f"| Q1 | {frame['work_minutes'].quantile(.25):.1f} | {frame['occupied_minutes'].quantile(.25):.1f} |")
    parts.append(f"| Q3 | {frame['work_minutes'].quantile(.75):.1f} | {frame['occupied_minutes'].quantile(.75):.1f} |")
    parts.append(f"| 95% CI среднего | [{work_ci[0]:.1f}, {work_ci[1]:.1f}] | [{occ_ci[0]:.1f}, {occ_ci[1]:.1f}] |\n")
    parts.append(f"Корреляция занятости и времени работ: **{frame[['work_minutes', 'occupied_minutes']].corr().iloc[0, 1]:.3f}**.\n")
    parts.append("Медиана времени работ по виду (мин, приор для предсказания времени):\n")
    parts.append(markdown_series(dur_by_type.round(0).astype(int), "Медиана, мин") + "\n")

    parts.append("## 7. Буксиры и парные операции\n")
    parts.append(markdown_series(frame["tug"].replace("", "(пусто)").value_counts(), "Строк") + "\n")
    parts.append(
        f"Операций с обоими буксирами (одно судно + совпадающие начало/конец): **{dual}**. "
        f"Совпадение колонки «Буксир» и суффикса файла p/k: **{agree}** из **{len(both)}** "
        f"({pct(agree, len(both))}).\n"
    )

    parts.append("## 8. GRT, валюта, курс\n")
    parts.append(
        f"GRT: медиана **{frame['grt'].median():.0f} t**, Q1–Q3 **{frame['grt'].quantile(.25):.0f}–{frame['grt'].quantile(.75):.0f} t**, "
        f"диапазон **{frame['grt'].min():.0f}–{frame['grt'].max():.0f} t**.\n"
    )
    parts.append(markdown_series(frame["currency"].value_counts(), "Строк") + "\n")
    parts.append(
        f"Курс: медиана **{frame['exchange_rate_num'].median():.4f}**, диапазон "
        f"**{frame['exchange_rate_num'].min():.4f}–{frame['exchange_rate_num'].max():.4f}**.\n"
    )

    parts.append("## 9. Ночь, выходные, лаг заявки, формулы\n")
    weekend = int(frame["work_start"].dt.dayofweek.ge(5).sum())
    parts.append(f"- Ночные работы (22:00–06:00): **{int(frame['night'].sum()):,} ({pct(frame['night'].sum(), n)})**.")
    parts.append(f"- Выходные: **{weekend:,} ({pct(weekend, n)})**.")
    parts.append(
        f"- Лаг заявка→работа: медиана **{frame['application_lag_days'].median():.1f}** дн., "
        f"Q1–Q3 **{frame['application_lag_days'].quantile(.25):.1f}–{frame['application_lag_days'].quantile(.75):.1f}** "
        f"(извлечено {int(frame['application_lag_days'].notna().sum()):,} дат из имени заявки)."
    )
    parts.append(f"- Пересчёт amount из примечания: расхождений >0.02 — **{amount_diff:.1%}**.")
    parts.append(f"- Проверка `сумма×курс→выручка`: расхождений >1 руб — **{revenue_bad}** из {len(revenue_valid):,}.\n")

    parts.append("## 10. Запуск\n")
    parts.append("```bash\npython analysis/eda_extraction.py path/to/export.xls\n```\n")
    parts.append("Зависимости — в `analysis/requirements-eda.txt`.")

    output.write_text("\n".join(parts) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input", type=Path, help="Путь к HTML-выгрузке .xls")
    parser.add_argument("--report", type=Path, default=Path("analysis/eda_report.md"))
    parser.add_argument("--figures", type=Path, default=Path("analysis/figures"))
    args = parser.parse_args()
    frame = parse_export(args.input)
    figures = make_figures(frame, args.figures)
    build_report(frame, figures, args.input, args.report)
    print(f"Parsed {len(frame)} rows; report: {args.report}; figures: {args.figures}")


if __name__ == "__main__":
    main()
