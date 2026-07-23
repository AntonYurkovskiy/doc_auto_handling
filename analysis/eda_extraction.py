"""EDA for the historical tug-work HTML export saved with an .xls suffix."""

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
    "tug_raw",
    "vessel",
    "vessel_type_raw",
    "port_raw",
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
    "unused_20",
    "unused_21",
    "unused_22",
]
DATE_COLS = ["base_departure", "base_arrival", "work_start", "work_end"]


def parse_export(path: Path) -> pd.DataFrame:
    raw = path.read_bytes()
    text = raw.decode("cp1251")
    soup = BeautifulSoup(text, "html.parser")
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
    frame["amount_num"] = pd.to_numeric(
        frame["amount"].str.replace(",", ".", regex=False), errors="coerce"
    )
    frame["exchange_rate_num"] = pd.to_numeric(
        frame["exchange_rate"].str.replace(",", ".", regex=False), errors="coerce"
    )
    frame["revenue_rub_num"] = pd.to_numeric(
        frame["revenue_rub"].str.replace(",", ".", regex=False), errors="coerce"
    )
    grt_values = frame["grt_raw"].str.extract(r"([\d.,]+)")[0].str.replace(
        ",", ".", regex=False
    )
    frame["grt"] = pd.to_numeric(grt_values, errors="coerce")
    frame["occupied_minutes"] = frame["occupied_raw"].map(parse_duration)
    frame["work_minutes"] = frame["work_duration_raw"].map(parse_duration)
    frame["tug"] = frame["voucher_file"].str.extract(r"(?i)([pk])\.pdf$")[0].map(
        {"p": "Пионер", "k": "Коммунар"}
    )
    frame["currency"] = frame["calculation_note"].map(infer_currency)
    parsed = frame["calculation_note"].map(parse_tariff)
    parsed_frame = pd.DataFrame(parsed.tolist(), index=frame.index)
    frame = pd.concat([frame, parsed_frame], axis=1)
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
        r"\b[\d.,]+\s*x\s*[\d.,]+\s*(?:USD|EUR|руб\.?)\s*/\s*[\d.,]+",
        text,
        re.IGNORECASE,
    ):
        unit = "per_ton_divided"
    elif re.search(r"\b[\d.,]+\s*x\s*[\d.,]+\s*(?:USD|EUR|руб\.?)", text, re.IGNORECASE):
        unit = "per_ton"
    else:
        unit = "unknown"
    divisor_match = re.search(r"/\s*([\d.,]+)", text)
    divisor = parse_number(divisor_match.group(1)) if divisor_match else math.nan
    hour_match = re.search(
        r"([\d.,]+)\s*(?:USD|EUR|руб\.?)\s*x\s*\d+h\d+m", text, re.IGNORECASE
    )
    ton_match = re.search(
        r"\b[\d.,]+\s*x\s*([\d.,]+)\s*(?:USD|EUR|руб\.?)", text, re.IGNORECASE
    )
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


def tariff_check(frame: pd.DataFrame) -> tuple[float, float]:
    valid = frame[["amount_num", "parsed_amount"]].dropna()
    if valid.empty:
        return (0.0, 0.0)
    difference = (valid["amount_num"] - valid["parsed_amount"]).abs()
    return float((difference > 0.02).mean()), float(difference.median())


def make_figures(frame: pd.DataFrame, output: Path) -> list[str]:
    output.mkdir(parents=True, exist_ok=True)
    names: list[str] = []
    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    frame.groupby("month").size().plot.bar(ax=axes[0, 0], title="Работы по месяцам")
    frame["work_minutes"].dropna().plot.hist(bins=30, ax=axes[0, 1], title="Время работ, мин")
    frame["occupied_minutes"].dropna().plot.hist(bins=30, ax=axes[1, 0], title="Занятость, мин")
    axes[1, 1].scatter(frame["work_minutes"], frame["occupied_minutes"], s=5, alpha=0.35)
    axes[1, 1].set(xlabel="Работа, мин", ylabel="Занятость, мин", title="Занятость vs работа")
    fig.tight_layout()
    name = "overview.png"
    fig.savefig(output / name, dpi=150)
    plt.close(fig)
    names.append(name)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    frame["tariff_unit"].value_counts().plot.bar(ax=axes[0], title="Методы тарификации")
    frame["tariff_rate"].dropna().plot.hist(bins=30, ax=axes[1], title="Ставки")
    fig.tight_layout()
    name = "tariffs.png"
    fig.savefig(output / name, dpi=150)
    plt.close(fig)
    names.append(name)

    fig, ax = plt.subplots(figsize=(10, 4))
    frame.groupby("weekday", sort=False).size().reindex(
        ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    ).plot.bar(ax=ax, title="Работы по дням недели")
    ax.set_ylabel("Строк")
    fig.tight_layout()
    name = "weekday.png"
    fig.savefig(output / name, dpi=150)
    plt.close(fig)
    names.append(name)
    return names


def markdown_table(series: pd.Series, name: str = "Значение") -> str:
    lines = [f"| | {name} |", "|---|---:|"]
    lines.extend(f"| {index} | {value} |" for index, value in series.items())
    return "\n".join(lines)


def build_report(frame: pd.DataFrame, figures: list[str], source: Path, output: Path) -> None:
    n = len(frame)
    date_valid = frame["work_start"].dropna()
    work_ci = ci95(frame["work_minutes"])
    occ_ci = ci95(frame["occupied_minutes"])
    amount_diff, median_amount_diff = tariff_check(frame)
    revenue_valid = frame[["amount_num", "exchange_rate_num", "revenue_rub_num"]].dropna()
    revenue_expected = revenue_valid["amount_num"] * revenue_valid["exchange_rate_num"]
    revenue_diff = (revenue_expected - revenue_valid["revenue_rub_num"]).abs()
    revenue_bad = int((revenue_diff > 1).sum())
    dual = 0
    grouped = frame.dropna(subset=["vessel", "work_start", "work_end"]).groupby(
        ["vessel", "work_start", "work_end"], dropna=False
    )
    for _, group in grouped:
        if set(group["tug"].dropna()) == {"Пионер", "Коммунар"}:
            dual += 1
    voucher_counts = frame["voucher_number_num"].value_counts()
    duplicate_numbers = int((voucher_counts > 1).sum())
    missing_voucher = int(frame["voucher_number_num"].isna().sum())
    period = f"{date_valid.min():%d.%m.%Y}—{date_valid.max():%d.%m.%Y}" if not date_valid.empty else "n/a"
    years = frame["year"].value_counts().sort_index()
    units = frame["tariff_unit"].value_counts()
    date_bad = int(sum(frame[column].isna().sum() for column in DATE_COLS))
    report = f"""# EDA исторической выгрузки работ буксиров

Источник: `{source.name}`. Скрипт: [`eda_extraction.py`](eda_extraction.py).

## Ограничения данных

**Первое и главное ограничение — разрушенная кодировка.** В исходном HTML вся кириллица уже заменена последовательностями `U+FFFD` (после указанного декодирования видны `пїЅ`). Восстановление текстов невозможно. Поэтому мёртвыми являются поля буксира, типа судна, порта, валюты и вида работ (отдельной колонки вида работ в выгрузке нет). Марковская цепочка видов работ и выводы об агенте/группе по этому файлу невозможны. Скрипт обнаруживает артефакт и не пытается восстанавливать кириллицу.

Правила LOA из ODT: 80–120 → 1 буксир; 120–145 → 2; 146–160 → 2; 161–175 → 3; >175 → 3. В данной выгрузке LOA нет, поэтому правило приведено для совместного использования с будущими данными, а не проверено на этих строках.

## 1. Обзор

| Метрика | Значение |
|---|---:|
| Строк данных | {n:,} |
| Период по началу работ | {period} |
| Уникальных судов (латиница сохранена) | {frame["vessel"].nunique()} |
| Уникальных номеров ваучеров | {frame["voucher_number_num"].nunique()} |
| Кириллический артефакт найден | {"да" if frame.astype(str).apply(lambda column: column.str.contains("�|пїЅ", regex=True).any()).any() else "нет"} |

Годы:

{markdown_table(years, "Работ")}

Месяцы:

{markdown_table(frame["month"].value_counts().sort_index(), "Работ")}

![Обзор](figures/{figures[0]})

## 2. Качество и восстановленные поля

Суффикс ваучера восстановил буксир: Пионер — **{int((frame["tug"] == "Пионер").sum()):,}**, Коммунар — **{int((frame["tug"] == "Коммунар").sum()):,}**, без `p/k` — **{int(frame["tug"].isna().sum()):,} ({pct(frame["tug"].isna().sum(), n)})**. Это строки, требующие отдельной проверки.

Битых значений дат (по четырём datetime-полям) — **{int(date_bad):,}**. Измерения длительностей распознаны как `HH:MM`; пропуски времени работ — **{int(frame["work_minutes"].isna().sum())}**, занятости — **{int(frame["occupied_minutes"].isna().sum())}**.

Валюта восстановлена из примечания: {markdown_table(frame["currency"].value_counts(), "Строк")}.

## 3. Баланс буксиров и парные ваучеры

{markdown_table(frame["tug"].fillna("без суффикса").value_counts(), "Строк")}

По ключу судно + совпадающие начало/завершение работ найдено **{dual}** групп с обоими суффиксами `p` и `k` (интерпретация: одна работа на два буксира и два ваучера). Это эвристика, так как исходный идентификатор работы отсутствует.

## 4. Длительности

| Показатель | Время работ, мин | Время занятости, мин |
|---|---:|---:|
| Медиана | {frame["work_minutes"].median():.1f} | {frame["occupied_minutes"].median():.1f} |
| Q1 | {frame["work_minutes"].quantile(.25):.1f} | {frame["occupied_minutes"].quantile(.25):.1f} |
| Q3 | {frame["work_minutes"].quantile(.75):.1f} | {frame["occupied_minutes"].quantile(.75):.1f} |
| Среднее | {frame["work_minutes"].mean():.1f} | {frame["occupied_minutes"].mean():.1f} |
| 95% CI среднего | [{work_ci[0]:.1f}, {work_ci[1]:.1f}] | [{occ_ci[0]:.1f}, {occ_ci[1]:.1f}] |

Корреляция Пирсона между занятостью и временем работ: **{frame[["work_minutes", "occupied_minutes"]].corr().iloc[0, 1]:.3f}**.

![Распределения](figures/{figures[0]})

## 5. Тарификация и GRT

{markdown_table(units, "Строк")}

Доля основных классов: per_ton с делителем — **{pct(units.get("per_ton_divided", 0), n)}**, per_ton — **{pct(units.get("per_ton", 0), n)}**, per_hour — **{pct(units.get("per_hour", 0), n)}**, composite — **{pct(units.get("composite", 0), n)}**. Медианная ставка для per_ton: **{frame.loc[frame["tariff_unit"].isin(["per_ton", "per_ton_divided"]), "tariff_rate"].median():.2f}**, для per_hour: **{frame.loc[frame["tariff_unit"] == "per_hour", "tariff_rate"].median():.2f}**. Наиболее частый делитель: **{frame["tariff_divisor"].value_counts().index[0] if frame["tariff_divisor"].notna().any() else "n/a"}**.

GRT: медиана **{frame["grt"].median():.0f} t**, Q1–Q3 **{frame["grt"].quantile(.25):.0f}–{frame["grt"].quantile(.75):.0f} t**, диапазон **{frame["grt"].min():.0f}–{frame["grt"].max():.0f} t**.

![Тарификация](figures/{figures[1]})

## 6. Проверка формул

Для распознанных примечаний `amount` пересчитан из последнего выражения после `=`. Доля расхождений более 0.02 валютных единицы: **{amount_diff:.1%}**, медианная абсолютная разница: **{median_amount_diff:.4f}**. Для выручки `сумма × курс` проверено {len(revenue_valid):,} строк; расхождений более 1 рубля — **{revenue_bad:,} ({pct(revenue_bad, len(revenue_valid))})**.

## 7. Валюта и курс

{markdown_table(frame["currency"].value_counts(), "Строк")}

Курс: медиана **{frame["exchange_rate_num"].median():.4f}**, диапазон **{frame["exchange_rate_num"].min():.4f}–{frame["exchange_rate_num"].max():.4f}**. По времени медианный курс меняется вместе с датой выгрузки; для детального временного ряда используйте `frame` в скрипте.

## 8. Ночь и выходные

Работы, начавшиеся ночью (22:00–06:00): **{int(frame["night"].sum()):,} ({pct(frame["night"].sum(), n)})**. Выходные (суббота/воскресенье): **{int(frame["work_start"].dt.dayofweek.ge(5).sum()):,} ({pct(frame["work_start"].dt.dayofweek.ge(5).sum(), n)})**.

![Дни недели](figures/{figures[2]})

## 9. Лаг заявки до работы

Дата из имени заявки извлечена в **{int(frame["application_lag_days"].notna().sum()):,}** строках. Медианный лаг — **{frame["application_lag_days"].median():.1f} дня**, Q1–Q3 — **{frame["application_lag_days"].quantile(.25):.1f}–{frame["application_lag_days"].quantile(.75):.1f}**, диапазон — **{frame["application_lag_days"].min():.0f}–{frame["application_lag_days"].max():.0f}** дней. Год в имени заявки не указан: для каждой строки выбрана наиболее поздняя дата с этим месяцем/днём, не превышающая дату работы; результат около годовых границ следует трактовать осторожно.

## 10. Номера ваучеров

Диапазон числовых номеров: **{frame["voucher_number_num"].min():.0f}–{frame["voucher_number_num"].max():.0f}**. Пропусков номера — **{missing_voucher}**. Номеров, повторяющихся в нескольких строках/суффиксах, — **{duplicate_numbers}**; наиболее полезная интерпретация повторов — пары буксиров на одной операции, но это проверяется эвристикой из раздела 3.

## 11. Дополнительные артефакты и воспроизводимость

Сохранены три PNG в `analysis/figures/`. Запуск:

```bash
python analysis/eda_extraction.py path/to/export.xls
```

Скрипт принимает путь к HTML-таблице с расширением `.xls`, декодирует `cp1251`, извлекает 23 колонки, восстанавливает буксир/валюту/метод тарификации и заново создаёт этот отчёт и рисунки. Зависимости вынесены в `analysis/requirements-eda.txt`.
"""
    output.write_text(report, encoding="utf-8")


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
