"""Экспорт итоговых работ в Excel/CSV (формат matching.xls) и импорт справочников."""

from __future__ import annotations

import io

import pandas as pd
from sqlalchemy.orm import Session

from app.models import Agent, Tug, Work
from app.services.calculation import format_hm, minutes_between

COLUMNS = [
    "#",
    "Буксир",
    "Объект работ",
    "Вид работ",
    "Агент",
    "Нр. ваучера/наряда",
    "Выход из базы",
    "Приход в базу",
    "Время занятости",
    "Начало работ",
    "Завершение работ",
    "Время работ",
    "Примечания",
    "Заявка",
    "Ваучер",
    "Брутто/нетто",
    "Сумма",
    "Валюта",
    "Курс",
    "Выручка, руб",
]


def _fmt_dt(value) -> str:  # noqa: ANN001
    return value.strftime("%d.%m.%Y %H:%M") if value else ""


def works_to_dataframe(db: Session) -> pd.DataFrame:
    works = db.query(Work).order_by(Work.id).all()
    rows = []
    for i, w in enumerate(works, start=1):
        tug_name = w.tug.name if w.tug else ""
        rows.append(
            {
                "#": i,
                "Буксир": tug_name,
                "Объект работ": w.object_name or "",
                "Вид работ": w.work_type or "",
                "Агент": w.agent or "",
                "Нр. ваучера/наряда": (w.voucher.number if w.voucher else "") or "",
                "Выход из базы": _fmt_dt(w.left_base_dt),
                "Приход в базу": _fmt_dt(w.arrived_base_dt),
                "Время занятости": format_hm(minutes_between(w.left_base_dt, w.arrived_base_dt)),
                "Начало работ": _fmt_dt(w.started_dt),
                "Завершение работ": _fmt_dt(w.finished_dt),
                "Время работ": format_hm(minutes_between(w.started_dt, w.finished_dt)),
                "Примечания": w.calc_note or "",
                "Заявка": (w.application.file_path if w.application else "") or "",
                "Ваучер": (w.voucher.file_path if w.voucher else "") or "",
                "Брутто/нетто": w.gross_tonnage or "",
                "Сумма": w.amount if w.amount is not None else "",
                "Валюта": w.currency or "",
                "Курс": w.cbr_rate if w.cbr_rate is not None else "",
                "Выручка, руб": w.revenue_rub if w.revenue_rub is not None else "",
            }
        )
    return pd.DataFrame(rows, columns=COLUMNS)


def export_excel(db: Session) -> bytes:
    df = works_to_dataframe(db)
    buf = io.BytesIO()
    with pd.ExcelWriter(buf, engine="openpyxl") as writer:
        df.to_excel(writer, index=False, sheet_name="Работы")
    return buf.getvalue()


def export_csv(db: Session) -> bytes:
    df = works_to_dataframe(db)
    return df.to_csv(index=False).encode("utf-8-sig")


def import_reference_excel(db: Session, data: bytes) -> dict[str, int]:
    """Импорт справочников буксиров/агентов из Excel.

    Ожидает листы 'Буксиры' (колонка 'Название') и 'Агенты' (колонка 'Название').
    Отсутствующие листы игнорируются.
    """
    added = {"tugs": 0, "agents": 0}
    xls = pd.ExcelFile(io.BytesIO(data))
    if "Буксиры" in xls.sheet_names:
        df = xls.parse("Буксиры")
        for name in df.get("Название", []):
            name = str(name).strip()
            if name and not db.query(Tug).filter_by(name=name).first():
                db.add(Tug(name=name))
                added["tugs"] += 1
    if "Агенты" in xls.sheet_names:
        df = xls.parse("Агенты")
        for name in df.get("Название", []):
            name = str(name).strip()
            if name and not db.query(Agent).filter_by(name=name).first():
                db.add(Agent(name=name))
                added["agents"] += 1
    db.commit()
    return added
