"""FastAPI-приложение: приём документов, ручной ввод, сопоставление, расчёт, экспорт."""

from __future__ import annotations

import shutil
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, Request, UploadFile
from fastapi.responses import HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db
from app.models import Agent, Application, Direction, DocStatus, Ship, Tug, Voucher, Work
from app.services import export as export_service
from app.services.application_parser import parse_application
from app.services.calculation import calculate, tug_count_from_joint
from app.services.matching import find_candidates

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))

app = FastAPI(title="Обработка заявок и ваучеров буксира")
app.mount("/static", StaticFiles(directory=str(BASE_DIR / "web" / "static")), name="static")


DEFAULT_TUGS = [("БК Коммунар", "k"), ("БК Пионер", "p")]
DEFAULT_AGENTS = ["Транс-Агро", "Содружество - Соя", "МореСервис"]


@app.on_event("startup")
def _startup() -> None:
    settings.files_dir.mkdir(parents=True, exist_ok=True)
    settings.incoming_applications_dir.mkdir(parents=True, exist_ok=True)
    settings.incoming_vouchers_dir.mkdir(parents=True, exist_ok=True)
    init_db()
    _seed_reference()


def _seed_reference() -> None:
    from app.database import SessionLocal

    db = SessionLocal()
    try:
        for name, code in DEFAULT_TUGS:
            if not db.query(Tug).filter_by(name=name).first():
                db.add(Tug(name=name, code=code))
        for name in DEFAULT_AGENTS:
            if not db.query(Agent).filter_by(name=name).first():
                db.add(Agent(name=name))
        db.commit()
    finally:
        db.close()


def _parse_form_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _save_upload(file: UploadFile, folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (file.filename or "upload.bin")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return str(dest)


# --- Дашборд ----------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
def dashboard(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        "index.html",
        {
            "request": request,
            "n_applications": db.query(Application).count(),
            "n_vouchers": db.query(Voucher).count(),
            "n_works": db.query(Work).count(),
        },
    )


# --- Заявки -----------------------------------------------------------------
@app.get("/applications", response_class=HTMLResponse)
def applications_list(request: Request, db: Session = Depends(get_db)):
    items = db.query(Application).order_by(Application.id.desc()).all()
    return templates.TemplateResponse(
        "applications_list.html", {"request": request, "items": items}
    )


@app.get("/applications/new", response_class=HTMLResponse)
def application_new(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        "application_form.html",
        {"request": request, "item": None, "agents": db.query(Agent).all()},
    )


@app.post("/applications/upload")
async def application_upload(
    file: UploadFile = File(...), db: Session = Depends(get_db)
):
    path = _save_upload(file, settings.incoming_applications_dir)
    parsed = parse_application(path)
    app_row = Application(
        status=DocStatus.needs_review,
        source="upload",
        sender=parsed.sender,
        subject=parsed.subject,
        received_at=parsed.received_at,
        direction=Direction(parsed.direction) if parsed.direction in Direction._value2member_map_
        else Direction.other,
        vessel_name=parsed.vessel_name,
        imo=parsed.imo,
        gross_tonnage=parsed.gross_tonnage,
        net_tonnage=parsed.net_tonnage,
        entry_datetime=parsed.entry_datetime,
        exit_datetime=parsed.exit_datetime,
        destination=parsed.destination,
        tugs_text=parsed.tugs_text,
        raw_text=parsed.raw_text,
        file_path=path,
    )
    db.add(app_row)
    db.commit()
    return RedirectResponse(f"/applications/{app_row.id}", status_code=303)


@app.post("/applications")
def application_create(
    db: Session = Depends(get_db),
    vessel_name: str = Form(""),
    imo: str = Form(""),
    agent: str = Form(""),
    direction: str = Form("прочее"),
    gross_tonnage: str = Form(""),
    entry_datetime: str = Form(""),
    exit_datetime: str = Form(""),
    destination: str = Form(""),
    application_id: str = Form(""),
):
    if application_id:
        item = db.get(Application, int(application_id))
        if item is None:
            return RedirectResponse("/applications", status_code=303)
    else:
        item = Application(source="manual")
        db.add(item)

    item.vessel_name = vessel_name or None
    item.imo = imo or None
    item.agent = agent or None
    item.direction = (
        Direction(direction) if direction in Direction._value2member_map_ else Direction.other
    )
    item.gross_tonnage = int(gross_tonnage) if gross_tonnage.strip().isdigit() else None
    item.entry_datetime = _parse_form_dt(entry_datetime)
    item.exit_datetime = _parse_form_dt(exit_datetime)
    item.destination = destination or None
    item.status = DocStatus.confirmed
    db.commit()
    _ensure_ship(db, item.vessel_name, item.imo)
    return RedirectResponse(f"/applications/{item.id}", status_code=303)


@app.get("/applications/{app_id}", response_class=HTMLResponse)
def application_detail(app_id: int, request: Request, db: Session = Depends(get_db)):
    item = db.get(Application, app_id)
    if item is None:
        return RedirectResponse("/applications", status_code=303)
    return templates.TemplateResponse(
        "application_form.html",
        {"request": request, "item": item, "agents": db.query(Agent).all()},
    )


# --- Ваучеры ----------------------------------------------------------------
@app.get("/vouchers", response_class=HTMLResponse)
def vouchers_list(request: Request, db: Session = Depends(get_db)):
    items = db.query(Voucher).order_by(Voucher.id.desc()).all()
    return templates.TemplateResponse("vouchers_list.html", {"request": request, "items": items})


@app.get("/vouchers/new", response_class=HTMLResponse)
def voucher_new(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        "voucher_form.html",
        {"request": request, "item": None, "tugs": db.query(Tug).all(),
         "agents": db.query(Agent).all()},
    )


@app.post("/vouchers/upload")
async def voucher_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    # Ваучер — скан-картинка; OCR рукописных полей появится в Фазе 3/6.
    # Пока сохраняем файл и открываем форму ручного ввода полей.
    path = _save_upload(file, settings.incoming_vouchers_dir)
    voucher = Voucher(status=DocStatus.needs_review, file_path=path)
    db.add(voucher)
    db.commit()
    return RedirectResponse(f"/vouchers/{voucher.id}", status_code=303)


@app.post("/vouchers")
def voucher_create(
    db: Session = Depends(get_db),
    voucher_id: str = Form(""),
    number: str = Form(""),
    tug_id: str = Form(""),
    vessel_name: str = Form(""),
    agent: str = Form(""),
    work_type: str = Form(""),
    left_base_dt: str = Form(""),
    arrived_base_dt: str = Form(""),
    started_dt: str = Form(""),
    finished_dt: str = Form(""),
    remarks: str = Form(""),
    joint_with: str = Form(""),
    is_ice: str = Form(""),
    escort_hours: str = Form(""),
):
    if voucher_id:
        item = db.get(Voucher, int(voucher_id))
        if item is None:
            return RedirectResponse("/vouchers", status_code=303)
    else:
        item = Voucher()
        db.add(item)

    item.number = number or None
    item.tug_id = int(tug_id) if tug_id.strip().isdigit() else None
    item.vessel_name = vessel_name or None
    item.agent = agent or None
    item.work_type = work_type or None
    item.left_base_dt = _parse_form_dt(left_base_dt)
    item.arrived_base_dt = _parse_form_dt(arrived_base_dt)
    item.started_dt = _parse_form_dt(started_dt)
    item.finished_dt = _parse_form_dt(finished_dt)
    item.remarks = remarks or None
    item.joint_with = joint_with or None
    item.is_ice = bool(is_ice)
    try:
        item.escort_hours = float(escort_hours.replace(",", ".")) if escort_hours.strip() else None
    except ValueError:
        item.escort_hours = None
    item.status = DocStatus.confirmed
    db.commit()
    return RedirectResponse(f"/vouchers/{item.id}", status_code=303)


@app.get("/vouchers/{voucher_id}", response_class=HTMLResponse)
def voucher_detail(voucher_id: int, request: Request, db: Session = Depends(get_db)):
    item = db.get(Voucher, voucher_id)
    if item is None:
        return RedirectResponse("/vouchers", status_code=303)
    return templates.TemplateResponse(
        "voucher_form.html",
        {"request": request, "item": item, "tugs": db.query(Tug).all(),
         "agents": db.query(Agent).all()},
    )


# --- Сопоставление ----------------------------------------------------------
@app.get("/applications/{app_id}/match", response_class=HTMLResponse)
def match_view(app_id: int, request: Request, db: Session = Depends(get_db)):
    application = db.get(Application, app_id)
    if application is None:
        return RedirectResponse("/applications", status_code=303)
    vouchers = db.query(Voucher).all()
    candidates = find_candidates(application, vouchers, min_score=0.2)
    return templates.TemplateResponse(
        "match.html",
        {"request": request, "application": application, "candidates": candidates},
    )


@app.post("/match")
def create_match(
    db: Session = Depends(get_db),
    application_id: int = Form(...),
    voucher_id: int = Form(...),
):
    application = db.get(Application, application_id)
    voucher = db.get(Voucher, voucher_id)
    if application is None or voucher is None:
        return RedirectResponse("/applications", status_code=303)

    work = Work(
        status=DocStatus.matched,
        application_id=application.id,
        voucher_id=voucher.id,
        tug_id=voucher.tug_id,
        object_name=voucher.vessel_name or application.vessel_name,
        work_type=voucher.work_type,
        agent=voucher.agent or application.agent,
        left_base_dt=voucher.left_base_dt,
        arrived_base_dt=voucher.arrived_base_dt,
        started_dt=voucher.started_dt,
        finished_dt=voucher.finished_dt,
        gross_tonnage=application.gross_tonnage,
        is_ice=voucher.is_ice,
        escort_hours=voucher.escort_hours,
    )
    db.add(work)
    application.status = DocStatus.matched
    voucher.status = DocStatus.matched
    db.commit()
    return RedirectResponse(f"/works?highlight={work.id}", status_code=303)


# --- Работы и расчёт --------------------------------------------------------
@app.get("/works", response_class=HTMLResponse)
def works_list(request: Request, db: Session = Depends(get_db)):
    items = db.query(Work).order_by(Work.id.desc()).all()
    return templates.TemplateResponse("works_list.html", {"request": request, "items": items})


@app.post("/works/{work_id}/calculate")
def work_calculate(work_id: int, db: Session = Depends(get_db)):
    work = db.get(Work, work_id)
    if work is None:
        return RedirectResponse("/works", status_code=303)
    try:
        result = calculate(
            agent=work.agent or "",
            work_type=work.work_type,
            gross_tonnage=work.gross_tonnage,
            started_dt=work.started_dt,
            finished_dt=work.finished_dt,
            left_base_dt=work.left_base_dt,
            arrived_base_dt=work.arrived_base_dt,
            is_ice=work.is_ice,
            escort_hours=work.escort_hours,
            tug_count=tug_count_from_joint(work.voucher.joint_with if work.voucher else None),
        )
        work.amount = result.amount
        work.currency = result.currency
        work.cbr_rate = result.cbr_rate
        work.revenue_rub = result.revenue_rub
        work.calc_note = result.calc_note
        work.status = DocStatus.calculated
    except Exception as exc:  # noqa: BLE001
        work.calc_note = f"Ошибка расчёта: {exc}"
        work.status = DocStatus.error
    db.commit()
    return RedirectResponse("/works", status_code=303)


# --- Экспорт / импорт -------------------------------------------------------
@app.get("/export/excel")
def export_excel(db: Session = Depends(get_db)):
    data = export_service.export_excel(db)
    return Response(
        content=data,
        media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition": "attachment; filename=works.xlsx"},
    )


@app.get("/export/csv")
def export_csv(db: Session = Depends(get_db)):
    data = export_service.export_csv(db)
    return Response(
        content=data,
        media_type="text/csv",
        headers={"Content-Disposition": "attachment; filename=works.csv"},
    )


def _ensure_ship(db: Session, name: str | None, imo: str | None) -> None:
    if not name:
        return
    if not db.query(Ship).filter_by(name=name).first():
        db.add(Ship(name=name, imo=imo))
        db.commit()
