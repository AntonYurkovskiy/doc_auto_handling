"""FastAPI-приложение: приём документов, ручной ввод, сопоставление, расчёт, экспорт."""

from __future__ import annotations

import imaplib
import shutil
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from fastapi import Depends, FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import FileResponse, HTMLResponse, RedirectResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from app.config import settings
from app.database import get_db, init_db
from app.models import (
    Agent,
    Application,
    Direction,
    DocStatus,
    Operation,
    OperationKind,
    OperationTug,
    PortCall,
    Tug,
    Vessel,
    Voucher,
    Work,
)
from app.services import export as export_service
from app.services.application_parser import parse_application
from app.services.calculation import calculate, tug_count_from_joint
from app.services.html_sanitize import sanitize_email_html
from app.services.imap_ingest import fetch_new_applications
from app.services.matching import find_candidates
from app.services.operations import (
    calculate_operation,
    escort_likely,
    recommended_tug_count,
)
from app.services.vessels import ensure_vessel
from app.services.voucher import predict_and_store
from app.services.voucher_fields import (
    VOUCHER_FIELDS,
    confirm_fields,
    predictions_by_field,
)
from app.services.voucher_files import (
    media_type_for,
    preview_kind,
    resolve_stored_file,
    store_upload,
)
from app.services.voucher_linking import link_voucher
from app.services.voucher_template import ensure_default_template

BASE_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(BASE_DIR / "web" / "templates"))
templates.env.filters["safe_email_html"] = sanitize_email_html

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


def _known_agents(db: Session) -> list[str]:
    return [row.name for row in db.query(Agent).all()]


def _parse_form_dt(value: str | None) -> datetime | None:
    if not value:
        return None
    for fmt in ("%Y-%m-%dT%H:%M", "%d.%m.%Y %H:%M"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    return None


def _parse_form_float(value: str) -> float | None:
    if not value.strip():
        return None
    try:
        return float(value.replace(",", "."))
    except ValueError:
        return None


def _parse_form_int(value: str) -> int | None:
    if not value.strip():
        return None
    try:
        return int(value)
    except ValueError:
        return None


def _save_upload(file: UploadFile, folder: Path) -> str:
    folder.mkdir(parents=True, exist_ok=True)
    dest = folder / (file.filename or "upload.bin")
    with open(dest, "wb") as out:
        shutil.copyfileobj(file.file, out)
    return str(dest)


@dataclass
class VoucherFormData:
    """Ручные поля карточки ваучера (одинаковы для «Сохранить» и «Подтвердить»)."""

    number: str
    tug_id: str
    vessel_name: str
    agent: str
    work_type: str
    left_base_dt: str
    arrived_base_dt: str
    started_dt: str
    finished_dt: str
    remarks: str
    joint_with: str
    is_ice: str
    escort_hours: str
    application_id: str


def voucher_form_data(
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
    application_id: str = Form(""),
) -> VoucherFormData:
    return VoucherFormData(
        number=number,
        tug_id=tug_id,
        vessel_name=vessel_name,
        agent=agent,
        work_type=work_type,
        left_base_dt=left_base_dt,
        arrived_base_dt=arrived_base_dt,
        started_dt=started_dt,
        finished_dt=finished_dt,
        remarks=remarks,
        joint_with=joint_with,
        is_ice=is_ice,
        escort_hours=escort_hours,
        application_id=application_id,
    )


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


@app.post("/mail/fetch", response_class=HTMLResponse)
def mail_fetch(request: Request, db: Session = Depends(get_db)):
    try:
        summary = fetch_new_applications(db)
        mail_message = (
            f"Приём почты: получено {summary.fetched}, "
            f"создано {summary.created}, дублей {summary.skipped_duplicates}, "
            f"вложений {summary.attachments_saved}."
        )
    except (RuntimeError, OSError, imaplib.IMAP4.error) as exc:
        mail_message = f"Ошибка приёма почты: {exc}"
    items = db.query(Application).order_by(Application.id.desc()).all()
    return templates.TemplateResponse(
        "applications_list.html",
        {"request": request, "items": items, "mail_message": mail_message},
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
    parsed = parse_application(path, known_agents=_known_agents(db))
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
        loa_m=parsed.loa_m,
        draft_m=parsed.draft_m,
        entry_datetime=parsed.entry_datetime,
        exit_datetime=parsed.exit_datetime,
        destination=parsed.destination,
        agent=parsed.agent,
        tugs_text=parsed.tugs_text,
        raw_text=parsed.raw_text,
        raw_html=parsed.raw_html,
        file_path=path,
    )
    db.add(app_row)
    db.commit()
    ensure_vessel(db, app_row.vessel_name, app_row.imo, loa_m=app_row.loa_m)
    return RedirectResponse(f"/applications/{app_row.id}", status_code=303)


@app.post("/applications")
def application_create(
    db: Session = Depends(get_db),
    vessel_name: str = Form(""),
    imo: str = Form(""),
    agent: str = Form(""),
    direction: str = Form("прочее"),
    gross_tonnage: str = Form(""),
    loa_m: str = Form(""),
    draft_m: str = Form(""),
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
    item.loa_m = _parse_form_float(loa_m)
    item.draft_m = _parse_form_float(draft_m)
    item.entry_datetime = _parse_form_dt(entry_datetime)
    item.exit_datetime = _parse_form_dt(exit_datetime)
    item.destination = destination or None
    item.status = DocStatus.confirmed
    db.commit()
    ensure_vessel(db, item.vessel_name, item.imo, loa_m=item.loa_m)
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
        {
            "request": request,
            "item": None,
            "tugs": db.query(Tug).all(),
            "agents": db.query(Agent).all(),
            "preview": "none",
            "voucher_file_url": None,
            "fields": [],
            "applications": db.query(Application).order_by(Application.id.desc()).all(),
        },
    )


@app.post("/vouchers/upload")
async def voucher_upload(file: UploadFile = File(...), db: Session = Depends(get_db)):
    stored = store_upload(
        file.file, file.filename, file.content_type, settings.incoming_vouchers_dir
    )
    voucher = Voucher(
        status=DocStatus.needs_review,
        template=ensure_default_template(db),
        file_path=str(stored.path),
        original_filename=stored.original_filename,
        content_type=stored.content_type,
        sha256=stored.sha256,
    )
    db.add(voucher)
    db.flush()
    predict_and_store(db, voucher)
    return RedirectResponse(f"/vouchers/{voucher.id}", status_code=303)


def _apply_voucher_form(item: Voucher, form: VoucherFormData) -> None:
    item.number = form.number or None
    item.tug_id = int(form.tug_id) if form.tug_id.strip().isdigit() else None
    item.vessel_name = form.vessel_name or None
    item.agent = form.agent or None
    item.work_type = form.work_type or None
    item.left_base_dt = _parse_form_dt(form.left_base_dt)
    item.arrived_base_dt = _parse_form_dt(form.arrived_base_dt)
    item.started_dt = _parse_form_dt(form.started_dt)
    item.finished_dt = _parse_form_dt(form.finished_dt)
    item.remarks = form.remarks or None
    item.joint_with = form.joint_with or None
    item.is_ice = bool(form.is_ice)
    item.escort_hours = _parse_form_float(form.escort_hours)
    application_id = _parse_form_int(form.application_id)
    item.application_id = application_id


@app.post("/vouchers")
def voucher_create(
    db: Session = Depends(get_db),
    voucher_id: str = Form(""),
    form: VoucherFormData = Depends(voucher_form_data),
):
    if voucher_id:
        parsed_id = _parse_form_int(voucher_id)
        item = db.get(Voucher, parsed_id) if parsed_id is not None else None
        if item is None:
            return RedirectResponse("/vouchers", status_code=303)
    else:
        item = Voucher(status=DocStatus.needs_review)
        db.add(item)

    _apply_voucher_form(item, form)
    if item.application_id is not None and item.template_id is None:
        item.template = ensure_default_template(db)
    if item.application is not None:
        predict_and_store(db, item, item.application)
    # Ручное сохранение без подтверждения не завершает проверку ваучера.
    if item.status in (DocStatus.new, DocStatus.needs_review):
        item.status = DocStatus.needs_review
    db.commit()
    return RedirectResponse(f"/vouchers/{item.id}", status_code=303)


@app.post("/vouchers/{voucher_id}/confirm")
def voucher_confirm(
    voucher_id: int,
    db: Session = Depends(get_db),
    form: VoucherFormData = Depends(voucher_form_data),
):
    item = db.get(Voucher, voucher_id)
    if item is None:
        return RedirectResponse("/vouchers", status_code=303)

    _apply_voucher_form(item, form)
    db.flush()
    if item.template_id is None:
        item.template = ensure_default_template(db)
    if item.application is not None:
        predict_and_store(db, item, item.application)
    confirm_fields(db, item, datetime.utcnow())
    link_voucher(db, item, item.application)
    item.status = DocStatus.confirmed
    item.reviewed_at = datetime.utcnow()
    db.commit()
    return RedirectResponse(f"/vouchers/{item.id}", status_code=303)


def _voucher_file(item: Voucher) -> Path:
    """Файл ваучера строго внутри каталога хранения (иначе 404)."""
    if not item.file_path:
        raise HTTPException(status_code=404, detail="У ваучера нет файла")
    try:
        return resolve_stored_file(
            Path(item.file_path).name, settings.incoming_vouchers_dir
        )
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc


@app.get("/vouchers/{voucher_id}/file")
def voucher_file(voucher_id: int, db: Session = Depends(get_db)):
    item = db.get(Voucher, voucher_id)
    if item is None:
        raise HTTPException(status_code=404, detail="Ваучер не найден")
    path = _voucher_file(item)
    return FileResponse(
        path,
        media_type=media_type_for(path, item.content_type),
        filename=item.original_filename or path.name,
        content_disposition_type="inline",
    )


@app.get("/files/vouchers/{filename:path}")
def voucher_file_by_name(filename: str):
    """Выдача файла по имени: только из settings.incoming_vouchers_dir."""
    try:
        path = resolve_stored_file(filename, settings.incoming_vouchers_dir)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    return FileResponse(
        path, media_type=media_type_for(path), content_disposition_type="inline"
    )


@app.get("/vouchers/{voucher_id}", response_class=HTMLResponse)
def voucher_detail(voucher_id: int, request: Request, db: Session = Depends(get_db)):
    item = db.get(Voucher, voucher_id)
    if item is None:
        return RedirectResponse("/vouchers", status_code=303)

    preview = "none"
    file_url = None
    if item.file_path:
        try:
            path = resolve_stored_file(
                Path(item.file_path).name, settings.incoming_vouchers_dir
            )
        except ValueError:
            path = None
        if path is not None:
            preview = preview_kind(path)
            file_url = f"/vouchers/{item.id}/file"

    predictions = predictions_by_field(item)
    return templates.TemplateResponse(
        "voucher_form.html",
        {
            "request": request,
            "item": item,
            "tugs": db.query(Tug).all(),
            "agents": db.query(Agent).all(),
            "preview": preview,
            "voucher_file_url": file_url,
            "fields": [(field, predictions.get(field.name)) for field in VOUCHER_FIELDS],
            "applications": db.query(Application).order_by(Application.id.desc()).all(),
        },
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


# --- Суда -------------------------------------------------------------------
@app.get("/vessels", response_class=HTMLResponse)
def vessels_list(request: Request, db: Session = Depends(get_db)):
    items = db.query(Vessel).order_by(Vessel.name).all()
    return templates.TemplateResponse("vessels_list.html", {"request": request, "items": items})


@app.get("/vessels/new", response_class=HTMLResponse)
def vessel_new(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        "vessel_form.html", {"request": request, "item": None, "portcalls": []}
    )


@app.post("/vessels")
def vessel_create(
    db: Session = Depends(get_db),
    vessel_id: str = Form(""),
    name: str = Form(""),
    imo: str = Form(""),
    flag: str = Form(""),
    loa_m: str = Form(""),
    beam_m: str = Form(""),
    grt: str = Form(""),
    nrt: str = Form(""),
):
    if not name.strip():
        return RedirectResponse("/vessels", status_code=303)
    if vessel_id:
        parsed_id = _parse_form_int(vessel_id)
        vessel = db.get(Vessel, parsed_id) if parsed_id is not None else None
        if vessel is None:
            return RedirectResponse("/vessels", status_code=303)
    else:
        vessel = Vessel(name=name.strip())
        db.add(vessel)

    vessel.name = name.strip()
    vessel.imo = imo.strip() or None
    vessel.flag = flag.strip() or None
    vessel.loa_m = _parse_form_float(loa_m)
    vessel.beam_m = _parse_form_float(beam_m)
    vessel.grt = _parse_form_int(grt)
    vessel.nrt = _parse_form_int(nrt)
    db.commit()
    return RedirectResponse(f"/vessels/{vessel.id}", status_code=303)


@app.get("/vessels/{vessel_id}", response_class=HTMLResponse)
def vessel_detail(vessel_id: int, request: Request, db: Session = Depends(get_db)):
    item = db.get(Vessel, vessel_id)
    if item is None:
        return RedirectResponse("/vessels", status_code=303)
    return templates.TemplateResponse(
        "vessel_form.html",
        {"request": request, "item": item, "portcalls": item.portcalls},
    )


# --- Судозаходы -------------------------------------------------------------
@app.get("/portcalls", response_class=HTMLResponse)
def portcalls_list(request: Request, db: Session = Depends(get_db)):
    items = db.query(PortCall).order_by(PortCall.id.desc()).all()
    return templates.TemplateResponse("portcalls_list.html", {"request": request, "items": items})


@app.get("/portcalls/new", response_class=HTMLResponse)
def portcall_new(request: Request, db: Session = Depends(get_db)):
    return templates.TemplateResponse(
        "portcall_form.html",
        {
            "request": request,
            "item": None,
            "vessels": db.query(Vessel).order_by(Vessel.name).all(),
            "directions": Direction,
            "statuses": DocStatus,
            "operation_kinds": OperationKind,
            "tugs": db.query(Tug).order_by(Tug.name).all(),
            "escort_likely": escort_likely,
            "recommended_tug_count": recommended_tug_count,
            "applications": [],
        },
    )


@app.post("/portcalls")
def portcall_create(
    db: Session = Depends(get_db),
    portcall_id: str = Form(""),
    vessel_id: str = Form(""),
    direction: str = Form("прочее"),
    status: str = Form("new"),
    agent: str = Form(""),
    eta: str = Form(""),
    etd: str = Form(""),
    berth_from: str = Form(""),
    berth_to: str = Form(""),
    purpose: str = Form(""),
    notes: str = Form(""),
):
    if portcall_id:
        item_id = _parse_form_int(portcall_id)
        item = db.get(PortCall, item_id) if item_id is not None else None
        if item is None:
            return RedirectResponse("/portcalls", status_code=303)
    else:
        item = PortCall(source="manual")
        db.add(item)

    vessel_item_id = _parse_form_int(vessel_id)
    vessel_exists = db.get(Vessel, vessel_item_id) if vessel_item_id is not None else None
    item.vessel_id = vessel_item_id if vessel_exists is not None else None
    item.direction = (
        Direction(direction) if direction in Direction._value2member_map_ else Direction.other
    )
    if status in DocStatus._value2member_map_:
        item.status = DocStatus(status)
    elif not portcall_id:
        item.status = DocStatus.new
    item.agent = agent.strip() or None
    item.eta = _parse_form_dt(eta)
    item.etd = _parse_form_dt(etd)
    item.berth_from = berth_from.strip() or None
    item.berth_to = berth_to.strip() or None
    item.purpose = purpose.strip() or None
    item.notes = notes.strip() or None
    db.commit()
    return RedirectResponse(f"/portcalls/{item.id}", status_code=303)


@app.get("/portcalls/{portcall_id}", response_class=HTMLResponse)
def portcall_detail(portcall_id: int, request: Request, db: Session = Depends(get_db)):
    item = db.get(PortCall, portcall_id)
    if item is None:
        return RedirectResponse("/portcalls", status_code=303)
    total_amount = sum(op.amount for op in item.operations if op.amount is not None)
    total_revenue = sum(op.revenue_rub for op in item.operations if op.revenue_rub is not None)
    currency = next(
        (op.currency for op in item.operations if op.currency), settings.ue_currency
    )
    return templates.TemplateResponse(
        "portcall_form.html",
        {
            "request": request,
            "item": item,
            "vessels": db.query(Vessel).order_by(Vessel.name).all(),
            "directions": Direction,
            "statuses": DocStatus,
            "operation_kinds": OperationKind,
            "tugs": db.query(Tug).order_by(Tug.name).all(),
            "escort_likely": escort_likely,
            "recommended_tug_count": recommended_tug_count,
            "applications": item.applications,
            "total_amount": round(total_amount, 2),
            "total_revenue": round(total_revenue, 2),
            "total_currency": currency,
        },
    )


@app.post("/portcalls/{portcall_id}/operations")
def operation_create(
    portcall_id: int,
    db: Session = Depends(get_db),
    kind: str = Form("прочее"),
    draft_m: str = Form(""),
    work_start: str = Form(""),
    work_end: str = Form(""),
    is_ice: str = Form(""),
    notes: str = Form(""),
):
    portcall = db.get(PortCall, portcall_id)
    if portcall is None:
        return RedirectResponse("/portcalls", status_code=303)
    operation_kind = (
        OperationKind(kind)
        if kind in OperationKind._value2member_map_
        else OperationKind.other
    )
    db.add(
        Operation(
            portcall_id=portcall_id,
            kind=operation_kind,
            draft_m=_parse_form_float(draft_m),
            work_start=_parse_form_dt(work_start),
            work_end=_parse_form_dt(work_end),
            is_ice=bool(is_ice),
            notes=notes.strip() or None,
        )
    )
    db.commit()
    return RedirectResponse(f"/portcalls/{portcall_id}", status_code=303)


@app.post("/operations/{operation_id}/calculate")
def operation_calculate(
    operation_id: int,
    db: Session = Depends(get_db),
    work_start: str = Form(""),
    work_end: str = Form(""),
    is_ice: str = Form(""),
):
    operation = db.get(Operation, operation_id)
    if operation is None:
        return RedirectResponse("/portcalls", status_code=303)

    parsed_start = _parse_form_dt(work_start)
    parsed_end = _parse_form_dt(work_end)
    if parsed_start is not None:
        operation.work_start = parsed_start
    if parsed_end is not None:
        operation.work_end = parsed_end
    operation.is_ice = bool(is_ice)
    db.commit()

    try:
        calculate_operation(db, operation)
    except (ValueError, NotImplementedError) as exc:
        operation.amount = None
        operation.revenue_rub = None
        operation.calc_note = f"Ошибка расчёта: {exc}"
        operation.calculated_at = datetime.utcnow()
        db.commit()
    return RedirectResponse(f"/portcalls/{operation.portcall_id}", status_code=303)


@app.post("/operations/{operation_id}/tugs")
def operation_tug_create(
    operation_id: int,
    db: Session = Depends(get_db),
    tug_id: str = Form(""),
    escort: str = Form(""),
):
    operation = db.get(Operation, operation_id)
    if operation is None:
        return RedirectResponse("/portcalls", status_code=303)
    parsed_tug_id = _parse_form_int(tug_id)
    tug = db.get(Tug, parsed_tug_id) if parsed_tug_id is not None else None
    if tug is None:
        return RedirectResponse(f"/portcalls/{operation.portcall_id}", status_code=303)

    wants_escort = bool(escort)
    link = (
        db.query(OperationTug)
        .filter_by(operation_id=operation_id, tug_id=tug.id)
        .first()
    )
    if link is None:
        link = OperationTug(operation_id=operation_id, tug_id=tug.id)
        db.add(link)
    if wants_escort and not link.escort:
        has_other_escort = (
            db.query(OperationTug)
            .filter(
                OperationTug.operation_id == operation_id,
                OperationTug.escort.is_(True),
                OperationTug.tug_id != tug.id,
            )
            .first()
            is not None
        )
        link.escort = not has_other_escort
    elif not wants_escort:
        link.escort = False
    db.commit()
    return RedirectResponse(f"/portcalls/{operation.portcall_id}", status_code=303)
