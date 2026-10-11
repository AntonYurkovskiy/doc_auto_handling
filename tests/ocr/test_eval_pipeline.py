"""Тесты сквозной оценки (T21): скан → кропы → распределения → декодер → порог. Только синтетика."""

from __future__ import annotations

import csv
import json
from collections.abc import Mapping
from datetime import date, datetime
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pytest

from app.ocr import layouts as boxes
from app.ocr.decoder import ROWS, DecoderModel
from app.ocr.number_decoder import NumberPriors
from app.ocr.pair_decoder import PairPriors
from app.ocr.variants import load_layouts as load_variants
from app.ocr.variants import save_layout as save_variant
from ocr_lab import eval_pipeline as ep
from ocr_lab import paths
from ocr_lab.evaluate import evaluate, load_truth
from ocr_lab.predictions import read_predictions
from tests.ocr.synth import PAGE_H, PAGE_W, make_form

VARIANT = "syn_v1"


def _write_png(path: Path, img: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(cv2.imencode(".png", img)[1].tobytes())


def _box_layout() -> boxes.Layout:
    items = [boxes.Box(boxes.VOUCHER_NUMBER, 900, 300, 1150, 400, "number")]
    for i, row in enumerate(boxes.DATE_ROWS):
        for j, part in enumerate(boxes.DATE_PARTS):
            x0, y0 = 300 + 200 * j, 1200 + 100 * i
            items.append(boxes.Box(f"{row}.{part}", x0, y0, x0 + 150, y0 + 90, "two_digit"))
    return boxes.Layout(VARIANT, "s", (PAGE_W, PAGE_H), tuple(items))


def _heads_output(number: int) -> np.ndarray:
    """Логиты трёх голов (`heads`), уверенно дающие `number`."""
    out = np.zeros((3, 11))
    digits = str(number).rjust(3)
    for i, ch in enumerate(digits):
        out[i, 10 if ch == " " else int(ch)] = 12.0
    return out


class FakeRecognizer:
    """Распределения, уверенно дающие заданные значения (вместо onnxruntime)."""

    def __init__(self, values: Mapping[str, int] | None = None, number: int = 12) -> None:
        self.values = dict(values or {})
        self.number_value = number
        self.calls = 0

    def digits(self, crops: Mapping[str, np.ndarray]) -> dict[str, list[tuple[int, float]]]:
        self.calls += 1
        assert all(c.ndim == 2 for c in crops.values())
        return {n: [(self.values.get(n, 1), 0.98), (2, 0.02)] for n in crops}

    def number(self, crop: np.ndarray) -> tuple[np.ndarray, str, Any]:
        return _heads_output(self.number_value), "heads", 1.0


@pytest.fixture()
def resources(tmp_path: Path) -> ep.Resources:
    layouts_dir = tmp_path / "layouts"
    image, mask = make_form(layout_seed=0)
    save_variant(layouts_dir / VARIANT, image, mask, {"tug_code": "s"})
    return ep.Resources(load_variants(layouts_dir), {VARIANT: _box_layout()}, FakeRecognizer())


@pytest.fixture()
def work(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Рабочий каталог OCR во временной папке (e2e, отчёты, флаги печати)."""
    w = tmp_path / "work"
    monkeypatch.setattr(paths, "WORK_DIR", w)
    monkeypatch.setattr(paths, "PRINTED_FLAGS_CSV", w / "labels" / "printed_flags.csv")
    return w


# --- run: скан → распределения ---------------------------------------------------------------


def test_recognize_scan_from_file(tmp_path: Path, resources: ep.Resources) -> None:
    scan, _ = make_form(layout_seed=0, fill_seed=3)
    path = tmp_path / "scans" / "2025_67к.png"  # кириллица в пути
    _write_png(path, scan)
    rec, bands = ep.recognize_scan(path, resources, tug_hint="s")
    assert rec["status"] == "ok" and rec["variant"] == VARIANT and rec["align"]["ok"]
    assert set(rec["sources"]) == set(boxes.SUBFIELD_NAMES)
    assert set(rec["inputs"]) == set(ep.DIGIT_SUBFIELDS)
    assert np.asarray(rec["number_output"]).shape == (3, 11) and rec["number_kind"] == "heads"
    assert {"load", "align", "crops", "digits", "number", "total"} <= set(rec["timing"])
    # Полосы: номер и 4 строки дат; полоса строки — объединение её боксов, с рамками (BGR).
    assert set(bands) == {boxes.VOUCHER_NUMBER, *ROWS}
    assert bands["left_base"].shape == (90, 750, 3)
    json.dumps(rec)  # запись сериализуется в JSONL


def test_recognize_scan_align_failed(tmp_path: Path, resources: ep.Resources) -> None:
    noise = np.random.default_rng(0).integers(0, 256, (PAGE_H, PAGE_W), dtype=np.uint8)
    path = tmp_path / "noise.png"
    _write_png(path, noise)
    rec, bands = ep.recognize_scan(path, resources)
    assert rec["status"] == "align_failed" and rec["reason"]
    assert bands == {} and "inputs" not in rec


def _manifest(path: Path, rows: list[dict[str, Any]]) -> None:
    cols = sorted({k for r in rows for k in r})
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=cols)
        writer.writeheader()
        writer.writerows(rows)


def test_run_split_resumes_and_survives_errors(
    tmp_path: Path, work: Path, resources: ep.Resources
) -> None:
    scan, _ = make_form(layout_seed=0, fill_seed=5)
    _write_png(tmp_path / "a.png", scan)
    manifest = tmp_path / "manifest.csv"
    _manifest(manifest, [
        {"scan_id": "2026_1k", "split": "val", "scan_path": str(tmp_path / "a.png"),
         "voucher_file": "1k.pdf"},
        {"scan_id": "2026_2k ", "split": "val", "scan_path": str(tmp_path / "missing.png"),
         "voucher_file": "2k.pdf"},
        {"scan_id": "2026_3k", "split": "test", "scan_path": str(tmp_path / "a.png"),
         "voucher_file": "3k.pdf"},
    ])
    stats = ep.run_split("val", resources, manifest=manifest, name="t")
    assert stats["ok"] == 1 and stats["error"] == 1
    recs = {r["scan_id"]: r for r in ep.read_jsonl(ep.scans_path("val", "t"))}
    assert set(recs) == {"2026_1k", "2026_2k"}  # пробел в scan_id срезан
    assert recs["2026_2k"]["reason"] == "FileNotFoundError"  # только тип, без содержимого
    assert (ep.bands_dir("t") / "2026_1k" / "left_base.png").exists()
    again = ep.run_split("val", resources, manifest=manifest, name="t")
    assert again["skipped"] == 2 and again["ok"] == 0
    stopped = ep.run_split("test", resources, manifest=manifest, name="t", max_minutes=0)
    assert stopped["stopped"] == 1


# --- decode: ядро + номер + пары -------------------------------------------------------------


TRUTH = {
    "left_base": (date(2026, 5, 4), 8, 0),
    "started_work": (date(2026, 5, 4), 8, 40),
    "finished_work": (date(2026, 5, 4), 10, 0),
    "arrived_base": (date(2026, 5, 4), 10, 30),
}


def _truth_row(sid: str, number: int, *, split: str = "val", pair: str = "") -> dict[str, Any]:
    row: dict[str, Any] = {
        "scan_id": sid, "split": split, "year": 2026, "tug_code": sid[-1],
        "voucher_number": number, "voucher_file": f"{number}{sid[-1]}.pdf", "pair_id": pair,
        "app_dt": "2026-05-04T06:00:00", "work_type": "", "chain_ok": "True",
        "has_hour24": "False", "scan_path": "x.pdf",
    }
    for r, (d, h, mi) in TRUTH.items():
        row.update({
            f"{r}_dt": datetime(d.year, d.month, d.day, h, mi).isoformat(), f"{r}_day": d.day,
            f"{r}_month": d.month, f"{r}_hour": h, f"{r}_minute": mi, f"{r}_year": d.year,
        })
    return row


def _scan_rec(sid: str, number: int, split: str = "val") -> dict[str, Any]:
    inputs = {}
    for r, (d, h, mi) in TRUTH.items():
        for part, v in (("day", d.day), ("month", d.month), ("hour", h), ("minute", mi)):
            alt = v + 1 if part != "month" else (v % 12) + 1
            inputs[f"{r}.{part}"] = [[v, 0.97], [alt, 0.03]]
    return {
        "scan_id": sid, "split": split, "status": "ok", "variant": VARIANT,
        "align": {"ok": True}, "sources": {}, "timing": {"total": 0.5},
        "inputs": inputs, "number_output": _heads_output(number).tolist(),
        "number_kind": "heads", "number_temperature": 1.0,
    }


def _assets() -> ep.DecodeAssets:
    return ep.DecodeAssets(DecoderModel.default(), NumberPriors.default(), PairPriors.default())


def _write_scans(split: str, name: str, recs: list[dict[str, Any]]) -> None:
    path = ep.scans_path(split, name)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(r) + "\n" for r in recs), encoding="utf-8")


def test_decode_split_writes_t13_and_details(tmp_path: Path, work: Path) -> None:
    manifest = tmp_path / "manifest.csv"
    rows = [
        _truth_row("2026_11k", 11, split="train"),
        _truth_row("2026_12k", 12, pair="P1"),
        _truth_row("2026_12p", 12, pair="P1"),
        _truth_row("2026_13k", 13),
    ]
    _manifest(manifest, rows)
    _write_scans("val", "t", [
        _scan_rec("2026_12k", 12), _scan_rec("2026_12p", 12),
        {"scan_id": "2026_13k", "split": "val", "status": "align_failed", "timing": {}},
    ])
    res = ep.decode_split("val", _assets(), manifest=manifest, name="t")
    assert res["n"] == 3 and res["pairs"] == 1 and res["statuses"] == {"ok": 2, "align_failed": 1}

    preds = {p.scan_id: p for p in read_predictions(ep.predictions_path("val", "t"))}
    assert preds["2026_13k"].records == [] and preds["2026_13k"].confidence is None
    metrics = evaluate(load_truth(manifest, "val"), list(preds.values()))
    # 2 из 3 ваучеров верны (третий не выровнен), номер 12 — по картинке и порядку.
    assert metrics["vouchers"]["rows"]["all"] == {"n": 3, "accuracy": pytest.approx(2 / 3)}
    assert metrics["vouchers"]["rows_number"]["all"]["accuracy"] == pytest.approx(2 / 3)

    details = {d["scan_id"]: d for d in ep.read_jsonl(ep.details_path("val", "t"))}
    d = details["2026_12k"]
    assert d["partner"] == "2026_12p" and d["number"] == 12 and d["p_core_final"] > 0.5
    assert ep.forms_from_json(d["forms"]) == TRUTH
    assert d["timing"]["total"] > d["timing"]["decode"] > 0

    outs, _ = ep.outcomes("val", name="t", manifest=manifest)
    by_id = {v.scan_id: v for v in outs}
    assert by_id["2026_12k"].rows_ok and not by_id["2026_13k"].rows_ok
    assert by_id["2026_13k"].conf == {"core": 0.0, "final": 0.0, "guarded": 0.0}
    assert by_id["2026_13k"].money["work_minutes"]  # нет записи — деньги меняются


# --- порог, деньги, этапы, расхождения -------------------------------------------------------


def _v(sid: str, ok: bool, conf: float, *, number_ok: bool = True, p_number: float = 1.0,
       split: str = "val") -> ep.VoucherOutcome:
    return ep.VoucherOutcome(
        scan_id=sid, split=split, status="ok", rows_ok=ok, number_ok=number_ok,
        wrong_rows=() if ok else ("left_base",),
        conf={"core": conf, "final": min(1.0, conf + 1e-4), "guarded": conf - 1e-6},
        p_number=p_number,
    )


def test_select_thresholds_and_accept_stats() -> None:
    val = [_v(f"v{i}", True, 0.9999) for i in range(30)] + [
        _v("bad", False, 0.99), _v("low", True, 0.5)
    ]
    sel = ep.select_thresholds(val, 0.995)
    assert sel["kind"] in ep.CONF_KINDS
    assert sel["per_kind_val"][sel["kind"]]["n_accepted"] == 30
    assert 0.99 < sel["t_val"] <= 1.0
    # Консервативный — не ниже 1 − 0,005 / 10 и не ниже порога val.
    assert sel["t_conservative"] == pytest.approx(max(0.9995, sel["t_val"]))
    fixed = ep.select_thresholds(val, 0.995, fixed=0.42)
    assert fixed["t_val"] == 0.42 and fixed["fixed_from_cli"]

    test = [_v("a", True, 0.9999), _v("b", False, 0.99995), _v("c", True, 0.1)]
    acc = ep.accept_stats(test, "core", 0.9995)
    assert acc["n_accepted"] == 2 and acc["n_errors"] == 1 and acc["errors"] == ["b"]
    assert acc["coverage"] == pytest.approx(2 / 3) and acc["precision"] == 0.5
    assert 0 < acc["ci_low"] < 0.5 < acc["ci_high"] < 1
    assert ep.accept_stats(test, "core", None)["n_accepted"] == 0


def test_money_changes() -> None:
    t = {r: datetime.combine(d, datetime.min.time()).replace(hour=h, minute=mi)
         for r, (d, h, mi) in TRUTH.items()}
    assert not any(ep.money_changes(t, t).values())
    late = {**t, "finished_work": t["finished_work"].replace(minute=50)}
    assert ep.money_changes(t, late) == {
        "work_minutes": True, "busy_minutes": False, "night": False, "fx_date": False
    }
    next_day = {**t, "finished_work": t["finished_work"].replace(day=5)}
    assert ep.money_changes(t, next_day)["fx_date"]
    night = {**t, "started_work": t["started_work"].replace(hour=2)}
    assert ep.money_changes(t, night)["night"]


def test_auto_stage() -> None:
    w = {"truth_rank": 1, "source": "line"}
    assert ep.auto_stage({"status": "align_failed"}, []) == "alignment"
    assert ep.auto_stage({"status": "ok"}, [{**w, "source": "row"}]) == "crop"
    assert ep.auto_stage({"status": "ok"}, [w]) == "decoder"
    assert ep.auto_stage({"status": "ok"}, [w, {**w, "truth_rank": 4}]) == "recognition"


def test_wrong_subfields_ranks_truth_in_image() -> None:
    row = _truth_row("2026_12k", 12)
    forms = {**TRUTH, "left_base": (date(2026, 5, 4), 9, 0)}
    detail = {
        "forms": ep._forms_json(forms), "sources": {"left_base.hour": "line"},
        "_inputs": {"left_base.hour": [[9, 0.6], [8, 0.3], [7, 0.1]]},
    }
    (w,) = ep.wrong_subfields(detail, row)
    assert w == {"subfield": "left_base.hour", "truth": 8, "pred": 9, "truth_rank": 2,
                 "truth_p": 0.3, "image_top1": 9, "source": "line"}


def test_disagreements_selection() -> None:
    items = {
        "val": [_v("a", False, 0.9999), _v("b", False, 0.9), _v("c", True, 0.99999)],
        "test": [_v("d", True, 0.99, number_ok=False, p_number=0.9999, split="test"),
                 _v("e", False, 0.99999, split="test")],
    }
    picked = ep.disagreements(items, "core", 0.999)
    assert [v.scan_id for v in picked] == ["e", "a", "d"]
    assert [v.scan_id for v in ep.disagreements(items, "core", 0.999, limit=1)] == ["e"]


def test_review_image(work: Path) -> None:
    sid = "2026_12k"
    for name, w in ((boxes.VOUCHER_NUMBER, 250), *((r, 750) for r in ROWS)):
        _write_png(ep.bands_dir("t") / sid / f"{name}.png", np.full((90, w, 3), 200, np.uint8))
    v = ep.VoucherOutcome(sid, "val", "ok", False, False, ("finished_work",),
                          {"core": 0.9999, "final": 1.0, "guarded": 0.9999}, 0.99)
    forms = {**TRUTH, "finished_work": (date(2026, 5, 4), 10, 50)}
    detail = {"forms": ep._forms_json(forms), "number": 21, "p_number": 0.99,
              "partner": "2026_12p"}
    img = ep.review_image(v, detail, _truth_row(sid, 12), "core", "t")
    assert img.width == 750 and max(img.size) <= 2000
    assert img.height > 5 * 90


# --- поправки истины -------------------------------------------------------------------------


def test_correction_rows_and_append(tmp_path: Path) -> None:
    sid = "2026_12k"
    pred = {**TRUTH, "finished_work": (date(2026, 5, 4), 10, 50)}
    details = {sid: {"forms": ep._forms_json(pred), "number": 21}}
    manifest_rows = {sid: {k: str(v) for k, v in _truth_row(sid, 12).items()}}
    verdicts = [
        {"scan_id": sid, "verdict": "model", "rows": "", "note": "на бланке 50"},
        {"scan_id": sid, "verdict": "truth", "rows": "", "note": ""},
    ]
    rows = ep.correction_rows(verdicts, details, manifest_rows)
    assert [(r["field"], r["export_value"], r["corrected_value"]) for r in rows] == [
        ("finished_work", "04.05.2026 10:00", "04.05.2026 10:50"),
        (boxes.VOUCHER_NUMBER, "12", "21"),
    ]
    assert all(r["status"] == "proposed" and r["source"] == "T21" for r in rows)
    only = ep.correction_rows([{**verdicts[0], "rows": "voucher_number"}], details, manifest_rows)
    assert [r["field"] for r in only] == [boxes.VOUCHER_NUMBER]

    path = tmp_path / "truth_corrections.csv"
    head = ("\ufeffscan_id,field,export_value,corrected_value,status,source,note\r\n"
            "2025_1k,arrived_base,05.01.2025 12:50,05.01.2025 13:40,CONFIRMED,plan,x").encode()
    path.write_bytes(head)  # без перевода строки в конце
    assert ep.append_corrections(path, rows) == 2
    assert ep.append_corrections(path, rows) == 0  # повторный запуск ничего не дописывает
    assert path.read_bytes().startswith(head + b"\r\n")  # старые байты не тронуты
    with path.open(encoding="utf-8-sig", newline="") as fh:
        got = list(csv.DictReader(fh))
    assert [r["status"] for r in got] == ["CONFIRMED", "proposed", "proposed"]
    assert got[1]["note"] == "на бланке 50"

    fresh = tmp_path / "new.csv"
    assert ep.append_corrections(fresh, rows[:1]) == 1
    assert fresh.read_bytes().startswith(b"\xef\xbb\xbfscan_id,field")


def test_form_text_hour24() -> None:
    assert ep.form_text((date(2026, 1, 5), 24, 0)) == "05.01.2026 24:00"
