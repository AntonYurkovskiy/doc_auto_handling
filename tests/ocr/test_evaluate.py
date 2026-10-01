"""Тесты формата предсказаний и модуля оценки на синтетической истине."""

from __future__ import annotations

import csv
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import pytest

from ocr_lab import evaluate as ev
from ocr_lab.predictions import (
    PARTS,
    ROWS,
    PredictionError,
    RecordCandidate,
    ScanPrediction,
    parse_prediction,
    read_predictions,
    record_from_fields,
    write_predictions,
)

# Строка бланка: (год, месяц, день, час, минута); час 24 — «24:00» на бланке.
RowSpec = tuple[int, int, int, int, int]

SCANS: list[tuple[str, str, str, int, dict[str, RowSpec]]] = [
    (
        "2026_101k", "val", "k", 101,
        {
            "left_base": (2026, 4, 10, 9, 10),
            "started_work": (2026, 4, 10, 10, 0),
            "finished_work": (2026, 4, 10, 12, 30),
            "arrived_base": (2026, 4, 10, 13, 40),
        },
    ),
    (
        # Переход через полночь, начало работ ночью.
        "2026_55p", "val", "p", 55,
        {
            "left_base": (2026, 4, 20, 22, 30),
            "started_work": (2026, 4, 20, 23, 10),
            "finished_work": (2026, 4, 21, 0, 40),
            "arrived_base": (2026, 4, 21, 1, 30),
        },
    ),
    (
        # «24:00» в окончании работ.
        "2026_120k", "val", "k", 120,
        {
            "left_base": (2026, 5, 2, 21, 0),
            "started_work": (2026, 5, 2, 22, 0),
            "finished_work": (2026, 5, 2, 24, 0),
            "arrived_base": (2026, 5, 3, 0, 40),
        },
    ),
    (
        "2025_7p", "train", "p", 7,
        {
            "left_base": (2025, 3, 1, 8, 0),
            "started_work": (2025, 3, 1, 9, 0),
            "finished_work": (2025, 3, 1, 10, 0),
            "arrived_base": (2025, 3, 1, 11, 0),
        },
    ),
]


def _dt(spec: RowSpec) -> datetime:
    y, mo, d, h, mi = spec
    if h == 24:
        return datetime(y, mo, d) + timedelta(days=1)
    return datetime(y, mo, d, h, mi)


@pytest.fixture
def manifest(tmp_path: Path) -> Path:
    path = tmp_path / "manifest.csv"
    cols = ["scan_id", "split", "tug_code", "year", "voucher_number", "has_hour24"]
    for r in ROWS:
        cols += [f"{r}_dt", f"{r}_day", f"{r}_month", f"{r}_year", f"{r}_hour", f"{r}_minute"]
    with path.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for scan_id, split, tug, number, rows in SCANS:
            rec: dict[str, object] = {
                "scan_id": scan_id,
                "split": split,
                "tug_code": tug,
                "year": rows["left_base"][0],
                "voucher_number": number,
                "has_hour24": any(s[3] == 24 for s in rows.values()),
            }
            for r, spec in rows.items():
                y, mo, d, h, mi = spec
                rec.update(
                    {
                        f"{r}_dt": _dt(spec).isoformat(),
                        f"{r}_day": d,
                        f"{r}_month": mo,
                        f"{r}_year": y,
                        f"{r}_hour": h,
                        f"{r}_minute": mi,
                    }
                )
            w.writerow(rec)
    return path


def _perfect(manifest: Path, split: str = "val") -> tuple[dict, list[ScanPrediction]]:
    truth = ev.load_truth(manifest, split)
    return truth, ev.perfect_predictions(truth)


def _set(pred: ScanPrediction, subfield: str, value: int) -> None:
    pred.fields[subfield] = [(value, 1.0)]


def _by_id(preds: list[ScanPrediction], scan_id: str) -> ScanPrediction:
    return next(p for p in preds if p.scan_id == scan_id)


def _assert_all_perfect(m: dict) -> None:
    for dim in m["subfields"].values():
        for s in dim.values():
            assert s["top1"] == 1.0 and s["top3"] == 1.0 and s["coverage"] == 1.0
            assert s["nll"] == pytest.approx(0.0) and s["ece"] == pytest.approx(0.0)
    for dim in m["timestamps"].values():
        for s in dim.values():
            assert s["accuracy"] == 1.0 and s["coverage"] == 1.0
    for crit in m["vouchers"].values():
        for s in crit.values():
            assert s["accuracy"] == 1.0
    money = m["money"]
    for key in ("abs_delta_work_minutes", "abs_delta_busy_minutes"):
        assert money[key]["max"] == 0 and money[key]["share_nonzero"] == 0.0
    assert money["night_flag_changed"]["share"] == 0.0
    assert money["finish_date_changed"]["share"] == 0.0
    thr = m["auto_accept"]["threshold"]
    assert thr["precision"] == 1.0 and thr["coverage"] == 1.0


def test_load_truth_split_and_hour24(manifest: Path) -> None:
    truth = ev.load_truth(manifest, "val")
    assert set(truth) == {"2026_101k", "2026_55p", "2026_120k"}
    t = truth["2026_120k"]
    assert t.has_hour24
    assert t.parts["finished_work.hour"] == 24
    assert t.rows["finished_work"] == datetime(2026, 5, 3, 0, 0)
    assert len(ev.load_truth(manifest, "all")) == 4


@pytest.mark.parametrize("with_records", [False, True])
def test_perfect_predictions_give_100(manifest: Path, with_records: bool) -> None:
    truth = ev.load_truth(manifest, "val")
    preds = ev.perfect_predictions(truth, with_records=with_records)
    m = ev.evaluate(truth, preds)
    _assert_all_perfect(m)
    assert m["n_matched"] == 3
    # Ваучеров мало: цель 99,5 % честно не подтверждена.
    thr = m["auto_accept"]["threshold"]
    assert not thr["confirmed"] and "мала" in thr["note"]


def test_cli_perfect_roundtrip(manifest: Path, tmp_path: Path) -> None:
    pred_path = tmp_path / "perfect.jsonl"
    out = tmp_path / "report"
    args = ["--manifest", str(manifest), "--split", "val"]
    assert ev.main(["--write-perfect", str(pred_path), "--with-records", *args]) == 0
    assert ev.main(["--pred", str(pred_path), "--out", str(out), *args]) == 0
    m = json.loads((out / "metrics.json").read_text(encoding="utf-8"))
    _assert_all_perfect(m)
    report = (out / "report.md").read_text(encoding="utf-8")
    assert "## Автоприём" in report and "100.00 %" in report


def test_one_wrong_minute(manifest: Path) -> None:
    truth, preds = _perfect(manifest)
    _set(_by_id(preds, "2026_101k"), "finished_work.minute", 40)
    m = ev.evaluate(truth, preds)
    assert m["subfields"]["subfield"]["finished_work.minute"]["top1"] == pytest.approx(2 / 3)
    assert m["subfields"]["subfield"]["finished_work.minute"]["top3"] == pytest.approx(2 / 3)
    assert m["timestamps"]["overall"]["all"]["accuracy"] == pytest.approx(11 / 12)
    assert m["timestamps"]["row"]["finished_work"]["accuracy"] == pytest.approx(2 / 3)
    assert m["vouchers"]["rows"]["all"]["accuracy"] == pytest.approx(2 / 3)
    assert m["vouchers"]["rows"]["p"]["accuracy"] == 1.0
    work = m["money"]["abs_delta_work_minutes"]
    assert work["max"] == 10 and work["share_nonzero"] == pytest.approx(1 / 3)
    assert m["money"]["abs_delta_busy_minutes"]["max"] == 0
    assert m["money"]["night_flag_changed"]["share"] == 0.0
    # Вероятность 1.0 у неверного ответа: NLL на пороге, калибровка плохая.
    s = m["subfields"]["subfield"]["finished_work.minute"]
    assert s["nll"] == pytest.approx(-math.log(ev.NLL_FLOOR) / 3)
    assert s["ece"] == pytest.approx(1 / 3)


def test_hour24_equals_midnight_next_day(manifest: Path) -> None:
    truth, preds = _perfect(manifest)
    pred = _by_id(preds, "2026_120k")
    # 00:00 следующих суток вместо «24:00»: подполя неверны, метка времени верна.
    _set(pred, "finished_work.hour", 0)
    _set(pred, "finished_work.day", 3)
    m = ev.evaluate(truth, preds)
    assert m["subfields"]["subfield"]["finished_work.hour"]["top1"] == pytest.approx(2 / 3)
    assert m["timestamps"]["overall"]["all"]["accuracy"] == 1.0
    assert m["vouchers"]["rows"]["all"]["accuracy"] == 1.0

    # 00:00 того же дня — это уже другая метка и другая дата курса.
    _set(pred, "finished_work.day", 2)
    m = ev.evaluate(truth, preds)
    assert m["timestamps"]["row"]["finished_work"]["accuracy"] == pytest.approx(2 / 3)
    assert m["money"]["finish_date_changed"]["share"] == pytest.approx(1 / 3)


def test_record_from_fields_converts_hour24() -> None:
    fields = {f"finished_work.{p}": [(v, 0.9)] for p, v in zip(PARTS, (31, 12, 24, 0), strict=True)}
    pred = ScanPrediction("s", "x", fields=fields)  # type: ignore[arg-type]
    rec = record_from_fields(pred, {"finished_work": 2025})
    assert rec.rows["finished_work"] == datetime(2026, 1, 1, 0, 0)
    assert rec.hour24 == ("finished_work",)
    assert rec.rows["left_base"] is None


def test_decoder_record_takes_priority(manifest: Path) -> None:
    truth, preds = _perfect(manifest)
    pred = _by_id(preds, "2026_120k")
    _set(pred, "finished_work.hour", 23)  # поля неверны, но запись декодера верна
    t = truth["2026_120k"]
    pred.records = [RecordCandidate(rows=dict(t.rows), hour24=("finished_work",), p=0.9)]
    m = ev.evaluate(truth, preds)
    assert m["timestamps"]["overall"]["all"]["accuracy"] == 1.0
    # Номер берётся из подполя, если декодер его не дал.
    assert m["vouchers"]["rows_number"]["all"]["accuracy"] == 1.0


def test_midnight_crossing_missed(manifest: Path) -> None:
    truth, preds = _perfect(manifest)
    # Распознаватель не увидел смену дня в окончании работ.
    _set(_by_id(preds, "2026_55p"), "finished_work.day", 20)
    m = ev.evaluate(truth, preds)
    work = m["money"]["abs_delta_work_minutes"]
    assert work["max"] == 90  # было 23:10 → 00:40, стало отрицательным → 0
    assert m["money"]["finish_date_changed"]["share"] == pytest.approx(1 / 3)
    assert m["money"]["abs_delta_busy_minutes"]["max"] == 0


def test_night_flag_change(manifest: Path) -> None:
    truth, preds = _perfect(manifest)
    _set(_by_id(preds, "2026_55p"), "started_work.hour", 21)
    m = ev.evaluate(truth, preds)
    assert m["money"]["night_flag_changed"]["share"] == pytest.approx(1 / 3)
    assert m["money"]["abs_delta_work_minutes"]["max"] == 120


def test_missing_prediction_counts_as_uncovered(manifest: Path) -> None:
    truth, preds = _perfect(manifest)
    preds = [p for p in preds if p.scan_id != "2026_55p"]
    m = ev.evaluate(truth, preds)
    assert m["subfields"]["overall"]["all"]["coverage"] == pytest.approx(2 / 3)
    assert m["timestamps"]["overall"]["all"]["coverage"] == pytest.approx(2 / 3)
    assert m["vouchers"]["rows"]["all"]["accuracy"] == pytest.approx(2 / 3)
    assert m["money"]["n_missing_work_interval"] == 1
    assert m["auto_accept"]["threshold"]["coverage"] == pytest.approx(2 / 3)


def test_printed_flags_slice(manifest: Path, tmp_path: Path) -> None:
    flags_path = tmp_path / "printed_flags.csv"
    flags_path.write_text(
        "scan_id,field,printed\n2026_101k,voucher_number,1\n2026_101k,left_base,0\n",
        encoding="utf-8",
    )
    flags = ev.load_printed_flags(flags_path)
    truth, preds = _perfect(manifest)
    m = ev.evaluate(truth, preds, printed_flags=flags)
    sl = m["subfields"]["printed"]
    assert sl["printed"]["n"] == 1
    assert sl["handwritten"]["n"] == 4
    assert sl["unknown"]["n"] == 3 * 17 - 5
    assert ev.load_printed_flags(tmp_path / "nope.csv") == {}


# --- Автоприём ----------------------------------------------------------------------------


def test_threshold_toy_not_confirmed() -> None:
    conf = [0.99, 0.98, 0.97, 0.96, 0.5]
    ok = [True, True, True, True, False]
    thr = ev.threshold_for_precision(conf, ok, target=0.9)
    assert thr.threshold == 0.96
    assert thr.coverage == pytest.approx(0.8) and thr.precision == 1.0
    assert not thr.confirmed and "мала" in thr.note
    assert thr.min_n_to_confirm == ev.min_n_to_confirm(0.9)
    assert thr.ci_low is not None and thr.ci_low < 0.9


def test_threshold_toy_confirmed() -> None:
    conf = [0.99] * 1000 + [0.3] * 10
    ok = [True] * 1000 + [False] * 10
    thr = ev.threshold_for_precision(conf, ok, target=0.995)
    assert thr.confirmed and thr.threshold == 0.99
    assert thr.n_accepted == 1000 and thr.coverage == pytest.approx(1000 / 1010)
    assert thr.ci_low is not None and thr.ci_low >= 0.995


def test_threshold_unreachable() -> None:
    thr = ev.threshold_for_precision([0.9, 0.8], [False, True], target=0.995)
    assert thr.threshold is None and not thr.confirmed
    assert "ни один порог" in thr.note


def test_wilson_and_min_n() -> None:
    low, high = ev.wilson_interval(0, 0)
    assert (low, high) == (0.0, 1.0)
    low, high = ev.wilson_interval(50, 100)
    assert low == pytest.approx(0.4038, abs=1e-3) and high == pytest.approx(0.5962, abs=1e-3)
    n = ev.min_n_to_confirm(0.995)
    assert ev.wilson_interval(n, n)[0] >= 0.995 > ev.wilson_interval(n - 1, n - 1)[0]


def test_curve_ties() -> None:
    curve = ev.coverage_precision_curve([0.9, 0.9, 0.5], [True, False, True], n_total=4)
    assert [p["n_accepted"] for p in curve] == [2, 3]
    assert curve[0]["precision"] == 0.5 and curve[1]["coverage"] == 0.75


# --- Формат JSONL -------------------------------------------------------------------------


def _line(**kw: object) -> dict:
    obj: dict = {"scan_id": "s1", "source": "t", "fields": {"left_base.hour": [[9, 0.9]]}}
    obj.update(kw)
    return obj


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    pred = parse_prediction(
        _line(
            records=[
                {
                    "left_base": "2026-07-02T09:10",
                    "finished_work": "2026-07-03T00:00",
                    "hour24": ["finished_work"],
                    "voucher_number": 243,
                    "p": 0.97,
                }
            ],
            confidence=0.97,
            margin=0.95,
        )
    )
    path = tmp_path / "p.jsonl"
    write_predictions(path, [pred])
    [back] = read_predictions(path)
    assert back == pred
    assert back.records[0].rows["finished_work"] == datetime(2026, 7, 3)


@pytest.mark.parametrize(
    "bad",
    [
        _line(fields={"left_base.second": [[1, 0.5]]}),
        _line(fields={"left_base.hour": [[9, 1.5]]}),
        _line(fields={"left_base.hour": [[9, 0.1], [8, 0.8]]}),
        _line(fields={"left_base.hour": [[9, 0.6], [8, 0.6]]}),
        _line(fields={"left_base.hour": [["9", 0.6]]}),
        _line(fields={"left_base.hour": [[9, 0.5], [9, 0.4]]}),
        _line(confidence=2),
        _line(records=[{"left_base": "2026-07-02T09:10", "hour24": ["left_base"]}]),
        _line(records=[{"bogus": 1}]),
        {"source": "t"},
    ],
)
def test_validation_rejects(bad: dict) -> None:
    with pytest.raises(PredictionError):
        parse_prediction(bad)


def test_duplicate_scan_id_rejected(tmp_path: Path) -> None:
    path = tmp_path / "dup.jsonl"
    path.write_text("\n".join(json.dumps(_line()) for _ in range(2)), encoding="utf-8")
    with pytest.raises(PredictionError, match="повтор"):
        read_predictions(path)


# --- Пути ---------------------------------------------------------------------------------


def test_paths_contract(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    from ocr_lab import paths

    assert paths.MANIFEST == paths.WORK_DIR / "manifest.csv"
    assert paths.CORRECTIONS_CSV == paths.WORK_DIR / "truth_corrections.csv"
    with pytest.raises(ValueError):
        paths.work_subdir("secret")
    monkeypatch.setattr(paths, "WORK_DIR", tmp_path / "ocr")
    assert paths.work_subdir("reports").is_dir()

    monkeypatch.setenv("HF_HOME", str(tmp_path / "hf_preset"))
    monkeypatch.delenv("TORCH_HOME", raising=False)
    monkeypatch.setattr(paths, "CACHE_DIR", tmp_path / "cache")
    env = paths.configure_model_caches()
    assert env["HF_HOME"] == str(tmp_path / "hf_preset")
    assert Path(env["TORCH_HOME"]) == tmp_path / "cache" / "torch"
