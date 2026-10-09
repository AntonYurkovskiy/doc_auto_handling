"""Тесты подбора приоров и весов декодера на синтетическом манифесте."""

from __future__ import annotations

import csv
import json
import math
from dataclasses import replace
from datetime import datetime, timedelta

import numpy as np
import pytest

from app.ocr.decoder import (
    CHAIN,
    PARTS,
    DecoderModel,
    _evidence,
    _score_scalar,
    decode,
    subfield,
)
from ocr_lab import fit_decoder_priors as fit
from ocr_lab.evaluate import evaluate, load_truth
from ocr_lab.predictions import read_predictions

ROW_FIELDS = ("dt", "day", "month", "year", "hour", "minute")


def _voucher(rng: np.random.Generator, left: datetime) -> dict[str, datetime]:
    legs = [int(rng.choice([10, 20, 30, 40, 60, 90])) for _ in range(3)]
    legs[1] = int(rng.choice([30, 60, 90, 120, 180]))
    times = [left]
    for minutes in legs:
        times.append(times[-1] + timedelta(minutes=minutes))
    return dict(zip(CHAIN, times, strict=True))


def _form(dt: datetime, hour24: bool) -> tuple[int, int, int, int, int]:
    if hour24:
        prev = dt - timedelta(days=1)
        return prev.day, prev.month, prev.year, 24, 0
    return dt.day, dt.month, dt.year, dt.hour, dt.minute


def _write_manifest(path, rng: np.random.Generator) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    start = datetime(2025, 1, 3, 0, 0)
    for i in range(260):
        day = start + timedelta(days=int(i * 1.9))
        left = day.replace(hour=int(rng.integers(5, 22)), minute=int(rng.choice(range(0, 60, 10))))
        times = _voucher(rng, left)
        split = "train" if day < datetime(2026, 3, 1) else "val"
        row = {
            "scan_id": f"{day.year}_{i + 1}k",
            "year": str(left.year),
            "split": split,
            "work_type": "швартовка" if i % 2 else "отшвартовка",
            "app_dt": (times["started_work"] - timedelta(hours=int(rng.integers(0, 8)))).isoformat(
                timespec="minutes"
            ),
            "chain_ok": "True",
        }
        for name in CHAIN:
            dt = times[name]
            hour24 = dt.hour == 0 and dt.minute == 0 and rng.random() < 0.5
            d, m, y, h, mi = _form(dt, hour24)
            row.update(
                {
                    f"{name}_dt": dt.isoformat(timespec="minutes"),
                    f"{name}_day": str(d),
                    f"{name}_month": str(m),
                    f"{name}_year": str(y),
                    f"{name}_hour": str(h),
                    f"{name}_minute": str(mi),
                }
            )
        rows.append(row)
    # Ваучер с нарушенной цепочкой не должен портить длительности.
    rows[0]["chain_ok"] = "False"
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return rows


def _noisy_fields(row: dict[str, str], rng: np.random.Generator) -> dict[str, list]:
    fields: dict[str, list] = {}
    ranges = {"day": (1, 31), "month": (1, 12), "hour": (0, 24), "minute": (0, 59)}
    for name in CHAIN:
        for part in PARTS:
            value = int(row[f"{name}_{part}"])
            lo, hi = ranges[part]
            other = int(rng.integers(lo, hi + 1))
            if other == value:
                other = value + 1 if value < hi else value - 1
            if rng.random() < 0.1:
                dist = [[other, 0.55], [value, 0.4]]  # ошибка top-1 картинки
            else:
                dist = [[value, 0.8], [other, 0.15]]
            fields[subfield(name, part)] = dist
    return fields


def test_fit_priors_and_weights_on_synthetic_manifest(tmp_path):
    rng = np.random.default_rng(7)
    manifest = tmp_path / "manifest.csv"
    rows = _write_manifest(manifest, rng)
    pred = tmp_path / "val_predictions.jsonl"
    with pred.open("w", encoding="utf-8") as fh:
        for row in rows:
            if row["split"] != "val":
                continue
            item = {"scan_id": row["scan_id"], "source": "digits_v0"}
            item["fields"] = _noisy_fields(row, rng)
            fh.write(json.dumps(item) + "\n")
    out = tmp_path / "models" / "decoder_priors_v0.json"

    code = fit.main(
        [
            "fit",
            "--manifest",
            str(manifest),
            "--pred",
            str(pred),
            "--out",
            str(out),
            "--grid",
            "0.5,1,2",
            "--rounds",
            "1",
            "--min-work-type",
            "50",
        ]
    )
    assert code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert data["meta"]["weights_fit_on"] == "val"
    assert data["meta"]["prior_stats"]["minutes_mult10_share"] == 1.0
    after = data["meta"]["fit_after"]
    before = data["meta"]["fit_before"]
    assert after["mean_logp"] >= before["mean_logp"]
    assert after["top1"] >= after["fields_top1"]

    model = DecoderModel.load(out)
    assert set(model.priors.durations_by_work_type) == {"швартовка", "отшвартовка"}
    # Эмпирика перевесила сглаживание: кратные 10 минуты почти всё.
    minute_p = np.exp(model.priors.minute_array)
    assert minute_p[::10].sum() > 0.99
    # Декодер с подобранной моделью читает val-ваучер.
    samples = fit.build_samples(
        [r for r in rows if r["split"] == "val"], fit.load_predictions(pred)
    )
    hits = sum(decode(s.inputs, s.context, model).top.forms() == s.forms for s in samples[:20])
    assert hits >= 17


def test_fit_priors_without_predictions_keeps_default_weights(tmp_path):
    rng = np.random.default_rng(3)
    manifest = tmp_path / "manifest.csv"
    _write_manifest(manifest, rng)
    out = tmp_path / "decoder_priors_v0.json"
    assert fit.main(["fit", "--manifest", str(manifest), "--out", str(out), "--no-weights"]) == 0
    model = DecoderModel.load(out)
    assert model.meta["weights_fit_on"] is None
    assert model.weights == DecoderModel.default().weights


def test_truth_forms_keeps_hour24_and_offsets():
    row = {
        "year": "2025",
        **{f"{name}_{f}": "" for name in CHAIN for f in ROW_FIELDS},
    }
    values = {
        "left_base": (31, 12, 2025, 22, 0),
        "started_work": (31, 12, 2025, 23, 0),
        "finished_work": (31, 12, 2025, 24, 0),
        "arrived_base": (1, 1, 2026, 0, 40),
    }
    for name, (d, m, y, h, mi) in values.items():
        row.update(
            {
                f"{name}_day": str(d),
                f"{name}_month": str(m),
                f"{name}_year": str(y),
                f"{name}_hour": str(h),
                f"{name}_minute": str(mi),
            }
        )
    forms = fit.truth_forms(row)
    assert forms is not None
    assert fit.form_minutes(forms) == [22 * 60, 23 * 60, 24 * 60, 1440 + 40]
    assert fit.chain_ok(row, forms)


def test_fit_priors_ties_hour24_to_midnight_and_counts_late_finish(tmp_path):
    rng = np.random.default_rng(11)
    rows = _write_manifest(tmp_path / "manifest.csv", rng)
    train = [r for r in rows if r["split"] == "train"]
    # Один бланк с Приходом на 20 минут раньше Окончания и один — на 2 часа (вне окна).
    for row, minutes in ((train[5], 20), (train[6], 120)):
        fin = datetime.fromisoformat(row["finished_work_dt"])
        arr = fin - timedelta(minutes=minutes)
        row.update(
            {
                "arrived_base_dt": arr.isoformat(timespec="minutes"),
                "arrived_base_day": str(arr.day),
                "arrived_base_month": str(arr.month),
                "arrived_base_year": str(arr.year),
                "arrived_base_hour": str(arr.hour),
                "arrived_base_minute": str(arr.minute),
                "chain_ok": "False",
            }
        )
    priors, stats = fit.fit_priors(train)
    for name in CHAIN:
        hour = np.exp(priors.hour_array(name))
        assert hour[24] == pytest.approx(hour[0])
        assert hour.sum() == pytest.approx(1.0)
    assert stats.late_finish == 1
    assert stats.late_finish_beyond == 1
    assert priors.late_finish_max == 30.0
    expected = (1 + 200 * 0.005) / (stats.train_vouchers + 200)
    assert priors.late_finish_logp == pytest.approx(math.log(expected / 30.0))


def test_bootstrap_support_rejects_gain_of_single_voucher():
    spread = np.full(51, 0.02)
    single = np.zeros(51)
    single[3] = 1.0
    mixed = np.zeros(51)
    mixed[:2] = [1.2, -0.9]
    assert fit.bootstrap_support(spread) == 1.0
    assert fit.bootstrap_support(single) < 0.8
    assert fit.bootstrap_support(mixed) < 0.9


def test_truth_outside_candidates_is_added_to_normalizer(tmp_path):
    rng = np.random.default_rng(5)
    rows = _write_manifest(tmp_path / "manifest.csv", rng)
    row = next(r for r in rows if r["split"] == "val")
    fields = {
        subfield(name, part): [(int(row[f"{name}_{part}"]), 0.97)]
        for name in CHAIN
        for part in PARTS
    }
    true_hour = int(row["left_base_hour"])
    wrong = true_hour - 1 if true_hour > 5 else true_hour + 1
    fields["left_base.hour"] = [(wrong, 0.6), (true_hour + 5, 0.39)]
    sample = fit.build_samples([row], {row["scan_id"]: fields})[0]
    model = DecoderModel.default()
    narrow = replace(model, search=replace(model.search, all_hours=False))
    logp, reachable = fit.truth_log_prob(narrow, sample)
    assert not reachable
    assert fit.LOG_P_CLIP < logp < 0.0
    result = decode(sample.inputs, sample.context, narrow)
    ev = _evidence(sample.inputs, narrow.search.prob_floor)
    score = _score_scalar(sample.forms, ev, sample.context, narrow)
    assert logp == pytest.approx(score - float(np.logaddexp(result.log_z, score)))
    wide_logp, wide_reach = fit.truth_log_prob(model, sample)
    assert wide_reach


def test_predict_split_writes_t13_jsonl_readable_by_evaluate(tmp_path):
    rng = np.random.default_rng(9)
    manifest = tmp_path / "manifest.csv"
    rows = _write_manifest(manifest, rng)
    val = [r for r in rows if r["split"] == "val"]
    inputs = {r["scan_id"]: _noisy_fields(r, rng) for r in val}
    inputs = {
        k: {f: [(int(v), float(p)) for v, p in d] for f, d in x.items()} for k, x in inputs.items()
    }
    del inputs[val[0]["scan_id"]]["left_base.minute"]  # пропуск подполя
    out = tmp_path / "decoder_v0" / "val_predictions.jsonl"
    numbers = {val[1]["scan_id"]: [[7, 0.9]]}
    stats = fit.predict_split(
        "val", DecoderModel.default(), inputs, fit.read_manifest(manifest), out, numbers=numbers
    )
    assert stats.n_scans == len(val)
    assert stats.n_no_candidates == 0
    assert stats.ms_median > 0
    preds = {p.scan_id: p for p in read_predictions(out)}
    assert set(preds) == {r["scan_id"] for r in val}
    first = preds[val[0]["scan_id"]]
    assert first.source == fit.SOURCE
    assert 1 <= len(first.records) <= 5
    assert first.confidence is not None and first.margin is not None
    assert "left_base.minute" not in first.fields
    assert preds[val[1]["scan_id"]].fields["voucher_number"] == [(7, 0.9)]
    raw = json.loads(out.read_text(encoding="utf-8").splitlines()[0])
    assert "missing_subfields" in raw["flags"]
    metrics = evaluate(load_truth(manifest, "val"), read_predictions(out))
    assert metrics["n_matched"] == len(val)
    assert metrics["vouchers"]["rows"]["all"]["accuracy"] > 0.5
    assert metrics["auto_accept"]["n_with_confidence"] == len(val)


def test_split_crops_strips_scan_id_and_skips_failed_alignment(tmp_path):
    manifest = tmp_path / "manifest.csv"
    with manifest.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scan_id", "split"])
        writer.writeheader()
        writer.writerows(
            [{"scan_id": "2026_223k ", "split": "test"}, {"scan_id": "x", "split": "val"}]
        )
    index = tmp_path / "crops_index.csv"
    with index.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=["scan_id", "subfield", "path", "align_ok"])
        writer.writeheader()
        writer.writerows(
            [
                {
                    "scan_id": "2026_223k ",
                    "subfield": "left_base.hour",
                    "path": "a.png",
                    "align_ok": "True",
                },
                {
                    "scan_id": "2026_223k ",
                    "subfield": "voucher_number",
                    "path": "b.png",
                    "align_ok": "True",
                },
                {
                    "scan_id": "2026_223k ",
                    "subfield": "left_base.day",
                    "path": "c.png",
                    "align_ok": "False",
                },
                {"scan_id": "x", "subfield": "left_base.hour", "path": "d.png", "align_ok": "True"},
            ]
        )
    crops = fit.split_crops("test", manifest=manifest, index_csv=index)
    assert crops == [("2026_223k", "left_base.hour", tmp_path / "a.png")]
