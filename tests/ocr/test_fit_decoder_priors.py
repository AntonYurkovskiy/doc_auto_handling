"""Тесты подбора приоров и весов декодера на синтетическом манифесте."""

from __future__ import annotations

import csv
import json
from datetime import datetime, timedelta

import numpy as np

from app.ocr.decoder import CHAIN, PARTS, DecoderModel, decode, subfield
from ocr_lab import fit_decoder_priors as fit

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
    assert fit.main(["--manifest", str(manifest), "--out", str(out)]) == 0
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
