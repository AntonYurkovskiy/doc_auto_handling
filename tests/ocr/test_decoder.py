"""Тесты совместного декодера записи ваучера (T19) на синтетических распределениях."""

from __future__ import annotations

import itertools
import math
import time as time_mod
from dataclasses import replace
from datetime import date, datetime, timedelta

import numpy as np
import pytest

from app.ocr.decoder import (
    CHAIN,
    FLAG_CROSSES_MIDNIGHT,
    FLAG_HOUR24,
    FLAG_LATE_FINISH,
    FLAG_MISSING,
    FLAG_NO_CANDIDATES,
    FLAG_TOP1_CHAIN_VIOLATION,
    PARTS,
    ROWS,
    DecodeContext,
    DecoderModel,
    DecoderPriors,
    SearchParams,
    _evidence,
    _score_scalar,
    candidate_space,
    decode,
    forms_from_values,
    subfield,
)

Truth = dict[str, tuple[int, int, int, int]]


def make_inputs(truth: Truth, conf: float = 0.9) -> dict[str, list[tuple[int, float]]]:
    """Уверенные распределения вокруг истины: верное значение + соседний «шум»."""
    inputs: dict[str, list[tuple[int, float]]] = {}
    alt = {"day": 1, "month": 1, "hour": 1, "minute": 10}
    hi = {"day": 31, "month": 12, "hour": 23, "minute": 59}
    for row, values in truth.items():
        for part, value in zip(PARTS, values, strict=True):
            other = value + alt[part] if value + alt[part] <= hi[part] else value - alt[part]
            inputs[subfield(row, part)] = [(value, conf), (other, (1 - conf) / 2)]
    return inputs


def top_dt(result) -> dict[str, datetime]:
    assert result.records, result.flags
    return {row: value.dt for row, value in result.top.rows.items()}


BASE_TRUTH: Truth = {
    "left_base": (5, 3, 9, 10),
    "started_work": (5, 3, 10, 0),
    "finished_work": (5, 3, 12, 30),
    "arrived_base": (5, 3, 13, 20),
}
CTX = DecodeContext(year=2026)


def test_basic_confident_record_and_contract():
    result = decode(make_inputs(BASE_TRUTH), CTX)
    assert top_dt(result) == {
        "left_base": datetime(2026, 3, 5, 9, 10),
        "arrived_base": datetime(2026, 3, 5, 13, 20),
        "started_work": datetime(2026, 3, 5, 10, 0),
        "finished_work": datetime(2026, 3, 5, 12, 30),
    }
    assert 0.5 < result.confidence <= 1.0
    assert math.isclose(result.margin, result.records[0].p - result.records[1].p)
    probs = [r.p for r in result.records]
    assert probs == sorted(probs, reverse=True)
    assert sum(probs) <= 1.0 + 1e-9
    assert set(result.marginals) == set(ROWS)
    for row in ROWS:
        options = result.marginals[row]
        assert 1 <= len(options) <= 3
        assert options[0].value == result.top.rows[row]
    payload = result.top.to_json()
    assert payload["left_base"] == "2026-03-05T09:10"
    assert payload["hour24"] == []
    assert result.missing == ()
    assert result.overridden == ()


# 1. Top-1 картинки нарушает цепочку, второй вариант — нет.
def test_chain_violation_picks_second_option():
    inputs = make_inputs(BASE_TRUTH)
    # Начало работ: картинка скорее читает 08, но 08:00 раньше выхода 09:10.
    inputs["started_work.hour"] = [(8, 0.6), (10, 0.35), (9, 0.05)]
    result = decode(inputs, CTX)
    assert result.top.rows["started_work"].hour == 10
    assert "started_work.hour" in result.overridden
    assert FLAG_TOP1_CHAIN_VIOLATION in result.flags
    for rec in result.records:
        dts = [rec.rows[r].dt for r in CHAIN]
        assert dts == sorted(dts)


# 2. Переход через полночь.
def test_midnight_crossing():
    truth: Truth = {
        "left_base": (5, 3, 23, 40),
        "started_work": (5, 3, 23, 50),
        "finished_work": (6, 3, 0, 30),
        "arrived_base": (6, 3, 1, 10),
    }
    result = decode(make_inputs(truth), CTX)
    dts = top_dt(result)
    assert dts["left_base"] == datetime(2026, 3, 5, 23, 40)
    assert dts["finished_work"] == datetime(2026, 3, 6, 0, 30)
    assert dts["arrived_base"] == datetime(2026, 3, 6, 1, 10)
    assert FLAG_CROSSES_MIDNIGHT in result.flags
    assert result.confidence > 0.5


# 3. Переход месяца и Нового года.
def test_month_crossing():
    truth: Truth = {
        "left_base": (31, 1, 22, 30),
        "started_work": (31, 1, 23, 20),
        "finished_work": (1, 2, 0, 40),
        "arrived_base": (1, 2, 1, 30),
    }
    dts = top_dt(decode(make_inputs(truth), CTX))
    assert dts["started_work"] == datetime(2026, 1, 31, 23, 20)
    assert dts["finished_work"] == datetime(2026, 2, 1, 0, 40)
    assert dts["arrived_base"] == datetime(2026, 2, 1, 1, 30)


def test_new_year_crossing():
    truth: Truth = {
        "left_base": (31, 12, 23, 0),
        "started_work": (31, 12, 23, 40),
        "finished_work": (1, 1, 0, 50),
        "arrived_base": (1, 1, 1, 40),
    }
    result = decode(make_inputs(truth), DecodeContext(year=2025))
    dts = top_dt(result)
    assert dts["left_base"] == datetime(2025, 12, 31, 23, 0)
    assert dts["finished_work"] == datetime(2026, 1, 1, 0, 50)
    assert dts["arrived_base"] == datetime(2026, 1, 1, 1, 40)


# 4. 24:00 с минутами 00 допустимо, 24:10 — нет.
def test_hour24_with_zero_minutes_is_allowed():
    truth: Truth = {
        "left_base": (5, 3, 21, 30),
        "started_work": (5, 3, 22, 20),
        "finished_work": (5, 3, 24, 0),
        "arrived_base": (6, 3, 0, 40),
    }
    result = decode(make_inputs(truth), CTX)
    top = result.top
    assert top.rows["finished_work"].hour == 24
    assert top.rows["finished_work"].form_date == date(2026, 3, 5)
    assert top.rows["finished_work"].dt == datetime(2026, 3, 6, 0, 0)
    assert top.hour24 == ("finished_work",)
    assert top.to_json()["finished_work"] == "2026-03-06T00:00"
    assert top.to_json()["hour24"] == ["finished_work"]
    assert FLAG_HOUR24 in result.flags


def test_hour24_with_nonzero_minutes_is_rejected():
    truth: Truth = {
        "left_base": (5, 3, 21, 30),
        "started_work": (5, 3, 22, 20),
        "finished_work": (5, 3, 23, 10),
        "arrived_base": (6, 3, 0, 40),
    }
    inputs = make_inputs(truth)
    inputs["finished_work.hour"] = [(24, 0.9), (23, 0.08), (22, 0.02)]
    inputs["finished_work.minute"] = [(10, 0.95), (0, 0.03), (20, 0.02)]
    result = decode(inputs, CTX)
    for rec in result.records:
        for row in ROWS:
            value = rec.rows[row]
            assert not (value.hour == 24 and value.minute != 0)
    assert (result.top.rows["finished_work"].hour, result.top.rows["finished_work"].minute) != (
        24,
        10,
    )
    for options in result.marginals.values():
        for option in options:
            assert not (option.value.hour == 24 and option.value.minute != 0)


# Мягкое правило «Окончание ≤ Приход» (T04): Приход раньше на 10–20 минут допустим.
def test_late_finish_within_window_is_soft():
    truth: Truth = {
        "left_base": (25, 5, 20, 30),
        "started_work": (25, 5, 20, 40),
        "finished_work": (25, 5, 21, 40),
        "arrived_base": (25, 5, 21, 20),
    }
    inputs = make_inputs(truth, conf=0.995)
    result = decode(inputs, CTX)
    top = result.top
    assert top.rows["finished_work"].dt == datetime(2026, 5, 25, 21, 40)
    assert top.rows["arrived_base"].dt == datetime(2026, 5, 25, 21, 20)
    assert FLAG_LATE_FINISH in result.flags
    assert FLAG_TOP1_CHAIN_VIOLATION in result.flags
    # Штраф есть: та же запись без нарушения оценивается выше.
    ok = {**truth, "arrived_base": (25, 5, 21, 50)}
    forms = forms_from_values({row: dict(zip(PARTS, ok[row], strict=True)) for row in CHAIN}, 2026)
    model = DecoderModel.default()
    ev = _evidence(make_inputs(ok, conf=0.995), model.search.prob_floor)
    assert _score_scalar(forms, ev, CTX, model) > top.score


def test_late_finish_beyond_window_is_rejected():
    truth: Truth = {
        "left_base": (25, 5, 20, 30),
        "started_work": (25, 5, 20, 40),
        "finished_work": (25, 5, 22, 40),
        "arrived_base": (25, 5, 21, 20),
    }
    result = decode(make_inputs(truth, conf=0.995), CTX)
    for rec in result.records:
        late = rec.rows["finished_work"].dt - rec.rows["arrived_base"].dt
        assert late <= timedelta(minutes=30)
    # Остальная цепочка жёсткая: Начало раньше Выхода не допускается даже на минуту.
    inputs = make_inputs(BASE_TRUTH, conf=0.995)
    inputs["started_work.hour"] = [(9, 0.995), (10, 0.003)]
    inputs["started_work.minute"] = [(0, 0.995), (10, 0.003)]
    for rec in decode(inputs, CTX).records:
        assert rec.rows["started_work"].dt >= rec.rows["left_base"].dt


# Час вне top-k картинки: перебор всех часов и цепочка вытягивают допустимое значение.
def test_hour_outside_image_list_is_recovered_by_chain():
    truth: Truth = {
        "left_base": (18, 5, 16, 50),
        "started_work": (18, 5, 17, 30),
        "finished_work": (18, 5, 18, 30),
        "arrived_base": (18, 5, 18, 40),
    }
    inputs = make_inputs(truth, conf=0.995)
    # Все часы из списка картинки нарушают цепочку (Выход позже Начала 17:30).
    inputs["left_base.hour"] = [(18, 0.97), (19, 0.02)]
    result = decode(inputs, CTX)
    assert result.records
    assert result.top.rows["left_base"].hour == 16
    assert "left_base.hour" in result.overridden
    narrow = DecoderModel.default()
    narrow = replace(narrow, search=replace(narrow.search, all_hours=False))
    assert decode(inputs, CTX, narrow).top.rows["left_base"].hour != 16


# 5. Минуты 45: уверенная картинка — принимается, неуверенная — побеждает 40 или 50.
def test_minute_45_confident_is_kept():
    inputs = make_inputs(BASE_TRUTH)
    inputs["finished_work.minute"] = [(45, 0.9995), (40, 0.0003), (50, 0.0002)]
    result = decode(inputs, CTX)
    assert result.top.rows["finished_work"].minute == 45


def test_minute_45_uncertain_snaps_to_multiple_of_10():
    inputs = make_inputs(BASE_TRUTH)
    inputs["finished_work.minute"] = [(45, 0.6), (40, 0.25), (50, 0.15)]
    result = decode(inputs, CTX)
    assert result.top.rows["finished_work"].minute in (40, 50)
    assert "finished_work.minute" in result.overridden


# 6. Время заявки на +6 суток запись не отбрасывает, а только слегка штрафует.
def test_far_application_time_only_penalizes():
    inputs = make_inputs(BASE_TRUTH)
    near = decode(inputs, DecodeContext(year=2026, app_dt=datetime(2026, 3, 5, 6, 0)))
    far = decode(inputs, DecodeContext(year=2026, app_dt=datetime(2026, 2, 27, 6, 0)))
    none = decode(inputs, CTX)
    assert top_dt(far) == top_dt(near) == top_dt(none)
    assert far.top.score < near.top.score
    # Штраф конечный и умеренный по сравнению с 4 голосами за день.
    assert near.top.score - far.top.score < 15.0
    assert far.confidence > 0.5
    assert "app_far" in far.flags


def test_application_time_beyond_prior_range_is_not_a_window():
    inputs = make_inputs(BASE_TRUTH)
    result = decode(inputs, DecodeContext(year=2026, app_dt=datetime(2026, 1, 1, 6, 0)))
    assert top_dt(result)["left_base"] == datetime(2026, 3, 5, 9, 10)


# 7. Пропущенное подполе: результат есть, уверенность ниже.
def test_missing_subfield_lowers_confidence():
    inputs = make_inputs(BASE_TRUTH)
    full = decode(inputs, CTX)
    del inputs["started_work.minute"]
    inputs["arrived_base.hour"] = []
    partial = decode(inputs, CTX)
    assert partial.records
    assert set(partial.missing) == {"started_work.minute", "arrived_base.hour"}
    assert FLAG_MISSING in partial.flags
    assert partial.confidence < full.confidence
    assert top_dt(partial)["left_base"] == datetime(2026, 3, 5, 9, 10)


def test_all_days_missing_uses_application_window():
    inputs = make_inputs(BASE_TRUTH)
    for row in ROWS:
        del inputs[subfield(row, "day")]
    no_app = decode(inputs, CTX)
    assert not no_app.records
    assert FLAG_NO_CANDIDATES in no_app.flags
    with_app = decode(inputs, DecodeContext(year=2026, app_dt=datetime(2026, 3, 5, 6, 0)))
    assert with_app.records
    assert with_app.confidence < 0.9


def test_day_votes_fix_single_misread_day():
    inputs = make_inputs(BASE_TRUTH)
    inputs["arrived_base.day"] = [(8, 0.7), (5, 0.25), (6, 0.05)]
    result = decode(inputs, CTX)
    assert result.top.rows["arrived_base"].form_date == date(2026, 3, 5)
    assert "arrived_base.day" in result.overridden


# 8. Совпадение с полным перебором на малом входе.
def _random_inputs(rng: np.random.Generator) -> tuple[dict, DecodeContext]:
    left_day = int(rng.integers(1, 29))
    month = int(rng.choice([1, 2, 12]))
    if month == 12 and rng.random() < 0.5:
        left_day = 31
    hours = sorted(int(h) for h in rng.integers(6, 25, size=4))
    truth_days = [left_day] * 4
    if rng.random() < 0.3:
        truth_days[3] = left_day + 1 if left_day < 28 else 1
    inputs: dict[str, list[tuple[int, float]]] = {}
    for row, day, hour in zip(CHAIN, truth_days, hours, strict=True):
        vals = {
            "day": [day, int(rng.integers(1, 32))],
            "month": [month, month % 12 + 1],
            "hour": [hour, int(rng.integers(0, 25))],
            "minute": [int(rng.choice([0, 10, 20, 30, 40, 50, 45, 5])), int(rng.integers(0, 60))],
        }
        for part in PARTS:
            if part != "minute" and rng.random() < 0.08:
                continue  # пропуск подполя (без минут: иначе перебор 60 значений)
            a, b = vals[part]
            p = float(rng.uniform(0.4, 0.95))
            dist = [(a, p)] if a == b else [(a, p), (b, float(rng.uniform(0, 1 - p)))]
            inputs[subfield(row, part)] = dist
    app = None
    if rng.random() < 0.5:
        app = datetime(2026 if month != 12 else 2025, month, min(left_day, 28), 6, 0)
    return inputs, DecodeContext(year=2026 if month != 12 else 2025, app_dt=app)


def _brute_force(inputs, context, model):
    space = candidate_space(inputs, context, model)
    ev = _evidence(inputs, model.search.prob_floor)
    per_row = {
        row: [
            (h, m) for h in space.hours[row] for m in space.minutes[row] if not (h == 24 and m != 0)
        ]
        for row in CHAIN
    }
    scored = []
    late_max = model.priors.late_finish_max  # мягкое правило «Окончание ≤ Приход»

    for base in space.base_dates:
        for pattern in space.patterns:
            for combo in itertools.product(*(per_row[r] for r in CHAIN)):
                t = [o * 1440 + h * 60 + m for o, (h, m) in zip(pattern, combo, strict=True)]
                if not (t[0] <= t[1] <= t[2] and t[3] >= t[2] - late_max):
                    continue  # нарушение цепочки: скалярная оценка дала бы -inf
                forms = {
                    row: (base + timedelta(days=o), h, m)
                    for row, o, (h, m) in zip(CHAIN, pattern, combo, strict=True)
                }
                s = _score_scalar(forms, ev, context, model)
                if math.isfinite(s):
                    scored.append((s, forms))
    return scored


@pytest.mark.parametrize("seed", range(12))
def test_matches_brute_force(seed):
    rng = np.random.default_rng(seed)
    inputs, context = _random_inputs(rng)
    search = SearchParams(
        top_k={"day": 2, "month": 2, "hour": 2, "minute": 2},
        extra_minutes=(),
        top_n=12,
        min_prob=0.0,
        all_hours=False,  # перебор по 25 часам слишком долог; ДП от этого не меняется
        max_base_dates=6,  # заодно проверяем отсечение базовых дат
    )
    priors = DecoderPriors.default()
    if seed % 3 == 0:
        priors = replace(priors, max_day_offset=2)
    model = DecoderModel(priors=priors, search=search).with_weights(
        day=float(rng.uniform(0.5, 1.5)), prior_duration=float(rng.uniform(0.5, 1.5))
    )
    result = decode(inputs, context, model)
    scored = _brute_force(inputs, context, model)
    if not scored:
        assert not result.records
        assert FLAG_NO_CANDIDATES in result.flags
        return
    scores = np.asarray([s for s, _ in scored])
    m = scores.max()
    log_z = m + math.log(np.exp(scores - m).sum())
    assert result.log_z == pytest.approx(log_z, abs=1e-8)

    order = np.argsort(-scores, kind="stable")
    n = min(model.search.top_n, len(scored))
    brute_top = scores[order[:n]]
    got = np.asarray([r.score for r in result.records])
    assert len(got) == n
    np.testing.assert_allclose(got, brute_top, atol=1e-8)
    for rec in result.records:
        assert rec.score == pytest.approx(
            _score_scalar(rec.forms(), _evidence(inputs, 1e-6), context, model), abs=1e-8
        )
        assert rec.p == pytest.approx(math.exp(rec.score - log_z))
    if n > 1 and brute_top[0] - brute_top[1] > 1e-9:
        assert result.top.forms() == scored[order[0]][1]

    # Маргиналы строк.
    for row in CHAIN:
        agg: dict[tuple, float] = {}
        for s, forms in scored:
            agg[forms[row]] = agg.get(forms[row], 0.0) + math.exp(s - log_z)
        best = sorted(agg.values(), reverse=True)[:3]
        got_m = [opt.p for opt in result.marginals[row]]
        np.testing.assert_allclose(got_m, best[: len(got_m)], atol=1e-9)
        for opt in result.marginals[row]:
            assert agg[opt.value.as_form()] == pytest.approx(opt.p, abs=1e-9)


@pytest.mark.parametrize("seed", range(6))
def test_top1_does_not_depend_on_top_n(seed):
    """Отсечение в k-best точное и при `top_n = 1` (порог равен самому максимуму)."""
    rng = np.random.default_rng(100 + seed)
    inputs, context = _random_inputs(rng)
    model = DecoderModel.default()
    wide = decode(inputs, context, model)
    narrow = decode(inputs, context, replace(model, search=replace(model.search, top_n=1)))
    if not wide.records:
        assert not narrow.records
        return
    assert len(narrow.records) == 1
    assert narrow.top.score == pytest.approx(wide.top.score, abs=1e-9)
    assert narrow.top.forms() == wide.top.forms()
    assert narrow.log_z == pytest.approx(wide.log_z, abs=1e-12)


# 9. Время работы.
def test_speed_under_30ms():
    rng = np.random.default_rng(0)
    inputs = make_inputs(BASE_TRUTH, conf=0.7)
    # Размазанные распределения: полный набор top-k.
    for name, dist in list(inputs.items()):
        value = dist[0][0]
        noise = [
            (int(v), float(p))
            for v, p in zip(rng.integers(0, 12, 3), [0.1, 0.05, 0.03], strict=True)
        ]
        inputs[name] = [(value, 0.6), *[(v, p) for v, p in noise if v != value]]
    context = DecodeContext(year=2026, app_dt=datetime(2026, 3, 5, 6, 0))
    decode(inputs, context)
    times = []
    for _ in range(20):
        t0 = time_mod.perf_counter()
        decode(inputs, context)
        times.append(time_mod.perf_counter() - t0)
    assert float(np.median(times)) < 0.030


def test_model_json_roundtrip(tmp_path):
    model = DecoderModel.default().with_weights(minute=1.5, prior_app=0.5)
    path = tmp_path / "decoder_priors_v0.json"
    model.save(path)
    loaded = DecoderModel.load(path)
    assert loaded.weights == model.weights
    assert loaded.priors.max_day_offset == model.priors.max_day_offset
    inputs = make_inputs(BASE_TRUTH)
    assert decode(inputs, CTX, loaded).top.score == pytest.approx(
        decode(inputs, CTX, model).top.score
    )
    assert DecoderModel.load(tmp_path / "missing.json").meta["source"] == "defaults_from_plan"


def test_forms_from_values_and_candidate_membership():
    values = {row: dict(zip(PARTS, BASE_TRUTH[row], strict=True)) for row in CHAIN}
    forms = forms_from_values(values, 2026)
    assert forms["left_base"] == (date(2026, 3, 5), 9, 10)
    space = candidate_space(make_inputs(BASE_TRUTH), CTX)
    assert space.contains(forms)
