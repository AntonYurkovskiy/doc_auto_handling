"""Тесты T20: декодер номера ваучера и совместное декодирование пары k/p (синтетика)."""

from __future__ import annotations

import json
import math
from collections.abc import Mapping, Sequence
from dataclasses import replace
from datetime import date, datetime, timedelta

import pytest

from app.ocr.decoder import (
    FLAG_LATE_FINISH,
    FLAG_OVERRIDE,
    PARTS,
    ROWS,
    DecodeContext,
    DecoderModel,
    decode,
    forms_from_values,
    rerank_result,
    subfield,
)
from app.ocr.number_decoder import (
    FLAG_NUMBER_FAR,
    FLAG_NUMBER_FILE_MISMATCH,
    FLAG_NUMBER_OVERRIDE,
    FLAG_NUMBER_USED,
    NumberContext,
    NumberHistoryItem,
    NumberPriors,
    decode_number,
    filename_hint,
    history_context,
)
from app.ocr.pair_decoder import (
    FLAG_PAIR_CHANGED,
    FLAG_PAIR_CONFIRMED,
    FLAG_PAIR_DIFFERS,
    FLAG_PAIR_SAME,
    PairContext,
    PairPriors,
    decode_pair,
    record_times,
)
from ocr_lab import fit_number_pairs as fnp

Truth = dict[str, tuple[int, int, int, int]]


# ---------------------------------------------------------------------------
# Номер
# ---------------------------------------------------------------------------


def scorer_from(probs: Mapping[int, float], floor: float = 1e-9):
    """Картинка номера: `{номер: p}` → `log P` кандидатов (прочим — пол)."""

    def score(cands: Sequence[str]) -> dict[str, float]:
        return {c: math.log(probs.get(int(c), floor)) for c in cands}

    top = sorted(probs.items(), key=lambda kv: -kv[1])
    return top, score


def test_sequence_prior_pulls_expected_number_when_image_unsure():
    # Картинка колеблется между 248 и 243 (стилизованная «3»), ожидаемый номер — 243.
    top, score = scorer_from({248: 0.5, 243: 0.4, 246: 0.1})
    ctx = NumberContext(expected=243, used=frozenset(range(1, 243)))
    result = decode_number(top, score, ctx)
    assert result.value == 243
    assert FLAG_NUMBER_OVERRIDE in result.flags
    assert 0.5 < result.confidence <= 1.0
    probs = [p for _, p in result.candidates]
    assert probs == sorted(probs, reverse=True)
    # Без приора порядка выигрывает картинка.
    assert decode_number(top, score, NumberContext()).value == 248


def test_used_number_is_penalized_but_wins_with_confident_image():
    used = frozenset(range(1, 244))  # 243 уже подтверждён
    # Неуверенная картинка: 243 чуть выше 244 — штраф за занятый номер решает.
    top, score = scorer_from({243: 0.55, 244: 0.45})
    weak = decode_number(top, score, NumberContext(expected=244, used=used))
    assert weak.value == 244
    # Очень уверенная картинка: копия «(2)» с тем же номером всё же возможна.
    top, score = scorer_from({243: 1.0 - 1e-12}, floor=1e-14)
    strong = decode_number(top, score, NumberContext(expected=244, used=used))
    assert strong.value == 243
    assert FLAG_NUMBER_USED in strong.flags
    # Штраф конечен: занятый номер остаётся среди кандидатов с ненулевой вероятностью.
    assert 243 in dict(weak.candidates)
    assert dict(weak.candidates)[243] > 0


def test_filename_prior_strong_and_copy_waives_used_penalty():
    top, score = scorer_from({247: 0.6, 241: 0.4})
    ctx = NumberContext(expected=245, filename="241p.pdf", tug_code="p")
    assert decode_number(top, score, ctx).value == 241
    # Код буксира в имени чужой — приор слабый, но порядок + картинка всё равно решают.
    hint = filename_hint("241k.pdf")
    assert hint is not None and hint.tug_code == "k"
    # Копия «(2)» уже подтверждённого номера не штрафуется за занятость.
    used = frozenset(range(1, 246))
    top, score = scorer_from({240: 0.5, 245: 0.5})
    res = decode_number(
        top, score, NumberContext(expected=246, used=used, filename="240p(2).pdf", tug_code="p")
    )
    assert res.value == 240
    assert FLAG_NUMBER_USED not in res.flags
    # Имя файла спорит с уверенной картинкой и порядком — флаг расхождения.
    top, score = scorer_from({250: 1.0 - 1e-12}, floor=1e-15)
    res = decode_number(top, score, NumberContext(expected=250, filename="25p.pdf", tug_code="p"))
    assert res.value == 250
    assert FLAG_NUMBER_FILE_MISMATCH in res.flags


def test_long_pause_allows_number_far_ahead():
    priors = NumberPriors.default()
    top, score = scorer_from({178: 0.9999, 148: 1e-4}, floor=1e-12)
    short = decode_number(top, score, NumberContext(expected=148, pause_days=0.5), priors)
    assert short.value == 148
    # После паузы в 27 дней номера могли уйти вперёд на ~30 — верит картинке.
    long = decode_number(top, score, NumberContext(expected=148, pause_days=27.0), priors)
    assert long.value == 178
    assert FLAG_NUMBER_FAR not in long.flags
    assert priors.span(27.0) >= 30
    assert priors.span(1.0) == 0


def test_number_without_crop_uses_sequence_and_filename():
    ctx = NumberContext(expected=12, filename="12k.pdf", tug_code="k")
    res = decode_number((), lambda c: {}, ctx)
    assert res.value == 12
    assert res.confidence > 0.99


# ---------------------------------------------------------------------------
# Утечка: приор номера не видит истинный номер текущего ваучера
# ---------------------------------------------------------------------------


def _history(current_number: int) -> list[NumberHistoryItem]:
    t0 = datetime(2026, 5, 1, 8, 0)
    items = [
        NumberHistoryItem("p", 2026, t0 + timedelta(days=i), 100 + i, f"2026_{100 + i}p")
        for i in range(5)
    ]
    # Текущий ваучер и более поздний — в той же истории.
    items.append(NumberHistoryItem("p", 2026, t0 + timedelta(days=5), current_number, "cur"))
    items.append(NumberHistoryItem("p", 2026, t0 + timedelta(days=6), 106, "2026_106p"))
    # Другой буксир и другой год не влияют.
    items.append(NumberHistoryItem("k", 2026, t0, 300, "2026_300k"))
    items.append(NumberHistoryItem("p", 2025, t0, 330, "2025_330p"))
    return items


def test_number_prior_does_not_see_current_truth():
    before = datetime(2026, 5, 6, 8, 0)
    ctx_a = history_context(_history(105), tug_code="p", year=2026, before=before, exclude="cur")
    ctx_b = history_context(_history(999), tug_code="p", year=2026, before=before, exclude="cur")
    assert ctx_a == ctx_b
    assert ctx_a.expected == 105  # max(100..104) + 1, без текущего и без более позднего
    assert ctx_a.used == frozenset(range(100, 105))
    assert ctx_a.pause_days == pytest.approx(1.0)
    # Даже если текущий ваучер в истории «раньше», исключение по ключу его убирает.
    later = datetime(2026, 5, 9)
    ctx_c = history_context(_history(150), tug_code="p", year=2026, before=later, exclude="cur")
    assert 150 not in ctx_c.used
    assert ctx_c.expected == 107
    # И итог декодера номера одинаков при любом истинном номере текущего ваучера.
    top, score = scorer_from({105: 0.3, 108: 0.7})
    res = [
        decode_number(top, score, NumberContext(expected=c.expected, used=c.used))
        for c in (ctx_a, ctx_b)
    ]
    assert res[0].candidates == res[1].candidates


def test_lab_number_context_excludes_current_voucher():
    rows = []
    for i, n in enumerate((1, 2, 3, 4)):
        rows.append(
            {
                "scan_id": f"2026_{n}p",
                "tug_code": "p",
                "year": "2026",
                "voucher_number": str(n),
                "voucher_file": f"{n}p.pdf",
                "left_base_dt": f"2026-01-0{i + 2}T10:00:00",
            }
        )
    history = fnp.history_from_manifest(rows)
    current = dict(rows[2])
    before = datetime(2026, 1, 4, 10, 0)  # ровно дата текущего — сам он «не раньше»
    ctx = fnp.number_context(current, history, before, use_filename=False)
    assert ctx.expected == 3 and ctx.used == frozenset({1, 2})
    assert ctx.filename is None
    # Подменили истинный номер текущего в манифесте — контекст тот же.
    rows[2] = {**rows[2], "voucher_number": "77", "voucher_file": "77p.pdf"}
    ctx2 = fnp.number_context(current, fnp.history_from_manifest(rows), before, use_filename=False)
    assert ctx2 == ctx
    # Даже с датой «после всех» текущий ваучер не попадает в историю.
    late = fnp.number_context(current, history, datetime(2026, 2, 1), use_filename=False)
    assert 3 not in late.used and late.expected == 5


# ---------------------------------------------------------------------------
# Пары
# ---------------------------------------------------------------------------


def make_inputs(
    truth: Truth,
    conf: float = 0.95,
    overrides: Mapping[str, list[tuple[int, float]]] | None = None,
) -> dict[str, list[tuple[int, float]]]:
    inputs: dict[str, list[tuple[int, float]]] = {}
    alt = {"day": 1, "month": 1, "hour": 1, "minute": 10}
    hi = {"day": 31, "month": 12, "hour": 23, "minute": 59}
    for row, values in truth.items():
        for part, value in zip(PARTS, values, strict=True):
            other = value + alt[part] if value + alt[part] <= hi[part] else value - alt[part]
            inputs[subfield(row, part)] = [(value, conf), (other, (1 - conf) / 2)]
    inputs.update(overrides or {})
    return inputs


def forms_of(truth: Truth, year: int = 2026):
    values = {
        row: dict(zip(("day", "month", "hour", "minute"), v, strict=True))
        for row, v in truth.items()
    }
    return forms_from_values(values, year)


TRUTH: Truth = {
    "left_base": (5, 3, 9, 10),
    "started_work": (5, 3, 10, 0),
    "finished_work": (5, 3, 12, 30),
    "arrived_base": (5, 3, 13, 20),
}
CTX = DecodeContext(year=2026)
MODEL = DecoderModel.default()


def test_pair_fixes_uncertain_minute_from_partner():
    # У ваучера A минуты Окончания читаются плохо: 40 чуть выше истинных 30.
    a_inputs = make_inputs(TRUTH, overrides={"finished_work.minute": [(40, 0.55), (30, 0.43)]})
    b_inputs = make_inputs(TRUTH, conf=0.97)
    alone = decode(a_inputs, CTX, MODEL)
    assert alone.top is not None and alone.top.rows["finished_work"].minute == 40
    res = decode_pair(a_inputs, b_inputs, PairContext(a=CTX, b=CTX), MODEL, PairPriors.default())
    assert res.b is not None
    assert res.a.top is not None and res.a.top.forms() == forms_of(TRUTH)
    assert res.b.top is not None and res.b.top.forms() == forms_of(TRUTH)
    assert res.p_same > 0.9
    assert FLAG_PAIR_SAME in res.flags
    assert FLAG_PAIR_CHANGED in res.a.flags and FLAG_PAIR_CHANGED not in res.b.flags
    assert res.a.confidence > alone.confidence
    # Контракт: вероятности по убыванию, не больше 1 в сумме, флаги пересчитаны.
    probs = [r.p for r in res.a.records]
    assert probs == sorted(probs, reverse=True) and sum(probs) <= 1 + 1e-9
    assert FLAG_OVERRIDE in res.a.flags and "finished_work.minute" in res.a.overridden


def test_pair_with_really_different_times_is_not_glued():
    # Пионер пришёл в базу на 2 часа позже: Окончание и Приход расходятся.
    b_truth = dict(TRUTH, finished_work=(5, 3, 14, 30), arrived_base=(5, 3, 15, 30))
    a_inputs = make_inputs(TRUTH)
    b_inputs = make_inputs(b_truth)
    res = decode_pair(a_inputs, b_inputs, PairContext(a=CTX), MODEL, PairPriors.default())
    assert res.b is not None
    assert res.a.top is not None and res.a.top.forms() == forms_of(TRUTH)
    assert res.b.top is not None and res.b.top.forms() == forms_of(b_truth)
    assert res.p_same < 0.05
    assert FLAG_PAIR_DIFFERS in res.flags
    # Одна различающаяся минута при уверенных картинках тоже не склеивается.
    c_truth = dict(TRUTH, left_base=(5, 3, 9, 0))
    res2 = decode_pair(
        make_inputs(TRUTH, conf=0.99), make_inputs(c_truth, conf=0.99), PairContext(a=CTX), MODEL
    )
    assert res2.b is not None and res2.b.top is not None
    assert res2.b.top.forms() == forms_of(c_truth)
    assert res2.a.top is not None and res2.a.top.forms() == forms_of(TRUTH)


def test_confirmed_partner_is_strong_prior():
    a_inputs = make_inputs(TRUTH, overrides={"finished_work.minute": [(40, 0.55), (30, 0.43)]})
    ctx = PairContext(a=CTX, b_confirmed=forms_of(TRUTH))
    res = decode_pair(a_inputs, None, ctx, MODEL)
    assert res.b is None
    assert res.a.top is not None and res.a.top.forms() == forms_of(TRUTH)
    assert FLAG_PAIR_CONFIRMED in res.flags and FLAG_PAIR_CONFIRMED in res.a.flags
    assert res.p_same > 0.9
    # Подтверждённая запись, которой нет в top-N ядра, добавляется в кандидаты.
    far = dict(TRUTH, arrived_base=(5, 3, 17, 50))
    a2 = make_inputs(TRUTH, overrides={"arrived_base.hour": [(13, 0.6), (17, 0.4)]})
    res2 = decode_pair(a2, None, PairContext(a=CTX, b_confirmed=forms_of(far)), MODEL)
    assert any(r.forms() == forms_of(far) for r in res2.a.records)
    with pytest.raises(ValueError):
        decode_pair(a_inputs, None, PairContext(a=CTX), MODEL)


def test_pair_link_values_and_rest_mass():
    pair = PairPriors.default()
    t = record_times(forms_of(TRUTH))
    assert pair.link(t, t) == pytest.approx(pair.same_logp)
    shifted = tuple(x + timedelta(minutes=10) if i == 0 else x for i, x in enumerate(t))
    assert pair.link(t, shifted) < pair.same_logp
    # Отрицательный и положительный сдвиг симметричны.
    back = tuple(x - timedelta(minutes=10) if i == 0 else x for i, x in enumerate(t))
    assert pair.link(t, shifted) == pytest.approx(pair.link(t, back))
    # Неуверенная сторона: масса «прочих» записей не даёт уверенности стать единицей.
    vague = make_inputs(TRUTH, conf=0.3)
    res = decode_pair(vague, vague, PairContext(a=CTX), MODEL, pair)
    assert res.a.confidence < 0.999


def test_rerank_result_recomputes_record_flags():
    late = dict(TRUTH, arrived_base=(5, 3, 12, 20))  # Приход раньше Окончания на 10 мин
    inputs = make_inputs(TRUTH)
    core = decode(inputs, CTX, MODEL)
    assert FLAG_LATE_FINISH not in core.flags
    from app.ocr.decoder import record_from_forms

    rec = record_from_forms(forms_of(late), score=0.0, p=0.7)
    out = rerank_result(core, [rec], inputs, CTX, extra_flags=["x"])
    assert out.confidence == pytest.approx(0.7) and out.margin == pytest.approx(0.7)
    assert FLAG_LATE_FINISH in out.flags and "x" in out.flags
    assert out.overridden == ("arrived_base.hour",)
    assert out.marginals == core.marginals
    empty = rerank_result(core, [], inputs, CTX)
    assert empty.records == () and "no_candidates" in empty.flags


# ---------------------------------------------------------------------------
# Сериализация и подбор приоров
# ---------------------------------------------------------------------------


def test_priors_json_roundtrip(tmp_path):
    number = NumberPriors.default()
    number.save(tmp_path / "n.json")
    loaded = NumberPriors.from_json(json.loads((tmp_path / "n.json").read_text("utf-8")))
    assert loaded == replace(number, meta=loaded.meta)
    pair = PairPriors.default()
    pair.save(tmp_path / "p.json")
    loaded_p = PairPriors.from_json(json.loads((tmp_path / "p.json").read_text("utf-8")))
    t = record_times(forms_of(TRUTH))
    other = (t[0], t[1], t[2] + timedelta(minutes=120), t[3] + timedelta(minutes=120))
    assert loaded_p.link(t, other) == pytest.approx(pair.link(t, other))
    assert loaded_p.rest_link == pytest.approx(pair.rest_link)


def _row(sid: str, tug: str, number: int, left: datetime, pair_id: str = "") -> dict[str, str]:
    row = {
        "scan_id": sid,
        "tug_code": tug,
        "year": str(left.year),
        "voucher_number": str(number),
        "voucher_file": f"{number}{tug}.pdf",
        "left_base_dt": left.isoformat(),
        "pair_id": pair_id,
        "split": "train",
        "app_dt": "",
        "work_type": "",
    }
    times = {
        "left_base": left,
        "started_work": left + timedelta(minutes=20),
        "finished_work": left + timedelta(minutes=80),
        "arrived_base": left + timedelta(minutes=100),
    }
    for r, dt in times.items():
        row[f"{r}_day"], row[f"{r}_month"], row[f"{r}_year"] = str(dt.day), str(dt.month), ""
        row[f"{r}_hour"], row[f"{r}_minute"] = str(dt.hour), str(dt.minute)
    return row


def test_fit_number_and_pair_priors_on_synthetic_manifest():
    rows = []
    t0 = datetime(2026, 1, 2, 8, 0)
    for i in range(40):
        left = t0 + timedelta(days=i)
        rows.append(_row(f"2026_{i + 1}k", "k", i + 1, left, pair_id=f"P{i:04d}"))
        rows.append(_row(f"2026_{i + 1}p", "p", i + 1, left, pair_id=f"P{i:04d}"))
    # У трёх пар Пионер вернулся на 2 часа позже.
    for i in range(3):
        row = rows[2 * i + 1]
        for r in ("finished_work", "arrived_base"):
            row[f"{r}_hour"] = str(int(row[f"{r}_hour"]) + 2)
    number, nstats = fnp.fit_number_priors(rows, fnp.history_from_manifest(rows))
    assert nstats.n == 80  # у первого ваучера года ожидаемый номер — 1
    assert nstats.offsets == {0: 80}
    assert number.short_value(0) > math.log(0.85)
    assert number.short_value(0) > number.short_value(1) > number.outside_logp
    assert nstats.used_hits == 0 and nstats.file_strong_mismatch == 0
    assert number.rate == pytest.approx(1.0)
    pair, pstats = fnp.fit_pair_priors(rows)
    assert pstats.n == 40 and pstats.same == 37
    assert pstats.patterns == {"0011": 3}
    assert math.exp(pair.same_logp) == pytest.approx((37 + 10 * 0.67) / 50)
    assert pair.pattern_logp[(0, 0, 1, 1)] == max(pair.pattern_logp.values())


def test_history_from_manifest_uses_filename_number_and_tug():
    rows = [
        _row("2025_12k", "k", 12, datetime(2025, 1, 14, 12, 50)),
        _row("2025_15p", "p", 15, datetime(2025, 1, 18, 18, 10)),
    ]
    rows[0]["voucher_number"] = "123"  # ошибка истины в манифесте
    rows[1]["tug_code"] = "k"  # буксир в манифесте не тот, что в имени файла
    hist = fnp.history_from_manifest(rows)
    assert [(h.tug_code, h.number) for h in hist] == [("k", 12), ("p", 15)]
    assert hist[0].dt.date() == date(2025, 1, 14)
    assert ROWS  # имена строк общие с ядром
