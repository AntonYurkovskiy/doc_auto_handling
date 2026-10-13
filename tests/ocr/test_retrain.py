"""Тесты чистой логики `ocr_lab.retrain` на синтетике (без обучения/торча/БД).

`decide_promotion` — критерий приёмки T28 («не хуже по покрытию при целевой точности»);
`default_version`/`current_version`/`VersionPaths`/`month`-утилиты — вспомогательная
логика версий цикла.
"""

from __future__ import annotations

from datetime import date

from ocr_lab.retrain import (
    PRECISION_TOLERANCE,
    VersionPaths,
    current_version,
    decide_promotion,
    default_version,
)


def _metrics(coverage: float | None, precision: float | None) -> dict:
    accept = {"coverage": coverage, "precision": precision}
    return {"splits": {"test": {"accept": {"t_val": accept}}}}


def test_promote_when_new_not_worse_on_both_metrics():
    new = _metrics(0.95, 0.996)
    current = _metrics(0.90, 0.995)

    decision = decide_promotion(new, current)

    assert decision.promote is True
    assert "не хуже" in decision.reason


def test_promote_when_metrics_tie_exactly():
    new = _metrics(0.90, 0.995)
    current = _metrics(0.90, 0.995)

    decision = decide_promotion(new, current)

    assert decision.promote is True


def test_reject_when_coverage_worse():
    new = _metrics(0.80, 0.999)
    current = _metrics(0.90, 0.995)

    decision = decide_promotion(new, current)

    assert decision.promote is False
    assert "покрытие хуже" in decision.reason


def test_reject_when_precision_worse_beyond_tolerance():
    new = _metrics(0.95, 0.90)
    current = _metrics(0.90, 0.995)

    decision = decide_promotion(new, current)

    assert decision.promote is False
    assert "точность хуже" in decision.reason


def test_promote_when_precision_drop_within_tolerance():
    new = _metrics(0.95, 0.995 - PRECISION_TOLERANCE / 2)
    current = _metrics(0.90, 0.995)

    decision = decide_promotion(new, current)

    assert decision.promote is True


def test_reject_when_threshold_missing_on_either_side():
    missing_new = _metrics(None, None)
    current = _metrics(0.90, 0.995)

    decision = decide_promotion(missing_new, current)

    assert decision.promote is False
    assert "нет порога" in decision.reason


def test_default_version_is_year_month_of_as_of():
    assert default_version(date(2026, 10, 13)) == "v202610"
    assert default_version(date(2026, 1, 1)) == "v202601"


def test_current_version_falls_back_to_v1_without_pointer_file(tmp_path):
    assert current_version(tmp_path / "missing.txt") == "v1"


def test_current_version_reads_pointer_file(tmp_path):
    pointer = tmp_path / "current_version.txt"
    pointer.write_text("v202609\n", encoding="utf-8")

    assert current_version(pointer) == "v202609"


def test_version_paths_layout(tmp_path):
    paths = VersionPaths("v202610", models_dir=tmp_path)

    assert paths.digits_dir == tmp_path / "digits_v202610"
    assert paths.number_dir == tmp_path / "number_v202610"
    assert paths.decoder_priors == tmp_path / "decoder_priors_v202610.json"
    assert paths.number_priors == tmp_path / "number_priors_v202610.json"
    assert paths.pair_priors == tmp_path / "pair_priors_v202610.json"
