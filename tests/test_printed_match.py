"""Тесты сверки прочитанных печатных строк со списками известных значений (T30)."""

from __future__ import annotations

import pytest

from app.services.printed_match import (
    MIN_SCORE,
    PrintedMatch,
    match_agent,
    match_vessel,
    match_work_type,
)

VESSELS = ["AMBAL", "SORMOVSKIY-53", "GT VELA", "ROSEBURG", "RIG", "MELATI 7", "POLA ANASTASIA"]


# --- судно ---------------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("m/v AMBAL", "AMBAL"),
        ("mv AMBA|", "AMBAL"),
        ("m/v SORMOVSKIY 53 |", "SORMOVSKIY-53"),
        ("m/v R0SEBURG .", "ROSEBURG"),  # ноль вместо «O»
        ("POLA ANASTAS1A", "POLA ANASTASIA"),  # единица вместо «I»
        ("МЕLATI 7", "MELATI 7"),  # кириллические «М» и «Е» вместо латинских
        ("м/в AMBAL", "AMBAL"),  # русская приставка
    ],
)
def test_match_vessel_tolerates_ocr_noise(text: str, expected: str) -> None:
    found = match_vessel(text, VESSELS)

    assert found is not None
    assert found.value == expected
    assert MIN_SCORE <= found.score <= 1.0


@pytest.mark.parametrize(
    "text",
    [
        "",
        None,
        "FRIGATE",  # «RIG» — короткое имя, внутри чужого слова не ищется
        "Tug «Kommunar» / «Буксир»",  # строка буксира вместо судна
        "QWXZ",
        "POLA",  # общее начало двух имён, кандидат не однозначен
    ],
)
def test_match_vessel_rejects_garbage_and_ambiguity(text: str | None) -> None:
    assert match_vessel(text, VESSELS + ["POLA MARIA"]) is None


def test_match_vessel_returns_none_for_empty_candidates() -> None:
    assert match_vessel("m/v AMBAL", []) is None


def test_match_vessel_does_not_pick_a_shorter_name_inside_garbage() -> None:
    # Найденное внутри длинной нечитаемой строки короткое имя не должно проходить сверку.
    assert match_vessel("ILXQZWKRTUVBNMPOQ AMBAL ZXCVBN", VESSELS) is None


def test_match_vessel_returns_known_spelling_not_ocr_text() -> None:
    found = match_vessel("  m/v   Sormovskiy-53  ", VESSELS)

    assert found == PrintedMatch("SORMOVSKIY-53", found.score if found else 0)


# --- агент ---------------------------------------------------------------------------------

AGENTS = ["Транс-Агро", "Терминал", "ТБК", "Светловский судоремонтный завод"]


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Транс-Агро", "Транс-Агро"),
        ("ООО «Транс-Агро»", "Транс-Агро"),
        ("ТРАНС АГРО", "Транс-Агро"),
        ("Агент: Терминал", "Терминал"),
        ("ТБК", "ТБК"),
        ("Светловскии судоремонтныи завод", "Светловский судоремонтный завод"),
    ],
)
def test_match_agent_finds_known_agent(text: str, expected: str) -> None:
    found = match_agent(text, AGENTS)

    assert found is not None and found.value == expected


@pytest.mark.parametrize("text", ["", None, "Название судна:", "me: m/v QWXZ -� 106�"])
def test_match_agent_rejects_garbage(text: str | None) -> None:
    assert match_agent(text, AGENTS) is None


# --- вид работ -----------------------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "expected"),
    [
        ("Швартовка", "Швартовка"),
        ("Отшвартовка судна к причалу № 8 ТСС", "Отшвартовка"),
        ("Отшвартовка + Сопровождение", "Отшвартовка + Сопровождение"),
        ("Перестановка из ПД-8", "Перестановка"),
        ("0тшвартовка", "Отшвартовка"),  # ноль вместо «О»
        ("Шварт0вка", "Швартовка"),
        ("Отшвартовка + Сопровожденне", "Отшвартовка + Сопровождение"),
    ],
)
def test_match_work_type_finds_contract_service(text: str, expected: str) -> None:
    found = match_work_type(text)

    assert found is not None and found.value == expected


@pytest.mark.parametrize("text", ["", None, "абвгд", "Tug Kommunar", "Date 2026 Time"])
def test_match_work_type_rejects_garbage(text: str | None) -> None:
    assert match_work_type(text) is None
