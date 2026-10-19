"""Сверка прочитанной печатной строки ваучера со списками известных значений.

Tesseract по печатным строкам бланка (судно, агент, вид работ) читает с ошибками в буквах и
иногда попадает не на ту строку. В форму такой текст не пишем: из прочитанной строки берём
**ближайшее известное значение** (агента из известных агентов, вид работ из стандартных, судно
из судов заявок и истории). Похожего нет — значения нет, поле остаётся пустым, а прочитанный
текст служит оператору подсказкой (решение В1/В2, `docs/ocr_tasks/open_questions.md`).

Только стандартная библиотека: `difflib` вместо внешней библиотеки нечёткого поиска.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterable
from dataclasses import dataclass
from difflib import SequenceMatcher

from app.services.calculation import match_contract_services

#: Минимальная схожесть прочитанной строки с известным значением. Подобрано перебором на 147
#: сканах val+test (`ocr_lab.eval_printed_fields --sweep`, судно среди заявок ±14 дней): верно
#: найдено / ошибочно — 0,65: 109/2; 0,7: 100/1; 0,8: 99/1; 0,85: 92/1. Взято 0,7: на
#: порог ниже растёт число ошибочных, выше — теряются верные.
MIN_SCORE = 0.7
#: Строже для списка «все известные суда»: без окна по датам похожих имён много.
STRICT_SCORE = 0.8
#: Насколько лучший кандидат должен опережать следующий (другое значение).
MIN_MARGIN = 0.1
#: Названия короче этой длины сверяем только целиком, не как кусок строки: «RIG» не должен
#: находиться внутри «FRIGATE».
SHORT_NAME_LEN = 5

#: Кириллические двойники латинских букв: Tesseract с `rus+eng` часто путает алфавиты.
_CYR_TO_LAT = str.maketrans("АВЕКМНОРСТХУавекмнорстху", "ABEKMHOPCTXYabekmhopctxy")
#: Похожие по начертанию символы сводим к одному классу.
_LOOKALIKE = str.maketrans({"0": "O", "1": "I", "L": "I", "5": "S", "8": "B", "2": "Z"})
#: Приставки типа судна перед названием: «m/v», «М/В», «т/х», «MT».
_VESSEL_PREFIX = re.compile(
    r"^\s*(?:m\s*/\s*[vt]|m[vt]\b|м\s*/\s*[вт]|т\s*/\s*х|мв\b|мт\b)\s*", re.I
)
#: Организационно-правовые формы и служебные слова перед названием агента.
_AGENT_NOISE = re.compile(r"\b(?:ооо|оао|зао|ао|пао|агент|agent)\b", re.I)


@dataclass(frozen=True)
class PrintedMatch:
    """Известное значение, найденное в прочитанной строке, и схожесть (0..1)."""

    value: str
    score: float


def _fold_latin(text: str) -> str:
    """Название судна -> `ABC123`: верхний регистр, кириллические двойники в латиницу.

    Кириллические буквы без латинских двойников («Д», «Ж») остаются как есть.
    """
    text = _VESSEL_PREFIX.sub("", text)
    text = text.translate(_CYR_TO_LAT).upper()
    return re.sub(r"[^A-Z0-9А-ЯЁ]", "", text).translate(_LOOKALIKE)


def _fold_cyrillic(text: str) -> str:
    """Название агента -> строчные кириллические буквы и цифры без пробелов и знаков."""
    text = _AGENT_NOISE.sub(" ", text.lower().replace("ё", "е"))
    return re.sub(r"[^a-zа-я0-9]", "", text)


def _ratio(a: str, b: str) -> float:
    return SequenceMatcher(None, a, b, autojunk=False).ratio()


def _partial_ratio(needle: str, haystack: str) -> float:
    """Схожесть `needle` с лучшим окном `haystack` такой же длины (±1 символ)."""
    if not needle or not haystack:
        return 0.0
    if len(haystack) <= len(needle) + 1:
        return _ratio(needle, haystack)
    best = 0.0
    n = len(needle)
    for width in (n - 1, n, n + 1):
        if width < 1:
            continue
        for start in range(0, len(haystack) - width + 1):
            best = max(best, _ratio(needle, haystack[start : start + width]))
            if best >= 1.0:
                return 1.0
    return best


def _share_chars(a: str, b: str) -> bool:
    """Дешёвый отсев: у похожих строк общих символов много."""
    common = sum(min(a.count(ch), b.count(ch)) for ch in set(a))
    return common >= 0.6 * min(len(a), len(b))


def _best_match(
    text: str,
    candidates: Iterable[str],
    fold: Callable[[str], str],
    *,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
    penalize_long_text: bool = False,
) -> PrintedMatch | None:
    folded_text = fold(text)
    if len(folded_text) < 2:
        return None
    scored: dict[str, float] = {}
    for candidate in candidates:
        value = " ".join(str(candidate).split())
        if not value or value in scored:
            continue
        folded = fold(value)
        if len(folded) < 2 or not _share_chars(folded, folded_text):
            continue
        if len(folded) < SHORT_NAME_LEN:
            score = _ratio(folded, folded_text)
        else:
            # Совпадение куска строки засчитывается тем слабее, чем меньшую часть строки
            # занимает имя: короткое имя внутри длинной нечитаемой строки находится случайно.
            coverage = min(1.0, len(folded) / len(folded_text)) if penalize_long_text else 1.0
            score = max(
                _ratio(folded, folded_text),
                _partial_ratio(folded, folded_text) * (0.5 + 0.5 * coverage),
            )
        scored[value] = score
    if not scored:
        return None
    ranked = sorted(scored.items(), key=lambda item: item[1], reverse=True)
    best_value, best_score = ranked[0]
    if best_score < min_score:
        return None
    # Два разных значения почти одинаково похожи — не угадываем.
    for other_value, other_score in ranked[1:]:
        if fold(other_value) != fold(best_value):
            if best_score - other_score < min_margin:
                return None
            break
    return PrintedMatch(best_value, round(best_score, 4))


def match_vessel(
    text: str | None,
    candidates: Iterable[str],
    *,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
) -> PrintedMatch | None:
    """Судно из известных (заявки, история); `None` — похожего нет."""
    if not text:
        return None
    return _best_match(
        text,
        candidates,
        _fold_latin,
        min_score=min_score,
        min_margin=min_margin,
        penalize_long_text=True,
    )


def match_agent(
    text: str | None,
    candidates: Iterable[str],
    *,
    min_score: float = MIN_SCORE,
    min_margin: float = MIN_MARGIN,
) -> PrintedMatch | None:
    """Агент из известных (группы тарифов, заявки, история); `None` — похожего нет."""
    if not text:
        return None
    return _best_match(text, candidates, _fold_cyrillic, min_score=min_score, min_margin=min_margin)


#: Основы названий услуг для нечёткого поиска слов (после точного regex из договора).
_SERVICE_STEMS: tuple[tuple[str, str], ...] = (
    ("отшвартовка", "Отшвартовка"),
    ("перешвартовка", "Перестановка"),
    ("перестановка", "Перестановка"),
    ("швартовка", "Швартовка"),
    ("сопровождение", "Сопровождение"),
    ("буксировка", "Буксировка"),
    ("околка", "Околка льда"),
)
_WORD = re.compile(r"[а-яёa-z]{5,}", re.I)


def _fuzzy_services(text: str) -> str | None:
    """Услуги по словам с опечатками: «0тшвартовка» -> Отшвартовка (порог 0,82)."""
    found: list[str] = []
    for word in _WORD.findall(text.lower().replace("ё", "е").replace("0", "о")):
        best_service, best = None, 0.0
        for stem, service in _SERVICE_STEMS:
            score = _ratio(stem, word)
            if score > best:
                best_service, best = service, score
        if best_service is not None and best >= 0.82 and best_service not in found:
            found.append(best_service)
    return " + ".join(found) if found else None


def match_work_type(text: str | None) -> PrintedMatch | None:
    """Вид работ: услуги договора по regex, затем по словам с опечатками; `None` — не нашли.

    Возвращается написание из договора (`CONTRACT_SERVICES`), а не из истории: в истории
    встречаются «ШВАРТОВКА» и «Швартовка», и разное написание не должно расползаться по форме.
    """
    if not text:
        return None
    exact = match_contract_services(text)
    if exact is not None:
        return PrintedMatch(exact, 1.0)
    fuzzy = _fuzzy_services(text)
    if fuzzy is not None:
        return PrintedMatch(fuzzy, 0.85)
    return None
