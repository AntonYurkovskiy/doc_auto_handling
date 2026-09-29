"""Единый формат предсказаний распознавателей: JSONL, одна строка на скан.

Пример строки:

    {"scan_id": "2026_243k", "source": "cnn_v0",
     "fields": {"left_base.hour": [[9, 0.93], [8, 0.05]], "voucher_number": [[243, 0.9]]},
     "records": [{"left_base": "2026-07-02T09:10", "arrived_base": "...",
                  "started_work": "...", "finished_work": "...",
                  "hour24": ["arrived_base"], "voucher_number": 243, "p": 0.97}],
     "confidence": 0.97, "margin": 0.95}

- `fields` — top-k по подполю: пары «целое значение, вероятность» по убыванию вероятности.
  Вероятность может быть `null`, если распознаватель её не даёт.
- `records` — top-N целых записей (их даёт декодер). Время строки — ISO без секунд;
  `24:00` дня D хранится как `00:00` дня D+1, а сама строка перечисляется в `hour24`.
- `confidence` и `margin` — только у декодера.
"""

from __future__ import annotations

import json
import math
from collections.abc import Iterable, Mapping
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Any

ROWS: tuple[str, ...] = ("left_base", "arrived_base", "started_work", "finished_work")
PARTS: tuple[str, ...] = ("day", "month", "hour", "minute")
VOUCHER_NUMBER = "voucher_number"
SUBFIELDS: tuple[str, ...] = tuple(f"{r}.{p}" for r in ROWS for p in PARTS) + (VOUCHER_NUMBER,)

# Допуск на округление при проверке порядка и диапазона вероятностей.
_EPS = 1e-9


class PredictionError(ValueError):
    """Строка предсказаний не соответствует формату."""


Candidate = tuple[int, float | None]


@dataclass
class RecordCandidate:
    """Целая запись ваучера: время четырёх строк и номер."""

    rows: dict[str, datetime | None]
    hour24: tuple[str, ...] = ()
    voucher_number: int | None = None
    p: float | None = None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {}
        for r in ROWS:
            dt = self.rows.get(r)
            out[r] = dt.isoformat(timespec="minutes") if dt is not None else None
        out["hour24"] = list(self.hour24)
        out["voucher_number"] = self.voucher_number
        out["p"] = self.p
        return out


@dataclass
class ScanPrediction:
    """Предсказание одного распознавателя для одного скана."""

    scan_id: str
    source: str
    fields: dict[str, list[Candidate]] = field(default_factory=dict)
    records: list[RecordCandidate] = field(default_factory=list)
    confidence: float | None = None
    margin: float | None = None

    def top1(self, subfield: str) -> int | None:
        cands = self.fields.get(subfield)
        return cands[0][0] if cands else None

    def to_json(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "scan_id": self.scan_id,
            "source": self.source,
            "fields": {k: [[v, p] for v, p in c] for k, c in self.fields.items()},
        }
        if self.records:
            out["records"] = [r.to_json() for r in self.records]
        if self.confidence is not None:
            out["confidence"] = self.confidence
        if self.margin is not None:
            out["margin"] = self.margin
        return out


def _check_prob(value: Any, where: str) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool) or not isinstance(value, int | float):
        raise PredictionError(f"{where}: вероятность должна быть числом")
    p = float(value)
    if math.isnan(p) or p < -_EPS or p > 1 + _EPS:
        raise PredictionError(f"{where}: вероятность вне [0, 1]")
    return min(max(p, 0.0), 1.0)


def _check_int(value: Any, where: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise PredictionError(f"{where}: значение должно быть целым")
    return value


def _parse_dt(value: Any, where: str) -> datetime | None:
    if value is None or value == "":
        return None
    if not isinstance(value, str):
        raise PredictionError(f"{where}: время должно быть строкой ISO")
    try:
        return datetime.fromisoformat(value)
    except ValueError as exc:
        raise PredictionError(f"{where}: не ISO-время") from exc


def _parse_candidates(raw: Any, where: str) -> list[Candidate]:
    if not isinstance(raw, list):
        raise PredictionError(f"{where}: ожидается список кандидатов")
    out: list[Candidate] = []
    for i, item in enumerate(raw):
        if not isinstance(item, list | tuple) or len(item) not in (1, 2):
            raise PredictionError(f"{where}[{i}]: кандидат — [значение, вероятность]")
        value = _check_int(item[0], f"{where}[{i}]")
        prob = _check_prob(item[1] if len(item) == 2 else None, f"{where}[{i}]")
        out.append((value, prob))
    probs = [p for _, p in out]
    if all(p is not None for p in probs):
        for a, b in zip(probs, probs[1:], strict=False):
            assert a is not None and b is not None
            if b > a + _EPS:
                raise PredictionError(f"{where}: кандидаты не по убыванию вероятности")
        if sum(p for p in probs if p is not None) > 1 + 1e-6:
            raise PredictionError(f"{where}: сумма вероятностей больше 1")
    values = [v for v, _ in out]
    if len(set(values)) != len(values):
        raise PredictionError(f"{where}: повтор значения среди кандидатов")
    return out


def _parse_record(raw: Any, where: str) -> RecordCandidate:
    if not isinstance(raw, Mapping):
        raise PredictionError(f"{where}: запись должна быть объектом")
    unknown = set(raw) - set(ROWS) - {"hour24", VOUCHER_NUMBER, "p"}
    if unknown:
        raise PredictionError(f"{where}: неизвестные ключи {sorted(unknown)}")
    rows = {r: _parse_dt(raw.get(r), f"{where}.{r}") for r in ROWS}
    hour24_raw = raw.get("hour24") or []
    if not isinstance(hour24_raw, list) or any(h not in ROWS for h in hour24_raw):
        raise PredictionError(f"{where}.hour24: ожидается список строк бланка")
    for r in hour24_raw:
        dt = rows[r]
        if dt is not None and (dt.hour, dt.minute) != (0, 0):
            raise PredictionError(f"{where}.hour24: {r} должен храниться как 00:00")
    number = raw.get(VOUCHER_NUMBER)
    return RecordCandidate(
        rows=rows,
        hour24=tuple(hour24_raw),
        voucher_number=None if number is None else _check_int(number, f"{where}.number"),
        p=_check_prob(raw.get("p"), f"{where}.p"),
    )


def parse_prediction(obj: Any) -> ScanPrediction:
    """Проверить и разобрать один объект JSONL. Ошибки — `PredictionError`."""
    if not isinstance(obj, Mapping):
        raise PredictionError("строка должна быть объектом")
    scan_id = obj.get("scan_id")
    if not isinstance(scan_id, str) or not scan_id:
        raise PredictionError("нет scan_id")
    source = obj.get("source")
    if not isinstance(source, str) or not source:
        raise PredictionError(f"{scan_id}: нет source")
    raw_fields = obj.get("fields") or {}
    if not isinstance(raw_fields, Mapping):
        raise PredictionError(f"{scan_id}: fields должен быть объектом")
    unknown = set(raw_fields) - set(SUBFIELDS)
    if unknown:
        raise PredictionError(f"{scan_id}: неизвестные подполя {sorted(unknown)}")
    fields = {k: _parse_candidates(v, f"{scan_id}.{k}") for k, v in raw_fields.items()}
    raw_records = obj.get("records") or []
    if not isinstance(raw_records, list):
        raise PredictionError(f"{scan_id}: records должен быть списком")
    records = [_parse_record(r, f"{scan_id}.records[{i}]") for i, r in enumerate(raw_records)]
    return ScanPrediction(
        scan_id=scan_id,
        source=source,
        fields=fields,
        records=records,
        confidence=_check_prob(obj.get("confidence"), f"{scan_id}.confidence"),
        margin=_check_prob(obj.get("margin"), f"{scan_id}.margin"),
    )


def read_predictions(path: Path) -> list[ScanPrediction]:
    """Прочитать JSONL с проверкой формата и уникальности `scan_id`."""
    out: list[ScanPrediction] = []
    seen: set[str] = set()
    with path.open(encoding="utf-8") as fh:
        for lineno, line in enumerate(fh, start=1):
            if not line.strip():
                continue
            try:
                pred = parse_prediction(json.loads(line))
            except (json.JSONDecodeError, PredictionError) as exc:
                raise PredictionError(f"{path.name}:{lineno}: {exc}") from exc
            if pred.scan_id in seen:
                raise PredictionError(f"{path.name}:{lineno}: повтор scan_id {pred.scan_id}")
            seen.add(pred.scan_id)
            out.append(pred)
    return out


def write_predictions(path: Path, predictions: Iterable[ScanPrediction]) -> int:
    """Записать предсказания в JSONL. Возвращает число строк."""
    path.parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with path.open("w", encoding="utf-8") as fh:
        for pred in predictions:
            fh.write(json.dumps(pred.to_json(), ensure_ascii=False) + "\n")
            n += 1
    return n


def row_datetime(year: int, month: int, day: int, hour: int, minute: int) -> datetime | None:
    """Время строки по подполям. `24:00` дня D → `00:00` дня D+1; невозможная дата → None."""
    if hour == 24:
        if minute != 0:
            return None
        base = row_datetime(year, month, day, 0, 0)
        return None if base is None else base + timedelta(days=1)
    try:
        return datetime(year, month, day, hour, minute)
    except ValueError:
        return None


def record_from_fields(pred: ScanPrediction, years: Mapping[str, int | None]) -> RecordCandidate:
    """Собрать запись из top-1 подполей.

    Год строки берётся из `years` (из манифеста): распознаватели его не читают.
    Переход через полночь не угадывается: день берётся как распознан.
    """
    rows: dict[str, datetime | None] = {}
    hour24: list[str] = []
    for r in ROWS:
        parts = [pred.top1(f"{r}.{p}") for p in PARTS]
        year = years.get(r)
        if year is None or any(v is None for v in parts):
            rows[r] = None
            continue
        day, month, hour, minute = (int(v) for v in parts if v is not None)
        rows[r] = row_datetime(year, month, day, hour, minute)
        if hour == 24 and rows[r] is not None:
            hour24.append(r)
    return RecordCandidate(
        rows=rows, hour24=tuple(hour24), voucher_number=pred.top1(VOUCHER_NUMBER)
    )


def best_record(pred: ScanPrediction, years: Mapping[str, int | None]) -> RecordCandidate:
    """Лучшая запись: первая из `records`, иначе собранная из top-1 подполей.

    Если в записи декодера нет номера, он берётся из top-1 подполя `voucher_number`.
    """
    if pred.records:
        rec = pred.records[0]
        if rec.voucher_number is None and pred.top1(VOUCHER_NUMBER) is not None:
            return RecordCandidate(
                rows=rec.rows,
                hour24=rec.hour24,
                voucher_number=pred.top1(VOUCHER_NUMBER),
                p=rec.p,
            )
        return rec
    return record_from_fields(pred, years)
