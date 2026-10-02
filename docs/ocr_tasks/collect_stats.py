"""Статистика запусков задач OCR-плана по логам `data/ocr/logs/*.jsonl`.

Для калибровки оценок (время, ходы, $ по прайсу, доля окна и недели) на будущие задачи.
Читает логи `run_task.py` (stream-json claude -p) и пишет `docs/ocr_tasks/calibration/runs.csv`.
Часы ПК переводились вручную, поэтому длительность берётся из события `result`
(`duration_ms`); если сессию оборвали, стоит оценка по времени файла (`wall_est`).

    python docs/ocr_tasks/collect_stats.py            # таблица + CSV
    python docs/ocr_tasks/collect_stats.py --task T07 # только одна задача
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from datetime import datetime
from pathlib import Path
from typing import Any

TASKS_DIR = Path(__file__).resolve().parent
REPO_ROOT = TASKS_DIR.parent.parent
LOG_DIR = REPO_ROOT / "data" / "ocr" / "logs"
OUT_CSV = TASKS_DIR / "calibration" / "runs.csv"
_NAME_RE = re.compile(r"^(\d{8})-(\d{6})_(T\d+|R|C\d)(_review|_fix)?_(claude|devin)\.jsonl$")

FIELDS = [
    "log",
    "task",
    "channel",
    "start",
    "wall_est_min",
    "duration_min",
    "segments",
    "results",
    "assistant_msgs",
    "turns",
    "tool_calls",
    "cost_usd",
    "outcome",
    "models",
    "input_tokens",
    "output_tokens",
    "cache_read_tokens",
    "cache_write_tokens",
    "five_hour_first",
    "five_hour_last",
    "seven_day_first",
    "seven_day_last",
    "api_retries",
    "auth_failed",
    "denials",
]


def _windows(event: dict[str, Any]) -> dict[str, Any]:
    info = event.get("rate_limit_info", event)
    return info.get("unifiedWindows") or {}


def analyse(path: Path) -> dict[str, Any] | None:
    match = _NAME_RE.match(path.name)
    if match is None:
        return None
    day, clock, task, suffix, channel = match.groups()
    start = datetime.strptime(day + clock, "%Y%m%d%H%M%S")
    row: dict[str, Any] = {
        "log": path.name,
        "task": task + (suffix or ""),
        "channel": channel,
        "start": start.isoformat(timespec="minutes"),
        "wall_est_min": round(
            (datetime.fromtimestamp(path.stat().st_mtime) - start).total_seconds() / 60, 1
        ),
    }
    if channel != "claude":
        row["outcome"] = "devin: stream-json нет, статистики нет"
        return row

    results: list[dict[str, Any]] = []
    inits = 0
    assistants_after_result = 0
    five: list[float] = []
    seven: list[float] = []
    tools: Counter[str] = Counter()
    assistants = retries = 0
    auth_failed = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            continue
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            inits += 1
        elif kind == "result":
            results.append(event)
            assistants_after_result = 0
        elif kind == "rate_limit_event":
            windows = _windows(event)
            if "five_hour" in windows and "utilization" in windows["five_hour"]:
                five.append(windows["five_hour"]["utilization"])
            if "seven_day" in windows and "utilization" in windows["seven_day"]:
                seven.append(windows["seven_day"]["utilization"])
        elif kind == "system" and event.get("subtype") == "api_retry":
            retries += 1
        elif kind == "assistant":
            assistants += 1
            assistants_after_result += 1
            if event.get("error") == "authentication_failed":
                auth_failed = True
            for block in event.get("message", {}).get("content", []) or []:
                if block.get("type") == "tool_use":
                    tools[block.get("name", "?")] += 1

    row.update(
        segments=inits,
        results=len(results),
        assistant_msgs=assistants,
        tool_calls=sum(tools.values()),
        five_hour_first=five[0] if five else "",
        five_hour_last=five[-1] if five else "",
        seven_day_first=seven[0] if seven else "",
        seven_day_last=seven[-1] if seven else "",
        api_retries=retries,
        auth_failed=int(auth_failed),
    )
    if not results:
        row["turns"] = assistants
        row["outcome"] = "оборвано (нет result)" + (
            ", 403 authentication_failed" if auth_failed else ""
        )
        return row
    # Сессия с фоновыми задачами выдаёт несколько result: суммируем сегменты.
    usage: dict[str, dict[str, Any]] = {}
    for item in results:
        for model, data in (item.get("modelUsage") or {}).items():
            total = usage.setdefault(model, {})
            for key, value in data.items():
                if isinstance(value, int | float):
                    total[key] = total.get(key, 0) + value
    last = results[-1]
    outcome = "ok" if not last.get("is_error") else str(last.get("subtype"))
    if assistants_after_result:
        outcome += f", затем оборвано ({assistants_after_result} сообщений после result)"
    row.update(
        duration_min=round(sum((r.get("duration_ms") or 0) for r in results) / 60000, 1),
        turns=sum((r.get("num_turns") or 0) for r in results),
        cost_usd=round(sum((r.get("total_cost_usd") or 0) for r in results), 2),
        outcome=outcome,
        models="+".join(sorted(usage)),
        input_tokens=int(sum(m.get("inputTokens", 0) for m in usage.values())),
        output_tokens=int(sum(m.get("outputTokens", 0) for m in usage.values())),
        cache_read_tokens=int(sum(m.get("cacheReadInputTokens", 0) for m in usage.values())),
        cache_write_tokens=int(sum(m.get("cacheCreationInputTokens", 0) for m in usage.values())),
        denials=sum(len(r.get("permission_denials") or []) for r in results),
    )
    return row


def main() -> None:
    parser = argparse.ArgumentParser(description="Статистика запусков OCR-задач")
    parser.add_argument("--task", help="например T07")
    args = parser.parse_args()
    rows = [row for path in sorted(LOG_DIR.glob("*.jsonl")) if (row := analyse(path))]
    if args.task:
        rows = [row for row in rows if row["task"].startswith(args.task.upper())]
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDS, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(
            f"{row['task']:<4} {row['start']} {row['channel']:<6} "
            f"мин={row.get('duration_min', row['wall_est_min'])!s:<6} "
            f"сегм={row.get('segments', '')!s:<2} сообщ={row.get('assistant_msgs', '')!s:<4} "
            f"ходов={row.get('turns', '')!s:<4} $={row.get('cost_usd', '')!s:<5} "
            f"5ч {row.get('five_hour_first', '')}→{row.get('five_hour_last', '')} "
            f"нед {row.get('seven_day_first', '')}→{row.get('seven_day_last', '')}  "
            f"{row['outcome']}"
        )
    print(f"\nCSV: {OUT_CSV.relative_to(REPO_ROOT)}")


if __name__ == "__main__":
    main()
