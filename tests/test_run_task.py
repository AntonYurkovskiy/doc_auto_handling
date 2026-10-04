"""Тесты раннера OCR-плана: выбор claude, сторож зависаний, повтор при 403."""

from __future__ import annotations

import importlib.util
import json
import sys
import time
from pathlib import Path

import pytest

_PATH = Path(__file__).resolve().parent.parent / "docs" / "ocr_tasks" / "run_task.py"
_spec = importlib.util.spec_from_file_location("ocr_run_task", _PATH)
assert _spec and _spec.loader
rt = importlib.util.module_from_spec(_spec)
sys.modules["ocr_run_task"] = rt
_spec.loader.exec_module(rt)


def _plan(code: str, channel: str = "claude") -> rt.RunPlan:
    return rt.RunPlan(
        channel=channel,
        executable=sys.executable,
        command=[sys.executable, "-u", "-c", code],
        prompt="",
        model="test",
        effort="",
        log_name="20260101-000000_T99_claude.jsonl",
    )


def _task() -> rt.Task:
    return rt.Task(id="T99", file="T99.md")


def test_find_claude_picks_newest_version(monkeypatch: pytest.MonkeyPatch) -> None:
    versions = {"old": (2, 1, 272), "new": (2, 1, 286), "mid": (2, 1, 284)}
    monkeypatch.setattr(rt, "_claude_candidates", lambda: ["old", "new", "mid"])
    monkeypatch.setattr(rt, "cli_version", lambda path: versions.get(path))
    rt.find_claude.cache_clear()
    try:
        assert rt.find_claude() == "new"
    finally:
        rt.find_claude.cache_clear()


def test_find_claude_ignores_broken_binaries(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "_claude_candidates", lambda: ["broken", "ok"])
    monkeypatch.setattr(rt, "cli_version", lambda path: (2, 1, 1) if path == "ok" else None)
    rt.find_claude.cache_clear()
    try:
        assert rt.find_claude() == "ok"
    finally:
        rt.find_claude.cache_clear()


def test_find_claude_falls_back_to_name(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(rt, "_claude_candidates", lambda: [])
    rt.find_claude.cache_clear()
    try:
        assert rt.find_claude() == "claude"
    finally:
        rt.find_claude.cache_clear()


def test_normal_run_returns_exit_code(tmp_path: Path) -> None:
    code = rt.run(_plan("print('hello'); raise SystemExit(3)"), _task(), log_dir=tmp_path)
    assert code == 3
    assert (tmp_path / "20260101-000000_T99_claude.jsonl").read_text().strip() == "hello"


def test_watchdog_stops_silent_session(tmp_path: Path) -> None:
    started = time.monotonic()
    code = rt.run(
        _plan("import time; print('start', flush=True); time.sleep(120)"),
        _task(),
        idle_timeout_min=0.05,  # 3 с
        log_dir=tmp_path,
    )
    assert code == rt.IDLE_TIMEOUT_CODE
    assert time.monotonic() - started < 30
    assert "start" in (tmp_path / "20260101-000000_T99_claude.jsonl").read_text()


def test_watchdog_off_with_zero_timeout(tmp_path: Path) -> None:
    code = rt.run(
        _plan("import time; time.sleep(4); print('done')"),
        _task(),
        idle_timeout_min=0,
        log_dir=tmp_path,
    )
    assert code == 0


def test_auth_failure_is_retried(tmp_path: Path) -> None:
    event = json.dumps(
        {"type": "assistant", "error": "authentication_failed", "message": {"content": []}}
    )
    code = rt.run(
        _plan(f"print({event!r}); raise SystemExit(1)"),
        _task(),
        api_retries=1,
        api_retry_delay_s=0,
        log_dir=tmp_path,
    )
    assert code == 1
    names = sorted(p.name for p in tmp_path.iterdir())
    assert names == [
        "20260101-000000_T99_claude.jsonl",
        "20260101-000000_T99_claude_retry1.jsonl",
    ]


def test_auth_retry_stops_after_success(tmp_path: Path) -> None:
    marker = tmp_path / "first_done"
    event = json.dumps(
        {"type": "assistant", "error": "authentication_failed", "message": {"content": []}}
    )
    code = rt.run(
        _plan(
            "import pathlib, sys\n"
            f"m = pathlib.Path({str(marker)!r})\n"
            "if m.exists():\n    print('ok'); sys.exit(0)\n"
            f"m.write_text('x'); print({event!r}); sys.exit(1)\n"
        ),
        _task(),
        api_retries=3,
        api_retry_delay_s=0,
        log_dir=tmp_path,
    )
    assert code == 0
    assert len([p for p in tmp_path.iterdir() if p.suffix == ".jsonl"]) == 2


def test_no_retry_without_auth_failure(tmp_path: Path) -> None:
    code = rt.run(
        _plan("raise SystemExit(5)"),
        _task(),
        api_retries=2,
        api_retry_delay_s=0,
        log_dir=tmp_path,
    )
    assert code == 5
    assert len(list(tmp_path.iterdir())) == 1


def _api_error_session(status_text: str) -> str:
    """Код сессии, которая печатает ошибку API так же, как `claude -p`, и выходит с 1."""
    assistant = json.dumps(
        {
            "type": "assistant",
            "error": "unknown",
            "message": {"content": [{"type": "text", "text": status_text}]},
        }
    )
    result = json.dumps(
        {
            "type": "result",
            "subtype": "success",
            "is_error": True,
            "terminal_reason": "api_error",
            "result": status_text,
        }
    )
    return f"print({assistant!r}); print({result!r}); raise SystemExit(1)"


@pytest.mark.parametrize(
    "text",
    [
        "API Error: 408 Request timeout",
        "API Error: 429 rate limited",
        "API Error: 503 Service Unavailable",
        "API Error: 529 Overloaded",
        "API Error: Connection reset by peer",
    ],
)
def test_transient_api_errors_are_retried(tmp_path: Path, text: str) -> None:
    rt.run(
        _plan(_api_error_session(text)),
        _task(),
        api_retries=1,
        api_retry_delay_s=0,
        log_dir=tmp_path,
    )
    assert len(list(tmp_path.iterdir())) == 2


@pytest.mark.parametrize(
    "text",
    [
        "API Error: 400 invalid request",
        "API Error: 413 request too large",
        "API Error: 404 model not found",
        "API Error: something unexpected",
    ],
)
def test_permanent_api_errors_are_not_retried(tmp_path: Path, text: str) -> None:
    rt.run(
        _plan(_api_error_session(text)),
        _task(),
        api_retries=2,
        api_retry_delay_s=0,
        log_dir=tmp_path,
    )
    assert len(list(tmp_path.iterdir())) == 1


def test_retry_delay_is_capped(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    sleeps: list[float] = []
    monkeypatch.setattr(rt.time, "sleep", sleeps.append)
    rt.run(
        _plan(_api_error_session("API Error: 408 Request timeout")),
        _task(),
        api_retries=4,
        api_retry_delay_s=1000,
        log_dir=tmp_path,
    )
    assert sleeps == [1000, 1800, 1800, 1800]


def test_cli_accepts_old_and_new_retry_flags() -> None:
    new = rt.parse_args(["--all", "--api-retries", "5", "--api-retry-delay", "7"])
    old = rt.parse_args(["--all", "--auth-retries", "5", "--auth-retry-delay", "7"])
    assert (new.api_retries, new.api_retry_delay) == (5, 7.0)
    assert (old.api_retries, old.api_retry_delay) == (5, 7.0)
