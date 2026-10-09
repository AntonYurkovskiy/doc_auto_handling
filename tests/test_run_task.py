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


# --- канал Devin: те же модели, что у Claude, стартовые повторы, чтение кусками ---


def _devin_task(**fields: str) -> rt.Task:
    base = {
        "id": "T98",
        "file": "T98.md",
        "path": "docs/ocr_tasks/prompts/T98.md",
        "channel": "claude_first",
        "claude_model": "claude-opus-5-5",
        "claude_effort": "high",
        "devin_model": "swe-2-high",
        "review": "devin:gpt-5-6-sol-high",
    }
    return rt.Task(**{**base, **fields})


def _devin_plan(task: rt.Task, *argv: str) -> rt.RunPlan:
    return rt.build_plan(task, rt.parse_args([task.id, "--channel", "devin", *argv]))


@pytest.fixture(autouse=True)
def _no_model_map(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("OCR_MODEL_MAP", raising=False)


def test_devin_uses_same_model_and_effort_as_claude() -> None:
    plan = _devin_plan(_devin_task())
    assert plan.model == "claude-opus-5-5-high"
    assert plan.command[plan.command.index("--model") + 1] == "claude-opus-5-5-high"
    assert plan.command[plan.command.index("--permission-mode") + 1] == "dangerous"


def test_devin_model_map_applies_before_effort() -> None:
    task = _devin_task(claude_model="claude-sonnet-5", claude_effort="high")
    plan = _devin_plan(task, "--model-map", "claude-sonnet-5=claude-sonnet-5-5")
    assert plan.model == "claude-sonnet-5-5-high"


def test_devin_effort_override_and_default() -> None:
    assert _devin_plan(_devin_task(), "--effort", "xhigh").model == "claude-opus-5-5-xhigh"
    assert _devin_plan(_devin_task(claude_effort="")).model == "claude-opus-5-5-medium"


def test_devin_header_mode_and_explicit_id() -> None:
    assert _devin_plan(_devin_task(), "--devin-models", "header").model == "swe-2-high"
    explicit = _devin_plan(
        _devin_task(), "--devin-models", "header", "--devin-model", "claude-fable-5-1-high"
    )
    assert explicit.model == "claude-fable-5-1-high"
    assert explicit.effort == ""


def test_devin_fix_escalates_effort() -> None:
    plan = _devin_plan(_devin_task(claude_effort="high"), "--fix")
    assert plan.model == "claude-opus-5-5-xhigh"


def test_devin_review_model_is_verbatim() -> None:
    plan = _devin_plan(_devin_task(), "--review")
    assert plan.model == "gpt-5-6-sol-high"


def test_devin_takes_claude_only_unless_kept() -> None:
    task = _devin_task(channel="claude_only")
    assert _devin_plan(task).model == "claude-opus-5-5-high"
    with pytest.raises(SystemExit):
        _devin_plan(task, "--keep-claude-only")


def test_claude_plan_is_unchanged() -> None:
    plan = rt.build_plan(_devin_task(), rt.parse_args(["T98", "--channel", "claude"]))
    assert plan.channel == "claude"
    assert plan.model == "claude-opus-5-5"
    assert plan.command[plan.command.index("--effort") + 1] == "high"


@pytest.mark.parametrize(
    "text",
    [
        "Error: Unknown model: 'claude-sonnet-5-5-low'\nAvailable: \n",
        "Error: Unknown model: 'sonnet'\nAvailable:",
        'Error: session/set_config_option (model) failed: Resource not found: {\n  "uri": '
        '"Model not found: claude-sonnet-5-5-medium. Available models: "\n}',
    ],
)
def test_devin_empty_registry_is_start_failure(text: str) -> None:
    assert rt.devin_start_failure(text)


@pytest.mark.parametrize(
    "text",
    [
        "Error: Unknown model: 'swe-2-hgih'\nAvailable: swe-2-high, swe-2-medium\n",
        "PONG",
        "Error: Refusing to run in an untrusted workspace: repo",
    ],
)
def test_devin_real_errors_are_not_start_failures(text: str) -> None:
    assert not rt.devin_start_failure(text)


def test_devin_fatal_hints() -> None:
    assert "доверие" in (rt.devin_fatal_hint("Refusing to run in an untrusted workspace") or "")
    assert rt.devin_fatal_hint("PONG") is None


_REGISTRY_ERROR = "Error: Unknown model: 'm'\nAvailable: \n"


def _devin_flaky_session(counter: Path, fail_times: int) -> str:
    return (
        "import pathlib, sys\n"
        f"c = pathlib.Path({str(counter)!r})\n"
        "n = len(c.read_text()) if c.exists() else 0\n"
        "c.write_text('x' * (n + 1))\n"
        f"if n < {fail_times}:\n"
        f"    sys.stdout.write({_REGISTRY_ERROR!r}); sys.exit(1)\n"
        "print('PONG')\n"
    )


def test_devin_start_failure_is_retried_into_same_log(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(rt, "git", lambda *args: "")
    counter = tmp_path / "count"
    code = rt.run(
        _plan(_devin_flaky_session(counter, 3), channel="devin"),
        _task(),
        devin_start_retries=5,
        devin_start_retry_delay_s=0,
        log_dir=tmp_path,
    )
    assert code == 0
    assert len(counter.read_text()) == 4
    logs = [p for p in tmp_path.iterdir() if p.suffix == ".jsonl"]
    assert len(logs) == 1
    assert logs[0].read_text().strip() == "PONG"


def test_devin_start_retry_gives_up(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rt, "git", lambda *args: "")
    counter = tmp_path / "count"
    code = rt.run(
        _plan(_devin_flaky_session(counter, 99), channel="devin"),
        _task(),
        devin_start_retries=2,
        devin_start_retry_delay_s=0,
        log_dir=tmp_path,
    )
    assert code != 0
    assert len(counter.read_text()) == 3  # первый старт и два повтора


def test_devin_real_error_is_not_retried(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    monkeypatch.setattr(rt, "git", lambda *args: "")
    code = rt.run(
        _plan(
            "import sys; sys.stdout.write(\"Error: Unknown model: 'x'\nAvailable: a, b\n\")"
            "; sys.exit(1)",
            channel="devin",
        ),
        _task(),
        devin_start_retries=5,
        devin_start_retry_delay_s=0,
        log_dir=tmp_path,
    )
    assert code == 1
    assert len(list(tmp_path.iterdir())) == 1


def test_devin_output_without_newlines_keeps_watchdog_quiet(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Devin печатает текст без переводов строки: сторож не должен считать это тишиной."""
    monkeypatch.setattr(rt, "git", lambda *args: "")
    code = rt.run(
        _plan(
            "import sys, time\n"
            "for _ in range(10):\n"
            "    sys.stdout.write('x'); sys.stdout.flush(); time.sleep(0.5)\n",
            channel="devin",
        ),
        _task(),
        idle_timeout_min=0.05,  # 3 с, а вывод занимает 5 с
        log_dir=tmp_path,
    )
    assert code == 0
    assert next(tmp_path.glob("*.jsonl")).read_text() == "x" * 10


_SILENT_DEVIN = "import time; time.sleep(6)"


def test_devin_silent_but_files_changing_is_not_idle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """Обучение в фоне: текста нет, но логи и чекпоинты обновляются — сессия жива."""
    monkeypatch.setattr(rt, "git", lambda *args: "")
    code = rt.run(
        _plan(_SILENT_DEVIN, channel="devin"),
        _task(),
        idle_timeout_min=0.05,  # 3 с, тишина длится 6 с
        activity_probe=time.time,
        log_dir=tmp_path,
    )
    assert code == 0


def test_devin_silent_and_files_stale_is_idle(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(rt, "git", lambda *args: "")
    code = rt.run(
        _plan("import time; time.sleep(120)", channel="devin"),
        _task(),
        idle_timeout_min=0.05,
        activity_probe=lambda: 0.0,
        log_dir=tmp_path,
    )
    assert code == rt.IDLE_TIMEOUT_CODE


def test_latest_file_activity_skips_heavy_dirs(tmp_path: Path) -> None:
    (tmp_path / "crops").mkdir()
    (tmp_path / "crops" / "a.png").write_text("x")
    (tmp_path / "models").mkdir()
    old = tmp_path / "models" / "history.csv"
    old.write_text("x")
    import os

    os.utime(old, (1000.0, 1000.0))
    assert rt.latest_file_activity(tmp_path) == 1000.0
    assert rt.latest_file_activity(tmp_path / "nothing") == 0.0
