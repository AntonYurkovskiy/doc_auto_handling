"""Запуск атомарной задачи OCR-плана в `claude -p` или `devin -p`.

Модель, effort, канал и лимит ходов берутся из шапки файла задачи
(`docs/ocr_tasks/prompts/T*.md`). Правила — из справочника моделей проекта:
только полные ID моделей, Fable в `claude -p` на Pro запрещён (оплата кредитами),
ревью — моделью другого вендора, эскалация — сначала effort, потом модель.

Примеры:
    python docs/ocr_tasks/run_task.py --status
    python docs/ocr_tasks/run_task.py T02 --dry-run
    python docs/ocr_tasks/run_task.py T02
    python docs/ocr_tasks/run_task.py T03 --channel devin
    python docs/ocr_tasks/run_task.py T16 --effort xhigh
    python docs/ocr_tasks/run_task.py T06 --review
    python docs/ocr_tasks/run_task.py T06 --fix
    python docs/ocr_tasks/run_task.py --summarize data/ocr/logs/<файл>.jsonl
    python docs/ocr_tasks/run_task.py --export-json > tasks.json
    python docs/ocr_tasks/run_task.py T16 --cloud-prompt       # текст для облачной сессии
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Any

TASKS_DIR = Path(__file__).resolve().parent
PROMPTS_DIR = TASKS_DIR / "prompts"
CLOUD_DIR = TASKS_DIR / "cloud"
REPO_ROOT = TASKS_DIR.parent.parent
LOG_DIR = REPO_ROOT / "data" / "ocr" / "logs"
PROGRESS_MD = TASKS_DIR / "PROGRESS.md"
DECISIONS_MD = TASKS_DIR / "DECISIONS.md"
REVIEWS_DIR = TASKS_DIR / "reviews"

# Минимальная версия Claude Code CLI для модели (справочник, § 14.2).
MIN_CLI_FOR_MODEL = {"claude-opus-5-5": (2, 1, 280)}
# На Pro Fable оплачивается только кредитами, а в -p списание идёт без вопроса (§ 1.3).
CREDIT_ONLY_PREFIXES = ("claude-fable",)
# Запасная модель при перегрузке API (529), не при лимите подписки (§ 18.2).
FALLBACK_FOR_OPUS = "claude-sonnet-5"
# Потолок сессии в $ по прайсу API для размера блока (model-routing.json v3).
BUDGET_BY_SIZE = {"S": 3.0, "M": 5.0, "L": 8.0, "XL": 10.0}
# Опасные git-операции запрещены в любой задаче: push, переписывание истории, удаление.
DISALLOWED_TOOLS = ",".join(
    [
        "Bash(git push *)",
        "Bash(git push)",
        "Bash(git reset --hard *)",
        "Bash(git clean *)",
        "Bash(git rebase *)",
        "Bash(git checkout -- *)",
        "Bash(git switch *)",
        "Bash(rm -rf *)",
    ]
)
# Цепочка эскалации в Claude Code: сначала effort, потом модель (§ 8.3).
ESCALATION = {
    ("claude-sonnet-5", "low"): ("claude-sonnet-5", "medium"),
    ("claude-sonnet-5", "medium"): ("claude-sonnet-5", "high"),
    ("claude-sonnet-5", "high"): ("claude-sonnet-5", "xhigh"),
    ("claude-sonnet-5", "xhigh"): ("claude-opus-5-5", "high"),
    ("claude-opus-5-5", "low"): ("claude-opus-5-5", "medium"),
    ("claude-opus-5-5", "medium"): ("claude-opus-5-5", "high"),
    ("claude-opus-5-5", "high"): ("claude-opus-5-5", "xhigh"),
    ("claude-opus-5-5", "xhigh"): ("claude-opus-5-5", "max"),
}
ESCALATION_AFTER_MAX = "Devin: --channel devin --devin-model claude-fable-5-1-high"

TASK_PROMPT = (
    "Execute task {id} of the OCR plan. First read docs/ocr_tasks/_common.md, then "
    "{path}. Follow them exactly and work autonomously: nobody can "
    "answer questions during this run. The task files are in Russian; write code comments, "
    "docs, commit messages and the PROGRESS.md entry in Russian."
)
CLOUD_PROMPT = (
    "Execute task {id} of the OCR plan in this cloud session. Read docs/ocr_tasks/_common.md, "
    "then docs/ocr_tasks/cloud/_cloud_overrides.md, which overrides it for the cloud, then "
    "{path}. Work autonomously and write code comments, docs, commit messages and the "
    "PROGRESS.md entry in Russian."
)
FIX_SUFFIX = (
    " This is a repeat run after a failed review: read docs/ocr_tasks/reviews/{id}.md first "
    "and fix every blocking finding without widening the task scope."
)
REVIEW_PROMPT = (
    "Review task {id} of the OCR plan. Read docs/ocr_tasks/_common.md, then "
    "docs/ocr_tasks/prompts/R_review.md with TASK set to {id}, then the task file "
    "{path}. Do not modify code: write docs/ocr_tasks/reviews/{id}.md "
    "in Russian and commit only that file."
)

_FRONTMATTER_RE = re.compile(r"\A---\s*\n(.*?)\n---\s*\n", re.DOTALL)
_TASK_HEADER_RE = re.compile(r"^##\s+([TC]\d{1,2})\b")
_STATUS_RE = re.compile(r"^-\s*Статус:\s*(\S+)")
_CHECKPOINT_RE = re.compile(r"^\s*-\s*\[[xXхХ]\]\s*(H\d)\b")


@dataclass
class Task:
    """Шапка файла задачи."""

    id: str
    file: str
    path: str = ""
    title: str = ""
    stage: str = ""
    depends_on: list[str] = field(default_factory=list)
    type: str = ""
    complexity: str = ""
    risk: str = ""
    needs_vision: bool = False
    needs_web: bool = False
    host: str = "local"
    size: str = "M"
    channel: str = "claude_first"
    claude_model: str = ""
    claude_effort: str = ""
    devin_model: str = ""
    review: str = ""
    human_checkpoint: str = ""
    max_turns: int = 150
    optional: bool = False


def _as_bool(value: str) -> bool:
    return value.strip().lower() in {"true", "yes", "1", "да"}


def _as_list(value: str) -> list[str]:
    value = value.strip().strip("[]")
    if value in {"", "—", "-"}:
        return []
    return [part.strip() for part in value.split(",") if part.strip()]


def parse_frontmatter(path: Path) -> dict[str, str]:
    """Плоская YAML-шапка `key: value` между строками `---`."""
    match = _FRONTMATTER_RE.match(path.read_text(encoding="utf-8"))
    if match is None:
        raise ValueError(f"В {path.name} нет шапки между строками ---")
    result: dict[str, str] = {}
    for line in match.group(1).splitlines():
        if ":" in line and not line.lstrip().startswith("#"):
            key, value = line.split(":", 1)
            result[key.strip()] = value.strip()
    return result


def load_tasks() -> dict[str, Task]:
    """Задачи из prompts/ (T00…T29, R) и cloud/ (C1, C2), ключ — id."""
    tasks: dict[str, Task] = {}
    paths = sorted(PROMPTS_DIR.glob("*.md")) + sorted(CLOUD_DIR.glob("C*.md"))
    for path in paths:
        meta = parse_frontmatter(path)
        task_id = meta.get("id") or path.stem.split("_", 1)[0]
        tasks[task_id] = Task(
            id=task_id,
            file=path.name,
            path=path.relative_to(REPO_ROOT).as_posix(),
            title=meta.get("title", ""),
            stage=meta.get("stage", ""),
            depends_on=_as_list(meta.get("depends_on", "")),
            type=meta.get("type", ""),
            complexity=meta.get("complexity", ""),
            risk=meta.get("risk", ""),
            needs_vision=_as_bool(meta.get("needs_vision", "false")),
            needs_web=_as_bool(meta.get("needs_web", "false")),
            host=meta.get("host", "local"),
            size=meta.get("size", "M"),
            channel=meta.get("channel", "claude_first"),
            claude_model=meta.get("claude_model", "").strip("—"),
            claude_effort=meta.get("claude_effort", "").strip("—"),
            devin_model=meta.get("devin_model", "").strip("—"),
            review=meta.get("review", "").strip("—"),
            human_checkpoint=meta.get("human_checkpoint", "").strip("—"),
            max_turns=int(meta.get("max_turns", "150") or 150),
            optional=_as_bool(meta.get("optional", "false")),
        )
    return tasks


def task_statuses() -> dict[str, str]:
    """Статусы задач из журнала: последняя строка `- Статус:` в разделе `## Txx`."""
    statuses: dict[str, str] = {}
    if not PROGRESS_MD.exists():
        return statuses
    current: str | None = None
    text = re.sub(r"<!--.*?-->", "", PROGRESS_MD.read_text(encoding="utf-8"), flags=re.DOTALL)
    for line in text.splitlines():
        header = _TASK_HEADER_RE.match(line)
        if header:
            current = header.group(1)
            continue
        status = _STATUS_RE.match(line)
        if status and current:
            statuses[current] = status.group(1).strip(".,;").lower()
    return statuses


def done_checkpoints() -> set[str]:
    """Ручные точки, отмеченные в DECISIONS.md как `- [x] H1`."""
    if not DECISIONS_MD.exists():
        return set()
    return {
        match.group(1)
        for line in DECISIONS_MD.read_text(encoding="utf-8").splitlines()
        if (match := _CHECKPOINT_RE.match(line))
    }


def unmet_dependencies(task: Task) -> list[str]:
    statuses = task_statuses()
    checkpoints = done_checkpoints()
    unmet = []
    for dep in task.depends_on:
        if dep.startswith("H"):
            if dep not in checkpoints:
                unmet.append(dep)
        elif statuses.get(dep) != "готово":
            unmet.append(dep)
    return unmet


def print_status(tasks: dict[str, Task]) -> None:
    statuses = task_statuses()
    rows = []
    for task in tasks.values():
        if task.id == "R":
            continue
        state = statuses.get(task.id)
        if state is None:
            unmet = unmet_dependencies(task)
            state = "ждёт: " + ", ".join(unmet) if unmet else "можно запускать"
        if task.optional:
            state += " (опц.)"
        claude = f"{task.claude_model}/{task.claude_effort}" if task.claude_model else "—"
        devin = task.devin_model or "—"
        rows.append((task.id, task.title[:48], task.channel, claude, devin, state))
    widths = [max(len(str(row[i])) for row in rows) for i in range(len(rows[0]))]
    for row in rows:
        print("  ".join(str(cell).ljust(width) for cell, width in zip(row, widths, strict=True)))


def cli_version(executable: str) -> tuple[int, ...] | None:
    try:
        output = subprocess.run(
            [executable, "--version"], capture_output=True, text=True, timeout=60
        ).stdout
    except (OSError, subprocess.SubprocessError):
        return None
    match = re.search(r"(\d+)\.(\d+)\.(\d+)", output)
    return tuple(int(part) for part in match.groups()) if match else None


def git(*args: str) -> str:
    return subprocess.run(
        ["git", *args], cwd=REPO_ROOT, capture_output=True, text=True, check=False
    ).stdout.strip()


def apply_model_map(model: str, mapping: str | None) -> str:
    """`OCR_MODEL_MAP="claude-sonnet-5=claude-sonnet-5-5"` — подмена без правки шапок."""
    for pair in (mapping or "").split(","):
        if "=" in pair:
            source, target = (part.strip() for part in pair.split("=", 1))
            if model == source:
                return target
    return model


@dataclass
class RunPlan:
    channel: str
    executable: str
    command: list[str]
    prompt: str
    model: str
    effort: str
    log_name: str


def build_plan(task: Task, args: argparse.Namespace) -> RunPlan:
    if args.review:
        channel, model, effort = _review_target(task, args)
        prompt = REVIEW_PROMPT.format(id=task.id, path=task.path)
        max_turns = 60
    else:
        channel = args.channel or ("devin" if task.channel.startswith("devin") else "claude")
        model = task.claude_model if channel == "claude" else task.devin_model
        effort = task.claude_effort if channel == "claude" else ""
        prompt = TASK_PROMPT.format(id=task.id, path=task.path)
        max_turns = task.max_turns
        if args.fix:
            prompt += FIX_SUFFIX.format(id=task.id)
            if channel == "claude":
                model, effort = ESCALATION.get((model, effort), (model, effort))
    if channel == "claude" and task.channel == "devin_only" and not args.review:
        print(f"[i] {task.id}: по справочнику это задача для Devin (бесплатная SWE-2).")
    if channel == "devin" and task.channel == "claude_only":
        raise SystemExit(f"{task.id}: только Claude (картинки или машина пользователя).")
    if args.devin_model and channel == "devin":
        model = args.devin_model
    if args.model:
        model = args.model
    if args.effort:
        effort = args.effort
    model = apply_model_map(model, args.model_map or os.environ.get("OCR_MODEL_MAP"))
    if not model:
        raise SystemExit(f"{task.id}: для канала {channel} в шапке не задана модель.")

    executable = shutil.which(channel) or channel
    suffix = "_review" if args.review else "_fix" if args.fix else ""
    stamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    log_name = f"{stamp}_{task.id}{suffix}_{channel}.jsonl"

    if channel == "claude":
        command = [
            executable,
            "-p",
            prompt,
            "--model",
            model,
            "--permission-mode",
            args.permission_mode,
            "--output-format",
            "stream-json",
            "--verbose",
            "--max-turns",
            str(args.max_turns or max_turns),
            "-n",
            f"OCR {task.id}{suffix}",
        ]
        if effort and not model.startswith("claude-haiku"):
            command += ["--effort", effort]
        if model.startswith("claude-opus"):
            command += ["--fallback-model", FALLBACK_FOR_OPUS]
        budget = _budget(task, args)
        if budget:
            command += ["--max-budget-usd", f"{budget:g}"]
        command += ["--disallowedTools", DISALLOWED_TOOLS]
    else:
        command = [executable, "-p", prompt, "--model", model, "--permission-mode", "auto"]
    return RunPlan(channel, executable, command, prompt, model, effort, log_name)


def _review_target(task: Task, args: argparse.Namespace) -> tuple[str, str, str]:
    """`devin:gpt-5-6-sol-high` или `claude:claude-sonnet-5/medium` → канал, модель, effort."""
    target = args.reviewer or task.review
    if not target or target == "human":
        raise SystemExit(f"{task.id}: ревью делает человек (review: {task.review or '—'}).")
    channel, _, rest = target.partition(":")
    if channel not in {"claude", "devin"}:
        raise SystemExit(f"Непонятный ревьюер: {target}")
    model, _, effort = rest.partition("/")
    if channel == "claude" and not effort:
        effort = "medium"
    return channel, model, effort


def _budget(task: Task, args: argparse.Namespace) -> float | None:
    if args.budget is None:
        return None
    if args.budget == "auto":
        return BUDGET_BY_SIZE.get(task.size, BUDGET_BY_SIZE["M"])
    return float(args.budget)


def preflight(task: Task, plan: RunPlan, args: argparse.Namespace) -> list[str]:
    problems: list[str] = []
    if shutil.which(plan.channel) is None:
        problems.append(f"не найден CLI `{plan.channel}` в PATH")
    branch = git("rev-parse", "--abbrev-ref", "HEAD")
    if branch in {"main", "master"} and not args.allow_main:
        problems.append(f"текущая ветка {branch}: создайте рабочую ветку или --allow-main")
    if git("status", "--porcelain", "--untracked-files=no") and not args.allow_dirty:
        problems.append("есть незакоммиченные изменения в отслеживаемых файлах (--allow-dirty)")
    if not args.review:
        unmet = unmet_dependencies(task)
        if unmet and not args.force:
            problems.append("не выполнены зависимости: " + ", ".join(unmet) + " (--force)")
    if plan.channel == "claude":
        if plan.model.startswith(CREDIT_ONLY_PREFIXES) and not args.allow_credits:
            problems.append(f"{plan.model} на Pro оплачивается кредитами (--allow-credits)")
        required = MIN_CLI_FOR_MODEL.get(plan.model)
        if required and shutil.which("claude"):
            version = cli_version(plan.executable)
            if version is None or version < required:
                need = ".".join(map(str, required))
                have = ".".join(map(str, version)) if version else "?"
                problems.append(f"{plan.model} требует claude >= {need}, сейчас {have}")
    if args.review and task.id != "R" and not git("log", "--oneline", f"--grep=OCR {task.id}:"):
        problems.append(f"нет коммитов `OCR {task.id}:` для ревью")
    if args.fix and not (REVIEWS_DIR / f"{task.id}.md").exists():
        problems.append(f"нет файла ревью docs/ocr_tasks/reviews/{task.id}.md для --fix")
    return problems


class StreamSummary:
    """Печать хода сессии `claude -p --output-format stream-json` и итоговой сводки."""

    def __init__(self, echo: bool = True) -> None:
        self.echo = echo
        self.result: dict[str, Any] | None = None
        self.rate_limit: dict[str, Any] | None = None
        self.init: dict[str, Any] | None = None
        self.tool_calls = 0

    def feed(self, line: str) -> None:
        try:
            event = json.loads(line)
        except json.JSONDecodeError:
            self._print(line.rstrip())
            return
        kind = event.get("type")
        if kind == "system" and event.get("subtype") == "init":
            self.init = event
            self._print(f"[init] model={event.get('model')} session={event.get('session_id')}")
        elif kind == "assistant":
            for block in event.get("message", {}).get("content", []):
                self._assistant_block(block)
        elif kind == "result":
            self.result = event
        elif kind == "rate_limit_event":
            self._rate_limit(event)

    def _assistant_block(self, block: dict[str, Any]) -> None:
        if block.get("type") == "text":
            text = " ".join(str(block.get("text", "")).split())
            if text:
                self._print(f"  > {text[:160]}")
        elif block.get("type") == "tool_use":
            self.tool_calls += 1
            data = block.get("input", {}) or {}
            detail = data.get("command") or data.get("file_path") or data.get("pattern") or ""
            detail = " ".join(str(detail).split())
            self._print(f"  $ {block.get('name')}: {detail[:140]}")

    def _rate_limit(self, event: dict[str, Any]) -> None:
        info = event.get("rate_limit_info", event)
        previous = _window_usage(self.rate_limit)
        self.rate_limit = info
        current = _window_usage(info)
        if current and current != previous:
            self._print(f"[quota] {current}")

    def _print(self, text: str) -> None:
        if self.echo and text:
            print(text, flush=True)

    def report(self) -> str:
        lines = ["", "=== Итог сессии ==="]
        result = self.result or {}
        if not result:
            lines.append("событие result не получено (сессия оборвалась?)")
        else:
            lines.append(
                f"итог: {result.get('subtype')}  ошибка: {result.get('is_error')}  "
                f"ходов: {result.get('num_turns')}  вызовов инструментов: {self.tool_calls}"
            )
            duration = (result.get("duration_ms") or 0) / 60000
            lines.append(
                f"время: {duration:.1f} мин  $ по прайсу API: {result.get('total_cost_usd')}"
            )
            models = sorted((result.get("modelUsage") or {}).keys())
            if models:
                lines.append("модели в сессии: " + ", ".join(models))
            denials = result.get("permission_denials") or []
            if denials:
                lines.append(f"отказов в разрешениях: {len(denials)}")
        usage = _window_usage(self.rate_limit)
        if usage:
            lines.append(f"квота: {usage}")
        return "\n".join(lines)


def _window_usage(info: dict[str, Any] | None) -> str:
    if not info:
        return ""
    windows = info.get("unifiedWindows") or {}
    parts = []
    for name in ("five_hour", "seven_day"):
        window = windows.get(name) or {}
        if "utilization" in window:
            parts.append(f"{name}={window['utilization']}")
    if info.get("status"):
        parts.append(f"status={info['status']}")
    return " ".join(parts)


def run(plan: RunPlan, task: Task, *, keep_api_key: bool = False) -> int:
    LOG_DIR.mkdir(parents=True, exist_ok=True)
    log_path = LOG_DIR / plan.log_name
    env = os.environ.copy()
    env["PYTHONUTF8"] = "1"
    if not keep_api_key:
        # ANTHROPIC_API_KEY в -p важнее подписки: сессия ушла бы в оплату по API (D-28).
        env.pop("ANTHROPIC_API_KEY", None)
    print(f"[run] {task.id} → {plan.channel} {plan.model} {plan.effort}".rstrip())
    print(f"[log] {log_path.relative_to(REPO_ROOT)}")
    summary = StreamSummary()
    with (
        log_path.open("w", encoding="utf-8") as log,
        subprocess.Popen(
            plan.command,
            cwd=REPO_ROOT,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            encoding="utf-8",
            errors="replace",
            bufsize=1,
        ) as process,
    ):
        assert process.stdout is not None
        for line in process.stdout:
            log.write(line)
            log.flush()
            if plan.channel == "claude":
                summary.feed(line)
            else:
                print(line, end="", flush=True)
        code = process.wait()
    if plan.channel == "claude":
        print(summary.report())
    print(f"код выхода: {code}")
    print(f"Проверьте: git log -1 --stat и раздел {task.id} в docs/ocr_tasks/PROGRESS.md")
    return code


def summarize(path: Path) -> None:
    summary = StreamSummary(echo=False)
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        summary.feed(line)
    print(summary.report())


def parse_args(argv: list[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Запуск задачи OCR-плана в claude/devin -p")
    parser.add_argument("task", nargs="?", help="id задачи, например T02")
    parser.add_argument("--list", "--status", dest="status", action="store_true")
    parser.add_argument("--export-json", action="store_true", help="шапки задач в JSON")
    parser.add_argument("--summarize", type=Path, help="сводка по готовому логу stream-json")
    parser.add_argument("--dry-run", action="store_true", help="показать команду и выйти")
    parser.add_argument(
        "--cloud-prompt", action="store_true", help="текст задания для облачной сессии"
    )
    parser.add_argument("--channel", choices=["claude", "devin"])
    parser.add_argument("--model", help="полный ID модели вместо шапки")
    parser.add_argument("--effort", choices=["low", "medium", "high", "xhigh", "max"])
    parser.add_argument("--devin-model", help="UID модели Devin вместо шапки")
    parser.add_argument("--model-map", help="подмена моделей: a=b,c=d (или OCR_MODEL_MAP)")
    parser.add_argument("--review", action="store_true", help="ревью по R_review.md")
    parser.add_argument("--reviewer", help="например devin:gpt-6-astra-high")
    parser.add_argument("--fix", action="store_true", help="повтор после ревью, ступень выше")
    parser.add_argument("--max-turns", type=int)
    parser.add_argument("--budget", help="auto (по размеру блока) или сумма в $")
    parser.add_argument("--permission-mode", default="auto")
    parser.add_argument("--force", action="store_true", help="игнорировать зависимости")
    parser.add_argument("--allow-main", action="store_true")
    parser.add_argument("--allow-dirty", action="store_true")
    parser.add_argument("--allow-credits", action="store_true", help="разрешить Fable")
    parser.add_argument("--keep-api-key", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")
    args = parse_args(sys.argv[1:] if argv is None else argv)
    tasks = load_tasks()
    if args.summarize:
        summarize(args.summarize)
        return 0
    if args.export_json:
        print(json.dumps([asdict(task) for task in tasks.values()], ensure_ascii=False, indent=2))
        return 0
    if args.status or not args.task:
        print_status(tasks)
        return 0
    task = tasks.get(args.task.upper())
    if task is None:
        raise SystemExit(f"Нет задачи {args.task}. Список: --status")
    if args.cloud_prompt:
        prompt = CLOUD_PROMPT.format(id=task.id, path=task.path)
        print(prompt)
        print(f'\nЗапуск из терминала: claude --cloud "{prompt}"')
        return 0
    plan = build_plan(task, args)
    problems = preflight(task, plan, args)
    print(f"[plan] {task.id} {task.title}")
    print(f"[plan] канал={plan.channel} модель={plan.model} effort={plan.effort or '—'}")
    print("[cmd] " + subprocess.list2cmdline(plan.command))
    if problems:
        print("[preflight] " + "\n[preflight] ".join(problems))
        if not args.dry_run:
            return 2
    if args.dry_run:
        return 0
    if args.fix and plan.effort == "max" and plan.channel == "claude":
        print(f"[i] следующая ступень после max — {ESCALATION_AFTER_MAX}")
    return run(plan, task, keep_api_key=args.keep_api_key)


if __name__ == "__main__":
    sys.exit(main())
