"""T28: ежемесячный цикл дообучения — манифест → кропы → обучение → калибровка/ONNX →
приоры → сквозная оценка на holdout → сравнение с текущей прод-моделью → продвижение.

Повторяет вручную выверенный рецепт T22 (см. `docs/ocr_tasks/PROGRESS.md`, T22, «для
следующих задач»), но на объединённом манифесте «история + подтверждённые ваучеры»
(`ocr_lab.manifest.combine`, holdout — последний полный месяц) и с автоматическим шагом
сравнения/продвижения.

Запуск (раз в месяц, из `.venv`): `python -m ocr_lab.retrain run`.
Инструкция для оператора — `docs/ocr_retrain.md`.

**Версии моделей.** Рантайм (`app/ocr/runtime.py`, T18/T22) уже умеет выбирать версию по
суффиксу каталога (`<models_dir>/digits_<version>`, …) через `OCR_MODEL_VERSION`/
`Settings.ocr_model_version` (`APP_OCR_MODEL_VERSION`) — переключение и откат уже сегодня
одна настройка, без перезапуска с другим `APP_OCR_MODELS_DIR`. Этот цикл использует ту же
схему: версия — `v<ГГГГММ>` (месяц запуска), артефакты лежат рядом с `v0`/`v1` в общем
`data/ocr/models/` (не в отдельном подкаталоге `vYYYYMM/`, как буквально написано в
промпте, — решение без человека, см. журнал T28). `data/ocr/models/current_version.txt` —
человекочитаемый указатель «какая версия сейчас рекомендована» (информационный: реальное
переключение прода — отдельное осознанное действие оператора, см. `docs/ocr_retrain.md`).

**Шаги с torch** (`train_digits`, `train_number`, `calibrate`, `export_onnx`) идут отдельным
процессом через `.venv-train` (там CUDA) — сам `retrain.py` запускают из `.venv` (там
`sqlalchemy`/`app.*` для выгрузки подтверждённых ваучеров). Лёгкие шаги (манифест, кропы,
приоры, сквозная оценка) вызываются прямо как функции `ocr_lab.*`, без сабпроцесса.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path
from typing import Any

from app.database import SessionLocal
from ocr_lab import cut_crops as cut_crops_mod
from ocr_lab import eval_pipeline as eval_pipeline_mod
from ocr_lab import export_confirmed
from ocr_lab import fit_decoder_priors as fit_decoder_priors_mod
from ocr_lab import fit_number_pairs as fit_number_pairs_mod
from ocr_lab import manifest as manifest_mod
from ocr_lab import pages as pages_mod
from ocr_lab import train_digits as train_digits_mod
from ocr_lab import train_number as train_number_mod
from ocr_lab.dataset import INDEX_CSV
from ocr_lab.paths import MODELS_DIR, REPO_ROOT, REPORTS_DIR, WORK_DIR, ensure_dir

RETRAIN_DIR = WORK_DIR / "retrain"
CURRENT_VERSION_FILE = MODELS_DIR / "current_version.txt"
DEFAULT_CURRENT_VERSION = "v1"
#: Рецепт T22 (см. докстринг) — те же конфиги обучения, что дали `v1`.
DEFAULT_MNIST = 3000
DEFAULT_SYNTH = 1000
DEFAULT_TARGET = 0.995
#: Допуск точности при сравнении версий — решение без человека: сравнивать `coverage`
#: при "не худшей" точности, а не требовать от новой версии строго большей точности на
#: шумной малой выборке (см. журнал T28).
PRECISION_TOLERANCE = 1e-9


class RetrainStepError(RuntimeError):
    """Шаг цикла завершился ошибкой (ненулевой код возврата внешнего процесса)."""


class RetrainIncomplete(RuntimeError):
    """Шаг остановился по `--max-minutes`. Повторите ту же команду — она продолжит (`--resume`)."""


# ---------------------------------------------------------------------------
# Версия цикла и пути
# ---------------------------------------------------------------------------


def default_version(as_of: date) -> str:
    """Версия цикла по умолчанию — месяц запуска: `v202610`."""
    return f"v{as_of:%Y%m}"


def current_version(path: Path = CURRENT_VERSION_FILE) -> str:
    """Версия, помеченная как текущая прод (указатель `current_version.txt`).

    Нет файла — `v1` (версия, замороженная T22; это же значение по умолчанию у
    `Settings.ocr_model_version`/`app.ocr.runtime.MODEL_VERSION`).
    """
    if path.is_file():
        text = path.read_text(encoding="utf-8").strip()
        if text:
            return text
    return DEFAULT_CURRENT_VERSION


@dataclass(frozen=True)
class VersionPaths:
    """Пути артефактов одной версии модели в общем `data/ocr/models/`."""

    version: str
    models_dir: Path = MODELS_DIR

    @property
    def digits_dir(self) -> Path:
        return self.models_dir / f"digits_{self.version}"

    @property
    def number_dir(self) -> Path:
        return self.models_dir / f"number_{self.version}"

    @property
    def decoder_priors(self) -> Path:
        return self.models_dir / f"decoder_priors_{self.version}.json"

    @property
    def number_priors(self) -> Path:
        return self.models_dir / f"number_priors_{self.version}.json"

    @property
    def pair_priors(self) -> Path:
        return self.models_dir / f"pair_priors_{self.version}.json"


# ---------------------------------------------------------------------------
# Подпроцесс .venv-train (шаги с torch)
# ---------------------------------------------------------------------------


def train_python() -> Path:
    """Интерпретатор для `train_digits`/`train_number`/`calibrate`/`export_onnx`.

    По умолчанию — `.venv-train` (CUDA, T05); переопределяется `OCR_TRAIN_PYTHON`. Если
    ни того, ни другого нет — текущий интерпретатор (медленнее, CPU, но работает: в `.venv`
    тоже стоит torch, см. `docs/ocr_tasks/CONTEXT.md`).
    """
    override = os.environ.get("OCR_TRAIN_PYTHON")
    if override:
        return Path(override)
    candidate = REPO_ROOT / ".venv-train" / "Scripts" / "python.exe"
    return candidate if candidate.is_file() else Path(sys.executable)


def _run_train_module(python: Path, module: str, args: list[str], *, log: Path | None = None
                      ) -> int:
    """Запустить `python -m <module> <args>` отдельным процессом (torch-шаги)."""
    cmd = [str(python), "-m", module, *args]
    env = dict(os.environ)
    env["PYTHONUTF8"] = "1"
    result = subprocess.run(cmd, cwd=str(REPO_ROOT), env=env, capture_output=True, text=True,
                            encoding="utf-8", errors="replace")
    output = (result.stdout or "") + (result.stderr or "")
    if log is not None:
        ensure_dir(log.parent)
        log.write_text(output, encoding="utf-8")
    if result.returncode not in (0, 3):
        tail = "\n".join(output.splitlines()[-40:])
        raise RetrainStepError(f"{module} {args[:1]}: код {result.returncode}\n{tail}")
    return result.returncode


# ---------------------------------------------------------------------------
# Шаг 1 — манифест цикла (история + подтверждённые ваучеры)
# ---------------------------------------------------------------------------


def _split_counts_to_json(counts: Counter[tuple[str, str]]) -> list[list[Any]]:
    """`Counter[(source, split)]` -> список `[source, split, n]` (ключ-кортеж не JSON-совместим)."""
    return [[source, split, n] for (source, split), n in counts.items()]


def _split_counts_from_json(rows: list[list[Any]]) -> Counter[tuple[str, str]]:
    return Counter({(str(source), str(split)): int(n) for source, split, n in rows})


def _combine_report_to_json(report: manifest_mod.CombineReport) -> dict[str, Any]:
    return {**report.__dict__, "split_counts": _split_counts_to_json(report.split_counts)}


def _combine_report_from_json(data: dict[str, Any]) -> manifest_mod.CombineReport:
    return manifest_mod.CombineReport(
        **{**data, "split_counts": _split_counts_from_json(data["split_counts"])}
    )


def step_manifest(version: str, *, as_of: date, force: bool = False
                  ) -> tuple[Path, manifest_mod.CombineReport]:
    """Собрать манифест цикла: история (:func:`manifest_mod.build`) + выгрузка
    подтверждённых ваучеров (:func:`export_confirmed.build`) → :func:`manifest_mod.combine`.

    Идемпотентен: если `manifest.csv` версии уже на диске и `force=False`, сборка и выгрузка
    (включая обращение к БД) пропускаются, отчёт объединения берётся из `combine_report.json`
    рядом.
    """
    out_dir = ensure_dir(RETRAIN_DIR / version)
    manifest_path = out_dir / "manifest.csv"
    report_json = out_dir / "combine_report.json"
    if manifest_path.is_file() and report_json.is_file() and not force:
        report = _combine_report_from_json(json.loads(report_json.read_text(encoding="utf-8")))
        return manifest_path, report

    history_df, history_report = manifest_mod.build()
    with SessionLocal() as db:
        app_df, export_report = export_confirmed.build(db)
    combined_df, combine_report = manifest_mod.combine(history_df, app_df, as_of=as_of)

    combined_df.to_csv(manifest_path, index=False, encoding="utf-8-sig")
    (out_dir / "manifest_report.md").write_text(
        manifest_mod.render_report(history_report), encoding="utf-8"
    )
    (out_dir / "export_confirmed_report.md").write_text(
        export_confirmed.render_report(export_report), encoding="utf-8"
    )
    (out_dir / "combine_report.md").write_text(
        manifest_mod.render_combine_report(combine_report, as_of=as_of), encoding="utf-8"
    )
    report_json.write_text(
        json.dumps(_combine_report_to_json(combine_report), ensure_ascii=False, indent=2,
                   default=list),
        encoding="utf-8",
    )
    return manifest_path, combine_report


# ---------------------------------------------------------------------------
# Шаг 2 — страницы и кропы (только новые сканы — идемпотентность "бесплатно")
# ---------------------------------------------------------------------------


def step_crops(manifest_path: Path, *, workers: int = 4) -> None:
    """Дорендерить страницы и докропить подполя новых сканов манифеста цикла.

    `ocr_lab.pages.build`/`ocr_lab.cut_crops.build` сами пропускают сканы, которые уже есть
    в кэше (`data/ocr/pages/`, `data/ocr/crops_index.csv`) — повторный вызов на том же
    манифесте почти не стоит времени, поэтому отдельного флага "только новые" не нужно.
    """
    pages_mod.build(manifest_path=manifest_path, out_dir=pages_mod.PAGES_DIR, workers=workers)
    cut_crops_mod.build(manifest=manifest_path, workers=workers)


# ---------------------------------------------------------------------------
# Шаги 3-4 — обучение digits/number (torch, .venv-train)
# ---------------------------------------------------------------------------


def _resume_flag(run_dir: Path) -> list[str]:
    return ["--resume"] if (run_dir / "last.pt").is_file() else []


def step_train_digits(
    version: str, manifest_path: Path, *, mnist: int = DEFAULT_MNIST,
    max_minutes: float | None = None, force: bool = False,
) -> Path:
    """Обучить (или продолжить) `digits_<version>`. Возвращает каталог продвинутой модели."""
    paths = VersionPaths(version)
    if (paths.digits_dir / "best.pt").is_file() and not force:
        return paths.digits_dir
    run_name = f"retrain_{version}_digits"
    run_dir = train_digits_mod.RUNS_DIR / run_name
    python = train_python()
    args = [
        "train", "--run", run_name, "--mnist", str(mnist),
        "--manifest", str(manifest_path), "--index-csv", str(INDEX_CSV),
        *(["--max-minutes", str(max_minutes)] if max_minutes is not None else []),
        *_resume_flag(run_dir),
    ]
    rc = _run_train_module(python, "ocr_lab.train_digits", args,
                           log=RETRAIN_DIR / version / "train_digits.log")
    if rc == 3:
        raise RetrainIncomplete(f"digits {version}: остановлено по --max-minutes, run={run_name}")
    _run_train_module(python, "ocr_lab.train_digits",
                      ["promote", "--run", run_name, "--dest", str(paths.digits_dir)])
    return paths.digits_dir


def step_train_number(
    version: str, manifest_path: Path, digits_dir: Path, *, synth: int = DEFAULT_SYNTH,
    max_minutes: float | None = None, force: bool = False,
) -> Path:
    """Обучить (или продолжить) `number_<version>` (CTC, бэкбон — из `digits_<version>`)."""
    paths = VersionPaths(version)
    if (paths.number_dir / "best.pt").is_file() and not force:
        return paths.number_dir
    run_name = f"retrain_{version}_number"
    run_dir = train_number_mod.RUNS_DIR / run_name
    python = train_python()
    args = [
        "train", "--run", run_name, "--kind", "ctc", "--synth", str(synth),
        "--digits-checkpoint", str(digits_dir / "best.pt"),
        "--manifest", str(manifest_path), "--index-csv", str(INDEX_CSV),
        *(["--max-minutes", str(max_minutes)] if max_minutes is not None else []),
        *_resume_flag(run_dir),
    ]
    rc = _run_train_module(python, "ocr_lab.train_number", args,
                           log=RETRAIN_DIR / version / "train_number.log")
    if rc == 3:
        raise RetrainIncomplete(f"number {version}: остановлено по --max-minutes, run={run_name}")
    _run_train_module(python, "ocr_lab.train_number",
                      ["promote", "--run", run_name, "--dest", str(paths.number_dir)])
    return paths.number_dir


# ---------------------------------------------------------------------------
# Шаги 5-6 — калибровка и экспорт ONNX (torch, .venv-train)
# ---------------------------------------------------------------------------


def step_calibrate(version: str, manifest_path: Path, *, force: bool = False) -> None:
    paths = VersionPaths(version)
    python = train_python()
    for kind, model_dir in (("digits", paths.digits_dir), ("number", paths.number_dir)):
        if (model_dir / "calibration.json").is_file() and not force:
            continue
        _run_train_module(
            python, "ocr_lab.calibrate",
            [kind, "--model-dir", str(model_dir), "--manifest", str(manifest_path),
             "--index-csv", str(INDEX_CSV)],
            log=RETRAIN_DIR / version / f"calibrate_{kind}.log",
        )


def step_export_onnx(version: str, *, trained_at: str, force: bool = False) -> None:
    paths = VersionPaths(version)
    python = train_python()
    for kind, model_dir in (("digits", paths.digits_dir), ("number", paths.number_dir)):
        if (model_dir / "model.onnx").is_file() and (model_dir / "meta.json").is_file() \
                and not force:
            continue
        _run_train_module(
            python, "ocr_lab.export_onnx",
            ["export", kind, "--model-dir", str(model_dir), "--trained-at", trained_at],
            log=RETRAIN_DIR / version / f"export_onnx_{kind}.log",
        )


# ---------------------------------------------------------------------------
# Шаг 7 — приоры декодера, номера и пары (без torch, в процессе)
# ---------------------------------------------------------------------------


def step_priors(version: str, manifest_path: Path, *, force: bool = False) -> None:
    paths = VersionPaths(version)
    if not paths.decoder_priors.is_file() or force:
        rc = fit_decoder_priors_mod.main([
            "fit", "--manifest", str(manifest_path), "--no-weights",
            "--out", str(paths.decoder_priors),
        ])
        if rc != 0:
            raise RetrainStepError(f"fit_decoder_priors fit: код {rc}")
    if (not paths.number_priors.is_file() or not paths.pair_priors.is_file()) or force:
        rc = fit_number_pairs_mod.main([
            "fit", "--manifest", str(manifest_path),
            "--number-out", str(paths.number_priors), "--pair-out", str(paths.pair_priors),
        ])
        if rc != 0:
            raise RetrainStepError(f"fit_number_pairs fit: код {rc}")


# ---------------------------------------------------------------------------
# Шаг 8 — сквозная оценка на holdout (без torch, в процессе)
# ---------------------------------------------------------------------------


def step_eval(
    label: str, version: str, manifest_path: Path, *, target: float = DEFAULT_TARGET,
    max_minutes: float | None = None, force: bool = False,
) -> dict[str, Any]:
    """Прогнать `eval_pipeline run/decode/report` для версии `version` под именем `label`.

    Возвращает разобранный `e2e_metrics.json` этого прогона. `label` различает "новую"
    версию и "текущую прод" версию на одном и том же `manifest_path` (одинаковый holdout).
    """
    paths = VersionPaths(version)
    report_md = REPORTS_DIR / f"e2e_{label}_report.md"
    report_json = report_md.with_suffix(".json")
    if report_json.is_file() and not force:
        return json.loads(report_json.read_text(encoding="utf-8"))

    common = ["--name", label, "--manifest", str(manifest_path)]
    rc = eval_pipeline_mod.main([
        *common, "run", "--split", "val,test", "--models-dir", str(paths.models_dir),
        "--digits", f"digits_{version}", "--number", f"number_{version}",
        *(["--max-minutes", str(max_minutes)] if max_minutes is not None else []),
    ])
    if rc != 0:
        raise RetrainStepError(f"eval_pipeline run ({label}): код {rc}")
    rc = eval_pipeline_mod.main([
        *common, "decode", "--split", "val,test",
        "--decoder-priors", str(paths.decoder_priors),
        "--number-priors", str(paths.number_priors), "--pair-priors", str(paths.pair_priors),
    ])
    if rc != 0:
        raise RetrainStepError(f"eval_pipeline decode ({label}): код {rc}")
    rc = eval_pipeline_mod.main([
        *common, "report", "--target", str(target), "--out", str(report_md),
    ])
    if rc != 0:
        raise RetrainStepError(f"eval_pipeline report ({label}): код {rc}")
    return json.loads(report_json.read_text(encoding="utf-8"))


# ---------------------------------------------------------------------------
# Шаг 9 — сравнение "новая против текущей прод" и решение о продвижении
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class PromotionDecision:
    """Решение шага 10 (сравнение) — не зависит от того, применили его или нет (`--dry-run`)."""

    promote: bool
    reason: str
    new_coverage: float | None
    new_precision: float | None
    current_coverage: float | None
    current_precision: float | None


def _test_accept(metrics: dict[str, Any]) -> dict[str, Any]:
    """`splits.test.accept.t_val` метрик `ocr_lab.eval_pipeline` (порог, выбранный на val)."""
    return metrics["splits"]["test"]["accept"]["t_val"]


def decide_promotion(
    new_metrics: dict[str, Any], current_metrics: dict[str, Any],
    *, precision_tolerance: float = PRECISION_TOLERANCE,
) -> PromotionDecision:
    """Продвигать новую версию, только если на `test` она не хуже текущей прод-версии.

    Критерий (план, п.3): «не хуже по покрытию при целевой точности» — обе версии уже
    выбрали свой порог на `val` под одну и ту же целевую точность (`eval_pipeline.report
    --target`), поэтому сравниваются их `coverage`/`precision` на `test` при этом пороге.
    Решение без человека: точность новой версии не обязана быть строго больше (шум малой
    выборки, см. T21/T22), достаточно не хуже `current.precision - precision_tolerance`,
    но `coverage` обязан быть не меньше. Любой `None` (нет принятых на val/test) —
    отказ: сравнивать нечего.
    """
    new = _test_accept(new_metrics)
    cur = _test_accept(current_metrics)
    new_cov, new_prec = new["coverage"], new["precision"]
    cur_cov, cur_prec = cur["coverage"], cur["precision"]
    if None in (new_cov, new_prec, cur_cov, cur_prec):
        return PromotionDecision(False, "нет порога/принятых ваучеров на val или test",
                                 new_cov, new_prec, cur_cov, cur_prec)
    if new_cov < cur_cov:
        return PromotionDecision(
            False, f"покрытие хуже: {new_cov:.4f} < {cur_cov:.4f}",
            new_cov, new_prec, cur_cov, cur_prec,
        )
    if new_prec < cur_prec - precision_tolerance:
        return PromotionDecision(
            False, f"точность хуже: {new_prec:.4f} < {cur_prec:.4f}",
            new_cov, new_prec, cur_cov, cur_prec,
        )
    return PromotionDecision(
        True, f"не хуже текущей: покрытие {new_cov:.4f} >= {cur_cov:.4f}, "
        f"точность {new_prec:.4f} >= {cur_prec:.4f} - {precision_tolerance:g}",
        new_cov, new_prec, cur_cov, cur_prec,
    )


# ---------------------------------------------------------------------------
# Оркестратор одного месячного цикла
# ---------------------------------------------------------------------------


@dataclass
class RetrainResult:
    version: str
    manifest_path: Path
    combine_report: manifest_mod.CombineReport
    decision: PromotionDecision
    promoted: bool
    new_metrics_path: Path
    current_metrics_path: Path


def run(
    *, as_of: date | None = None, version: str | None = None, dry_run: bool = False,
    force: bool = False, max_minutes: float | None = None, mnist: int = DEFAULT_MNIST,
    synth: int = DEFAULT_SYNTH, target: float = DEFAULT_TARGET, workers: int = 4,
    current: str | None = None,
) -> RetrainResult:
    """Один проход ежемесячного цикла (план, «Этап 6»). См. докстринг модуля по шагам.

    `dry_run=True` — все шаги выполняются как обычно (включая обучение и оценку), но
    продвижение (шаг 10) не применяется: указатель `current_version.txt` не трогается,
    решение только записывается в отчёт. Используется для сухого прогона на истории
    (критерий приёмки T28) и для проверки цикла без риска подвинуть прод.
    """
    as_of = as_of or datetime.now().date()
    version = version or default_version(as_of)
    current_ver = current or current_version()

    manifest_path, combine_report = step_manifest(version, as_of=as_of, force=force)
    step_crops(manifest_path, workers=workers)

    digits_dir = step_train_digits(version, manifest_path, mnist=mnist,
                                   max_minutes=max_minutes, force=force)
    step_train_number(version, manifest_path, digits_dir, synth=synth,
                      max_minutes=max_minutes, force=force)
    step_calibrate(version, manifest_path, force=force)
    step_export_onnx(version, trained_at=as_of.isoformat(), force=force)
    step_priors(version, manifest_path, force=force)

    new_metrics = step_eval(f"retrain_{version}_new", version, manifest_path, target=target,
                            max_minutes=max_minutes, force=force)
    current_metrics = step_eval(f"retrain_{version}_vs_{current_ver}", current_ver, manifest_path,
                                target=target, max_minutes=max_minutes, force=force)
    decision = decide_promotion(new_metrics, current_metrics)

    promoted = False
    if decision.promote and not dry_run:
        ensure_dir(CURRENT_VERSION_FILE.parent)
        CURRENT_VERSION_FILE.write_text(version + "\n", encoding="utf-8")
        promoted = True

    out_dir = ensure_dir(RETRAIN_DIR / version)
    summary = {
        "version": version, "as_of": as_of.isoformat(), "current_version": current_ver,
        "dry_run": dry_run, "promoted": promoted, "decision": decision.__dict__,
        "combine": _combine_report_to_json(combine_report),
    }
    (out_dir / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2, default=list), encoding="utf-8"
    )
    _write_cycle_report(out_dir / "report.md", summary)

    return RetrainResult(
        version=version, manifest_path=manifest_path, combine_report=combine_report,
        decision=decision, promoted=promoted,
        new_metrics_path=REPORTS_DIR / f"e2e_retrain_{version}_new_report.json",
        current_metrics_path=REPORTS_DIR / f"e2e_retrain_{version}_vs_{current_ver}_report.json",
    )


def _write_cycle_report(path: Path, summary: dict[str, Any]) -> None:
    decision = summary["decision"]
    lines = [
        f"# Цикл дообучения {summary['version']} ({summary['as_of']})",
        "",
        f"- режим: {'сухой прогон (--dry-run)' if summary['dry_run'] else 'обычный'}",
        f"- текущая прод-версия (сравнение): {summary['current_version']}",
        f"- решение о продвижении: {'да' if decision['promote'] else 'нет'} — {decision['reason']}",
        f"- применено (`current_version.txt` обновлён): {'да' if summary['promoted'] else 'нет'}",
        f"- новая версия, test: покрытие {decision['new_coverage']}, "
        f"точность {decision['new_precision']}",
        f"- текущая версия, test: покрытие {decision['current_coverage']}, "
        f"точность {decision['current_precision']}",
        "",
        "## Манифест цикла",
        "",
    ]
    for source, split, n in sorted(summary["combine"]["split_counts"]):
        lines.append(f"- {source} / {split}: {n}")
    ensure_dir(path.parent)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="python -m ocr_lab.retrain", description=__doc__.splitlines()[0]
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p_run = sub.add_parser("run", help="один проход ежемесячного цикла (все шаги)")
    p_run.add_argument("--as-of", type=date.fromisoformat, default=None,
                       help="ГГГГ-ММ-ДД, по умолчанию — сегодня")
    p_run.add_argument("--version", default=None, help="по умолчанию v<ГГГГММ> от --as-of")
    p_run.add_argument("--current", default=None,
                       help="версия для сравнения, по умолчанию current_version.txt (иначе v1)")
    p_run.add_argument("--dry-run", action="store_true",
                       help="не применять продвижение (current_version.txt не меняется)")
    p_run.add_argument("--force", action="store_true", help="пересчитать готовые шаги заново")
    p_run.add_argument("--max-minutes", type=float, default=None,
                       help="на каждый тяжёлый шаг (обучение/оценка) отдельно")
    p_run.add_argument("--mnist", type=int, default=DEFAULT_MNIST)
    p_run.add_argument("--synth", type=int, default=DEFAULT_SYNTH)
    p_run.add_argument("--target", type=float, default=DEFAULT_TARGET)
    p_run.add_argument("--workers", type=int, default=4)

    args = parser.parse_args(argv)
    if args.command != "run":
        return 1

    try:
        result = run(
            as_of=args.as_of, version=args.version, dry_run=args.dry_run, force=args.force,
            max_minutes=args.max_minutes, mnist=args.mnist, synth=args.synth,
            target=args.target, workers=args.workers, current=args.current,
        )
    except RetrainIncomplete as exc:
        print(f"остановлено по --max-minutes: {exc}", file=sys.stderr)
        print("повторите ту же команду — цикл продолжит с этого шага", file=sys.stderr)
        return 3
    except RetrainStepError as exc:
        print(f"ошибка шага: {exc}", file=sys.stderr)
        return 2

    print(f"версия {result.version}: продвижение — "
         f"{'да' if result.promoted else 'нет'} ({result.decision.reason})")
    print(f"манифест цикла -> {result.manifest_path}")
    print(f"отчёт -> {RETRAIN_DIR / result.version / 'report.md'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
