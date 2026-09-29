"""Единственное место с путями OCR-лаборатории.

Пути вычисляются при импорте из переменных окружения:
- `OCR_DATASET_CSV` — выгрузка истины (по умолчанию
  `data/historical/analysis/reconciled_dataset.csv`);
- `OCR_WORK_DIR` — каталог производных артефактов (по умолчанию `data/ocr`).

Каталоги создаются по требованию функциями `ensure_dir` и `work_subdir`, а не при импорте.
"""

from __future__ import annotations

import os
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent

DATASET_CSV = Path(
    os.environ.get(
        "OCR_DATASET_CSV",
        str(REPO_ROOT / "data" / "historical" / "analysis" / "reconciled_dataset.csv"),
    )
)

WORK_DIR = Path(os.environ.get("OCR_WORK_DIR", str(REPO_ROOT / "data" / "ocr")))

SUBDIRS = (
    "pages",
    "layouts",
    "crops",
    "sheets",
    "review",
    "labels",
    "models",
    "reports",
    "logs",
)

PAGES_DIR = WORK_DIR / "pages"
LAYOUTS_DIR = WORK_DIR / "layouts"
CROPS_DIR = WORK_DIR / "crops"
SHEETS_DIR = WORK_DIR / "sheets"
REVIEW_DIR = WORK_DIR / "review"
LABELS_DIR = WORK_DIR / "labels"
MODELS_DIR = WORK_DIR / "models"
REPORTS_DIR = WORK_DIR / "reports"
LOGS_DIR = WORK_DIR / "logs"

MANIFEST = WORK_DIR / "manifest.csv"
CORRECTIONS_CSV = WORK_DIR / "truth_corrections.csv"
PRINTED_FLAGS_CSV = LABELS_DIR / "printed_flags.csv"

# Кэши моделей (HF, torch). На ПК их нужно держать на E:, путь задаёт `OCR_CACHE_DIR`.
CACHE_DIR = Path(os.environ.get("OCR_CACHE_DIR", str(WORK_DIR / "cache")))


def ensure_dir(path: Path) -> Path:
    """Создать каталог (с родителями), если его нет, и вернуть путь."""
    path.mkdir(parents=True, exist_ok=True)
    return path


def work_subdir(name: str) -> Path:
    """Подкаталог `WORK_DIR` из `SUBDIRS`; создаётся при первом обращении."""
    if name not in SUBDIRS:
        raise ValueError(f"Неизвестный подкаталог OCR: {name}")
    return ensure_dir(WORK_DIR / name)


def configure_model_caches() -> dict[str, str]:
    """Выставить `HF_HOME` и `TORCH_HOME`, если они не заданы.

    Вызывать до импорта transformers и torch. Уже заданные переменные не трогает.
    Возвращает итоговые значения обеих переменных.
    """
    defaults = {"HF_HOME": CACHE_DIR / "hf", "TORCH_HOME": CACHE_DIR / "torch"}
    for name, path in defaults.items():
        if not os.environ.get(name):
            os.environ[name] = str(ensure_dir(path))
    return {name: os.environ[name] for name in defaults}
