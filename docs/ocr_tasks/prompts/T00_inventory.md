---
id: T00
title: Инвентаризация локального состояния и CONTEXT.md
stage: 0
depends_on: H0
type: research
complexity: medium
risk: safe
needs_vision: false
needs_web: false
host: local
size: S
channel: claude_first
claude_model: claude-sonnet-5
claude_effort: high
devin_model: claude-sonnet-5-medium
review: —
human_checkpoint: —
max_turns: 80
optional: false
---

# T00. Инвентаризация → `docs/ocr_tasks/CONTEXT.md`

## Зачем

Каждая следующая задача стартует в новой сессии. Чтобы она не тратила ходы на разведку,
собери в `docs/ocr_tasks/CONTEXT.md` проверенные факты о репозитории, текущем OCR-коде,
данных и окружении.

Часть кода и документов есть только на этой машине, на GitHub их нет:
- `app/services/voucher_ocr.py`, `app/services/voucher_trocr.py`;
- `data/historical/predict_ocr/ocr_regions.txt`;
- `docs/ocr_dates_plan.md` и соседние документы.

## Что собрать

1. **Git.**
   - Текущая ветка и `git status --short`, сгруппированный: изменённые, неотслеживаемые,
     игнорируемые каталоги данных.
   - Что отличается от `origin/main`: `git diff --stat origin/main` плюс неотслеживаемые
     файлы кода.
   - Какие документы плана отслеживает git.
2. **Текущий OCR-код.** Прочитай `voucher_ocr.py`, `voucher_trocr.py` и всё, что их вызывает
   (поиск `ocr` и `trocr` в `app/`, `scripts/`, `tests/`). Запиши:
   - публичные функции и классы с точными сигнатурами;
   - откуда они вызываются: роут, `predict_and_store`, скрипт;
   - какую модель TrOCR грузят: HF id, где лежит кэш, сколько занимает;
   - как режутся регионы: формат `ocr_regions.txt`, координаты, ресайз в 384×384;
   - что и куда пишется: `VoucherFieldPrediction.source`, `confidence`, кандидаты;
   - какие тесты это покрывают.
3. **Данные.**
   - Дерево `data/historical/`: каталоги, число файлов, объём. Содержимое не нужно.
   - `reconciled_dataset.csv`:
     - путь, кодировка, число строк;
     - колонки с долей заполненности;
     - для колонок с датой и временем — **формат** значения (например
       `ДД.ММ.ГГГГ ЧЧ:ММ`), без самих значений;
     - как записана полночь, если такие значения есть (`24:00`? `00:00`?);
     - уникальные значения `tug`;
     - распределение по годам.
   - Какая колонка содержит время из заявки (нужно приорам декодера): имя и формат.
   - `voucher_scan_path`:
     - сколько путей существует на диске;
     - общие корневые каталоги и расширения файлов;
     - на выборке из 30 файлов — число страниц PDF, размер страницы или изображения
       в пикселях, DPI.
   - Формат `ocr_regions.txt`. Это координаты, короткий файл можно показать целиком.
4. **Окружение.**
   - Версия Windows; Python в `.venv`.
   - Ключевые пакеты: `torch` (есть ли CUDA в сборке), `torchvision`, `transformers`,
     `numpy`, `opencv*`, `pypdfium2`, `pdfplumber`, `Pillow`, `onnxruntime`, `pandas`.
   - `nvidia-smi`: драйвер и версия CUDA.
   - Свободное место на C: и E:.
   - Кэш HF: `HF_HOME`, `~/.cache/huggingface`, размер.
   - `claude --version`; `devin --version`, если установлен.
5. **Базовые проверки.** Запусти `ruff check .`, `mypy app`, `pytest -q`. Зафиксируй,
   что падает уже сейчас: имя теста и причина в одну строку.
6. **`.gitignore`.** Проверь через `git check-ignore -v`, что `data/historical/` и
   `data/ocr/` не попадут в git. Если попадут — допиши правила в `.gitignore`.
7. **Дефект настроек TrOCR.** `voucher_trocr.py` читает `settings.trocr_enabled`,
   `trocr_model`, `trocr_device`, `trocr_max_new_tokens`, а в закоммиченном
   `app/config.py` их нет (см. CONTEXT.md, раздел 2). Проверь рабочую копию. Если полей нет —
   добавь их в `Settings` (по умолчанию: `trocr_enabled=True`,
   `trocr_model="kazars24/trocr-base-handwritten-ru"`, `trocr_device=""`,
   `trocr_max_new_tokens=32`) и тест, который вызывает `trocr_image` при
   `trocr_enabled=False`. Это единственная правка кода в T00.
8. **Сверка с планом.** Сравни факты с `docs/ocr_dates_plan.md`: 774 скана и 779 строк,
   2 ваучера третьего буксира, GTX 1050, 13,5 ГБ на C:. Расхождения выпиши списком.

## Результат

- `docs/ocr_tasks/CONTEXT.md` — заполни разделы шаблона, который уже лежит в файле.
  Пиши факты, а не планы. Команды приводи в виде, готовом к копированию.
- `.gitignore` — только если понадобилось.
- Больше ничего не меняй: это разведка.

## Критерии приёмки

- В CONTEXT.md есть все 8 пунктов. Для каждого ненайденного — пометка «не найдено» и где
  искал.
- Сигнатуры OCR-функций переписаны точно: имя, параметры, типы.
- В файле нет значений из реальных записей: имён судов и агентов, конкретных дат и времён
  ваучеров.

## Отчёт

Раздел в `PROGRESS.md` по шаблону из `_common.md`.
Коммит: `OCR T00: инвентаризация и CONTEXT.md`.
