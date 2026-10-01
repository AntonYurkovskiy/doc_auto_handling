# Контекст для задач OCR

Факты о локальной машине, данных и коде. Файл **заполняет T00**, следующие задачи
дополняют его, когда факт меняется. Реальных записей здесь нет: только форматы,
счётчики и пути.

Ниже — то, что известно из `origin/main` и плана на 2026-09-29. Всё с пометкой
«(проверить в T00)» T00 подтверждает или исправляет.

## 1. Репозиторий

- Монолит FastAPI + SQLite (WAL), без Alembic. Новые колонки добавляются через
  `_add_missing_columns` в `app/database.py`.
- Ваучеры:
  - `app/services/voucher.py` — приоры и кандидаты, включая сетку `candidate_datetimes`
    ±3 дня с шагом 10 минут;
  - `voucher_fields.py` — поля карточки и подтверждение;
  - `voucher_template.py` — один шаблон `baltiyskie_buksiry`, 12 строковых регионов
    по бланку 243k (Коммунар);
  - `voucher_number.py` — `parse_voucher_number`, `predict_next`;
  - `voucher_linking.py`, `voucher_files.py`.
- Модели БД: `Voucher`, `VoucherFieldPrediction` (`predicted_value`, `confidence`,
  `source`, `confirmed_value`; колонки top-k нет), `VoucherTemplate`, `VoucherRegion`.
- Проверки: `ruff check .`, `mypy app`, `pytest` (см. README).
- Только локально, на GitHub нет (проверить в T00): `app/services/voucher_ocr.py`,
  `app/services/voucher_trocr.py`, `data/historical/predict_ocr/ocr_regions.txt`,
  документы плана.

## 2. Текущий OCR-код

Прочитано из ветки `ocr/local-sources` (коммит 1f04bba). T00 сверяет с рабочей копией.

- `app/services/voucher_ocr.py`:
  - `load_voucher_image(voucher, *, dpi=300) -> PIL.Image | None` — картинка или первая
    страница PDF через `pypdfium2`, RGB;
  - `crop_region(image, region: VoucherRegion) -> PIL.Image` — кроп по нормализованным
    `center_x/center_y/width/height`, **строка целиком**;
  - `ocr_image(image, lang="rus+eng") -> (text, conf)` — Tesseract (`pytesseract`),
    контраст ×2; путь к бинарнику — `settings.tesseract_cmd`;
  - `ocr_voucher_regions(voucher) -> {region.name: (text, conf)}` — для
    `HANDWRITTEN_FIELDS` (4 строки дат) сначала TrOCR, при `None` — Tesseract; остальные
    регионы — Tesseract.
- `app/services/voucher_trocr.py`:
  - модель `kazars24/trocr-base-handwritten-ru` (VisionEncoderDecoder), ленивая загрузка
    с блокировкой, устройство cuda/cpu;
  - `trocr_image(image) -> (text, conf)`, conf — средняя max-вероятность токена;
  - процессор сжимает кроп строки в 384×384 — источник проблемы из плана.
  - **Дефект:** читает `settings.trocr_enabled`, `trocr_model`, `trocr_device`,
    `trocr_max_new_tokens`, а в `app/config.py` на ветке этих полей нет. Проверка
    `settings.trocr_enabled` стоит вне `try` → `AttributeError` при распознавании
    рукописных строк. Тесты это не ловят: `ocr_image`/`load_voucher_image` подменены.
    Возможно, в рабочей копии `config.py` есть незакоммиченные поля — проверить в T00.
- `app/services/voucher.py`: `predict_fields(..., ocr_values)` и
  `predict_and_store(..., ocr_values=None)` — если `ocr_values` не передан, вызывает
  `ocr_voucher_regions`. OCR-текст → `_ocr_predicted_value` (для дат — `_parse_datetime`),
  `source="ocr"`, `confidence` — уверенность OCR, кандидаты = OCR + приоры.
  Сетка `candidate_datetimes` осталась.
- `app/services/voucher_fields.py`: `apply_predictions_to_voucher(db, voucher)` — переносит
  предсказания в поля ваучера.
- `app/main.py`: `POST /vouchers/{voucher_id}/recognize` — перезапуск OCR и заполнение формы.
- Тесты: `tests/test_voucher_ocr.py` (Tesseract и TrOCR подменены).
- Зависимости: в `requirements.txt` добавлен `pytesseract==0.3.13`; torch и transformers
  там не перечислены — стоят в `.venv` вручную.
- `data/historical/predict_ocr/ocr_regions.txt` в ветку не попал.

## 3. Данные

- Истина: `data/historical/analysis/reconciled_dataset.csv`, лежит вне git.
  По плану — 779 строк на 774 скана, 2 ваучера третьего буксира.
- Колонки выгрузки (из `analysis/reconcile.py`, ветка `analysis/eda-historical`):
  `tug`, `vessel`, `work_type`, `agent`, `voucher_number`, `base_departure`,
  `base_arrival`, `work_start`, `work_end`, `application_file`, `voucher_file`, `grt_raw`, …
- Колонки сверки: `voucher_scan_path`, `email_path`, `base_year` и поля письма.
- Соответствие строк бланка: `left_base` ← `base_departure`,
  `arrived_base` ← `base_arrival`, `started_work` ← `work_start`,
  `finished_work` ← `work_end`.
- (заполняет T00: форматы дат, колонка времени заявки, где лежат сканы, форматы файлов,
  страницы, DPI)

## 4. Окружение

- По плану: Windows, GTX 1050 4 ГБ (Pascal, sm_61, только FP32), torch в `.venv` без CUDA,
  на C: свободно около 13,5 ГБ. `HF_HOME` планируется перенести на E:.
- Claude Code CLI должен быть ≥ 2.1.280 (Opus 5.5). По справочнику моделей на ПК стояла
  2.1.272.
- (заполняет T00: версии пакетов, `nvidia-smi`, место на дисках, кэши)

## 5. Базовые проверки

(заполняет T00: что падает до начала работ)

## 6. Пути и команды

- Производные артефакты — `data/ocr/`. Пути в коде — через `ocr_lab/paths.py` (после T02).
- Запуск задачи: `python docs/ocr_tasks/run_task.py T02`. Статус плана:
  `python docs/ocr_tasks/run_task.py --status`.
- (заполняет T00: точные команды для `.venv`, `.venv-train`)

## 7. Расхождения с планом

(заполняет T00)
