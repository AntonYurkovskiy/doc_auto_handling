# Журнал задач OCR-плана

Сюда каждая задача `T*` дописывает раздел по шаблону из `_common.md`, § 8.
Раннер (`run_task.py --status`) считает задачу выполненной, если в её разделе стоит
строка `- Статус: готово`.

Правила:
- новые разделы — только в конец;
- чужие разделы не редактировать;
- исправления — отдельной строкой в своём разделе.

<!-- Шаблон раздела (копировать ниже):

## Tnn — <название задачи> (<дата>)
- Статус: <готово | частично | заблокировано>
- Сделано: …
- Цифры: …
- Решения без человека: …
- Вопросы к человеку: …
- Для следующих задач: …

-->

## T13 — Формат предсказаний, метрики и отчёт оценки (облако) (2026-09-29)
- Статус: частично
- Сделано:
  - `ocr_lab/predictions.py` — формат JSONL (dataclasses `ScanPrediction`, `RecordCandidate`),
    чтение/запись с валидацией, сборка записи из top-1 подполей (`record_from_fields`,
    `best_record`), константы `ROWS`, `PARTS`, `SUBFIELDS`;
  - `ocr_lab/evaluate.py` — загрузчик истины `load_truth` (единственное место, читающее
    manifest.csv), метрики подполей/меток времени/ваучеров/денег, автоприём
    (`coverage_precision_curve`, `threshold_for_precision`, интервал Уилсона), отчёт
    `report.md` + `metrics.json`, CLI с режимом `--write-perfect`;
  - минимальные `ocr_lab/__init__.py` и `ocr_lab/paths.py` по контракту T02 (T02 их дополнит);
  - `tests/ocr/__init__.py`, `tests/ocr/test_evaluate.py` — 30 тестов на синтетике;
  - `.gitignore`: добавлен `data/ocr/`.
- Цифры: на синтетическом манифесте «идеальные» предсказания (из полей и из records) дают
  100 % по всем метрикам, NLL = 0, ECE = 0, Δ минут = 0. Проверки: ruff, mypy app — чисто,
  pytest — 118 passed. Прогона на реальных данных нет: в облаке нет manifest.csv (T02 не
  выполнена) — поэтому статус «частично».
- Решения без человека:
  - `paths.py` сделан раньше T02 (задача зависит от T02, которой ещё нет); API: `REPO_ROOT`,
    `DATASET_CSV` (`OCR_DATASET_CSV`), `WORK_DIR` (`OCR_WORK_DIR`), `*_DIR` для `SUBDIRS`,
    `MANIFEST`, `CORRECTIONS_CSV`, `PRINTED_FLAGS_CSV`, `ensure_dir`, `work_subdir`,
    `configure_model_caches` (кэш по умолчанию — `OCR_CACHE_DIR` или `WORK_DIR/cache`;
    путь на E: из CONTEXT.md пока неизвестен).
  - Год строки — из `<row>_year` манифеста (иначе `year`): распознаватели год не читают.
  - Критерий верности для автоприёма — «все 4 строки верны» (номер не входит).
    Покрытие считается от всех ваучеров сплита. Порог — максимальное покрытие при нижней
    границе Уилсона ≥ цели; если такого нет — по точечной оценке с `confirmed=false` и
    пояснением. Для 99,5 % нужно ≥ 765 принятых ваучеров без ошибок, поэтому на val/test
    (~сотни сканов) цель в принципе не подтверждается — отчёт пишет это прямо.
  - NLL: если истины нет среди top-k, вероятность берётся 1e-6. ECE — 10 равных корзин
    по top-1 вероятности. Непредсказанные подполя в NLL/ECE не входят (они видны в покрытии).
  - Скан истины без предсказания считается непокрытым и неверным во всех метриках.
  - `printed_flags.csv` ожидается с колонками `scan_id, field, printed`; `field` — подполе,
    строка бланка (на все её части) или `voucher_number`. Без файла срез = `unknown`.
  - Деньги: `minutes_between` и `_is_night` из `app/services/calculation.py` (без сети).
    Отрицательный интервал обрезается до 0, как в приложении.
  - `data/ocr/` добавлен в `.gitignore`, чтобы производные данные не попали в git.
  - Зависимостей не добавлено (только stdlib + app). В облаке пришлось поставить
    `requirements-dev.txt` через pip: окружение было пустым.
- Вопросы к человеку:
  - выполнить на ПК после T02: `python -m ocr_lab.evaluate --write-perfect data/ocr/reports/perfect_val.jsonl --split val`
    и `python -m ocr_lab.evaluate --pred data/ocr/reports/perfect_val.jsonl --split val` —
    ожидается 100 % на реальном манифесте; после этого статус можно считать «готово».
- Для следующих задач:
  - JSONL — одна строка на скан: `{"scan_id", "source", "fields": {"<row>.<part>" |
    "voucher_number": [[value:int, p:float|null], ...] по убыванию p}, "records": [{"left_base":
    ISO|null, ..., "hour24": [rows], "voucher_number": int|null, "p": float|null}],
    "confidence", "margin"}`. `24:00` дня D → `00:00` дня D+1 и строка в `hour24`.
    Писать через `ocr_lab.predictions.write_predictions`.
  - Оценка: `python -m ocr_lab.evaluate --pred <file.jsonl> --split val [--out <dir>]
    [--manifest ...] [--printed-flags ...] [--target 0.995]`; отчёт по умолчанию
    `data/ocr/reports/eval_<source>_<split>/{report.md,metrics.json}`.
  - Если T02 назовёт колонки манифеста иначе, править только `load_truth`.
