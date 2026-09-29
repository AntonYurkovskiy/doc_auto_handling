---
id: T26
title: Встраивание в приложение и БД (top-k, уверенность, автоприём)
stage: 5
depends_on: T25
type: db-migration
complexity: medium
risk: reversible
needs_vision: false
needs_web: false
host: local
size: M
channel: claude_first
claude_model: claude-sonnet-5
claude_effort: high
devin_model: claude-sonnet-5-medium
review: devin:gpt-5-6-sol-high
human_checkpoint: —
max_turns: 130
optional: false
---

# T26. Интеграция в `voucher_ocr.py`, `voucher.py` и БД

## Зачем

План, «Этап 5»:
- в `voucher_ocr.py` цепочку «строка → TrOCR» заменить на новый пайплайн (T25);
- сетку `candidate_datetimes` убрать: её работу делает декодер;
- в `VoucherFieldPrediction` добавить колонку для top-k: сейчас кандидаты живут только
  в памяти.

## Что сделать

1. **Разберись, как сейчас устроено**: CONTEXT.md, раздел «Текущий OCR-код»,
   `app/services/voucher_ocr.py`, `app/services/voucher.py`, роуты `/vouchers/upload` и
   `/vouchers/{id}` в `app/main.py`. Прежде чем менять, запиши план правок в журнал.
2. **`voucher_ocr.py`** (сейчас: `ocr_voucher_regions` → TrOCR/Tesseract по строковым
   регионам, см. CONTEXT.md):
   - для 4 строк дат и `voucher_number` вызывать `app.ocr.pipeline.recognize_voucher`;
   - печатные поля (буксир, судно, агент, вид работ) оставить на Tesseract, как сейчас;
   - сохранить совместимость роута `POST /vouchers/{voucher_id}/recognize` и
     `apply_predictions_to_voucher`;
   - собрать `RecognitionContext` из ваучера, заявки и истории номеров (буксир, год) из БД;
   - пару — по заявке, если у неё есть ваучер другого буксира.
   - TrOCR убрать из рабочего пути. Если он остаётся для сравнения — только за флагом
     настроек, выключенным по умолчанию. Зависимости, которые больше не нужны рантайму
     (transformers и т.п.), перечисли в журнале. Удалять их из `requirements.txt` без
     подтверждения человека нельзя: запиши вопрос.
3. **`voucher.py`:**
   - для datetime-полей кандидаты и предсказания берутся из результата распознавания;
   - `candidate_datetimes`, `DATE_WINDOW_DAYS`, `MINUTE_STEP` удалить или оставить только
     как запасной вариант, если распознавание недоступно (`no_models`). Выбери и обоснуй.
     Обнови `tests/test_voucher_prediction.py`: он импортирует эти константы;
   - `predict_fields` уже принимает `ocr_values` и ставит `source="ocr"`; расширь этот путь
     под top-k и уверенность записи, а не пиши параллельный;
   - `source` предсказания: `ocr` или `prior`;
   - `confidence` — вероятность выбранного значения поля;
   - подтверждённые оператором значения не перезаписывать, как и сейчас.
4. **БД** (`app/models.py`, `app/database.py`):
   - `VoucherFieldPrediction.topk` — TEXT с JSON `[[значение, вероятность], ...]`;
   - уровень ваучера: `Voucher.ocr_confidence` (FLOAT) и `Voucher.ocr_status` (VARCHAR),
     или обоснованная альтернатива;
   - миграция через `_add_missing_columns`, как принято в проекте (Alembic нет);
   - проверь на копии реальной `data/app.db`: `cp` во временный файл,
     `APP_DATABASE_URL=sqlite:///...`. Оригинал не трогать.
5. **Автоприём.**
   - Флаг `APP_OCR_AUTO_CONFIRM` по умолчанию **выключен**: значения предзаполняются,
     ваучер помечается «высокая уверенность», но статус не меняется.
   - Включённый флаг — поведение из ответа человека на вопрос Q3 в `DECISIONS.md`.
     Если ответа нет, оставь выключенным и запиши вопрос.
6. **Синхронный вызов при загрузке** (< 2 с). Ошибка распознавания не должна ломать
   загрузку: тогда ваучер получает предсказания на приорах, как сейчас.
7. **Тесты:**
   - `tests/test_web_vouchers.py` и `tests/test_voucher_prediction.py` с подменённым
     пайплайном: загрузка → предсказания с `topk` → подтверждение;
   - миграция на старой схеме: создать таблицы без новых колонок и вызвать `init_db`.

## Критерии приёмки

- Полный `pytest` зелёный. Изменения старых тестов объяснены в журнале.
- Ручная проверка через `TestClient`: 3 реальных скана из `test` проходят загрузку,
  а карточка показывает предсказания. Скриншоты не нужны, достаточно текста в журнале.
- Миграция идемпотентна: второй `init_db` ничего не меняет.

## Отчёт

Журнал по шаблону. Коммит: `OCR T26: встраивание распознавания в приложение и БД`.
