---
id: T25
title: Рантайм-пайплайн распознавания для приложения
stage: 5
depends_on: T22
type: api-contract
complexity: medium
risk: safe
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
max_turns: 110
optional: false
---

# T25. `app/ocr/pipeline.py` — одна функция для приложения

## Зачем

План, «Этап 5»: цепочку «строка → TrOCR» заменить на
«вариант бланка → выравнивание → кропы → модель → декодер».
Инференс идёт на CPU внутри приложения, отдельный сервис не нужен. Все части уже есть
в `app/ocr/`. Здесь их собирают за одним контрактом, который вызовет приложение (T26).

## Что сделать

1. **Контракт**:
   ```python
   recognize_voucher(path, *, tug_hint=None, context: RecognitionContext) -> VoucherRecognition
   ```
   - **`RecognitionContext`:** год, время из заявки, вид работ, ожидаемый следующий номер
     или история номеров (буксир, год), подтверждённый или ожидающий ваучер партнёра
     для пары, имя файла.
   - **`VoucherRecognition`:**
     - вариант бланка и метрики выравнивания;
     - по каждому полю ваучера (4 datetime и номер) — top-3 с вероятностями;
     - top-N записей;
     - `confidence`, `margin`;
     - `auto_accept` — по порогу из настроек;
     - пути к сохранённым кропам подполей: для показа в UI (T27);
     - время по этапам;
     - статус: `ok`, `no_models`, `align_failed`, `error`, с понятным текстом.
2. **Настройки** в `app/config.py` (`Settings`), с префиксом `APP_`:
   - `ocr_models_dir` — по умолчанию `data/ocr/models/…_v1` из T22;
   - `ocr_layouts_dir` — эталоны, `data/ocr/layouts`;
   - `ocr_crops_dir` — кропы для UI, например `data/files/ocr_crops`;
   - `ocr_auto_accept_threshold`;
   - `ocr_enabled`.
   JSON-макеты боксов (`app/ocr/layouts/*.json`) лежат в пакете.
3. **Деградация без падений.** Нет моделей или эталонов → `status=no_models`, и
   приложение работает как сейчас, на приорах. Выравнивание не удалось → запасной путь
   (T11), если он есть, иначе `align_failed`.
4. **Производительность.**
   - Модели и эталоны загружаются один раз на процесс, с блокировкой.
   - Цель — не больше 2 с на ваучер на CPU от файла до результата. Замерь на 20 реальных
     сканах из `test`.
5. **Сохранение кропов** — внутри разрешённого каталога. Безопасные имена — как в
   `app/services/voucher_files.py`.
6. **Тесты** `tests/ocr/test_pipeline.py`, без реальных моделей:
   - синтетический бланк и подменённый runtime (фиктивные распределения) → проверка
     контракта;
   - нет моделей → `no_models`;
   - плохой файл → `error` без исключения наружу.

## Критерии приёмки

- `app/ocr/` не импортирует torch и transformers (проверь тестом или grep).
- Время на ваучер на реальных сканах записано в журнал.
- Проверки из `_common.md` зелёные.

## Отчёт

Журнал по шаблону. Коммит: `OCR T25: рантайм-пайплайн распознавания`.
