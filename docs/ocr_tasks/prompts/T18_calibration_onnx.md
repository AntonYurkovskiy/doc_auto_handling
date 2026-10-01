---
id: T18
title: Калибровка вероятностей и экспорт моделей в ONNX
stage: 2
depends_on: T16, T17
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
max_turns: 100
optional: false
---

# T18. Калибровка и ONNX для рантайма на CPU

## Зачем

1. Декодер складывает логарифмы вероятностей модели с приорами. Если модель
   самоуверенна, приоры не сработают. Нужна калибровка: temperature scaling на `val`.
2. Приложение работает на CPU без torch. Модели должны работать через `onnxruntime`,
   и результат должен совпадать с torch.

## Что сделать

1. **Temperature scaling** (`ocr_lab/calibrate.py`):
   - отдельная температура для каждой головы (`tens`, `units`, головы номера), подбирается
     на `val` по NLL (LBFGS по одному скаляру);
   - замерь ECE и NLL до и после;
   - сохрани диаграмму надёжности в PNG (`data/ocr/reports/calibration_*.png`);
   - `test` не трогать.
2. **Экспорт в ONNX** (`ocr_lab/export_onnx.py`) для `digits` и `number`:
   - opset 17;
   - динамическая ось батча;
   - температуры либо зашиты в граф, либо лежат в `meta.json` — выбери одно и опиши.
3. **Контракт рантайма** в `data/ocr/models/<name>_v0/meta.json`, рядом с `model.onnx`:
   - размер входа, фон, нормализация, порядок голов и классов;
   - температуры;
   - хэш коммита кода, хэш манифеста, дата обучения.
4. **`app/ocr/runtime.py`** — лёгкая обёртка над onnxruntime, без torch:
   - ленивая загрузка сессии один раз на процесс;
   - потокобезопасность;
   - `predict_digits(crops) -> dict[subfield, list[(value, prob)]]`;
   - `score_number_candidates(crop, candidates) -> dict[str, logp]`.
   Предобработку (вписывание, нормализацию) продублируй в numpy строго по `meta.json`.
5. **Паритет.** На 200 кропах `val` сравни torch и onnxruntime: максимальное расхождение
   вероятностей меньше 1e-4, совпадение top-1 — 100 %.
6. **Скорость на CPU:** один ваучер (16 кропов цифр и номер одним батчем) — цель < 150 мс.
7. **Откалиброванные предсказания.** Перегенерируй `val_predictions.jsonl` и
   `test_predictions.jsonl` для обеих моделей через onnxruntime с температурой. На них
   T19 подбирает веса декодера.
8. **Зависимости.** Добавь `onnxruntime` в `requirements.txt` (рантайм) с `==`.

## Критерии приёмки

- ECE после калибровки не хуже, чем до. Цифры в журнале.
- Паритет и время замерены и записаны в журнал.
- `app/ocr/runtime.py` покрыт тестом без реальных моделей: крошечную ONNX-модель собрать
  в тесте через `onnx.helper` или подменить сессию.
- Проверки из `_common.md` зелёные.

## Отчёт

Журнал по шаблону. Модели не коммитятся.
Коммит: `OCR T18: калибровка и экспорт в ONNX`.
