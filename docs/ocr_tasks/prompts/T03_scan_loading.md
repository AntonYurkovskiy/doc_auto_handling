---
id: T03
title: Загрузка сканов и кэш страниц
stage: 0
depends_on: T02
type: api-contract
complexity: medium
risk: safe
needs_vision: false
needs_web: false
host: local
size: M
channel: claude_first
claude_model: claude-sonnet-5
claude_effort: medium
devin_model: swe-2-high
review: devin:gpt-5-6-sol-high
human_checkpoint: —
max_turns: 90
optional: false
---

# T03. Загрузка сканов (`app/ocr/io.py`) и кэш страниц

## Зачем

Сканы приходят как PDF из CamScanner или как картинки. Выравнивание, нарезка и
рантайм должны получать одно и то же: серое изображение `uint8` предсказуемого
масштаба. Загрузчик нужен и приложению, поэтому он живёт в `app/ocr/`, без torch.

## Что сделать

1. **`app/ocr/__init__.py` и `app/ocr/io.py`:**
   - `load_scan(path, *, dpi=200, page=0) -> np.ndarray` — серое `uint8`, форма HxW.
     - PDF рендерить через `pypdfium2`. Он уже стоит как зависимость `pdfplumber`,
       проверь это; poppler не нужен.
     - JPG, PNG, TIFF, WEBP, BMP открывать через Pillow и применять
       `ImageOps.exif_transpose`.
     - Многостраничный PDF: по умолчанию страница 0. Число страниц отдавай отдельной
       функцией `page_count(path)`.
   - `normalize_width(img, width=1654) -> np.ndarray` — 1654 px это ширина A4 при
     200 dpi. Пропорции сохранять, интерполяция `INTER_AREA` при уменьшении.
   - Ошибки — понятные исключения с именем файла, без содержимого.
2. **`ocr_lab/pages.py`**, запуск `python -m ocr_lab.pages build [--workers 4] [--limit N]`:
   - для каждой строки манифеста сохранить `data/ocr/pages/<scan_id>.png` — серое
     изображение, ширина 1654;
   - идемпотентность: готовые файлы пропускать;
   - параллельность через `multiprocessing` с защитой `if __name__ == "__main__":`.
     На Windows используется spawn, поэтому функции должны быть на верхнем уровне модуля;
   - отчёт `data/ocr/reports/pages_report.md`:
     - сколько сделано и пропущено, ошибки по `scan_id`;
     - распределение числа страниц;
     - исходные размеры и DPI;
     - доля альбомных страниц (ориентация);
     - время на страницу.
   - лист-превью `data/ocr/sheets/pages_sample.png`: 48 случайных страниц, миниатюры
     с подписью `scan_id`. Он нужен задаче T07, здесь смотреть его не обязательно.
3. **Зависимости.** В `requirements.txt` добавь с `==` только то, что реально импортирует
   `app/ocr/io.py`: `numpy`, `opencv-python-headless`, `pypdfium2`, `Pillow`. Версии возьми
   из `.venv`, если пакеты уже стоят (см. CONTEXT.md).
4. **Тесты** `tests/ocr/test_io.py`, только синтетика:
   - PDF из Pillow (`img.save(path, "PDF")`) — одна и две страницы;
   - JPEG с EXIF-поворотом;
   - PNG в оттенках серого и RGB;
   - несуществующий файл.

## Критерии приёмки

- `python -m ocr_lab.pages build` прошёл по всем сканам манифеста. Ошибки перечислены
  в отчёте и не роняют прогон.
- `app/ocr/io.py` не импортирует torch и transformers.
- Проверки из `_common.md` зелёные.

## Отчёт

Журнал по шаблону. В «Для следующих задач» — API `load_scan` и `normalize_width`, масштаб
страниц в кэше, заметные особенности сканов (повороты, поля, лишние страницы).
Коммит: `OCR T03: загрузка сканов и кэш страниц`.
