# Контекст для задач OCR

Факты о локальной машине, данных и коде. Файл заполнен в T00 (2026-10-01,
рабочая копия на коммите `425f61c`, ветка `ocr/handwritten-dates`). Реальных
записей здесь нет: только форматы, счётчики и пути.

## 1. Репозиторий

**Git.** Ветка `ocr/handwritten-dates`, `git status --short` — чисто (нет ни
изменённых, ни неотслеживаемых файлов кода; неотслеживаемое — только служебные
каталоги из `.gitignore`: `.venv/`, `*_cache/`, `__pycache__/`, `data/app.db*`,
`data/incoming/**`, `data/ocr/`). `git diff --stat origin/main` — **84 файла**,
+12965/−91 строк: вся ветка целиком расходится с `origin/main` (там ветки с OCR
ещё не было). Основные группы: `app/ocr/` (новый пакет — `align.py`, `decoder.py`),
правки `app/services/voucher*.py` и `app/config.py`, `app/main.py`; новый пакет
`ocr_lab/` (`__init__.py`, `evaluate.py`, `fit_decoder_priors.py`, `predictions.py`);
`tests/ocr/` целиком новый; весь `docs/ocr_tasks/` (этот план, промпты T00–T29,
`run_task.py`) и `docs/ocr_dates_plan.md`, `ocr_dates_guide.md`, `docs/base_logic.md`,
`docs/class_vs_ocr.md`; `voucher_names.csv` (1557 строк, маппинг имя файла → буксир).
Документы плана **все отслеживаются git** (`git ls-files` подтверждает: `docs/base_logic.md`,
`docs/class_vs_ocr.md`, `docs/ocr_dates_plan.md`, `ocr_dates_guide.md`, весь
`docs/ocr_tasks/**` кроме реальных данных) — секретов/данных среди них нет, это
документация и код.

- Монолит FastAPI + SQLite (WAL), без Alembic. Новые колонки добавляются через
  `_add_missing_columns` в `app/database.py`.
- Ваучеры:
  - `app/services/voucher.py` — приоры и кандидаты, включая сетку `candidate_datetimes`
    ±3 дня (`DATE_WINDOW_DAYS=3`) с шагом 10 минут (`MINUTE_STEP=10`);
  - `voucher_fields.py` — поля карточки и подтверждение;
  - `voucher_template.py` — один шаблон `baltiyskie_buksiry`, 12 строковых регионов
    по бланку 243k (Коммунар); координаты совпадают с `ocr_regions.txt` построчно
    (индекс файла + 1 = `order` региона);
  - `voucher_number.py` — `parse_voucher_name`, `parse_voucher_number`, `predict_next`
    (кириллица и дубли номеров уже разобраны в T01, см. PROGRESS.md);
  - `voucher_linking.py`, `voucher_files.py`.
- Модели БД (`app/models.py`): `Voucher`, `VoucherFieldPrediction` (`predicted_value`,
  `predicted_normalized_value`, `confidence`, `source`, `confirmed_value`,
  `confirmed_at`; колонки top-k нет — кандидаты только в памяти `FieldPrediction`),
  `VoucherTemplate`, `VoucherRegion` (`center_x/center_y/width/height`, нормализованные).
- Проверки: `ruff check .`, `mypy app`, `pytest` (см. README).
- `app/services/voucher_ocr.py`, `app/services/voucher_trocr.py` и
  `data/historical/predict_ocr/ocr_regions.txt` в `origin/main` до сих пор нет
  (`git diff --stat origin/main` показывает их как добавленные в этой ветке) —
  они живут только здесь и попадут в main вместе с этой веткой.

## 2. Текущий OCR-код (проверено в рабочей копии, коммит `425f61c`)

- `app/services/voucher_ocr.py`:
  - `load_voucher_image(voucher: Voucher, *, dpi: int = 300) -> Image.Image | None` —
    картинка или первая страница PDF через `pypdfium2` (`RENDER_DPI=300`), RGB;
  - `crop_region(image: Image.Image, region: VoucherRegion) -> Image.Image` — кроп по
    нормализованным `center_x/center_y/width/height`, **строка целиком**, с клампом
    координат к границам картинки;
  - `ocr_image(image: Image.Image, lang: str = "rus+eng") -> tuple[str | None, float | None]` —
    Tesseract (`pytesseract.image_to_data`, откат на `image_to_string`), препроцесс —
    grayscale + контраст ×2 (`ImageEnhance.Contrast`, коэффициент 2.0); путь к бинарнику —
    `settings.tesseract_cmd`; `_MIN_CONFIDENCE = 0.3`;
  - `ocr_voucher_regions(voucher: Voucher) -> dict[str, tuple[str | None, float | None]]` —
    для `HANDWRITTEN_FIELDS = frozenset({"left_base", "arrived_base", "started_work",
    "finished_work"})` сначала TrOCR (`trocr_image`), при `(None, None)` — Tesseract;
    остальные регионы (`tugboat`, `vessel`, `agent`, `work_type`, `voucher_number`,
    `remarks`, `joint_with_line_1/2`) — сразу Tesseract.
- `app/services/voucher_trocr.py`:
  - модель `kazars24/trocr-base-handwritten-ru` (`VisionEncoderDecoderModel` +
    `TrOCRProcessor`), ленивая загрузка с `threading.Lock`, кэш на уровне модуля
    (`_processor`, `_model`, `_device`, `_load_failed`);
  - `trocr_image(image: Image.Image) -> tuple[str | None, float | None]` — при
    `settings.trocr_enabled=False` сразу `(None, None)`; иначе `generate()` с
    `max_new_tokens=settings.trocr_max_new_tokens`, `output_scores=True`; confidence —
    средняя `softmax(...).max()` по шагам генерации (`_sequence_confidence`);
    устройство — `settings.trocr_device` или авто `cuda`/`cpu`;
  - процессор ресайзит кроп строки в 384×384 — источник проблемы из плана (вся строка
    сжимается в квадрат).
  - **Дефект из плана T00 уже устранён в рабочей копии.** `app/config.py` содержит
    все четыре поля `Settings`: `trocr_enabled: bool = True`,
    `trocr_model: str = "kazars24/trocr-base-handwritten-ru"`, `trocr_device: str = ""`,
    `trocr_max_new_tokens: int = 32` — ровно значения по умолчанию из промпта T00.
    Тест `tests/test_voucher_ocr.py::test_trocr_image_returns_none_when_disabled`
    уже проверяет `trocr_image` при `trocr_enabled=False`. Правка внесена коммитом
    `5c5ebdc` («Handwritten voucher fields: TrOCR config, …», тот же коммит добавил
    `trocr_image`/`ocr_voucher_regions`). Действие по п. 7 промпта не требуется,
    код T00 не менялся.
- `app/services/voucher.py`:
  - `predict_fields(voucher, application, history, ocr_values=None) -> list[FieldPrediction]`,
    `predict_and_store(db, voucher, application=None, history=None, ocr_values=None)
    -> list[VoucherFieldPrediction]` — если `ocr_values` не передан, вызывает
    `ocr_voucher_regions`. Порядок источников в `predict_fields`: детерминированное
    значение (`_trusted_value`: поле ваучера → имя файла → заявка) → OCR
    (`_ocr_predicted_value`, для дат — `_parse_datetime`) → приор
    (`_preferred_value`). `source` ∈ `{prior, filename, application, ocr}` (константы
    `PREDICTION_SOURCE_*`). Сетка `candidate_datetimes` (±3 дня, шаг 10 мин) всё ещё
    используется как источник кандидатов для полей дат/времени, когда нет OCR/trusted
    значения.
  - `DATETIME_FORMAT = "%Y-%m-%d %H:%M"`, `DATE_WINDOW_DAYS = 3`, `MINUTE_STEP = 10`.
  - `_application_datetime(application) -> datetime | None` = `entry_datetime or
    exit_datetime or received_at` поля `Application`.
- `app/services/voucher_fields.py`: `apply_predictions_to_voucher(db, voucher) -> None`
  переносит `predicted_value` в поля ваучера; `confirm_fields(db, voucher, confirmed_at)`
  — обратное (значения формы → `confirmed_value`); `VOUCHER_FIELDS` — 12 полей,
  имена совпадают с `VoucherRegion.name`.
- `app/main.py`: `POST /vouchers/{voucher_id}/recognize` — создаёт шаблон при
  необходимости и вызывает `predict_and_store(db, item)` (строка ~477).
- Тесты: `tests/test_voucher_ocr.py` (20 тестов: crop, Tesseract, TrOCR-фоллбек,
  `predict_fields` с разными источниками, `apply_predictions_to_voucher`). Tesseract
  и TrOCR подменены через `monkeypatch`, реальные веса не грузятся. `tests/conftest.py`
  принудительно выставляет `APP_TROCR_ENABLED=false` до импорта `app.config`, чтобы
  pytest не скачивал веса (~1,4 ГБ) и не грузил torch.
- Зависимости: в `requirements.txt` — `pytesseract==0.3.13`, `torch==2.5.1`,
  `transformers==4.46.3`, `numpy==2.4.6`, `opencv-python-headless==5.0.0.93`.
  `torchvision` и `onnxruntime` не объявлены нигде (ещё не нужны: `onnxruntime`
  появится в T18). `requirements-train.txt` **не существует** — пока нет задачи,
  которая развела бы рантайм и обучение по файлам (только `_common.md` упоминает
  такое разделение как конвенцию).
- `data/historical/predict_ocr/ocr_regions.txt` в git не попал (см. п. 3 и `.gitignore`).

## 3. Данные

- **Истина лежит не там, где её ждёт код.** Репозиторий — `E:\projects\doc_auto_handling\doc_auto_handling`,
  а `data/historical/` физически находится на уровень выше:
  `E:\projects\doc_auto_handling\data\historical` (955 МБ). Внутри репозитория
  каталога `data/historical` нет вообще. **Важно для T02**: `ocr_lab/paths.py` по
  умолчанию берёт `DATASET_CSV = REPO_ROOT/data/historical/analysis/reconciled_dataset.csv`
  — этот путь **не существует** без дополнительных действий. Нужно либо выставлять
  `OCR_DATASET_CSV=E:/projects/doc_auto_handling/data/historical/analysis/reconciled_dataset.csv`
  в окружении/`.env`, либо сделать junction
  `mklink /J "data\historical" "..\data\historical"` внутри репозитория. Пути
  `voucher_scan_path` внутри самого CSV — уже абсолютные (`E:\projects\doc_auto_handling\
  data\historical\vauchers\...`) и резолвятся независимо от того, где лежит сам CSV.
  `git check-ignore -v data/historical` **до правки** не находил правила (путь не
  игнорировался бы, появись он внутри репозитория) — `git check-ignore -v data/ocr`
  уже был покрыт (`.gitignore:21`). Добавил `data/historical/` в `.gitignore` на
  случай локальной копии/junction внутри репозитория (см. раздел 6 этого файла и
  п. 8 плана ниже).
- Дерево `data/historical/` (верхний уровень): `analysis/` (`reconciled_dataset.csv`,
  `reconcile_report.md`), `orders/` (несколько подпапок с выгрузками почты),
  `orders_index.csv`, `predict_ocr/` (разметка боксов + `vauchers/` с частью PDF),
  `vauchers/2025/`, `vauchers/2026/` (сканы по годам), `voucher_review_sample_*.zip`
  (несколько архивов ревью), `*.xls` выгрузка, `.odt` с правилами номеров.
- `reconciled_dataset.csv`:
  - путь: `E:\projects\doc_auto_handling\data\historical\analysis\reconciled_dataset.csv`;
  - кодировка: UTF-8 с BOM (`\xef\xbb\xbf`);
  - всего строк: **3255**, колонок: **55**. Это сырая сверка писем+ваучеров+выгрузки,
    а не финальный манифест: строк со `voucher_scan_path` (т.е. с привязанным сканом)
    — **779** на **774** уникальных файла (4 файла встречаются по 2–3 раза в строках —
    видимо, повторные/связанные работы). Это ровно совпадает с цифрами плана
    (774 скана, 779 строк).
  - `reconcile_report.md` поясняет происхождение 3255: строк выгрузки всего 3255,
    ваучеров в файловом индексе 780, связано по имени+году 774; остальные ~2476
    строк выгрузки — без скана (письма/заявки без ваучера и наоборот). Для OCR-плана
    нужны именно те 779 строк, где `voucher_scan_path` заполнен.
  - заполненность (среди всех 3255 строк, не только с сканом): ключевые колонки
    выгрузки (`tug`, `vessel`, `work_type`, `agent`, `voucher_number`,
    `base_departure`, `base_arrival`, `work_start`, `work_end`, `occupied_raw`,
    `work_duration_raw`, `grt_raw`, `amount`, `currency`, `exchange_rate`,
    `revenue_rub`, `base_year`) — 100%; `voucher_scan_path` — 23,9% (779/3255);
    колонки писем (`email_path`, `subject`, `from`, `date`, `vessel_eml`, `date_raw`,
    `time_raw`, `berth`, `imo`, `grt`, `nrt`, `cargo`, …) — ≈30,2–30,6%;
    `report`/`edit`/`delete`/`security_level` — 0% (служебные, не используются);
    `purpose` — 6,6%. Среди 779 строк со сканом у всех колонок выгрузки
    (`tug`…`revenue_rub`) заполненность 100%, кроме `application_file` (99,9%),
    `voucher_file` (99,8%), `voucher_name_key`/`voucher_key` (99,8%), `order_key`
    (99,9%).
  - формат колонок даты/времени (среди 779 строк со сканом), без самих значений:
    - `base_departure`, `base_arrival`, `work_start`: `ДД.ММ.ГГГГ ЧЧ:ММ` почти везде
      (776, 778, 778 из 779), у малого числа строк — только `ДД.ММ.ГГГГ` без времени
      (3, 1, 1 соответственно — видимо, время не найдено при извлечении);
    - `work_end`: `ДД.ММ.ГГГГ ЧЧ:ММ` у всех 779 строк;
    - `occupied_raw`, `work_duration_raw`: `ЧЧ:ММ` у всех 779 (посчитанные интервалы);
    - **полночь**: подстрок `24:` или `00:00` в `base_departure/base_arrival/
      work_start/work_end` среди 779 строк **нет ни одной** — согласуется с планом
      («в истине нет ни одного 00:00»), но и `24:00` в самой выгрузке тоже не
      встретилось (вопрос «как пишут на бланке» остаётся открытым до визуальной
      проверки сканов, выгрузка просто не содержит такого формата).
  - колонка времени из заявки (приор декодера): `date_raw` (`ДД.ММ` у 495/779,
    `Д.ММ` у 160/779 — без года, год берётся из `base_year`) + `time_raw`
    (`ЧЧ:ММ`, заполнено у 655/779 = 84%). Это сырой аналог `entry_datetime` /
    `exit_datetime` / `received_at` из `Application` (см. `_application_datetime`
    в `voucher.py`), посчитанный при сверке писем.
  - уникальные значения `tug` (779 строк со сканом): `БК Коммунар` (399),
    `БК Пионер` (378), `МБ Лигер` (2, третий буксир из плана). Среди всех 3255
    строк встречаются ещё `Test` и `-` (строки без привязанного скана — не входят
    в выборку OCR).
  - распределение по годам (`base_year`, 779 строк со сканом): 2025 — 466, 2026 — 313.
    Других лет нет.
- `voucher_scan_path` (779 строк со сканом):
  - все 774 уникальных пути существуют на диске (774/774);
  - единый корень: `E:\projects\doc_auto_handling\data\historical\vauchers\`
    (поддиректории `2025\`, `2026\`);
  - расширение файла — только `.pdf` (774/774, других форматов нет);
  - на случайной выборке 30 файлов (seed=42): все 30 — однострочные PDF с 1
    страницей, размер страницы 595×842 pt (A4), что при рендере приложением
    (`RENDER_DPI=300`, как в `load_voucher_image`) даёт растр **2479×3508 px**.
- Формат `ocr_regions.txt` (`E:\projects\doc_auto_handling\data\historical\predict_ocr\ocr_regions.txt`,
  12 строк + 1 строка-комментарий, целиком):
  ```
  YOLO формат, как я понимаю для сопоставления с файлом labels.txt нужно к индексу прибавить 1

  0 0.577564 0.313522 0.646226 0.039623
  1 0.586020 0.360063 0.656017 0.037107
  2 0.551305 0.397484 0.789535 0.041509
  3 0.529052 0.435220 0.794876 0.042767
  5 0.553530 0.487421 0.532291 0.040252
  6 0.556646 0.524214 0.534961 0.037107
  8 0.553976 0.599371 0.534961 0.038994
  7 0.556646 0.560377 0.536742 0.035220
  10 0.544629 0.643082 0.712095 0.048428
  9 0.657674 0.677044 0.516269 0.044654
  11 0.496118 0.722327 0.899910 0.042138
  4 0.570888 0.261635 0.141529 0.046541
  ```
  Формат YOLO: `class_id center_x center_y width height`, нормализовано на размер
  картинки `243k.jpg` (лежит рядом). `class_id + 1` — порядок (`order`) региона в
  `DEFAULT_VOUCHER_REGIONS` (`voucher_template.py`); координаты в коде совпадают с
  этим файлом поточечно. Рядом — `labels.txt` и архив разметки
  `labels_ocr_regions_2026-08-07-02-15-56.zip`, и папка `predict_ocr/vauchers/` с
  частью PDF (подвыборка для разметки боксов, не весь датасет).

## 4. Окружение

- Windows 10 Home, сборка `10.0.19045.6456` (совпадает с системным окружением).
- Python в `.venv`: **3.11.9**.
- Каталога `.venv-train` нет — T05 (GPU/окружение для обучения) ещё не выполнена.
- Ключевые пакеты в `.venv` (`pip list`):
  - `torch` **2.5.1+cpu** — сборка **без CUDA** (`torch.cuda.is_available()=False`,
    `torch.version.cuda=None`). Это расходится с ожиданием из плана «поставить
    сборку torch с CUDA, в которой есть sm_61» — такая сборка пока не поставлена
    (это задача T05);
  - `torchvision` — **не установлен** (`ModuleNotFoundError`);
  - `transformers` 4.46.3;
  - `numpy` 2.4.6;
  - `opencv-python-headless` (`cv2`) — **не установлен**, хотя объявлен в
    `requirements.txt==5.0.0.93`. Из-за этого падает сбор тестов `tests/ocr/test_align.py`
    (см. раздел 5);
  - `pypdfium2` 5.12.1;
  - `pdfplumber` 0.11.4;
  - `Pillow` 12.3.0;
  - `onnxruntime` — не установлен (ещё не требуется, появится в T18);
  - `pandas` 2.2.3;
  - `pytesseract` 0.3.13.
  - Вывод: `.venv` отстаёт от `requirements.txt` (нет `opencv-python-headless`),
    нужен `pip install -r requirements.txt` перед задачами, которые используют
    `app/ocr/align.py` (T06+ тесты) локально.
- `nvidia-smi`: драйвер **446.14**, **CUDA 11.0** (в заголовке драйвера), карта
  **GeForce GTX 1050, 4096 MiB**, сейчас используется 77 MiB, WDDM, Pascal (sm_61),
  подтверждает цифры плана.
- Свободное место: **C: — 1,0 ГБ свободно** из 127,4 ГБ (план ожидал ≈13,5 ГБ —
  **существенное расхождение, см. раздел 7**); **E: — 255,3 ГБ** свободно из
  1000,2 ГБ (репозиторий и исторические данные — на E:, это хорошо).
- Кэш HF: `HF_HOME` не задан (используется каталог по умолчанию
  `C:\Users\User\.cache\huggingface`, ≈619 МБ: `trocr-base-handwritten-ru` и
  `faster-whisper-small`). Учитывая 1 ГБ свободного места на C:, перенос
  `HF_HOME`/`TORCH_HOME` на E: (через `OCR_CACHE_DIR`, уже поддержано
  `ocr_lab/paths.configure_model_caches`) нужен раньше, чем планировалось — любая
  новая загрузка весов рискует не поместиться на C:.
- `claude --version` → **2.1.284 (Claude Code)** — условие H0 (≥ 2.1.280) выполнено.
- `devin --version` → **devin 3000.10.48 (fcf7ba39)** — установлен.

## 5. Базовые проверки

- `ruff check .` → **чисто** (`All checks passed!`).
- `mypy app` → **2 ошибки, не связанные с OCR** (были уже до T00, не чинить):
  - `app/services/imap_ingest.py:236` — `Argument 1 to "store" of "IMAP4" has
    incompatible type "bytes"; expected "str"`;
  - `app/main.py:254` — `Incompatible types in assignment (expression has type
    "list[Application]", variable has type "list[Voucher]")`.
- `pytest -q` → **падает сбор** `tests/ocr/test_align.py` (`ModuleNotFoundError:
  No module named 'cv2'`, т.к. `opencv-python-headless` не установлен в `.venv`,
  см. раздел 4). С `--continue-on-collection-errors`: **188 passed**, 1 ошибка
  сбора, 13 предупреждений (deprecation у `@app.on_event` и `TemplateResponse` —
  не блокируют). Чтобы прогнать `tests/ocr/test_align.py`, нужно сначала
  `pip install -r requirements.txt` (или хотя бы `opencv-python-headless`).

## 6. Пути и команды

- Производные артефакты — `data/ocr/` (внутри репозитория, в `.gitignore`).
  Пути в коде — через `ocr_lab/paths.py` (`REPO_ROOT`, `DATASET_CSV`, `WORK_DIR`,
  `CACHE_DIR`, `SUBDIRS`, `MANIFEST`, `CORRECTIONS_CSV`, `PRINTED_FLAGS_CSV`,
  `ensure_dir`, `work_subdir`, `configure_model_caches`).
- **Истина не на дефолтном пути** (см. раздел 3) — перед любой командой `ocr_lab`,
  которая читает `DATASET_CSV`, выставляй:
  ```bash
  export OCR_DATASET_CSV="E:/projects/doc_auto_handling/data/historical/analysis/reconciled_dataset.csv"
  ```
  (или создай junction `data\historical` → `..\data\historical` средствами Windows).
- Установка/синхронизация зависимостей рантайма (нужно хотя бы раз, не установлено
  сейчас):
  ```bash
  .venv/Scripts/python -m pip install -r requirements.txt
  ```
- Базовые проверки (из `.venv`, с `PYTHONUTF8=1`, раннер выставляет сам):
  ```bash
  .venv/Scripts/python -m ruff check .
  .venv/Scripts/python -m mypy app
  .venv/Scripts/python -m pytest -q
  ```
- Запуск задачи плана: `python docs/ocr_tasks/run_task.py T02`. Статус плана:
  `python docs/ocr_tasks/run_task.py --status`.
- `.venv-train/Scripts/python` появится после T05 — пока используем `.venv`
  (torch там CPU-only).

## 7. Расхождения с планом

Сверка с `docs/ocr_dates_plan.md` (774 скана / 779 строк, 2 ваучера третьего
буксира, GTX 1050, 13,5 ГБ на C:):

- **774 скана, 779 строк, 2 ваучера МБ Лигер** — подтверждено точно (раздел 3).
- **GTX 1050 4 ГБ, Pascal/sm_61** — подтверждено (`nvidia-smi`, раздел 4).
- **13,5 ГБ свободно на C:** — **не подтвердилось**: сейчас свободен **1,0 ГБ**.
  Это критично для T05 (установка torch с CUDA, десятки-сотни МБ — ГБ) и для
  любых новых HF-загрузок. Решение без человека: перенос кэшей на E: теперь не
  рекомендация, а необходимое условие перед T05; подробности — в разделе 4.
- **`data/historical/` внутри репозитория** — план и `_common.md` описывают его
  как подкаталог репозитория (`data/historical/**`), но физически он находится
  на уровень выше (`E:\projects\doc_auto_handling\data\historical`), вне
  `E:\projects\doc_auto_handling\doc_auto_handling`. `ocr_lab/paths.py` (уже
  написан в T02/T13 заранее) по умолчанию его не найдёт — нужен `OCR_DATASET_CSV`
  или junction (раздел 3, 6). Это не было явно оговорено в плане/`_common.md`.
- **torch с CUDA (sm_61) ещё не поставлен** — в `.venv` стоит `torch==2.5.1+cpu`.
  Это ожидаемо (ставит T05), просто фиксирую фактическое состояние на момент T00.
- **Дефект `trocr_enabled`/`trocr_model`/`trocr_device`/`trocr_max_new_tokens`,
  описанный в исходном `CONTEXT.md` (п. 7 промпта T00)** — уже исправлен в рабочей
  копии коммитом `5c5ebdc`, до начала этой сессии. Правка по п. 7 промпта не
  потребовалась.
- **`opencv-python-headless` объявлен в `requirements.txt`, но не установлен в
  `.venv`** — ломает сбор `tests/ocr/test_align.py`. План этого не упоминает
  явно, но это нужно для T06+ локально (раздел 4–5).
- **`requirements-train.txt` не существует** — конвенция `_common.md` («обучение
  — в requirements-train.txt») ещё не применена, т.к. обучающих зависимостей пока
  не добавляли.
