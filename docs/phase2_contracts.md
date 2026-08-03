# Фаза 2 — контракты модулей

Базовая ветка: `devin/1785313834-phase2-vessels-portcalls` (off `main`).
Фундамент (модуль D) уже в этой ветке: модели `Vessel` и `PortCall`, поле
`Application.portcall_id`, лёгкая миграция SQLite в `app/database.py`.

Каждый субагент ветвится ОТ этой ветки, редактирует ТОЛЬКО свои файлы, не трогает
чужие модули. Прогоняет локально `ruff check .` и `pytest` перед сдачей.

## Модели-фундамент (уже готовы, не менять)

`app/models.py`:

```python
class Vessel(Base):            # таблица vessels
    id: int
    name: str                  # unique
    imo: str | None            # unique
    flag: str | None
    loa_m: float | None
    beam_m: float | None
    draft_m: float | None
    grt: int | None
    nrt: int | None
    created_at: datetime
    portcalls: list[PortCall]

class PortCall(Base):          # таблица portcalls
    id: int
    status: DocStatus          # default new
    vessel_id: int | None      # FK vessels.id
    direction: Direction       # вход / выход / прочее
    agent: str | None
    eta: datetime | None       # приход/начало
    etd: datetime | None       # отход/конец
    berth_from: str | None
    berth_to: str | None
    purpose: str | None
    notes: str | None
    source: str | None         # history / email / manual
    created_at: datetime
    vessel: Vessel | None
    applications: list[Application]

# Application получил:
#   portcall_id: int | None  (FK portcalls.id)
#   portcall: PortCall | None
```

`Direction` — enum со значениями `вход`, `выход`, `прочее`.
`DocStatus` — enum: new, parsed, needs_review, confirmed, matched, calculated, done, error, cancelled.

БД-сессии: `from app.database import SessionLocal, get_db, init_db`.

## Модуль A — загрузчик истории

Файл: `scripts/load_history.py` (CLI) + при необходимости
`app/services/history_loader.py` (логика для переиспользования).

Вход: `reconciled_dataset.csv` (utf-8-sig), формируется `analysis/reconcile.py`.
Колонки датасета (важные для загрузки):

- `vessel` — имя судна (из строки выгрузки);
- поля из тела письма (могут быть пустыми): `vessel_name`, `imo`, `flag`,
  `loa_m`, `beam_m`, `draft_fore_m`, `draft_aft_m`, `grt`, `nrt`, `port_from`,
  `port_to`, `purpose`, `berth`, `direction`, `agent`;
- `work_type`, `agent`, `base_departure`, `base_arrival`, `work_start`,
  `work_end`, `application_file`, `voucher_file`, `voucher_scan_path`,
  `email_path`, `grt_raw`.

Задача загрузчика:
1. Upsert `Vessel` по (`imo` если есть, иначе `name`). Имя брать из `vessel_name`
   (тело письма) или `vessel`. Числа парсить безопасно (запятая→точка, пусто→None).
   `draft_m` = `draft_aft_m` или `draft_fore_m`. Заполнять только пустые поля
   (не затирать уже известные непустыми).
2. Создать `PortCall` на строку датасета: `vessel_id`, `direction` (маппинг
   вход/выход/прочее; пусто→прочее), `agent`, `eta`=`work_start`|`base_departure`,
   `etd`=`work_end`|`base_arrival`, `berth_from`/`berth_to` из `berth`/`port_from`/
   `port_to`, `purpose`=`purpose`|`work_type`, `source="history"`,
   `status=DocStatus.confirmed`.
3. Идемпотентность: повторный запуск не должен плодить дубли судов; для портколлов
   допускается ключ дедупа (vessel_id, eta, direction) — пропускать существующие.

CLI: `python scripts/load_history.py <reconciled_dataset.csv> [--db-url sqlite:///...]`.
Печатать сводку: создано/обновлено судов, создано портколлов, пропущено.

Тесты: `tests/test_history_loader.py` — на маленьком синтетическом CSV (создать в
тесте, НЕ коммитить реальные данные): проверить upsert судна, создание портколла,
идемпотентность второго запуска.

Контракт функции:
```python
def load_history(db: Session, rows: Iterable[dict]) -> dict[str, int]: ...
# возвращает {"vessels_created", "vessels_updated", "portcalls_created", "skipped"}
```

## Модуль C — предсказатель номера ваучера

Файл: `app/services/voucher_number.py` + `tests/test_voucher_number.py`.

Правило (подтверждено пользователем): нумерация ваучеров сбрасывается 1 января и
идёт отдельно по каждому буксиру. Номер = порядковый счётчик в пределах
(буксир, год). Имена файлов вида `100k.pdf` (k/p — код буксира), `262k(2).pdf` —
маркер копии `(N)` игнорировать.

Контракт:
```python
def predict_next(db: Session, tug_id: int, year: int) -> int: ...
# max существующего номера ваучера для (tug_id, year) + 1; если нет — 1

def parse_voucher_number(name: str) -> tuple[int | None, str | None]: ...
# "262k(2).pdf" -> (262, "k"); "100p.pdf" -> (100, "p")
```

Год ваучера определять по `Voucher.started_dt`/`left_base_dt` (год даты). Буксир —
по `Voucher.tug_id`. Учитывать только ваучеры с распознанным числовым номером.

Тесты: синтетические `Voucher` в in-memory/temp SQLite: несколько ваучеров одного
буксира за год → predict_next = max+1; другой год → счётчик независим; другой
буксир → независим; пустая история → 1; парсинг имён с `(N)` и кодами k/p.

НЕ трогать `app/main.py` и шаблоны — интеграцию в UI сделает модуль E/оркестратор.

## Модуль E — веб-UI vessels/portcalls

Файлы: правки `app/main.py` (только НОВЫЕ роуты в конце секций), новые шаблоны
`app/web/templates/vessels_list.html`, `vessel_form.html`, `portcalls_list.html`,
`portcall_form.html`, ссылки в `base.html` навигацию. Стиль — как в существующих
шаблонах (наследуют `base.html`, класс-стили из `static/style.css`).

Роуты (GET, чтение; POST для правки, как у applications/vouchers):
- `GET /vessels` — список судов (name, imo, flag, loa_m, grt, nrt, кол-во заходов).
- `GET /vessels/{id}` — карточка судна + его портколлы.
- `POST /vessels` — создать/править (форма).
- `GET /portcalls` — список судозаходов (судно, direction, eta, etd, agent, status).
- `GET /portcalls/{id}` — карточка захода + связанные заявки.
- `POST /portcalls` — создать/править.

Использовать существующие паттерны из `main.py`: `templates.TemplateResponse`,
`Depends(get_db)`, `_parse_form_dt`, `RedirectResponse(..., status_code=303)`.
Добавить пункты меню «Суда» и «Судозаходы» в `base.html`.

НЕ менять модели, расчёт, matching, парсеры. Только роуты чтения/правки и шаблоны.

## Общие правила

- Кириллица в строках — норма; файлы UTF-8.
- Не коммитить реальные данные (CSV/EML/PDF/сканы/*.db).
- Перед сдачей: `ruff check .` и `pytest` зелёные.
- НЕ открывать PR. Закоммитить и запушить свою ветку от базовой, сообщить имя
  ветки и итоговые интерфейсы. Интеграцию и PR делает оркестратор.
