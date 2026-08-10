# Контракты следующего этапа ваучеров

## Цель

Сценарий пользователя: загрузить скан ваучера → увидеть оригинал и
предзаполненные поля → исправить при необходимости → подтвердить. Судозаходы,
операции и расчёт формируются внутренней логикой, а не отдельными формами.

## Модули

1. `voucher_ingest`
   - принимает PDF/JPG/PNG;
   - сохраняет оригинал и создаёт `Voucher(status=new|needs_review)`;
   - не изменяет исходный файл.

2. `voucher_prediction`
   - использует `VoucherTemplate`/`VoucherRegion`;
   - возвращает `list[VoucherFieldPrediction]`;
   - поля печатной части получают кандидатов из заявки/истории;
   - рукописные поля получают ограниченный набор кандидатов и confidence;
   - подтверждённое значение хранится отдельно от предсказания.

3. `voucher_review`
   - показывает оригинал;
   - редактирует значения полей;
   - при подтверждении записывает `confirmed_value`, `reviewed_at`,
     `Voucher.status=confirmed`.

4. `application_normalization`
   - сохраняет исходное HTML отдельно от нормализованных полей;
   - извлекает агента, даты входа/выхода, LOA и максимальную осадку нос/корма.

5. `operation_participants`
   - разделяет всех фактических участников операции и наши тарифицируемые
     буксиры;
   - хранит время работы каждого нашего буксира;
   - использует общее число участников для швартовки/отшвартовки;
   - считает перестановку отдельно по каждому нашему буксиру.

## Основные поля ваучера

`number`, `tug_id`, `vessel_name`, `agent`, `work_type`,
`left_base_dt`, `arrived_base_dt`, `started_dt`, `finished_dt`,
`joint_with`, `remarks`, `file_path`, `original_filename`, `content_type`,
`sha256`, `template_id`, `application_id`, `operation_id`, `status`.

## Регионы шаблона

Эталон `baltiyskie_buksiry`, версия `1`, координаты нормализованы в `0..1`.
Классы: `tugboat`, `vessel`, `agent`, `work_type`, `voucher_number`,
`left_base`, `arrived_base`, `started_work`, `finished_work`, `remarks`,
`joint_with_line_1`, `joint_with_line_2`.

## UI-контракты

- `POST /vouchers/upload` — multipart-файл, redirect на карточку ваучера.
- `GET /vouchers/{voucher_id}` — оригинал + поля + предсказания.
- `POST /vouchers` — сохранить ручные значения.
- `POST /vouchers/{voucher_id}/confirm` — подтвердить ваучер.
- `GET /files/{path}` — отдавать только файлы из разрешённого каталога.
