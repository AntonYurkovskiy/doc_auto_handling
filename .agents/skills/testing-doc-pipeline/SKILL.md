---
name: testing-doc-pipeline
description: End-to-end test the doc_auto_handling pipeline (заявка → ваучер → сопоставление → расчёт → экспорт). Use when verifying UI/calculation changes in this repo.
---

# Testing the doc_auto_handling pipeline

Local FastAPI + SQLite app. Golden path: upload заявка → create voucher → match → calculate → export.

## Setup
```bash
cd <repo> && python3 -m venv .venv && . .venv/bin/activate && pip install -r requirements-dev.txt
ruff check . && mypy app && pytest -q     # all should be green
rm -f data/app.db*                         # start from a clean DB
uvicorn app.main:app --host 127.0.0.1 --port 8000 --log-level warning &
```
App at http://127.0.0.1:8000. Maximize Chrome before recording:
`wmctrl -r :ACTIVE: -b add,maximized_vert,maximized_horz`.

## Pre-load a заявка (setup, before recording)
Uploading via the browser file picker is painful. Instead seed it via curl so the
parsed fields are already visible in the UI:
```bash
curl -F "file=@/path/to/sample.eml" http://127.0.0.1:8000/applications/upload
```
Sample `.eml` files come from the user's attachments (e.g. WL LADOGA → GRT 24199).

## Cyrillic input — IMPORTANT
The `computer` tool's `type` action does NOT enter Cyrillic (keyboard layout has no
Cyrillic keys) — the field stays empty. Workaround that works: click the field with
the `computer` tool to focus it, then type via shell:
`DISPLAY=:0 xdotool type --delay 80 --clearmodifiers "Отшвартовка"`.
Verify with `read_dom` afterwards; retype if a char dropped. Latin/digits type fine
through the normal `type` action. `xclip` is not installed.

## datetime-local fields are finicky
The year segment can accept 6 digits and get corrupted (e.g. `202605`). Fix by
clicking the year segment directly and typing exactly 4 digits, and set the time by
clicking the hour segment and typing `HHMM` + `AM/PM`. Zoom to verify each segment.
Note: per_ton work types (швартовка/отшвартовка) ignore times for the amount, so
imperfect times don't affect that assertion — but calc still needs a non-null finished
date for the CBR rate lookup.

## Key calculation to assert
- per_ton (GRT ≥2000): `amount = GRT × rate / tug_count`. `tug_count` comes from the
  voucher «Совместно с …» field: empty → 1 (no division), 1 tug → /2, 2 tugs → /3.
  Example: GRT 24199, rate 0.50, «Совместно с»=БК Пионер → 24199×0.50/2 = **6049.75 USD**.
  Note string contains `/ 2 букс.`. A broken divisor would show 12099.50.
- Ледовые условия: manual checkbox on the voucher; note shows `(лёд)`. rate_ice is a
  placeholder `None`, so ice may not change the amount unless real ice rates are set.
- Часы сопровождения: adds `escort_rate × hours` on top — leave empty when asserting a
  clean per_ton amount.
- Revenue: `amount × cbr_rate` (live cbr.ru; RUB → 1.0). Needs network to cbr.ru and
  isdayoff.ru — verify reachability first, since calc raises on network failure.

## Export check
Download CSV from /works (button «Экспорт CSV»), then read the file:
`iconv -f UTF-8 ~/Downloads/works.csv`. Columns follow matching.xls.

## Devin Secrets Needed
None. Only outbound access to cbr.ru and isdayoff.ru.
