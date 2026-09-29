# Локальный пайплайн извлечения дат/времени из бланков (без API)

Гайд под вашу задачу: бланк с фиксированной вёрсткой (Left/Arrived home base, Started/Finished work), где день, часы и минуты — рукописные, часть текста — печатная. 700 размеченных документов, только локальные модели.

## 0. Ключевая рамка задачи

Это **не** задача общего "понимания документов" (как парсинг произвольных PDF или инвойсов). У вас:
- фиксированная вёрстка (один и тот же бланк);
- значения — почти всегда только цифры (день, часы, минуты), плюс, возможно, месяц/год;
- 700 документов дают, если резать по полям, тысячи обучающих примеров на *очень* узкую подзадачу.

Это резко сужает выбор моделей и снижает требования к "интеллекту" модели — задача ближе к распознаванию рукописных индексов/чеков, чем к document understanding общего вида.

Важная вводная из актуального (2026) состояния открытых OCR-моделей: даже лучшие открытые модели 2026 года — Surya, GOT-OCR 2.0, Baidu Unlimited-OCR, dots.ocr — рукописный текст "из коробки" распознают заметно хуже печатного; это называют главным нерешённым разрывом в open-source OCR. Значит, дообучение на ваших 700 документах — не опция "для качества", а обязательное условие, и я закладываю это в рекомендации ниже.

## 1. Рекомендуемая архитектура — два трека

### Трек A (обязательная база, вне зависимости от модели): выравнивание шаблона + кроп полей

Раз вёрстка фиксирована — выровняйте скан по референсному шаблону (гомография по ключевым точкам) и вырежьте фиксированные боксы полей (день/часы/минуты × 4 строки). Это не ML, а классическое CV, но именно это даёт наибольший прирост точности почти для любой модели дальше по пайплайну: рукописная цифра, вырезанная крупным планом, распознаётся сильно надёжнее, чем та же цифра, потерянная где-то в углу полностраничного скана после ресайза.

```python
# preprocess.py — выравнивание по ORB-фичам + гомография, затем кроп фиксированных боксов
import cv2
import numpy as np
import json
from pathlib import Path

REF_TEMPLATE_PATH = "template_reference.png"  # пустой бланк-эталон

# Заполняете один раз, посмотрев пиксельные координаты полей на эталоне
FIELD_BOXES = {
    "left_home_base_day":     (185, 40,  230, 75),
    "left_home_base_hour":    (430, 40,  475, 75),
    "left_home_base_min":     (560, 40,  610, 75),
    "arrived_home_base_day":  (185, 95,  230, 130),
    "arrived_home_base_hour": (430, 95,  475, 130),
    "arrived_home_base_min":  (560, 95,  610, 130),
    "started_work_day":       (185, 150, 230, 185),
    "started_work_hour":      (430, 150, 475, 185),
    "started_work_min":       (560, 150, 610, 185),
    "finished_work_day":      (185, 205, 230, 240),
    "finished_work_hour":     (430, 205, 475, 240),
    "finished_work_min":      (560, 205, 610, 240),
}

orb = cv2.ORB_create(4000)

def align_to_template(scan_gray, ref_gray):
    kp1, des1 = orb.detectAndCompute(ref_gray, None)
    kp2, des2 = orb.detectAndCompute(scan_gray, None)
    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des1, des2), key=lambda m: m.distance)
    good = matches[: max(50, int(len(matches) * 0.15))]
    if len(good) < 10:
        raise ValueError("Недостаточно совпадений для надёжного выравнивания")
    src = np.float32([kp1[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp2[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, mask = cv2.findHomography(dst, src, cv2.RANSAC, 5.0)
    inlier_ratio = float(mask.sum()) / len(mask) if mask is not None else 0.0
    if H is None or inlier_ratio < 0.4:
        raise ValueError(f"Низкая уверенность выравнивания (inliers={inlier_ratio:.2f})")
    h, w = ref_gray.shape[:2]
    return cv2.warpPerspective(scan_gray, H, (w, h)), inlier_ratio

def crop_fields(aligned_img, pad=4):
    crops, (h, w) = {}, aligned_img.shape[:2]
    for name, (x1, y1, x2, y2) in FIELD_BOXES.items():
        crops[name] = aligned_img[max(0,y1-pad):min(h,y2+pad), max(0,x1-pad):min(w,x2+pad)]
    return crops
```

Практический момент: если у вас нет единого «эталонного» бланка, возьмите один из самых чистых сканов как референс. Документы, у которых `inlier_ratio` низкий (плохое выравнивание — скан перекошен, обрезан иначе и т.п.), — не выбрасывайте, а откладывайте в отдельную папку "требует ручной разметки боксов" — таких обычно 3–7% и на них можно либо натренировать вторую версию боксов, либо просто размечать их вручную.

### Трек B (модель распознавания): с чего начать и куда расти

Два реалистичных варианта, от простого к более устойчивому:

**B1. Компактная VLM с дообучением (рекомендую как основной путь).** Вместо классического OCR-движка — маленькая vision-language модель, дообученная генерировать сразу структурированный ответ (JSON) по картинке поля/бланка. Работает без отдельного шага "детекция текста → распознавание", устойчивее к вариациям, и именно этот подход сейчас (2026) даёт лучшее соотношение точности и трудозатрат для документов со смешанной разметкой.

**B2. Классический HTR (CRNN / TrOCR) — лёгкий вариант для CPU или как второй независимый "проверяющий" движок.** Раз словарь — это фактически 10 цифр, задача сильно проще типичного OCR, и такая модель работает быстро даже без GPU.

Дальше — обе ветки подробно.

## 2. Сравнение моделей (актуально на сентябрь 2026)

| Модель | Размер | Лицензия | Почему подходит | Слабое место |
|---|---|---|---|---|
| **Florence-2-base** (Microsoft) | 0.23B | MIT | Есть прямой прецедент дообучения на *такой же* по духу задаче — чтение показаний счётчика по фото (138 train / 30 test примеров, тот же формат "узкое числовое поле → значение"). Быстро учится даже на CPU/Colab, минимум инфраструктуры. | Слабее общее визуальное понимание, чем у современных VLM — но для узкого поля это не критично |
| **Qwen2.5-VL-7B / Qwen3-VL-4B–8B** | 4–8B | Apache 2.0 (уточняйте лицензию для конкретного чекпоинта — у младших вариантов Qwen2.5-VL лицензия отличалась от 7B/72B) | Самая зрелая экосистема дообучения (Unsloth, HF PEFT+TRL, готовые LoRA-примеры именно под извлечение структурированных полей из бланков/чеков). Сильные визуальные приоры — устойчивее к грязным сканам. | Нужен GPU хотя бы 8–12 ГБ VRAM для комфортного QLoRA; инференс тяжелее, чем у Florence-2 |
| **PaddleOCR-VL-0.9B** | 0.9B (ERNIE-4.5-0.3B + NaViT-энкодер) | Apache 2.0 | "Родная" переменная разрешающая способность энкодера — не режет картинку на фиксированные тайлы, что обычно помогает с мелким рукописным текстом. Очень быстрый (на RTX 3090 — сотни страниц в минуту). | Официальный пайплайн дообучения (ERNIEKit) более сырой — в issue-трекере есть репорты, что дообучение версии 1.5 давало результат хуже базовой 0.9B; используйте как второй кандидат для сравнения, не как единственную ставку |
| **TrOCR-handwritten** (Microsoft, small/base) | 62M / 334M | MIT | Специально предобучен на рукописном тексте (IAM dataset), минимальный вес, отличная скорость на CPU. Хороший второй/проверяющий движок. | Хуже держит "грязные" реальные вариации (свет, наклон, качество скана) — но у вас это частично снимается Треком A |
| **Донат/Donut, CRNN с нуля** | — | MIT / — | Работоспособны, но по факту вытеснены VLM-based подходами для смешанных документов в 2026 — держите в уме как fallback | Больше ручной инженерии на выходе |

## 3. Подготовка данных

1. **Разметка боксов** (если ещё не сделано) — Label Studio или CVAT (локально, open source), по одному бланку размечаете 12 боксов один раз, дальше — гомография.
2. **Формат ground truth** — на документ:
```json
{"left_home_base": {"date": "2026-09-08", "time": "23:20"},
 "arrived_home_base": {"date": "2026-09-08", "time": "24:00"},
 "started_work": {"date": "2026-09-08", "time": "23:30"},
 "finished_work": {"date": "2026-09-08", "time": "23:50"}}
```
3. **Split** — 80/10/10 (train/val/test). Если несколько бланков могут быть от одной смены/партии со очень похожим почерком одного человека — делите по человеку/партии, а не случайно, иначе валидация будет завышать реальную точность.
4. **Аугментация** — обязательна при 700 документах: небольшой поворот (±3–7°), джиттер яркости/контраста, лёгкий шум/блюр (имитация скана), небольшие сдвиги кропа. Без этого модель за 15–30 эпох переобучится на конкретные почерки/сканы.
5. **Обратите внимание на конвенцию "24:00"** — на вашем примере "Arrived home base" час = 24, минуты = 00. Это стандартная конвенция в сменных/морских журналах для обозначения полуночи *в рамках той же даты*, а не переноса на следующий день. Не "исправляйте" это на 00:00 в постобработке — заложите это как легальное значение (см. код валидации ниже).
6. **Важно перепроверить**: печатный порядок строк на бланке (Left → Arrived → Started → Finished) в вашем примере **не совпадает** с хронологией событий (23:20 → 24:00 → 23:30 → 23:50 — не монотонно). Похоже, что реальная последовательность — Left (23:20) → Started (23:30) → Finished (23:50) → Arrived (24:00), т.е. "Arrived home base" — это возврат *после* работы. Я не могу быть уверен на одном примере, но это значит: **не закладывайте** в валидацию жёсткое правило "left ≤ arrived ≤ started ≤ finished" — оно будет ложно браковать корректные записи. Проверьте реальную семантику по нескольким документам из своих 700 перед тем, как писать кросс-полевые правила.

## 4. Гиперпараметры и обучение

### 4.1 Florence-2 — стартовая точка (рекомендую начать здесь)

| Параметр | Значение | Комментарий |
|---|---|---|
| Базовый чекпоинт | `microsoft/Florence-2-base-ft` | Если GPU слабый — base, не large |
| Метод | LoRA (r=16, alpha=32, dropout=0.05) | Full fine-tune возможен на RTX 3060+ (12 ГБ), но LoRA безопаснее против переобучения на 700 документах |
| Таргет-модули LoRA | `q_proj,k_proj,v_proj,out_proj,fc1,fc2` | Проверьте реальные имена модулей через `model.named_modules()` — могут отличаться от версии к версии |
| Optimizer | AdamW | — |
| LR | 1e-4 (LoRA) / 1e-6…1e-5 (full fine-tune) | Для full fine-tune в опубликованных рецептах используют cosine decay без warmup |
| Batch size | 4–16 (под VRAM) | Кропы маленькие — батч можно держать большим |
| Эпохи | старт 10–15, ранняя остановка по **field exact-match на валидации**, не по loss | Проверенный рецепт на аналогичной задаче (счётчики) сходился за ~7 эпох |
| Аугментация | обязательна | см. п.3 |

```python
# train_florence2.py — скелет LoRA-дообучения (адаптируйте пути/промпты под себя)
import json, torch
from pathlib import Path
from torch.utils.data import Dataset, DataLoader
from transformers import AutoModelForCausalLM, AutoProcessor, get_scheduler
from peft import LoraConfig, get_peft_model

MODEL_ID = "microsoft/Florence-2-base-ft"
TASK_PROMPT = "<TIMESHEET_JSON>"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

class TimesheetDataset(Dataset):
    # manifest.jsonl: {"image_path": "crops/0001/started_work_day.png", "target": "08"}
    # либо, для варианта "весь бланк одним проходом":
    # {"image_path": "aligned/0001.png", "target": "{...полный json...}"}
    def __init__(self, manifest_path, processor):
        self.rows = [json.loads(l) for l in Path(manifest_path).read_text().splitlines() if l.strip()]
        self.processor = processor
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        r = self.rows[i]
        return r["image_path"], TASK_PROMPT, r["target"]

def collate(batch, processor):
    from PIL import Image
    paths, prompts, targets = zip(*batch)
    images = [Image.open(p).convert("RGB") for p in paths]
    inputs = processor(text=list(prompts), images=images, return_tensors="pt", padding=True)
    labels = processor.tokenizer(list(targets), return_tensors="pt", padding=True,
                                  return_token_type_ids=False).input_ids
    labels[labels == processor.tokenizer.pad_token_id] = -100
    inputs["labels"] = labels
    return inputs

def build_model():
    model = AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE)
    cfg = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05,
                      target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"],
                      task_type="CAUSAL_LM")
    return get_peft_model(model, cfg)

def train(manifest_path, epochs=15, lr=1e-4, batch_size=8):
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    model = build_model(); model.train()
    ds = TimesheetDataset(manifest_path, processor)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=True,
                     collate_fn=lambda b: collate(b, processor))
    opt = torch.optim.AdamW(model.parameters(), lr=lr)
    steps = epochs * len(dl)
    sched = get_scheduler("cosine", opt, num_warmup_steps=int(0.03*steps), num_training_steps=steps)
    for ep in range(epochs):
        running = 0.0
        for batch in dl:
            batch = {k: v.to(DEVICE) for k, v in batch.items()}
            loss = model(**batch).loss
            loss.backward(); opt.step(); sched.step(); opt.zero_grad()
            running += loss.item()
        print(f"epoch {ep+1}/{epochs} - loss {running/len(dl):.4f}")
        # TODO: здесь — валидационный проход с метрикой field exact-match,
        # чекпоинт сохранять только при улучшении именно этой метрики.
    model.save_pretrained("florence2-timesheet-lora")

if __name__ == "__main__":
    train("train_manifest.jsonl")
```

### 4.2 Qwen2.5-VL / Qwen3-VL через LoRA/QLoRA — путь для повышения устойчивости

Если после Florence-2 точность/устойчивость к реальным сканам недостаточна — следующий шаг, не первый (дороже по инфраструктуре).

| Параметр | Значение | Комментарий |
|---|---|---|
| Базовый чекпоинт | Qwen2.5-VL-7B-Instruct или Qwen3-VL-4B/8B-Instruct | Размер — по доступной VRAM |
| Метод | QLoRA, 4-bit NF4 | Через Unsloth — самый низкий порог входа, есть готовые ноутбуки под Qwen-VL |
| LoRA rank / alpha | r=16–32, alpha=2×r | На узкой повторяющейся задаче (700 документов одного шаблона) высокий rank (128, который встречается в рецептах для больших общих SFT-датасетов) скорее навредит — переобучится на конкретные образцы |
| LR | 1e-4…2e-4, подбирать по сетке (5e-5 / 1e-4 / 2e-4) | Вот и есть предметный "подбор гиперпараметров", который вы просили — 3 запуска на подвыборке достаточно, чтобы увидеть, какой LR даёт лучшую **val exact-match**, а не только low loss |
| Эпохи | 8–15+, ранняя остановка по val exact-match | На реальном примере дообучения Qwen2-VL-7B под структурированное извлечение (чеки, LoRA, 600 изображений) 3 эпохи довели формат JSON до 100% валидности, но точность самих значений осталась низкой — 3 эпохи учат "как отвечать", а не "что писать". Закладывайте больше эпох именно для точности значений, а не только структуры |
| Разрешение картинки | не давайте фреймворку агрессивно даунсемплить — либо кормите кроп поля, либо явно поднимите `max_pixels` | Мелкий рукописный текст на полностраничном скане легко "теряется" при пережатии |

Инструменты: **Unsloth** (`unsloth.ai`) — минимум кода, готовые ноутбуки под Qwen3-VL/Qwen2.5-VL, 4-bit квантование на лету; альтернатива — HF `transformers` + `peft` + `trl.SFTTrainer` напрямую, если нужен полный контроль.

### 4.3 Классический HTR (Трек B2) — CPU-friendly вариант или "второй голос"

Раз словарь полей — практически только цифры, обучить с нуля лёгкий CRNN (CNN + BiLSTM + CTC, словарь 0-9 + blank) — минуты на CPU, доли секунды на инференс. Либо дообучить `microsoft/trocr-small-handwritten` (HF `Seq2SeqTrainer`, LR ~5e-5, batch 8–16, 15–30 эпох, ранняя остановка по CER на валидации).

Практическая польза этого трека даже если основной — VLM: используйте его как **независимую проверку**. Если CRNN/TrOCR и VLM согласны в значении поля — высокая уверенность, автоматически принимаете. Если расходятся — в очередь на ручную проверку. Это дёшево реализовать и резко снижает риск тихих ошибок в проде.

## 5. Постобработка и валидация (правила поверх модели)

Даже без единого лишнего процента ML-точности здесь можно поймать большую часть ошибок распознавания через простые проверки диапазонов и конвенцию 24:00:

```python
# validate.py
from dataclasses import dataclass, field
from typing import Optional

@dataclass
class FieldResult:
    value: Optional[int]; raw: str; valid: bool
    issues: list = field(default_factory=list)

def check_range(raw, lo, hi, name):
    raw = raw.strip()
    if not raw.isdigit():
        return FieldResult(None, raw, False, [f"{name}: '{raw}' не число"])
    v = int(raw)
    return FieldResult(v, raw, lo <= v <= hi,
                        [] if lo <= v <= hi else [f"{name}: {v} вне [{lo},{hi}]"])

def check_hour(raw, minute_val):
    # час=24 — легальный маркер конца суток, ЕСЛИ минуты=00 (см. п.3 про конвенцию)
    res = check_range(raw, 0, 23, "hour")
    if not res.valid and res.value == 24 and minute_val == 0:
        return FieldResult(24, raw, True, ["hour=24 принят как конец суток (00:00)"])
    return res

def validate_record(fields: dict) -> dict:
    """fields: {"left_home_base": {"day":"08","month":"09","year":"2026","hour":"23","min":"20"}, ...}
    Намеренно НЕ проверяет монотонность left<=arrived<=started<=finished —
    на вашем образце печатный порядок строк не совпадает с хронологией событий."""
    out, days_seen = {}, []
    for row_name, row in fields.items():
        month = check_range(row["month"], 1, 12, "month")
        year = check_range(row["year"], 2000, 2100, "year")
        minute = check_range(row["min"], 0, 59, "minute")
        day = check_range(row["day"], 1, 31, "day")
        hour = check_hour(row["hour"], minute.value)
        if day.valid: days_seen.append(day.value)
        out[row_name] = {"day": day, "month": month, "year": year, "hour": hour,
                          "minute": minute,
                          "row_valid": all(r.valid for r in (day, month, year, hour, minute))}
    flags = []
    if days_seen and (max(days_seen) - min(days_seen) > 1):
        flags.append(f"день расходится больше чем на 1 между строками {sorted(set(days_seen))} — вероятна ошибка распознавания")
    out["_document_flags"] = flags
    out["_needs_review"] = bool(flags) or any(
        not r["row_valid"] for k, r in out.items() if k != "_document_flags")
    return out
```

Проверено на ваших реальных значениях с картинки (08.09.2026, включая час=24/мин=00) — проходит без ложных срабатываний.

## 6. Развёртывание (полностью локально)

Важное терминологическое уточнение: "без API" в вашем требовании я читаю как "без обращения к чужим облачным сервисам" (OpenAI, Google Vision, AWS Textract и т.п.). Локальный REST-эндпоинт на вашей же машине/VPS, который вызывает n8n через HTTP-нод — это не нарушает это требование, это просто удобный интерфейс к вашей же модели.

**Для Florence-2 / любой HF-модели:**
```python
# serve.py — запуск: uvicorn serve:app --host 0.0.0.0 --port 8008
from io import BytesIO
import torch, json
from fastapi import FastAPI, File, UploadFile
from PIL import Image
from transformers import AutoModelForCausalLM, AutoProcessor
from peft import PeftModel
from validate import validate_record

MODEL_ID, ADAPTER_PATH = "microsoft/Florence-2-base-ft", "florence2-timesheet-lora"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
app = FastAPI(title="timesheet-date-extractor")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
model = PeftModel.from_pretrained(
    AutoModelForCausalLM.from_pretrained(MODEL_ID, trust_remote_code=True).to(DEVICE),
    ADAPTER_PATH).eval()

@app.post("/extract")
async def extract(file: UploadFile = File(...)):
    image = Image.open(BytesIO(await file.read())).convert("RGB")
    inputs = processor(text="<TIMESHEET_JSON>", images=image, return_tensors="pt").to(DEVICE)
    with torch.no_grad():
        ids = model.generate(**inputs, max_new_tokens=64, num_beams=3)
    raw = processor.batch_decode(ids, skip_special_tokens=True)[0]
    try:
        parsed = json.loads(raw)
    except json.JSONDecodeError:
        return {"error": "невалидный JSON от модели", "raw": raw}
    return {"fields": parsed, "validation": validate_record(parsed)}
```
Заверните в Docker (у вас уже есть опыт с VPS-инфраструктурой под остальные проекты) — так деплой воспроизводим и его несложно перенести на тот же VPS, где крутится остальная автоматизация.

**Для Qwen2.5-VL/Qwen3-VL:** после LoRA-дообучения — либо смёрджить адаптер в базу и раздавать через `vLLM` (если есть GPU и нужен throughput, свой OpenAI-совместимый локальный эндпоинт), либо сконвертировать в GGUF и поднять одной командой через `llama.cpp`/`Ollama` — тоже локальный OpenAI-совместимый сервер, без выхода во внешний интернет.

**Интеграция с вашим стеком:** раз у вас уже n8n — эндпоинт `/extract` вызывается обычной HTTP Request нодой из воркфлоу (скан пришёл → POST на локальный сервис → JSON с датами → дальше по пайплайну), без необходимости встраивать Python-инференс прямо в n8n.

## 7. Метрики и процесс проверки

- **CER (character error rate)** по каждому полю — грубая метрика на старте.
- **Exact-match по полю** (день/месяц/год/час/минута отдельно) — интерпретируемее для чисел.
- **Full-record accuracy** — % документов, где ВСЕ поля дат распознаны верно — это, скорее всего, ваша целевая бизнес-метрика.
- Раз время для вас не критично — считайте и репортите точность по датам отдельно от точности по времени; не давайте плохому времени "портить" общий отчёт о качестве.
- На каждом чекпоинте сохраняйте модель по val exact-match, а не по loss (см. пример из CORD выше — loss может красиво падать, а точность значений — нет).

## 8. План действий по шагам

1. Разметить боксы полей на 1 эталонном бланке, прогнать Трек A (`preprocess.py`) на всех 700 — отсеять/отложить те, что не выровнялись.
2. Быстрый baseline без дообучения (готовый TrOCR-handwritten или даже эталонная VLM без тюнинга) на кропах — проверить, что пайплайн выравнивания/кропа вообще работает и дать точку отсчёта.
3. Дообучить Florence-2-base (LoRA) на кропах — первая реальная цифра точности, дёшево итерировать.
4. Если точности/устойчивости не хватает — эскалация на Qwen2.5-VL/Qwen3-VL (QLoRA, Unsloth).
5. Добавить слой валидации (`validate.py`) поверх любой из моделей.
6. Поднять как локальный REST-сервис (FastAPI/Ollama/vLLM), подключить к n8n.
7. Завести очередь ручной проверки для документов с `_needs_review = true` — это дешевле, чем гнаться за 100% автоматической точностью с первого раза.

## Полезные источники (проверено, реальные ссылки)

- Официальный туториал дообучения Florence-2 (HF blog): https://huggingface.co/blog/finetune-florence2
- PaddleOCR-VL — репозиторий и документация по использованию: https://github.com/paddlepaddle/paddleocr/blob/main/docs/version3.x/pipeline_usage/PaddleOCR-VL.en.md
- Реальный пример QLoRA-дообучения Qwen2-VL-7B под структурированное извлечение из чеков (метрики до/после, на которые я ссылаюсь в п.4.2): https://huggingface.co/saliousk/qwen2vl-cord-lora
- Unsloth — доки по локальному запуску и дообучению Qwen-моделей (4-bit, минимум VRAM): https://unsloth.ai/docs/models/qwen3.5
- Qwen3-VL, готовые GGUF-веса для локального инференса через llama.cpp/Ollama: https://huggingface.co/Qwen/Qwen3-VL-8B-Instruct-GGUF
