# Установка зависимостей OCR-плана (один раз, руками)

Зачем: сеть через VPN медленная (≈0,1–0,15 МБ/с), и если агент ставит пакеты сам, он
тратит квоту и часы на скачивание. Здесь собраны все известные заранее зависимости (не только тяжёлые), которые понадобятся задачам `T00`–`T29`
(`docs/ocr_tasks/`). Выполните блоки по порядку в PowerShell, один раз; агентам
останется только проверить, что всё на месте.

Составлено по: `requirements.txt`, `requirements-dev.txt`, `docs/ocr_tasks/CONTEXT.md`
(§ 4–7) и промптам T05, T11, T14, T16, T17, T18, T23, T24. Версии, помеченные
«(проверить)», я не смог проверить без интернета: после установки прогоните блок 8.

## Сколько скачивать

| Блок | Что | Объём (прим.) | Нужно для |
|---|---|---|---|
| 1 | Драйвер NVIDIA (обязательно для GPU) | 0,6–0,8 ГБ | T05, T14, T16, T17 на GPU |
| 2 | Кэши и временные каталоги на E: | — | все |
| 3 | Дозаполнить `.venv` приложения | ~0,1 ГБ | T06+, T18 |
| 4 | `.venv-train` с torch+CUDA | 3–4 ГБ | T05, T14–T17, T23 |
| 5 | Веса TrOCR (докачать) | ~1,3 ГБ | приложение, T14 |
| 6 | Веса ImageNet (ResNet18, MobileNetV3-S) | ~60 МБ | T16 |
| 7 | DINOv2-small (опционально) | ~90 МБ | T23 (опц.) |

Без GPU (блок 1 пропущен): блоки 4–7 всё равно нужны, но torch ставится CPU-версией
(см. примечание в блоке 4), и объём падает до ~0,3 ГБ.

## 0. Перед началом

Все команды — из корня репозитория, `E:\projects\doc_auto_handling\doc_auto_handling`.
C: у вас почти полон (на момент проверки 16 ГБ свободно, а в `CONTEXT.md` было 1 ГБ),
поэтому кэш pip и временные файлы переводим на E:. Окно PowerShell то же на весь процесс.

```powershell
cd E:\projects\doc_auto_handling\doc_auto_handling
New-Item -ItemType Directory -Force E:\ocr_cache\pip, E:\ocr_cache\hf, E:\ocr_cache\torch, E:\ocr_tmp | Out-Null
$env:PIP_CACHE_DIR = "E:\ocr_cache\pip"
$env:TEMP = "E:\ocr_tmp"; $env:TMP = "E:\ocr_tmp"
$env:HF_HOME = "E:\ocr_cache\hf"; $env:TORCH_HOME = "E:\ocr_cache\torch"
$env:PIP_DEFAULT_TIMEOUT = "120"; $env:PIP_RETRIES = "10"
```

Чтобы кэши моделей и далее смотрели на E: (иначе при любой новой загрузке веса уйдут
на C:), один раз сохраните их для пользователя (подействует в новых окнах):

```powershell
setx HF_HOME E:\ocr_cache\hf
setx TORCH_HOME E:\ocr_cache\torch
setx OCR_CACHE_DIR E:\ocr_cache
setx PIP_CACHE_DIR E:\ocr_cache\pip
```

Если скачивание обрывается, просто повторите ту же команду: pip докачивает уже
загруженные колёса из кэша `E:\ocr_cache\pip`.

## 1. Драйвер NVIDIA (обязательно, если нужен GPU)

Сейчас стоит драйвер **446.14** (CUDA 11.0). Колёса PyTorch с CUDA (cu118/cu126) на нём
работать не будут, поэтому T05 остановится на вопросе про драйвер. Обновите драйвер
вручную: скачайте актуальный драйвер для **GeForce GTX 1050** (Windows 10 64-bit) на
nvidia.com → Drivers, установите и перезагрузите компьютер. Проверка:

```powershell
nvidia-smi
```

В шапке должно быть `CUDA Version: 12.x` (нужно ≥ 12.6 для cu126). Pascal (GTX 1050) в
свежих драйверах пока поддерживается, но если установщик скажет, что карта не
поддерживается, возьмите последнюю версию из ветки, где она есть, и пропишите мне номер.

## 2. Каталог и Python для `.venv-train`

Тот же Python 3.11, что у `.venv` (не системный 3.12):

```powershell
& "$env:LOCALAPPDATA\Programs\Python\Python311\python.exe" -m venv .venv-train
```

## 3. Дозаполнить `.venv` приложения (CPU, без обучения)

`opencv-python-headless` уже стоит (Devin его поставил). Остальное из
`requirements.txt` и dev-набора, плюс `onnxruntime` для T18:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pip install onnxruntime onnx
```

`requirements-dev.txt` подтягивает и `requirements.txt` (включая `torch==2.5.1` CPU и
`transformers`; они уже стоят, повторно качаться не будут). `onnxruntime` и `onnx` затем
закрепите версиями в `requirements.txt` (это сделает T18).

## 4. `.venv-train`: torch + CUDA и остальное для обучения

Версии torch/torchvision и индекс проверить: T05 требует сборку, где есть `sm_61`
(Pascal). По PyTorch 2.7 индекс **cu126** содержит sm_61 (проверить, как требует T05;
если нет, откатитесь на 2.6.0 и укажите мне).

```powershell
.\.venv-train\Scripts\python.exe -m pip install --upgrade pip
.\.venv-train\Scripts\python.exe -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu126
```

Это самый большой кусок (≈ 3 ГБ). Остальное идёт с обычного PyPI. Колесо OpenCV уже
скачано, ставим его локально без сети:

```powershell
.\.venv-train\Scripts\python.exe -m pip install E:\ocr_tmp\opencv_python_headless-5.0.0.93-cp37-abi3-win_amd64.whl
.\.venv-train\Scripts\python.exe -m pip install numpy==2.4.6 pillow==12.3.0 pypdfium2==5.12.1 pandas==2.2.3 scikit-learn matplotlib onnx onnxruntime transformers==4.46.3 pytest==8.3.4
```

Примечание для варианта без GPU: вместо строки с torch выполните
`... -m pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu`
(≈ 0,25 ГБ). `albumentations` не ставим: аугментации пишутся на cv2 и numpy (T15).

## 5. Веса TrOCR (докачать)

В кэше C: (`C:\Users\User\.cache\huggingface\hub\models--kazars24--trocr-base-handwritten-ru`)
лежат только токенайзер и конфиги, а файл весов остался как `.incomplete`: модель
не докачана. Качаем сразу в кэш на E:, чтобы агенту не пришлось ждать (≈ 1,3 ГБ):

```powershell
.\.venv-train\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('kazars24/trocr-base-handwritten-ru', max_workers=2))"
```

Если `huggingface_hub` не нашёлся, он тянется вместе с `transformers`. При обрыве
повторите команду: докачка продолжится.

## 6. Веса ImageNet для T16 (ResNet18 и MobileNetV3-Small)

```powershell
.\.venv-train\Scripts\python.exe -c "import torchvision as tv; tv.models.resnet18(weights='IMAGENET1K_V1'); tv.models.mobilenet_v3_small(weights='IMAGENET1K_V1'); print('ok')"
```

Веса лягут в `E:\ocr_cache\torch\hub\checkpoints` (≈ 47 МБ и 10 МБ).

## 7. DINOv2-small (только если будете делать T23, опционально)

```powershell
.\.venv-train\Scripts\python.exe -c "from huggingface_hub import snapshot_download; print(snapshot_download('facebook/dinov2-small'))"
```

## 8. Проверка: всё ли поставилось

```powershell
nvidia-smi
.\.venv-train\Scripts\python.exe -c "import torch,torchvision,cv2,onnx,onnxruntime,sklearn,matplotlib,transformers; print('torch',torch.__version__,'cuda',torch.cuda.is_available(),'arch',torch.cuda.get_arch_list() if torch.cuda.is_available() else '-')"
.\.venv\Scripts\python.exe -c "import cv2,onnxruntime,pypdfium2,pytesseract; print('venv ok')"
& "C:\Program Files\Tesseract-OCR\tesseract.exe" --list-langs
```

Ожидаемо: `cuda True` и в списке архитектур есть `sm_61`; у Tesseract — `eng`, `osd`,
`rus` (они уже есть).

Если `sm_61` в списке нет, значит выбранная сборка torch не поддерживает Pascal:
напишите, какую версию поставили, и я подберу нужную.

## 9. Убрать временное

После успешной проверки можно удалить `E:\ocr_tmp` (там лежат скачанные Devin куски
и колесо OpenCV на ~120 МБ):

```powershell
Remove-Item -Recurse -Force E:\ocr_tmp
```

## Что НЕ нужно ставить

- Tesseract: уже стоит (`C:\Program Files\Tesseract-OCR`, языки eng/osd/rus).
- Claude Code и Devin CLI: уже стоят.
- `albumentations`: не используется (аугментации на cv2 и numpy).
- Драйверы или системные переменные агенты сами не меняют (это правило плана).
