---
id: T05
title: Окружение для обучения на GTX 1050 (CUDA, sm_61)
stage: 0
depends_on: T00
type: infra
complexity: medium
risk: reversible
needs_vision: false
needs_web: true
host: local
size: M
channel: claude_first
claude_model: claude-sonnet-5
claude_effort: high
devin_model: claude-sonnet-5-medium
review: —
human_checkpoint: —
max_turns: 90
optional: true
---

# T05. Отдельное окружение для обучения: `.venv-train`

## Зачем

В `.venv` приложения стоит torch без CUDA. Маленькая CNN учится и на CPU, но на GPU
итерации в 5–10 раз быстрее, а TrOCR (T14) на CPU медленный. Окружение для обучения
держим **отдельно**, чтобы не сломать приложение. Задача опциональна: без неё T14 и T16
идут на CPU.

## Что сделать

1. **Драйвер.** Проверь его через `nvidia-smi`. Если драйвер слишком старый для CUDA 12.x
   (или 11.8), ничего не ставь: запиши в «Вопросы к человеку» инструкцию по обновлению
   драйвера и остановись. Драйверы сам не устанавливай.
2. **Окружение.** Создай `.venv-train` той же версии Python, что `.venv`.
3. **torch и torchvision со сборкой CUDA, в которой есть `sm_61`.**
   - Начни с индекса `https://download.pytorch.org/whl/cu126`. Насколько известно, начиная
     с PyTorch 2.8 сборки CUDA 12.8+ не включают Pascal (sm_50–sm_61), а cu126 включает.
     Не верь этому на слово — проверь.
   - Проверка обязательна: `torch.cuda.is_available()`, `'sm_61' in torch.cuda.get_arch_list()`,
     умножение матриц на GPU и один шаг обучения маленькой свёрточной сети.
   - Если свежая версия не содержит sm_61, откатывайся на старшую, в которой он есть
     (cu126 или cu118). Точные версии и индекс сверь по
     `https://pytorch.org/get-started/previous-versions/`.
4. **`requirements-train.txt`** — точные версии и строка `--index-url` или
   `--extra-index-url` с комментарием. Что поставить:
   `torch`, `torchvision`, `numpy`, `opencv-python-headless`, `Pillow`, `pypdfium2`,
   `pandas`, `scikit-learn`, `matplotlib`, `onnx`, `onnxruntime`, `transformers`
   (для T14), `pytest`.
   albumentations не ставь: аугментации пишем на cv2 и numpy (T15).
5. **Кэши.** `HF_HOME` и `TORCH_HOME` должны смотреть на E:.
   - Скрипты `ocr_lab` делают это сами через `ocr_lab/paths.py` (T02). Если T02 ещё не
     выполнена, запиши только команду для человека.
   - Системные переменные сам не меняй. В CONTEXT.md запиши готовую команду
     `setx HF_HOME E:\...` на случай, если человек захочет.
6. **Замер скорости.** ResNet18 и MobileNetV3-Small, вход 1×64×128, батч 128,
   forward + backward.
   - Замерь изображений в секунду на GPU и на CPU, пиковую VRAM.
   - Какой батч влезает в 4 ГБ в FP32.

## Критерии приёмки

- `.venv-train/Scripts/python -c "import torch; print(torch.cuda.get_arch_list())"`
  показывает `sm_61`, и тестовый шаг обучения проходит на GPU.
- `.venv` приложения не изменён: сравни `pip freeze` до и после.
- В CONTEXT.md есть раздел «Окружение обучения»: команды активации, версии, замеры
  скорости.

## Отчёт

Журнал по шаблону. Коммит: `requirements-train.txt`, CONTEXT.md, журнал.
Сообщение: `OCR T05: окружение для обучения с CUDA sm_61`.
