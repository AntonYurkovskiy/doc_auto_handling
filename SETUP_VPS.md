# Запуск очереди OCR-плана на VPS

Зачем: слабый интернет через VPN делает работу на ПК уязвимой (обрывы, сон, долгие
установки). На VPS быстрая сеть, сессии живут в `tmux` и переживают обрыв соединения.
Порядок: подготовить сервер (блоки 1–4), один раз перенести код и данные (блок 5),
поставить окружение (блок 6), проверить (блок 7), запустить очередь (блок 8).

Предполагаемый размер: GPU не нужен. Метод (геометрия → малые CNN → декодер) идёт на CPU,
инференс в проде остаётся на ПК через ONNX.

## Что заказать

| Параметр | Значение |
|---|---|
| ОС | Ubuntu 24.04 LTS (22.04 тоже подойдёт), x86_64 |
| vCPU | 8 (минимум 4) |
| RAM | 16 ГБ (32 ГБ, если захотите гонять T14 и обучение параллельно) |
| Диск | 60–80 ГБ NVMe (сейчас проект весит ≈12 ГБ, из них 5,6 ГБ было CUDA-torch, на CPU он ≈0,5 ГБ) |
| Сеть | от 100 Мбит/с |
| GPU | не нужен |

**Регион.** Выберите страну, откуда доступны Claude и ваш аккаунт (иначе смысл VPS
теряется). **Данные.** Сканы и реальные записи уйдут на чужой сервер: это ваше решение по
конфиденциальности. Итоговые модели (ONNX) маленькие и вернутся на ПК.

## 0. Что подготовить на своём ПК заранее

1. SSH-ключ (если нет): `ssh-keygen -t ed25519` (в PowerShell). Публичный ключ
   `%USERPROFILE%\.ssh\id_ed25519.pub` укажите при заказе VPS.
2. Дальше везде `USER@IP` — ваш логин и IP сервера (`ubuntu@203.0.113.10`).

## 1. Первый вход и безопасность (на VPS)

```bash
ssh USER@IP
sudo apt update && sudo apt -y upgrade
sudo apt -y install ufw fail2ban
sudo ufw allow OpenSSH && sudo ufw --force enable
sudo sed -i 's/^#\?PasswordAuthentication.*/PasswordAuthentication no/' /etc/ssh/sshd_config
sudo systemctl restart ssh
```

Если вход по ключу ещё не проверен в соседнем окне, последние две строки не выполняйте:
иначе можете потерять доступ.

## 2. Системные пакеты

```bash
sudo apt -y install git tmux curl unzip rsync build-essential \
    tesseract-ocr tesseract-ocr-rus tesseract-ocr-eng
tesseract --list-langs
```

В списке должны быть `eng` и `rus`.

## 3. Python 3.11 через uv

Репозиторий закреплён на Python 3.11 (на Ubuntu 24.04 по умолчанию 3.12).

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
source ~/.local/bin/env 2>/dev/null || export PATH="$HOME/.local/bin:$PATH"
uv python install 3.11
```

## 4. Claude Code на сервере

```bash
curl -fsSL https://claude.ai/install.sh | bash
export PATH="$HOME/.local/bin:$PATH"
claude --version        # нужно 2.1.280 или новее
claude                  # затем /login: откроется ссылка, войдите в браузере на ПК и вставьте код
```

Войдите тем же аккаунтом с подпиской Pro. Раннер на сервере запускайте **без**
`--channel devin`: подмена модели Devin написана под Windows. Задачи пойдут на Claude.
Квота та же, что и на ПК: лимиты Pro общие (параллельно двух очередей не запускайте).

## 5. Перенос проекта и данных (один раз)

Пересылаемый объём: код с историей git и каталог `data/` без кэшей моделей и без
окружений. Исторические данные ≈ 0,95 ГБ, производные `data/ocr` ≈ 0,7 ГБ. Кэши
моделей (4,5 ГБ) переносить не нужно: на VPS они скачаются за минуты.

На ПК, в Git Bash из корня репозитория (`E:/projects/doc_auto_handling/doc_auto_handling`).
Каталог `data/historical` лежит на уровень выше репозитория, поэтому кладём его внутрь
`data/` на сервере, куда его и ждёт `ocr_lab/paths.py`:

```bash
cd /e/projects/doc_auto_handling/doc_auto_handling
# 1) код и история git (без окружений и тяжёлых артефактов)
tar --exclude=.venv --exclude=.venv-train --exclude=./data --exclude=__pycache__ \
    --exclude=.mypy_cache --exclude=.ruff_cache --exclude=.pytest_cache \
    -czf /e/code.tar.gz .
# 2) производные данные OCR (без кэша моделей)
tar --exclude=data/ocr/cache -czf /e/ocr_data.tar.gz data/ocr
# 3) исторические данные (лежат уровнем выше)
tar -czf /e/historical.tar.gz -C /e/projects/doc_auto_handling data/historical
```

Передача. Через VPN скорость мала, поэтому режем на части по 100 МБ: если связь оборвётся,
повторяются только недокачанные части.

```bash
cd /e && split -b 100m code.tar.gz part_code_ && split -b 100m ocr_data.tar.gz part_ocr_ && split -b 100m historical.tar.gz part_hist_
for f in part_*; do scp -o ServerAliveInterval=30 "$f" USER@IP:/home/USER/; done
```

Если какая-то часть оборвалась, перешлите её отдельной командой `scp part_XX USER@IP:/home/USER/`.

На VPS собираем обратно:

```bash
mkdir -p ~/doc_auto_handling && cd ~/doc_auto_handling
cat ~/part_code_* | tar -xzf -
cat ~/part_ocr_*  | tar -xzf -
cat ~/part_hist_* | tar -xzf -
rm ~/part_*
git status --short | head        # дерево чистое или с уже известными правками
git branch --show-current        # ocr/handwritten-dates
```

Ничего не пушьте на GitHub отсюда: пуш в раннере запрещён, а код с данными в чужой
репозиторий отправлять не нужно. Результаты на ПК вернёте блоком 9.

## 6. Окружения

В `docs/ocr_tasks/_common.md` пути записаны как `.venv/Scripts/python` (Windows). На Linux
каталог называется `bin`, поэтому делаем ссылку `Scripts → bin`: так все команды из
промптов работают без правок.

```bash
cd ~/doc_auto_handling
export UV_PYTHON=3.11
# окружение приложения (CPU)
uv venv .venv --python 3.11
uv pip install --python .venv/bin/python torch==2.5.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv/bin/python -r requirements-dev.txt onnxruntime onnx
ln -sfn bin .venv/Scripts

# окружение обучения (CPU)
uv venv .venv-train --python 3.11
uv pip install --python .venv-train/bin/python torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cpu
uv pip install --python .venv-train/bin/python numpy==2.4.6 pillow==12.3.0 pypdfium2==5.12.1 pandas==2.2.3 \
    opencv-python-headless==5.0.0.93 scikit-learn matplotlib onnx onnxruntime transformers==4.46.3 pytest==8.3.4
ln -sfn bin .venv-train/Scripts
```

Версии `torch==2.5.1` и `torch==2.7.1` на CPU-индексе я заранее не проверял
«(проверить)»: если `pip` скажет, что версии нет, поставьте ближайшую доступную и
сообщите её агенту в журнале.

Веса моделей (быстро, на сервере сеть нормальная):

```bash
export HF_HOME=$HOME/ocr_cache/hf TORCH_HOME=$HOME/ocr_cache/torch OCR_CACHE_DIR=$HOME/ocr_cache
mkdir -p $HF_HOME $TORCH_HOME
.venv-train/bin/python -c "from huggingface_hub import snapshot_download as s; s('kazars24/trocr-base-handwritten-ru'); s('facebook/dinov2-small')"
.venv-train/bin/python -c "import torchvision as tv; tv.models.resnet18(weights='IMAGENET1K_V1'); tv.models.mobilenet_v3_small(weights='IMAGENET1K_V1'); print('ok')"
printf 'export HF_HOME=$HOME/ocr_cache/hf\nexport TORCH_HOME=$HOME/ocr_cache/torch\nexport OCR_CACHE_DIR=$HOME/ocr_cache\nexport PYTHONUTF8=1\n' >> ~/.bashrc
```

DINOv2 нужен только для T23 (опциональной). Задачу T05 (GPU) на сервере пропускаем: она
опциональна и очередь сама её не берёт.

## 7. Проверка

```bash
cd ~/doc_auto_handling
.venv/Scripts/python -c "import cv2, onnxruntime, pypdfium2, pytesseract; print('venv ok')"
.venv-train/Scripts/python -c "import torch, torchvision; print(torch.__version__, torch.cuda.is_available())"   # False нормально
APP_TROCR_ENABLED=false .venv/Scripts/python -m pytest -q --ignore=tests/ocr -p no:warnings | tail -2
.venv/Scripts/python docs/ocr_tasks/run_task.py --status | head -12
```

## 8. Запуск очереди в tmux

```bash
cd ~/doc_auto_handling
tmux new -s ocr
export PATH="$HOME/.local/bin:$PATH" PYTHONUTF8=1
python3 docs/ocr_tasks/run_task.py --all --allow-dirty
```

Отключиться и оставить работать: `Ctrl+b`, затем `d`. Вернуться: `tmux attach -t ocr`.
Статус: `python3 docs/ocr_tasks/run_task.py --status`.

Очередь остановится на ручных точках (H1 после T04, H2 после T10 и т. д.), на первой
ошибке и при упоре в лимит окна. Запустите её той же командой, и она продолжит с первой
незавершённой задачи.

Что на Linux работает иначе:
- Канал Devin на сервере не используем (см. блок 4).
- Приёмку T29 на Windows (`H5`) делайте на ПК.
- Тексты промптов упоминают Windows и `E:\...`; агент адаптирует их сам, а ссылки
  `.venv/Scripts/python` работают благодаря ссылке из блока 6.

## 9. Забрать результаты на ПК

Работа лежит в git-ветке на сервере, данные и модели в `data/ocr/`. Код и журнал
заберите коммитами, а крупные артефакты архивом.

```bash
# на VPS
cd ~/doc_auto_handling
git bundle create ~/ocr.bundle ocr/handwritten-dates
tar -czf ~/ocr_models.tar.gz data/ocr/models data/ocr/reports 2>/dev/null
```

```powershell
# на ПК (PowerShell)
scp USER@IP:/home/USER/ocr.bundle E:\
scp USER@IP:/home/USER/ocr_models.tar.gz E:\
cd E:\projects\doc_auto_handling\doc_auto_handling
git fetch E:\ocr.bundle ocr/handwritten-dates:ocr/from-vps
git merge ocr/from-vps
```

Модели и отчёты распакуйте в `data/ocr/` на ПК (`tar -xzf E:\ocr_models.tar.gz`).
Журнал `docs/ocr_tasks/PROGRESS.md` приедет вместе с коммитами, и раннер на ПК увидит
выполненные задачи.

## 10. Если что-то пошло не так

- **`требует claude >= 2.1.280`:** на сервере стоит старый `claude`: выполните
  `claude update`.
- **`Not logged in`:** `claude`, затем `/login`.
- **Диск заполнился:** проверьте `du -sh ~/doc_auto_handling/data/ocr ~/ocr_cache`.
- **Сессия оборвалась:** `tmux attach -t ocr`, если tmux жив; иначе запустите блок 8 заново.
- **Сервер больше не нужен:** заберите блок 9 и удалите VPS у провайдера (данные на нём
  реальные).
