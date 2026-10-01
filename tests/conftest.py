"""Общие настройки тестов."""

import os

# Тесты не должны скачивать веса TrOCR (~1,4 ГБ) и грузить torch: без этого pytest
# на машине с APP_TROCR_ENABLED=true в .env надолго зависает. Переменная окружения
# важнее .env, а conftest.py выполняется до импорта app.config.
os.environ["APP_TROCR_ENABLED"] = "false"
