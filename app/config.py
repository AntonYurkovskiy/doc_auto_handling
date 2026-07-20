"""Конфигурация приложения и ТАРИФНЫЕ КОНСТАНТЫ.

ВАЖНО: реальные ставки — коммерческая тайна и здесь НЕ хранятся.
Ниже стоят ПЛЕЙСХОЛДЕРЫ (условные числа), чтобы работала логика расчёта.
Замени значения `rate` на реальные — логика останется прежней.

Единицы тарификации (из прайса агента):
    per_ton       — за 1 регистровую тонну (умножаем на GRT из заявки)
    per_hour      — в час за каждый буксир (умножаем на часы работы, пропорционально минутам)
    per_operation — за 1 операцию (фиксированная сумма; зависит от будни/праздник+ночь)

Проверено на примерах: почасовая тарификация ПРОПОРЦИОНАЛЬНА минутам,
например 590 у.е. * (70 мин / 60) = 688.33.
"""

from __future__ import annotations

from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class Settings(BaseSettings):
    """Настройки приложения (можно переопределить через .env)."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="APP_", extra="ignore")

    database_url: str = f"sqlite:///{DATA_DIR / 'app.db'}"

    incoming_applications_dir: Path = DATA_DIR / "incoming" / "applications"
    incoming_vouchers_dir: Path = DATA_DIR / "incoming" / "vouchers"
    files_dir: Path = DATA_DIR / "files"

    # 1 у.е. == 1 USD (подтверждено примерами). Валюта берётся из тарифа услуги.
    ue_currency: str = "USD"

    # Ночное время (для тарифов «за операцию» у судов < 2000 GRT).
    night_start_hour: int = 22  # включительно
    night_end_hour: int = 6  # до 06:00

    # Граница валовой вместимости, разделяющая тарифные схемы.
    gross_tonnage_threshold: int = 2000

    # IMAP для приёма заявок по почте (Фаза 2). Оставь пустым, если не используешь.
    imap_host: str = ""
    imap_user: str = ""
    imap_password: str = ""
    imap_folder: str = "INBOX"
    application_sender: str = "agency@sodru.com"


settings = Settings()


# --- ТАРИФНЫЕ КОНСТАНТЫ (ПЛЕЙСХОЛДЕРЫ) --------------------------------------
# Структура: RATES[агент][вид_работ] = правило.
# Поля правила:
#   unit:      per_ton | per_hour | per_operation
#   currency:  USD | RUB
#   divisor:   делитель итоговой суммы (см. вопрос про "/3"); по умолчанию 1
#   rate:      ставка для per_ton / per_hour (у.е.)
#   rate_ice:  ставка в ледовых условиях (если None — используется rate)
#   rate_weekday / rate_holiday_night: для per_operation
#   escort_rate: доп. ставка сопровождения (у.е./час), если применимо
#
# Значения — УСЛОВНЫЕ. Проставь реальные цены.
RATES: dict[str, dict[str, dict]] = {
    "Транс-Агро": {
        "швартовка": {"unit": "per_ton", "currency": "USD", "rate": 0.50,
                      "rate_ice": None, "divisor": 1},
        "отшвартовка": {"unit": "per_ton", "currency": "USD", "rate": 0.50,
                        "rate_ice": None, "divisor": 1},
        "перестановка": {"unit": "per_hour", "currency": "USD", "rate": 100.0,
                         "rate_ice": None, "divisor": 1},
        "сопровождение": {"unit": "per_hour", "currency": "USD", "rate": 1508.0,
                          "rate_ice": None, "divisor": 1},
        "обслуживание судна": {"unit": "per_hour", "currency": "USD", "rate": 590.0,
                               "rate_ice": None, "divisor": 1},
        "обслуживание морских сооружений": {"unit": "per_hour", "currency": "USD",
                                            "rate": 590.0, "rate_ice": None, "divisor": 1},
        "околка льда": {"unit": "per_hour", "currency": "USD", "rate": 200.0,
                        "rate_ice": None, "divisor": 1},
    },
    # Другой договор (пример): расчёт в рублях, курс = 1.0
    "МореСервис": {
        "отшвартовка": {"unit": "per_hour", "currency": "RUB", "rate": 64000.0,
                        "rate_ice": None, "divisor": 1},
        "швартовка": {"unit": "per_hour", "currency": "RUB", "rate": 64000.0,
                      "rate_ice": None, "divisor": 1},
    },
}

# Тарифы «за операцию» для судов < 2000 GRT (плейсхолдеры).
# Ключ: (агент, вид_работ) -> {"weekday": сумма, "holiday_night": сумма, currency, ice-варианты}
RATES_PER_OPERATION: dict[tuple[str, str], dict] = {
    ("Транс-Агро", "швартовка"): {
        "currency": "USD", "weekday": 1000.0, "holiday_night": 1500.0,
        "weekday_ice": 1200.0, "holiday_night_ice": 1800.0,
    },
    ("Транс-Агро", "отшвартовка"): {
        "currency": "USD", "weekday": 1000.0, "holiday_night": 1500.0,
        "weekday_ice": 1200.0, "holiday_night_ice": 1800.0,
    },
    ("Транс-Агро", "перестановка"): {
        "currency": "USD", "weekday": 800.0, "holiday_night": 1200.0,
        "weekday_ice": 1000.0, "holiday_night_ice": 1400.0,
    },
}

# Соответствие терминов заявки -> нормализованный вид работ.
# заявка «вход» == швартовка; «выход»/«перешвартовка (не ТСС_)» == отшвартовка;
# «перешвартовка» == перестановка (перестановка судна).
WORK_TYPE_ALIASES: dict[str, str] = {
    "вход": "швартовка",
    "швартовка": "швартовка",
    "выход": "отшвартовка",
    "отшвартовка": "отшвартовка",
    "перешвартовка": "перестановка",
    "перестановка": "перестановка",
    "сопровождение": "сопровождение",
    "обслуживание судна": "обслуживание судна",
    "обслуживание морских сооружений": "обслуживание морских сооружений",
    "околка льда": "околка льда",
}
