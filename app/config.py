"""Конфигурация приложения и тарифы группы A."""

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


class TariffsGroupA(BaseSettings):
    """Тарифы Транс-Агро, загружаемые из переменных A_* в .env."""

    model_config = SettingsConfigDict(env_file=".env", env_prefix="A_", extra="ignore")

    escort_vessel_meeting_departure_cargo_canal: float = 0
    escort_vessel_meeting_departure_cargo_canal_ice: float = 0
    barge_towing_canal: float = 0
    barge_towing_canal_ice: float = 0
    mooring_unmooring_gt_2000_above: float = 0
    mooring_unmooring_gt_2000_above_ice: float = 0
    vessel_repositioning_gt_2000_above: float = 0
    vessel_repositioning_gt_2000_above_ice: float = 0
    mooring_unmooring_gt_below_2000_weekdays: float = 0
    mooring_unmooring_gt_below_2000_holiday_night: float = 0
    mooring_unmooring_gt_below_2000_ice_weekdays: float = 0
    mooring_unmooring_gt_below_2000_ice_holiday_night: float = 0
    vessel_repositioning_gt_below_2000_weekdays: float = 0
    vessel_repositioning_gt_below_2000_holiday_night: float = 0
    vessel_repositioning_gt_below_2000_ice_weekdays: float = 0
    vessel_repositioning_gt_below_2000_ice_holiday_night: float = 0
    escort_towing_gt_below_2000_ice_weekdays: float = 0
    escort_towing_gt_below_2000_ice_holiday_night: float = 0
    ice_breaking_tug_vessel_approach_departure: float = 0
    vessel_maintenance_services: float = 0
    offshore_facilities_maintenance_services: float = 0


tariffs_a = TariffsGroupA()

GROUP_A_AGENTS = {"Транс-Агро"}
GROUP_B_AGENTS = {"Терминал", "Содружество-Соя"}


def agent_group(agent: str | None) -> str:
    if agent in GROUP_A_AGENTS:
        return "A"
    if agent in GROUP_B_AGENTS:
        return "B"
    return "C"

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
    "буксировка баржи": "буксировка баржи",
    "обслуживание судна": "обслуживание судна",
    "обслуживание морских сооружений": "обслуживание морских сооружений",
    "околка льда": "околка льда",
}
