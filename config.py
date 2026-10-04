from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Telegram
    BOT_TOKEN: str

    # Google Gemini
    GEMINI_API_KEY: str
    GEMINI_MODEL: str = "gemini-3.8-flash"
    GEMINI_FALLBACK_MODEL: str = "gemini-3.1-flash-lite"
    GEMINI_MAX_RETRIES: int = 5
    GEMINI_RETRY_DELAY: float = 2.0

    # Часовой пояс для расписания (IANA name).
    # Все времена в команде /schedule интерпретируются в этом поясе.
    TIMEZONE: str = "Europe/Moscow"

    # Дайджест
    DEFAULT_DIGEST_INTERVAL_HOURS: int = 4
    POSTS_PER_CHANNEL: int = 25
    MAX_NEWS_IN_DIGEST: int = 25
    MAX_POSTS_TO_AI: int = 50
    DIGEST_LANGUAGE: str = "ru"
    DB_PATH: str = "bot_data.db"

    # Веб-новости — по умолчанию ОТКЛЮЧЕНЫ.
    # Бот работает только с постами из каналов пользователя.
    INCLUDE_WEB_NEWS: bool = False
    WEB_NEWS_TOPIC: str = "главные мировые и российские новости дня"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
