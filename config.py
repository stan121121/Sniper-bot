from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Telegram
    BOT_TOKEN: str

    # DeepSeek
    DEEPSEEK_API_KEY: str
    DEEPSEEK_MODEL: str = "deepseek-flash"

    # Дайджест
    DEFAULT_DIGEST_INTERVAL_HOURS: int = 4
    POSTS_PER_CHANNEL: int = 20
    MAX_NEWS_IN_DIGEST: int = 10
    MAX_POSTS_TO_AI: int = 50          # ← ограничение постов для AI
    DIGEST_LANGUAGE: str = "ru"
    DB_PATH: str = "bot_data.db"

    # Веб-новости
    INCLUDE_WEB_NEWS: bool = True
    WEB_NEWS_TOPIC: str = "главные мировые и российские новости дня"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"


settings = Settings()
