from pydantic_settings import BaseSettings

class Settings(BaseSettings):
    BOT_TOKEN: str
    OPENROUTER_API_KEY: str
    OPENROUTER_MODEL: str = "anthropic/claude-3.5-sonnet"

    DEFAULT_DIGEST_INTERVAL_HOURS: int = 4
    POSTS_PER_CHANNEL: int = 20
    MAX_NEWS_IN_DIGEST: int = 10
    DIGEST_LANGUAGE: str = "ru"
    DB_PATH: str = "bot_data.db"

    # Веб-новости через AI
    INCLUDE_WEB_NEWS: bool = True
    WEB_NEWS_TOPIC: str = "главные мировые и российские новости дня"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

settings = Settings()
