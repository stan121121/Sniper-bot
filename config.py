from pydantic_settings import BaseSettings


class Settings(BaseSettings):
    # Telegram
    BOT_TOKEN: str

    # ── LLM Provider (выбор провайдера и модели) ──────────────────
    # Допустимые значения: "gemini", "deepseek", "qwen", "openrouter"
    LLM_PROVIDER: str = "gemini"

    # Название основной модели (зависит от провайдера)
    # Примеры: "gemini-3.8-flash", "deepseek-flash", "qwen-max", "deepseek/deepseek-v4-flash"
    LLM_MODEL: str = "gemini-3.8-flash"
    
    # Резервная модель (для fallback при 503/429)
    LLM_FALLBACK_MODEL: str = "gemini-3.1-flash-lite"

    # ── API Ключи (используется только ключ выбранного провайдера) ──
    GEMINI_API_KEY: str = ""
    DEEPSEEK_API_KEY: str = ""
    QWEN_API_KEY: str = ""
    OPENROUTER_API_KEY: str = ""

    # ── Настройки Retry / Fallback ──────────────────────────────
    LLM_MAX_RETRIES: int = 5
    LLM_RETRY_DELAY: float = 2.0

    # ── Часовой пояс для расписания ─────────────────────────────
    TIMEZONE: str = "Europe/Moscow"

    # ── Дайджест ────────────────────────────────────────────────
    DEFAULT_DIGEST_INTERVAL_HOURS: int = 4
    POSTS_PER_CHANNEL: int = 20
    MAX_NEWS_IN_DIGEST: int = 15
    MAX_POSTS_TO_AI: int = 70
    POST_TEXT_LIMIT: int = 500
    DIGEST_LANGUAGE: str = "ru"
    DB_PATH: str = "bot_data.db"

    # ── Веб-новости (ОТКЛЮЧЕНЫ) ─────────────────────────────────
    INCLUDE_WEB_NEWS: bool = False
    WEB_NEWS_TOPIC: str = "главные мировые и российские новости дня"

    class Config:
        env_file = ".env"
        env_file_encoding = "utf-8"

    # ── Хелперы для получения конфигурации провайдера ────────────
    def get_api_key(self) -> str:
        mapping = {
            "gemini": self.GEMINI_API_KEY,
            "deepseek": self.DEEPSEEK_API_KEY,
            "qwen": self.QWEN_API_KEY,
            "openrouter": self.OPENROUTER_API_KEY,
        }
        key = mapping.get(self.LLM_PROVIDER.lower(), "")
        if not key:
            raise ValueError(f"API key для провайдера '{self.LLM_PROVIDER}' не задан.")
        return key

    def get_base_url(self) -> str:
        """Возвращает OpenAI-совместимый base_url для выбранного провайдера."""
        mapping = {
            "gemini": "https://generativelanguage.googleapis.com/v1beta/openai/",
            "deepseek": "https://api.deepseek.com",
            "qwen": "https://dashscope-intl.aliyuncs.com/compatible-mode/v1",
            "openrouter": "https://openrouter.ai/api/v1",
        }
        return mapping.get(self.LLM_PROVIDER.lower(), "")


settings = Settings()
