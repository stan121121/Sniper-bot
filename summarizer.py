"""
summarizer.py — Google Gemini через Interactions API (google-genai SDK).
Модель по умолчанию: gemini-3.8-flash.

Изменения по сравнению с предыдущей версией:
  - Переход с generate_content() на Interactions API (client.aio.interactions.create)
  - Модель обновлена до gemini-3.8-flash
  - summarize_posts и fetch_web_news возвращают (items, error)
  - Корректная обработка system_instruction и generation_config
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from google import genai
from config import settings

logger = logging.getLogger(__name__)

# Gemini-клиент (единый на весь модуль)
_client = genai.Client(api_key=settings.GEMINI_API_KEY)


@dataclass
class DigestItem:
    title: str
    summary: str
    importance: int
    channel: str
    url: str
    source_type: str = "telegram"   # "telegram" | "web"


# ── Обработка ошибок ─────────────────────────────────────────────

_ERROR_HINTS = {
    429: "⚠️ Превышен лимит запросов Gemini. Попробуй позже (бесплатный тариф: ~15 запросов/мин).",
    403: "❌ Неверный GEMINI_API_KEY или API не включён. Проверь ключ: https://aistudio.google.com/app/apikey",
    400: "❌ Ошибка в запросе к Gemini. Возможно, превышен размер контекста.",
    404: "❌ Модель не найдена. Проверь GEMINI_MODEL (актуальная: gemini-3.8-flash).",
}


async def _gemini(system: str, user: str, max_tokens: int = 2000) -> str:
    """
    Вызов Gemini через Interactions API.
    Бросает RuntimeError с понятным сообщением при ошибках.
    """
    try:
        interaction = await _client.aio.interactions.create(
            model=settings.GEMINI_MODEL,
            input=user,
            system_instruction=system,
            generation_config={
                "temperature": 0.3,
                "max_output_tokens": max_tokens,
            },
        )

        raw = interaction.output_text.strip()

        # Очистка от markdown-бэктиков
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]

        return raw.strip()

    except Exception as e:
        status = getattr(e, "status_code", None) or getattr(e, "code", None)
        if status and status in _ERROR_HINTS:
            raise RuntimeError(_ERROR_HINTS[status]) from e
        logger.error("Gemini error: %s", e)
        raise RuntimeError(f"⚠️ Ошибка Gemini API: {e}") from e


def _fmt_posts(posts) -> str:
    lines = []
    for i, p in enumerate(posts, 1):
        date_str = p.date.strftime("%d.%m %H:%M")
        lines.append(
            f"[{i}] {p.channel_title} (@{p.channel}) | {date_str}\n"
            f"    {p.text[:400].replace(chr(10), ' ')}\n"
            f"    {p.url}"
        )
    return "\n\n".join(lines)


# ── 1. Фильтрация Telegram-постов ────────────────────────────────

async def summarize_posts(posts) -> tuple[list[DigestItem], Optional[str]]:
    """
    Возвращает (items, error).
    error != None при сбое API — тогда items пустой, и scheduler НЕ помечает
    посты как seen, чтобы повторить попытку при следующем запуске.
    """
    if not posts:
        return [], None

    # Ограничиваем количество постов, передаваемых в AI
    posts = sorted(posts, key=lambda p: p.date, reverse=True)
    posts = posts[: settings.MAX_POSTS_TO_AI]

    system = (
        "Ты редактор новостного дайджеста. Из потока постов выбери ТОЛЬКО важные, "
        "отфильтровав рекламу, репосты без ценности, мелкие события, дубли.\n"
        "СТРОГО JSON-массив. Без пояснений, без markdown-бэктиков."
    )
    user = (
        f"Вот {len(posts)} постов. Выбери не более {settings.MAX_NEWS_IN_DIGEST} важных.\n"
        'Для каждой: "title"(80 симв.), "summary"(2-3 предл.), "importance"(1-10), "channel", "url"\n\n'
        f"Посты:\n{_fmt_posts(posts)}\n\nJSON: [{{...}}, ...]"
    )

    try:
        raw = await _gemini(system, user)
        items = [
            DigestItem(
                title=d.get("title", ""),
                summary=d.get("summary", ""),
                importance=int(d.get("importance", 5)),
                channel=d.get("channel", ""),
                url=d.get("url", ""),
                source_type="telegram",
            )
            for d in json.loads(raw)
            if isinstance(d, dict)
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("TG digest: %d items", len(items))
        return items, None

    except RuntimeError as e:
        logger.error("summarize_posts: %s", e)
        return [], str(e)
    except Exception as e:
        logger.error("summarize_posts unexpected: %s", e)
        return [], f"⚠️ Ошибка: {e}"


# ── 2. Веб-новости ───────────────────────────────────────────────

async def fetch_web_news(
    topic: str = "главные новости дня",
    lang: str = "ru",
) -> tuple[list[DigestItem], Optional[str]]:
    """
    Возвращает (items, error).
    error != None, если API недоступен — бот отправит пользователю подсказку.
    """
    lang_str = "русский" if lang == "ru" else "english"
    system = (
        "Ты редактор новостного дайджеста. Составь список важных новостей "
        "на основе своих знаний. Для каждой укажи реальный источник (Reuters, BBC, РИА и т.д.).\n"
        "СТРОГО JSON-массив. Без пояснений, без markdown-бэктиков."
    )
    user = (
        f"Составь {settings.MAX_NEWS_IN_DIGEST} важных новостей по теме: {topic}.\n"
        f"Язык: {lang_str}.\n"
        'Для каждой: "title"(80 симв.), "summary"(2-3 предл.), "importance"(1-10), '
        '"source"(название СМИ), "url"(если знаешь, иначе "")\n\n'
        "JSON: [{...}, ...]"
    )

    try:
        raw = await _gemini(system, user)
        items = [
            DigestItem(
                title=d.get("title", ""),
                summary=d.get("summary", ""),
                importance=int(d.get("importance", 5)),
                channel=d.get("source", "Web"),
                url=d.get("url", ""),
                source_type="web",
            )
            for d in json.loads(raw)
            if isinstance(d, dict)
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("Web news: %d items", len(items))
        return items, None

    except RuntimeError as e:
        logger.error("fetch_web_news: %s", e)
        return [], str(e)
    except Exception as e:
        logger.error("fetch_web_news unexpected: %s", e)
        return [], None


# ── 3. Итог дня ──────────────────────────────────────────────────

async def generate_day_summary(items: list[DigestItem], lang: str = "ru") -> str:
    if not items:
        return ""

    digest_text = "\n".join(
        f"• [{i.importance}/10] {i.title} — {i.summary}" for i in items
    )
    lang_str = "русский" if lang == "ru" else "english"

    try:
        summary = await _gemini(
            f"Ты аналитик. Напиши ИТОГ ДНЯ — 3-5 предложений: что главное произошло, "
            f"тренды, на что обратить внимание. Язык: {lang_str}. Стиль: деловой.",
            f"Новости дня:\n{digest_text}\n\nНапиши ИТОГ ДНЯ:",
            max_tokens=400,
        )
        return summary.strip()

    except RuntimeError as e:
        logger.warning("day_summary skipped: %s", e)
        return ""
    except Exception as e:
        logger.error("day_summary error: %s", e)
        return ""


# ── 4. Форматирование ─────────────────────────────────────────────

def _he(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_IMP_EMOJI = {10: "🔴", 9: "🔴", 8: "🟠", 7: "🟠", 6: "🟡", 5: "🟡"}


def _item_html(item: DigestItem) -> str:
    emoji = _IMP_EMOJI.get(item.importance, "🟢")
    icon = "🌐" if item.source_type == "web" else "📣"
    link = f' | <a href="{item.url}">Читать →</a>' if item.url else ""
    return (
        f'{emoji} <b>{_he(item.title)}</b>\n'
        f'{_he(item.summary)}\n'
        f'<i>{icon} {_he(item.channel)}</i>{link}\n'
    )


def format_digest_message(
    tg_items: list[DigestItem],
    web_items: list[DigestItem],
    day_summary: str,
    api_error: Optional[str] = None,
    lang: str = "ru",
) -> str:
    parts = []

    if tg_items:
        parts.append("📱 <b>Из ваших каналов</b>\n")
        for item in tg_items:
            parts.append(_item_html(item))

    if web_items:
        parts.append("\n🌐 <b>Важные новости из интернета</b>\n")
        for item in web_items:
            parts.append(_item_html(item))

    if not parts:
        return "📭 Новостей нет — всё тихо." if lang == "ru" else "📭 No news — all quiet."

    if day_summary:
        parts.append(f"\n\n📊 <b>Итог дня</b>\n<i>{_he(day_summary)}</i>")

    if api_error:
        parts.append(f"\n\n⚠️ <i>{_he(api_error)}</i>")

    model_short = settings.GEMINI_MODEL.replace("gemini-", "Gemini ")
    parts.append(f"\n🤖 <i>{_he(model_short)}</i>")

    return "\n".join(parts)
