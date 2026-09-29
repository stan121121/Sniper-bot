"""
summarizer.py — DeepSeek API (OpenAI-совместимый) для фильтрации,
генерации веб-новостей и итога дня.
"""
import json
import logging
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)

# DeepSeek OpenAI-совместимый клиент
_client = AsyncOpenAI(
    api_key=settings.DEEPSEEK_API_KEY,
    base_url="https://api.deepseek.com",
)


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
    402: "❌ Нет баланса на DeepSeek. Пополни счёт: https://platform.deepseek.com/top_up",
    401: "❌ Неверный DEEPSEEK_API_KEY.",
    429: "⚠️ Превышен лимит запросов DeepSeek. Попробуй позже.",
    503: "⚠️ DeepSeek временно недоступен.",
}


async def _deepseek(system: str, user: str, max_tokens: int = 2000) -> str:
    """
    Вызов DeepSeek Chat Completions.
    Бросает исключение с понятным сообщением при ошибках.
    """
    try:
        response = await _client.chat.completions.create(
            model=settings.DEEPSEEK_MODEL,
            max_tokens=max_tokens,
            temperature=0.3,
            messages=[
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        )
        raw = response.choices[0].message.content.strip()
        # Очистка от markdown-бэктиков
        if "```" in raw:
            parts = raw.split("```")
            raw = parts[1] if len(parts) > 1 else parts[0]
            if raw.startswith("json"):
                raw = raw[4:]
        return raw.strip()

    except Exception as e:
        status = getattr(e, "status_code", None) or getattr(e, "http_status", None)
        if status and status in _ERROR_HINTS:
            raise RuntimeError(_ERROR_HINTS[status]) from e
        logger.error("DeepSeek error: %s", e)
        raise RuntimeError(f"⚠️ Ошибка DeepSeek API: {e}") from e


def _fmt_posts(posts) -> str:
    lines = []
    for i, p in enumerate(posts, 1):
        date_str = p.date.strftime("%d.%m %H:%M")
        lines.append(
            f"[{i}] {p.channel_title} (@{p.channel}) | {date_str}\n"
            f"    {p.text[:400].replace(chr(10),' ')}\n"
            f"    {p.url}"
        )
    return "\n\n".join(lines)


# ── 1. Фильтрация Telegram-постов ────────────────────────────────

async def summarize_posts(posts) -> list[DigestItem]:
    if not posts:
        return []

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
        raw = await _deepseek(system, user)
        items = [
            DigestItem(
                title=d.get("title", ""), summary=d.get("summary", ""),
                importance=int(d.get("importance", 5)),
                channel=d.get("channel", ""), url=d.get("url", ""),
                source_type="telegram",
            )
            for d in json.loads(raw) if isinstance(d, dict)
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("TG digest: %d items", len(items))
        return items
    except RuntimeError as e:
        logger.error("summarize_posts: %s", e)
        return []
    except Exception as e:
        logger.error("summarize_posts unexpected: %s", e)
        return []


# ── 2. Веб-новости ───────────────────────────────────────────────

async def fetch_web_news(topic: str = "главные новости дня", lang: str = "ru") -> tuple[list[DigestItem], Optional[str]]:
    """
    Возвращает (items, error_msg).
    error_msg != None если API недоступен.
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
        raw = await _deepseek(system, user)
        items = [
            DigestItem(
                title=d.get("title", ""), summary=d.get("summary", ""),
                importance=int(d.get("importance", 5)),
                channel=d.get("source", "Web"), url=d.get("url", ""),
                source_type="web",
            )
            for d in json.loads(raw) if isinstance(d, dict)
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("Web news: %d items", len(items))
        return items, None
    except RuntimeError as e:
        hint = str(e)
        logger.error("fetch_web_news: %s", e)
        return [], hint
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
        summary = await _deepseek(
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

    model_short = settings.DEEPSEEK_MODEL.replace("deepseek-", "")
    parts.append(f"\n🤖 <i>DeepSeek {model_short}</i>")

    return "\n".join(parts)
