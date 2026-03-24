"""
summarizer.py — OpenRouter: фильтрация + веб-новости + итог дня
Мягкая обработка ошибок API (402, 429, 500 и т.д.)
"""
import json
import logging
from dataclasses import dataclass, field
from typing import Optional

import httpx
from config import settings

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"


def _auth_headers() -> dict:
    return {
        "Authorization": f"Bearer {settings.OPENROUTER_API_KEY}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/newsdigestbot",
        "X-Title": "News Digest Bot",
    }


@dataclass
class DigestItem:
    title: str
    summary: str
    importance: int
    channel: str
    url: str
    source_type: str = "telegram"   # "telegram" | "web"


# ── OpenRouter helper ─────────────────────────────────────────────

class OpenRouterError(Exception):
    def __init__(self, status: int, msg: str):
        self.status = status
        super().__init__(f"HTTP {status}: {msg}")

_ERROR_HINTS = {
    402: "❌ Нет баланса на OpenRouter. Пополни счёт: https://openrouter.ai/credits",
    401: "❌ Неверный OPENROUTER_API_KEY.",
    429: "⚠️ Превышен лимит запросов OpenRouter. Попробуй позже.",
    503: "⚠️ OpenRouter временно недоступен.",
}

async def _openrouter(system: str, user: str, max_tokens: int = 2000) -> str:
    """
    Вызов OpenRouter API.
    Бросает OpenRouterError с понятным сообщением при ошибках.
    """
    payload = {
        "model": settings.OPENROUTER_MODEL,
        "max_tokens": max_tokens,
        "temperature": 0.3,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user",   "content": user},
        ],
    }
    async with httpx.AsyncClient(timeout=60.0) as http:
        resp = await http.post(OPENROUTER_URL, json=payload, headers=_auth_headers())

    if not resp.is_success:
        hint = _ERROR_HINTS.get(resp.status_code, f"HTTP {resp.status_code}")
        logger.error("OpenRouter %d: %s", resp.status_code, resp.text[:200])
        raise OpenRouterError(resp.status_code, hint)

    raw = resp.json()["choices"][0]["message"]["content"].strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


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
        raw = await _openrouter(system, user)
        items = [
            DigestItem(
                title=d.get("title",""), summary=d.get("summary",""),
                importance=int(d.get("importance",5)),
                channel=d.get("channel",""), url=d.get("url",""),
                source_type="telegram",
            )
            for d in json.loads(raw) if isinstance(d, dict)
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("TG digest: %d items", len(items))
        return items
    except OpenRouterError as e:
        logger.error("summarize_posts: %s", e)
        return []
    except Exception as e:
        logger.error("summarize_posts unexpected: %s", e)
        return []


# ── 2. Веб-новости (AI генерирует из своих знаний) ───────────────

async def fetch_web_news(topic: str = "главные новости дня", lang: str = "ru") -> tuple[list[DigestItem], Optional[str]]:
    """
    Возвращает (items, error_msg).
    error_msg != None если API недоступен — бот отправит пользователю подсказку.
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
        raw = await _openrouter(system, user)
        items = [
            DigestItem(
                title=d.get("title",""), summary=d.get("summary",""),
                importance=int(d.get("importance",5)),
                channel=d.get("source","Web"), url=d.get("url",""),
                source_type="web",
            )
            for d in json.loads(raw) if isinstance(d, dict)
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("Web news: %d items", len(items))
        return items, None
    except OpenRouterError as e:
        hint = str(e).split(": ", 1)[-1]   # берём текст после "HTTP NNN: "
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
        summary = await _openrouter(
            f"Ты аналитик. Напиши ИТОГ ДНЯ — 3-5 предложений: что главное произошло, "
            f"тренды, на что обратить внимание. Язык: {lang_str}. Стиль: деловой.",
            f"Новости дня:\n{digest_text}\n\nНапиши ИТОГ ДНЯ:",
            max_tokens=400,
        )
        return summary.strip()
    except OpenRouterError as e:
        logger.warning("day_summary skipped: %s", e)
        return ""
    except Exception as e:
        logger.error("day_summary error: %s", e)
        return ""


# ── 4. Форматирование ─────────────────────────────────────────────

def _he(t: str) -> str:
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

_IMP_EMOJI = {10:"🔴",9:"🔴",8:"🟠",7:"🟠",6:"🟡",5:"🟡"}

def _item_html(item: DigestItem) -> str:
    emoji = _IMP_EMOJI.get(item.importance, "🟢")
    icon  = "🌐" if item.source_type == "web" else "📣"
    link  = f' | <a href="{item.url}">Читать →</a>' if item.url else ""
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

    # Показываем ошибку API как предупреждение (не крашим дайджест)
    if api_error:
        parts.append(f"\n\n⚠️ <i>{_he(api_error)}</i>")

    model_short = settings.OPENROUTER_MODEL.split("/")[-1]
    parts.append(f"\n🤖 <i>{_he(model_short)}</i>")

    return "\n".join(parts)
