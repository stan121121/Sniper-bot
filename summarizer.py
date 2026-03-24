"""
summarizer.py — OpenRouter: фильтрация новостей + веб-поиск + итог дня
"""
import json
import logging
from dataclasses import dataclass

import httpx
from config import settings

logger = logging.getLogger(__name__)

OPENROUTER_URL = "https://openrouter.ai/api/v1/chat/completions"
HEADERS = {
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


# ── Вспомогательные функции ──────────────────────────────────────

def _he(t: str) -> str:
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

async def _openrouter(system: str, user: str, max_tokens: int = 2000) -> str:
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
        resp = await http.post(OPENROUTER_URL, json=payload, headers=HEADERS)
        resp.raise_for_status()
    raw = resp.json()["choices"][0]["message"]["content"].strip()
    # убираем markdown-бэктики если модель их вставила
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


# ── 1. Фильтрация Telegram-постов ───────────────────────────────

FILTER_SYSTEM = (
    "Ты редактор новостного дайджеста. Из потока постов выбери ТОЛЬКО важные новости, "
    "отфильтровав: рекламу, репосты без ценности, мелкие события, дубли.\n"
    "Отвечай СТРОГО JSON-массивом без пояснений и без markdown-бэктиков."
)
FILTER_USER = (
    "Вот {count} постов. Выбери не более {max_n} самых важных.\n"
    "Для каждой новости верни:\n"
    '  "title"(до 80 симв.), "summary"(2-3 предл.), "importance"(1-10), "channel", "url"\n\n'
    "Посты:\n{posts}\n\n"
    "Верни JSON: [{{...}}, ...]"
)

async def summarize_posts(posts) -> list[DigestItem]:
    if not posts:
        return []
    try:
        raw = await _openrouter(
            FILTER_SYSTEM,
            FILTER_USER.format(
                count=len(posts),
                max_n=settings.MAX_NEWS_IN_DIGEST,
                posts=_fmt_posts(posts),
            ),
        )
        items = [
            DigestItem(
                title=d.get("title",""),
                summary=d.get("summary",""),
                importance=int(d.get("importance", 5)),
                channel=d.get("channel",""),
                url=d.get("url",""),
                source_type="telegram",
            )
            for d in json.loads(raw)
            if isinstance(d, dict)
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("Telegram digest: %d items", len(items))
        return items
    except Exception as e:
        logger.error("summarize_posts error: %s", e)
        return []


# ── 2. Веб-новости (AI сам ищет важное из интернета) ────────────

WEB_SYSTEM = (
    "Ты редактор новостного дайджеста. Твоя задача — составить список "
    "самых важных мировых новостей прямо сейчас (на основе своих знаний и "
    "обучающих данных). Для каждой новости укажи реальный источник (Reuters, BBC, RIA и т.д.).\n"
    "Отвечай СТРОГО JSON-массивом без пояснений и без markdown-бэктиков."
)
WEB_USER = (
    "Составь список из {max_n} важных новостей дня по теме: {topic}.\n"
    "Язык ответа: {lang}.\n"
    "Для каждой: "
    '"title"(до 80 симв.), "summary"(2-3 предл.), "importance"(1-10), '
    '"source"(название СМИ), "url"(официальный URL статьи если знаешь, иначе "")\n\n'
    "Верни JSON: [{{...}}, ...]"
)

async def fetch_web_news(topic: str = "главные новости дня", lang: str = "ru") -> list[DigestItem]:
    """AI генерирует топ-новости из своих знаний с указанием источника."""
    try:
        raw = await _openrouter(
            WEB_SYSTEM,
            WEB_USER.format(
                max_n=settings.MAX_NEWS_IN_DIGEST,
                topic=topic,
                lang="русский" if lang == "ru" else "english",
            ),
        )
        items = [
            DigestItem(
                title=d.get("title",""),
                summary=d.get("summary",""),
                importance=int(d.get("importance", 5)),
                channel=d.get("source","Web"),
                url=d.get("url",""),
                source_type="web",
            )
            for d in json.loads(raw)
            if isinstance(d, dict)
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("Web news: %d items for topic '%s'", len(items), topic)
        return items
    except Exception as e:
        logger.error("fetch_web_news error: %s", e)
        return []


# ── 3. Итог дня ─────────────────────────────────────────────────

DAY_SYSTEM = (
    "Ты аналитик. Тебе дан список новостей за день. "
    "Напиши краткое аналитическое заключение (ИТОГ ДНЯ) — 3-5 предложений: "
    "что главное произошло, какие тренды, на что обратить внимание. "
    "Язык: {lang}. Стиль: деловой, без воды."
)
DAY_USER = "Новости дня:\n{digest}\n\nНапиши ИТОГ ДНЯ:"

async def generate_day_summary(items: list[DigestItem], lang: str = "ru") -> str:
    if not items:
        return ""
    digest_text = "\n".join(
        f"• [{i.importance}/10] {i.title} — {i.summary}"
        for i in items
    )
    try:
        summary = await _openrouter(
            DAY_SYSTEM.format(lang="русский" if lang == "ru" else "english"),
            DAY_USER.format(digest=digest_text),
            max_tokens=500,
        )
        return summary.strip()
    except Exception as e:
        logger.error("generate_day_summary error: %s", e)
        return ""


# ── 4. Форматирование сообщений ──────────────────────────────────

_IMPORTANCE_EMOJI = {10:"🔴",9:"🔴",8:"🟠",7:"🟠",6:"🟡",5:"🟡"}

def _item_html(item: DigestItem) -> str:
    emoji = _IMPORTANCE_EMOJI.get(item.importance, "🟢")
    source_icon = "🌐" if item.source_type == "web" else "📣"
    link = f' | <a href="{item.url}">Читать →</a>' if item.url else ""
    return (
        f'{emoji} <b>{_he(item.title)}</b>\n'
        f'{_he(item.summary)}\n'
        f'<i>{source_icon} {_he(item.channel)}</i>{link}\n'
    )


def format_digest_message(
    tg_items: list[DigestItem],
    web_items: list[DigestItem],
    day_summary: str,
    lang: str = "ru",
) -> str:
    parts = []

    # Telegram-блок
    if tg_items:
        parts.append("📱 <b>Из ваших каналов</b>\n")
        for item in tg_items:
            parts.append(_item_html(item))

    # Web-блок
    if web_items:
        parts.append("\n🌐 <b>Важные новости из интернета</b>\n")
        for item in web_items:
            parts.append(_item_html(item))

    if not parts:
        return "📭 Новостей нет — всё тихо." if lang == "ru" else "📭 No news — all quiet."

    # Итог дня
    if day_summary:
        parts.append(f"\n\n📊 <b>Итог дня</b>\n<i>{_he(day_summary)}</i>")

    model_short = settings.OPENROUTER_MODEL.split("/")[-1]
    parts.append(f"\n\n🤖 <i>{_he(model_short)}</i>")

    return "\n".join(parts)
