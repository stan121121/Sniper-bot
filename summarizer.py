"""
summarizer.py — Google Gemini через стабильный generate_content API.

Изменения:
  - УДАЛЕН «Итог дня» (generate_day_summary).
  - Расширен охват: MAX_POSTS_TO_AI=70, MAX_NEWS_IN_DIGEST=15, POST_TEXT_LIMIT=500.
  - Промпты на английском для экономии токенов, вывод — на русском.
  - Устойчивый парсинг JSON, retry + fallback.
"""
import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

from google import genai
from config import settings

logger = logging.getLogger(__name__)

_client = genai.Client(api_key=settings.GEMINI_API_KEY)


# ── Модель данных ────────────────────────────────────────────────

@dataclass
class DigestItem:
    title: str
    summary: str
    why: str = ""
    framing: str = ""
    importance: int = 5
    channel: str = ""
    url: str = ""
    source_type: str = "telegram"


# ── Обработка ошибок ─────────────────────────────────────────────

_ERROR_HINTS = {
    400: "❌ Gemini отклонил запрос (400). Проверь размер контекста или формат.",
    429: "⚠️ Лимит запросов Gemini. Попробуй позже.",
    403: "❌ Неверный GEMINI_API_KEY или API не включён.",
    404: "❌ Модель не найдена. Проверь GEMINI_MODEL.",
    503: "⚠️ Gemini временно перегружен. Повторяю запрос...",
}

_RETRYABLE_STATUS = {429, 503, 500, 502, 504}


# ── Устойчивый парсинг JSON ──────────────────────────────────────

def _extract_json_array(raw: str) -> list[dict]:
    """Парсит JSON-массив из ответа Gemini, выдерживая обрезанные ответы."""
    if not raw:
        return []

    cleaned = raw.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else parts[0]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # Fallback: обход сбалансированных скобок
    items = []
    depth = 0
    start = None
    for i, ch in enumerate(cleaned):
        if ch == "{":
            if depth == 0:
                start = i
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0 and start is not None:
                chunk = cleaned[start:i + 1]
                try:
                    obj = json.loads(chunk)
                    if isinstance(obj, dict):
                        items.append(obj)
                except json.JSONDecodeError:
                    try:
                        obj = json.loads(_repair_json_chunk(chunk))
                        if isinstance(obj, dict):
                            items.append(obj)
                    except json.JSONDecodeError:
                        logger.debug("Skipped malformed chunk: %s", chunk[:120])
                start = None

    if items:
        logger.info("Recovered %d objects from partial JSON", len(items))
    else:
        logger.warning("Could not extract any JSON objects from response")

    return items


def _repair_json_chunk(chunk: str) -> str:
    """Правки типичных ошибок JSON."""
    fixed = re.sub(r"(?<![\\])'", '"', chunk)
    fixed = re.sub(r",\s*}", "}", fixed)
    return fixed


def _clean_text(text: str) -> str:
    """Убирает эмодзи и опасные для JSON символы."""
    text = re.sub(r'[\U00010000-\U0010ffff]', '', text)
    text = text.replace('"', "'").replace("\\", "")
    return text.strip()


# ── Вызов Gemini ─────────────────────────────────────────────────

async def _gemini_call(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool = True,
) -> str:
    """Один вызов Gemini через стабильный generate_content."""
    config = {
        "system_instruction": system,
        "temperature": 0.3,
        "max_output_tokens": max_tokens,
    }
    if json_mode:
        config["response_mime_type"] = "application/json"

    response = await _client.aio.models.generate_content(
        model=model,
        contents=user,
        config=config,
    )
    return (response.text or "").strip()


async def _gemini(
    system: str,
    user: str,
    max_tokens: int = 2000,
    json_mode: bool = True,
) -> str:
    """Вызов Gemini с retry и fallback."""
    models = [settings.GEMINI_MODEL, settings.GEMINI_FALLBACK_MODEL]
    last_error = None

    for model_idx, model in enumerate(models):
        if model_idx > 0:
            logger.info("Falling back to model: %s", model)

        for attempt in range(settings.GEMINI_MAX_RETRIES):
            try:
                result = await _gemini_call(model, system, user, max_tokens, json_mode)
                if model_idx > 0:
                    logger.info("Fallback model %s succeeded", model)
                return result

            except Exception as e:
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                last_error = e

                if status and status not in _RETRYABLE_STATUS:
                    hint = _ERROR_HINTS.get(status, f"HTTP {status}")
                    logger.error("Non-retryable error %s: %s", status, hint)
                    break

                if attempt < settings.GEMINI_MAX_RETRIES - 1:
                    delay = settings.GEMINI_RETRY_DELAY * (2 ** attempt)
                    delay += random.uniform(0, 1)
                    logger.warning(
                        "Gemini %s error (attempt %d/%d), retrying in %.1fs: %s",
                        status, attempt + 1, settings.GEMINI_MAX_RETRIES, delay, e
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error(
                        "Model %s failed after %d attempts: %s",
                        model, settings.GEMINI_MAX_RETRIES, e
                    )

    status = getattr(last_error, "status_code", None) or getattr(last_error, "code", None)
    if status and status in _ERROR_HINTS:
        raise RuntimeError(_ERROR_HINTS[status]) from last_error
    raise RuntimeError(f"⚠️ Gemini API error after all attempts: {last_error}") from last_error


# ── Форматирование постов для промпта ────────────────────────────

def _fmt_posts(posts) -> str:
    """Компактное представление постов для промпта."""
    limit = settings.POST_TEXT_LIMIT
    lines = []
    for i, p in enumerate(posts, 1):
        date_str = p.date.strftime("%d.%m %H:%M")
        text = _clean_text(p.text[:limit]).replace(chr(10), " ")
        lines.append(f"[{i}] @{p.channel} | {date_str} | {text} | URL: {p.url}")
    return "\n".join(lines)


# ── 1. Фильтрация Telegram-постов ────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an editor of a personal news digest from Telegram channels.\n"
    "Your task: select important posts from the PROVIDED LIST and compress them "
    "into short, informative summaries.\n\n"

    "=== HARD RULES ===\n\n"

    "1. SOURCE OF TRUTH — only the post texts in the list.\n"
    "   - Do NOT add facts, names, numbers, dates or events not present in the posts.\n"
    "   - Do NOT use your own world knowledge.\n"
    "   - Do NOT search the internet.\n"
    "   - If something is missing in a post, do not invent it.\n\n"

    "2. DEDUPLICATION.\n"
    "   - If several posts describe the same event, merge them into one entry.\n"
    "   - Preserve differences in framing: if channels present the event "
    "differently, note it in the 'framing' field.\n"
    "   - In 'channel' list all channels separated by comma.\n"
    "   - In 'url' use the link of the most informative post.\n\n"

    "3. FILTERING.\n"
    "   Discard: ads, low-value reposts, memes, greetings, duplicates, weather, "
    "sports without significant context, celebrity news, local incidents "
    "without broader significance.\n\n"

    "4. PRIORITIES (what counts as important):\n"
    "   1) Technology & AI — products, policy, business\n"
    "   2) Business, startups, entrepreneurship\n"
    "   3) Economics: markets, macro, trade, rates\n"
    "   4) Political decisions with real-world consequences\n"
    "   5) Regulatory changes (taxes, laws, sanctions)\n"
    "   6) High-quality analysis that shifts perspective\n\n"

    "5. LANGUAGE REQUIREMENT (CRITICAL).\n"
    "   - ALL text field VALUES must be in RUSSIAN: title, summary, why, framing.\n"
    "   - Field names and JSON structure stay in English.\n"
    "   - The 'channel' field keeps @usernames as-is.\n"
    "   - The 'url' field keeps URLs as-is.\n\n"

    "6. FORMAT.\n"
    "   - Output STRICTLY a JSON array of objects.\n"
    "   - No markdown, no backticks, no explanations before or after JSON.\n\n"

    "=== OBJECT SCHEMA ===\n\n"
    "{\n"
    '  "title":      "up to 80 chars, RUSSIAN, specific, no clickbait",\n'
    '  "summary":    "1-2 sentences, RUSSIAN, only facts from the post",\n'
    '  "why":        "1 sentence, RUSSIAN, why it matters",\n'
    '  "framing":    "differences in framing between channels, RUSSIAN, or empty string",\n'
    '  "importance": 1-10,\n'
    '  "channel":    "@channel1, @channel2",\n'
    '  "url":        "link from post or empty string"\n'
    "}\n\n"

    "=== LIMITS ===\n\n"
    "• Return no more than the requested number of important news items.\n"
    "• If fewer qualify, return fewer — do NOT pad.\n"
    "• If all posts are garbage, return [].\n"
)


async def summarize_posts(posts) -> tuple[list[DigestItem], Optional[str]]:
    """Отбирает важные посты ТОЛЬКО из переданного списка."""
    if not posts:
        return [], None

    posts = sorted(posts, key=lambda p: p.date, reverse=True)
    posts = posts[: settings.MAX_POSTS_TO_AI]

    user = (
        f"Analyze {len(posts)} posts from user's Telegram channels.\n"
        f"Select no more than {settings.MAX_NEWS_IN_DIGEST} most important "
        "and return a JSON array.\n"
        "REMINDER: all text values in Russian.\n\n"
        f"Posts:\n{_fmt_posts(posts)}\n\n"
        "JSON:"
    )

    # Щедрый лимит: 700 базовых + 350 на каждую новость
    max_tokens = 700 + settings.MAX_NEWS_IN_DIGEST * 350

    try:
        raw = await _gemini(_SYSTEM_PROMPT, user, max_tokens=max_tokens, json_mode=True)
        data = _extract_json_array(raw)

        items = []
        for d in data:
            try:
                items.append(DigestItem(
                    title=str(d.get("title", ""))[:200],
                    summary=str(d.get("summary", ""))[:1000],
                    why=str(d.get("why", ""))[:500],
                    framing=str(d.get("framing", ""))[:500],
                    importance=int(d.get("importance", 5)),
                    channel=str(d.get("channel", "")),
                    url=str(d.get("url", "")),
                    source_type="telegram",
                ))
            except (ValueError, TypeError) as e:
                logger.debug("Skipped malformed item: %s (%s)", d, e)

        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("TG digest: %d items (from %d posts)", len(items), len(posts))
        return items, None

    except RuntimeError as e:
        logger.error("summarize_posts: %s", e)
        return [], str(e)
    except Exception as e:
        logger.error("summarize_posts unexpected: %s", e, exc_info=True)
        return [], f"⚠️ Error: {e}"


# ── 2. Веб-новости (не используется, оставлено для совместимости) ─

async def fetch_web_news(
    topic: str = "главные новости дня",
    lang: str = "ru",
) -> tuple[list[DigestItem], Optional[str]]:
    """По умолчанию НЕ вызывается. Оставлено для совместимости."""
    system = (
        "You are a news digest editor. Compile a list of important news.\n"
        "Output a JSON array. All text values in Russian. No markdown."
    )
    user = (
        f"Compile {settings.MAX_NEWS_IN_DIGEST} important news on: {topic}.\n"
        'Fields: "title", "summary", "why", "importance" (1-10), "source", "url".\n'
        "JSON:"
    )

    try:
        raw = await _gemini(
            system, user,
            max_tokens=500 + settings.MAX_NEWS_IN_DIGEST * 300,
            json_mode=True,
        )
        data = _extract_json_array(raw)
        items = [
            DigestItem(
                title=str(d.get("title", "")),
                summary=str(d.get("summary", "")),
                why=str(d.get("why", "")),
                importance=int(d.get("importance", 5)),
                channel=str(d.get("source", "Web")),
                url=str(d.get("url", "")),
                source_type="web",
            )
            for d in data
        ]
        items.sort(key=lambda x: x.importance, reverse=True)
        return items, None
    except RuntimeError as e:
        logger.error("fetch_web_news: %s", e)
        return [], str(e)
    except Exception as e:
        logger.error("fetch_web_news unexpected: %s", e)
        return [], None


# ── 3. Форматирование для Telegram ───────────────────────────────

def _he(t: str) -> str:
    return t.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


_IMP_EMOJI = {10: "🔴", 9: "🔴", 8: "🟠", 7: "🟠", 6: "🟡", 5: "🟡"}


def _item_html(item: DigestItem) -> str:
    emoji = _IMP_EMOJI.get(item.importance, "🟢")
    icon = "🌐" if item.source_type == "web" else "📣"
    link = f' | <a href="{item.url}">Читать →</a>' if item.url else ""

    block = f'{emoji} <b>{_he(item.title)}</b>\n'
    block += f'{_he(item.summary)}\n'

    if item.why:
        block += f'<b>Почему важно:</b> {_he(item.why)}\n'

    if item.framing:
        block += f'<b>Разные взгляды:</b> <i>{_he(item.framing)}</i>\n'

    block += f'<i>{icon} {_he(item.channel)}</i>{link}\n'
    return block


def format_digest_message(
    tg_items: list[DigestItem],
    web_items: list[DigestItem],
    api_error: Optional[str] = None,
    lang: str = "ru",
) -> str:
    """
    Собирает финальное сообщение дайджеста.
    «Итог дня» УДАЛЁН — параметр day_summary больше не принимается.
    """
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

    if api_error:
        parts.append(f"\n\n⚠️ <i>{_he(api_error)}</i>")

    model_short = settings.GEMINI_MODEL.replace("gemini-", "Gemini ")
    parts.append(f"\n🤖 <i>{_he(model_short)}</i>")

    return "\n".join(parts)
