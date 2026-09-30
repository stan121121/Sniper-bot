"""
summarizer.py — Google Gemini через стабильный generate_content API.

Ключевые изменения:
  - Возврат с экспериментального Interactions API на models.generate_content
  - Увеличен max_tokens для summarize_posts (3500 вместо 2000)
  - Устойчивый парсинг JSON: если ответ обрезан, извлекаем целые объекты
  - Модель работает СТРОГО с постами каналов (без интернета)
  - Retry с экспоненциальной задержкой + fallback на резервную модель
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


@dataclass
class DigestItem:
    title: str
    summary: str
    importance: int
    channel: str
    url: str
    source_type: str = "telegram"


# ── Обработка ошибок ─────────────────────────────────────────────

_ERROR_HINTS = {
    429: "⚠️ Превышен лимит запросов Gemini. Попробуй позже.",
    403: "❌ Неверный GEMINI_API_KEY или API не включён.",
    400: "❌ Ошибка в запросе к Gemini. Возможно, превышен размер контекста.",
    404: "❌ Модель не найдена. Проверь GEMINI_MODEL (актуальная: gemini-3.8-flash).",
    503: "⚠️ Gemini временно перегружен. Повторяю запрос...",
}

_RETRYABLE_STATUS = {429, 503, 500, 502, 504}


# ── Устойчивый парсинг JSON ──────────────────────────────────────

def _extract_json_array(raw: str) -> list[dict]:
    """
    Пытается распарсить JSON-массив из ответа Gemini.
    Если ответ обрезан — извлекает все целые объекты { ... } через regex.
    Возвращает список словарей (пустой, если ничего не удалось).
    """
    if not raw:
        return []

    # 1. Чистим markdown-обёртки
    cleaned = raw.strip()
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else parts[0]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    # 2. Пробуем обычный парсинг
    try:
        data = json.loads(cleaned)
        if isinstance(data, list):
            return [d for d in data if isinstance(d, dict)]
        if isinstance(data, dict):
            return [data]
    except json.JSONDecodeError:
        pass

    # 3. Fallback: вытаскиваем целые объекты {...} через regex.
    #    Работает даже если массив обрезан и не закрыт.
    items = []
    # Ищем сбалансированные фигурные скобки вручную
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
                    # Пробуем починить типичные проблемы
                    fixed = _repair_json_chunk(chunk)
                    try:
                        obj = json.loads(fixed)
                        if isinstance(obj, dict):
                            items.append(obj)
                    except json.JSONDecodeError:
                        logger.debug("Skipped malformed JSON chunk: %s", chunk[:120])
                start = None

    if items:
        logger.info("Recovered %d objects from partial JSON", len(items))
    else:
        logger.warning("Could not extract any JSON objects from response")

    return items


def _repair_json_chunk(chunk: str) -> str:
    """Простые правки типичных ошибок: одиночные кавычки, висячие запятые."""
    # Одиночные кавычки → двойные (грубо, но помогает в 90% случаев)
    fixed = re.sub(r"(?<![\\])'", '"', chunk)
    # Убираем висячие запятые перед } 
    fixed = re.sub(r",\s*}", "}", fixed)
    return fixed


# ── Вызов Gemini ─────────────────────────────────────────────────

async def _gemini_call(model: str, system: str, user: str, max_tokens: int) -> str:
    """Один вызов Gemini через стабильный generate_content."""
    response = await _client.aio.models.generate_content(
        model=model,
        contents=user,
        config={
            "system_instruction": system,
            "temperature": 0.3,
            "max_output_tokens": max_tokens,
            # Просим Gemini вернуть строго JSON
            "response_mime_type": "application/json",
        },
    )
    return (response.text or "").strip()


async def _gemini(system: str, user: str, max_tokens: int = 2000) -> str:
    """
    Вызов Gemini с retry и fallback.
    Сначала пробует GEMINI_MODEL, при 503/429 — экспоненциальная задержка,
    затем fallback на GEMINI_FALLBACK_MODEL.
    """
    models = [settings.GEMINI_MODEL, settings.GEMINI_FALLBACK_MODEL]
    last_error = None

    for model_idx, model in enumerate(models):
        if model_idx > 0:
            logger.info("Falling back to model: %s", model)

        for attempt in range(settings.GEMINI_MAX_RETRIES):
            try:
                result = await _gemini_call(model, system, user, max_tokens)
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
    raise RuntimeError(f"⚠️ Ошибка Gemini API после всех попыток: {last_error}") from last_error


def _fmt_posts(posts) -> str:
    lines = []
    for i, p in enumerate(posts, 1):
        date_str = p.date.strftime("%d.%m %H:%M")
        text = p.text[:300].replace(chr(10), " ")
        lines.append(
            f"[{i}] @{p.channel} | {date_str}\n"
            f"{text}\n"
            f"URL: {p.url}"
        )
    return "\n\n".join(lines)


# ── 1. Фильтрация Telegram-постов (только каналы пользователя) ───

async def summarize_posts(posts) -> tuple[list[DigestItem], Optional[str]]:
    """
    Отбирает важные посты ТОЛЬКО из переданного списка.
    Возвращает (items, error).
    """
    if not posts:
        return [], None

    posts = sorted(posts, key=lambda p: p.date, reverse=True)
    posts = posts[: settings.MAX_POSTS_TO_AI]

    system = (
        "Ты редактор новостного дайджеста. Отбери важные посты из списка "
        "и сожми их до коротких резюме.\n\n"
        "ПРАВИЛА:\n"
        "1. Используй ТОЛЬКО информацию из постов. Не добавляй факты из своих знаний.\n"
        "2. Не ищи новости в интернете.\n"
        "3. Отфильтруй рекламу, репосты без ценности, дубли.\n"
        "4. Поле url бери строго из поста.\n"
        "5. Отвечай МАКСИМАЛЬНО КРАТКО: title до 60 символов, summary 1-2 предложения.\n"
        "6. Ответ — JSON-массив объектов. Без markdown, без пояснений."
    )
    user = (
        f"Постов: {len(posts)}. Выбери не более {settings.MAX_NEWS_IN_DIGEST} важных.\n\n"
        "Формат каждого объекта:\n"
        '{"title": "...", "summary": "...", "importance": 7, "channel": "@...", "url": "..."}\n\n'
        f"Посты:\n{_fmt_posts(posts)}\n\n"
        "Верни JSON-массив:"
    )

    # Даём щедрый лимит, чтобы JSON не обрезался
    max_tokens = 500 + settings.MAX_NEWS_IN_DIGEST * 250

    try:
        raw = await _gemini(system, user, max_tokens=max_tokens)
        data = _extract_json_array(raw)

        items = []
        for d in data:
            try:
                items.append(DigestItem(
                    title=str(d.get("title", ""))[:200],
                    summary=str(d.get("summary", ""))[:1000],
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
        return [], f"⚠️ Ошибка: {e}"


# ── 2. Веб-новости (не используется, оставлено для совместимости) ─

async def fetch_web_news(
    topic: str = "главные новости дня",
    lang: str = "ru",
) -> tuple[list[DigestItem], Optional[str]]:
    """По умолчанию НЕ вызывается. Оставлено для совместимости."""
    lang_str = "русский" if lang == "ru" else "english"
    system = (
        "Ты редактор новостного дайджеста. Составь список важных новостей.\n"
        "Ответ — JSON-массив. Без markdown."
    )
    user = (
        f"Составь {settings.MAX_NEWS_IN_DIGEST} важных новостей по теме: {topic}.\n"
        f"Язык: {lang_str}.\n"
        'Поля: "title", "summary", "importance" (1-10), "source", "url".\n'
        "JSON:"
    )

    try:
        raw = await _gemini(system, user, max_tokens=500 + settings.MAX_NEWS_IN_DIGEST * 250)
        data = _extract_json_array(raw)
        items = [
            DigestItem(
                title=str(d.get("title", "")),
                summary=str(d.get("summary", "")),
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


# ── 3. Итог дня ──────────────────────────────────────────────────

async def generate_day_summary(items: list[DigestItem], lang: str = "ru") -> str:
    if not items:
        return ""

    digest_text = "\n".join(
        f"• [{i.importance}/10] {i.title} — {i.summary}" for i in items
    )
    lang_str = "русский" if lang == "ru" else "english"

    system = (
        f"Ты аналитик. Напиши ИТОГ ДНЯ — 3-5 предложений. Используй ТОЛЬКО "
        f"перечисленные новости. Язык: {lang_str}. Стиль: деловой. "
        f"Без markdown, только обычный текст."
    )
    user = f"Новости дня:\n{digest_text}\n\nНапиши ИТОГ ДНЯ:"

    try:
        summary = await _gemini(system, user, max_tokens=600)
        # Убираем возможные markdown-артефакты
        summary = summary.strip().strip('"').strip("'")
        return summary
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
