"""
summarizer.py — Google Gemini через Interactions API (google-genai SDK).
Модель по умолчанию: gemini-3.8-flash.
Fallback: gemini-3.1-flash-lite при перегрузке.

Изменения:
  - Добавлен retry с экспоненциальной задержкой
  - Добавлен fallback на более лёгкую модель
  - Обработка 503 UNAVAILABLE
  - summarize_posts и fetch_web_news возвращают (items, error)
"""
import asyncio
import json
import logging
import random
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
    429: "⚠️ Превышен лимит запросов Gemini. Попробуй позже (бесплатный тариф: ~15 запросов/мин).",
    403: "❌ Неверный GEMINI_API_KEY или API не включён. Проверь ключ: https://aistudio.google.com/app/apikey",
    400: "❌ Ошибка в запросе к Gemini. Возможно, превышен размер контекста.",
    404: "❌ Модель не найдена. Проверь GEMINI_MODEL (актуальная: gemini-3.8-flash).",
    503: "⚠️ Gemini временно перегружен. Повторяю запрос...",
}

# Коды, при которых стоит повторить запрос
_RETRYABLE_STATUS = {429, 503, 500, 502, 504}


async def _gemini_call(model: str, system: str, user: str, max_tokens: int) -> str:
    """Один вызов Gemini с указанной моделью."""
    interaction = await _client.aio.interactions.create(
        model=model,
        input=user,
        system_instruction=system,
        generation_config={
            "temperature": 0.3,
            "max_output_tokens": max_tokens,
        },
    )
    raw = interaction.output_text.strip()
    if "```" in raw:
        parts = raw.split("```")
        raw = parts[1] if len(parts) > 1 else parts[0]
        if raw.startswith("json"):
            raw = raw[4:]
    return raw.strip()


async def _gemini(system: str, user: str, max_tokens: int = 2000) -> str:
    """
    Вызов Gemini с retry и fallback.
    Сначала пробует основную модель GEMINI_MODEL.
    При 503 — экспоненциальная задержка и повтор.
    После исчерпания попыток — fallback на GEMINI_FALLBACK_MODEL.
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

                # Не retryable ошибки — сразу пробуем fallback
                if status and status not in _RETRYABLE_STATUS:
                    hint = _ERROR_HINTS.get(status, f"HTTP {status}")
                    logger.error("Non-retryable error %s: %s", status, hint)
                    break  # переходим к следующей модели

                # Retryable ошибка — ждём и повторяем
                if attempt < settings.GEMINI_MAX_RETRIES - 1:
                    delay = settings.GEMINI_RETRY_DELAY * (2 ** attempt)
                    delay += random.uniform(0, 1)  # jitter
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

    # Все модели и попытки исчерпаны
    status = getattr(last_error, "status_code", None) or getattr(last_error, "code", None)
    if status and status in _ERROR_HINTS:
        raise RuntimeError(_ERROR_HINTS[status]) from last_error
    raise RuntimeError(f"⚠️ Ошибка Gemini API после всех попыток: {last_error}") from last_error


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
    if not posts:
        return [], None

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
