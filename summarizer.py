"""
summarizer.py — Google Gemini через стабильный generate_content API.

Ключевые особенности:
  - Модель работает СТРОГО с постами каналов пользователя.
  - Улучшенный промпт: интерес-профиль, дедупликация, why/framing.
  - Устойчивый парсинг JSON: если ответ обрезан, извлекаем целые объекты.
  - Retry с экспоненциальной задержкой + fallback на резервную модель.
  - json_mode: JSON включается только там, где нужен (summarize_posts),
    а для «Итога дня» используется чистый текст.
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
    429: "⚠️ Превышен лимит запросов Gemini. Попробуй позже.",
    403: "❌ Неверный GEMINI_API_KEY или API не включён.",
    400: "❌ Ошибка в запросе к Gemini. Возможно, превышен размер контекста.",
    404: "❌ Модель не найдена. Проверь GEMINI_MODEL (актуальная: gemini-3.8-flash).",
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

    # Fallback: обход сбалансированных фигурных скобок
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
                        logger.debug("Skipped malformed JSON chunk: %s", chunk[:120])
                start = None

    if items:
        logger.info("Recovered %d objects from partial JSON", len(items))
    else:
        logger.warning("Could not extract any JSON objects from response")

    return items


def _repair_json_chunk(chunk: str) -> str:
    """Правки типичных ошибок JSON: одиночные кавычки, висячие запятые."""
    fixed = re.sub(r"(?<![\\])'", '"', chunk)
    fixed = re.sub(r",\s*}", "}", fixed)
    return fixed


def _extract_plain_text(raw: str) -> str:
    """
    Извлекает чистый текст из ответа Gemini.
    Если модель всё же вернула JSON-объект (вида {"daily_summary": "..."}),
    вытаскиваем значение первого строкового поля.
    """
    if not raw:
        return ""

    cleaned = raw.strip()

    # Убираем markdown-обёртки
    if "```" in cleaned:
        parts = cleaned.split("```")
        cleaned = parts[1] if len(parts) > 1 else parts[0]
        if cleaned.startswith("json"):
            cleaned = cleaned[4:]
        cleaned = cleaned.strip()

    # Если это JSON-объект — вытаскиваем первое строковое значение
    if cleaned.startswith("{"):
        try:
            data = json.loads(cleaned)
            if isinstance(data, dict):
                for key in ("daily_summary", "summary", "text", "итог", "итог_дня"):
                    if key in data and isinstance(data[key], str):
                        return data[key].strip()
                # Если ключ неизвестен — берём первое строковое значение
                for v in data.values():
                    if isinstance(v, str):
                        return v.strip()
        except json.JSONDecodeError:
            pass

    # Если это JSON-массив — склеиваем строки
    if cleaned.startswith("["):
        try:
            data = json.loads(cleaned)
            if isinstance(data, list):
                parts = [str(x) for x in data if isinstance(x, (str, int, float))]
                if parts:
                    return " ".join(parts).strip()
        except json.JSONDecodeError:
            pass

    # Иначе — просто текст
    return cleaned.strip().strip('"').strip("'")


# ── Вызов Gemini ─────────────────────────────────────────────────

async def _gemini_call(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool = True,
) -> str:
    """
    Один вызов Gemini через стабильный generate_content.

    json_mode=True  — модель обязана вернуть валидный JSON.
    json_mode=False — модель возвращает обычный текст (для «Итога дня»).
    """
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
    """
    Вызов Gemini с retry и fallback.
    json_mode прокидывается в _gemini_call.
    """
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
    raise RuntimeError(f"⚠️ Ошибка Gemini API после всех попыток: {last_error}") from last_error


# ── Форматирование постов для промпта ────────────────────────────

def _fmt_posts(posts) -> str:
    lines = []
    for i, p in enumerate(posts, 1):
        date_str = p.date.strftime("%d.%m %H:%M")
        text = p.text[:400].replace(chr(10), " ")
        lines.append(
            f"[{i}] @{p.channel} | {date_str}\n"
            f"{text}\n"
            f"URL: {p.url}"
        )
    return "\n\n".join(lines)


# ── 1. Фильтрация Telegram-постов ────────────────────────────────

_SYSTEM_PROMPT = (
    "Ты — редактор персонального новостного дайджеста из Telegram-каналов.\n"
    "Твоя задача — отобрать важные посты из ПРЕДОСТАВЛЕННОГО СПИСКА "
    "и сжать их до коротких, информативных резюме.\n\n"

    "═══ ЖЁСТКИЕ ПРАВИЛА ═══\n\n"

    "1. ИСТОЧНИК ИНФОРМАЦИИ — только тексты постов из списка.\n"
    "   • НЕ добавляй факты, имена, числа, даты и события, которых нет в постах.\n"
    "   • НЕ используй свои знания о мире.\n"
    "   • НЕ ищи новости в интернете.\n"
    "   • Если в посте чего-то нет — не додумывай.\n\n"

    "2. ДЕДУПЛИКАЦИЯ.\n"
    "   • Если несколько постов описывают одно событие — объедини их в одну запись.\n"
    "   • Сохрани различия в подаче: если каналы трактуют событие по-разному, "
    "отметь это в поле framing.\n"
    "   • В поле channel укажи все каналы через запятую.\n"
    "   • В поле url — ссылку на самый содержательный пост.\n\n"

    "3. ФИЛЬТРАЦИЯ.\n"
    "   Отбрасывай: рекламу, репосты без добавленной ценности, мемы, "
    "поздравления, дубли, погоду, спорт без значимого контекста, "
    "светскую хронику, локальные происшествия без широкого значения.\n\n"

    "4. ПРИОРИТЕТЫ (что считать важным):\n"
    "   1) Технологии и AI — продукты, политика, бизнес\n"
    "   2) Бизнес, стартапы, предпринимательство\n"
    "   3) Экономика: рынки, макро, торговля, ставки\n"
    "   4) Политические решения с реальными последствиями\n"
    "   5) Регуляторные изменения (налоги, законы, санкции)\n"
    "   6) Качественные аналитические посты, меняющие взгляд на тему\n\n"

    "5. ФОРМАТ.\n"
    "   • Отвечай СТРОГО JSON-массивом объектов.\n"
    "   • Без markdown, без бэктиков, без пояснений до или после JSON.\n"
    "   • Все текстовые поля — на русском языке.\n\n"

    "═══ СХЕМА ОБЪЕКТА ═══\n\n"
    "{\n"
    '  "title":      "до 80 символов, конкретный, без кликбейта",\n'
    '  "summary":    "1–2 предложения: что произошло. Только факты из поста.",\n'
    '  "why":        "1 предложение: почему это важно.",\n'
    '  "framing":    "различия в подаче между каналами (или пустая строка)",\n'
    '  "importance": 1-10,\n'
    '  "channel":    "@канал1, @канал2",\n'
    '  "url":        "ссылка из поста или пустая строка"\n'
    "}\n\n"

    "═══ ОГРАНИЧЕНИЯ ═══\n\n"
    "• Не более MAX_NEWS важных новостей (см. пользовательский запрос).\n"
    "• Если подходящих постов меньше — верни меньше, НЕ добивай список мусором.\n"
    "• Если все посты — мусор, верни пустой массив [].\n"
    "• Не более 10 объектов в ответе, даже если постов много.\n"
)


async def summarize_posts(posts) -> tuple[list[DigestItem], Optional[str]]:
    """Отбирает важные посты ТОЛЬКО из переданного списка."""
    if not posts:
        return [], None

    posts = sorted(posts, key=lambda p: p.date, reverse=True)
    posts = posts[: settings.MAX_POSTS_TO_AI]

    user = (
        f"Проанализируй {len(posts)} постов из Telegram-каналов пользователя.\n"
        f"Отбери не более {settings.MAX_NEWS_IN_DIGEST} самых важных "
        "и верни JSON-массив по схеме.\n\n"
        f"Посты:\n{_fmt_posts(posts)}\n\n"
        "Ответ — JSON-массив:"
    )

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
        'Поля: "title", "summary", "why", "importance" (1-10), "source", "url".\n'
        "JSON:"
    )

    try:
        raw = await _gemini(
            system, user,
            max_tokens=700 + settings.MAX_NEWS_IN_DIGEST * 350,
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


# ── 3. Итог дня (чистый текст, без JSON) ─────────────────────────

async def generate_day_summary(items: list[DigestItem], lang: str = "ru") -> str:
    """
    Генерирует «Итог дня» как обычный текст.
    json_mode=False, чтобы модель не оборачивала ответ в JSON.
    """
    if not items:
        return ""

    digest_text = "\n".join(
        f"• [{i.importance}/10] {i.title} — {i.summary}" for i in items
    )
    lang_str = "русский" if lang == "ru" else "english"

    system = (
        f"Ты аналитик новостей. Напиши связный ИТОГ ДНЯ — 3-5 предложений "
        f"о том, что главное произошло. Используй ТОЛЬКО перечисленные ниже "
        f"новости, не добавляй факты из своих знаний.\n\n"
        f"Язык: {lang_str}.\n"
        f"Стиль: деловой, без вводных фраз вроде «сегодня», «в этот день».\n"
        f"Формат: обычный текст. НЕ используй JSON, кавычки, markdown, "
        f"заголовки, ключи вида \"daily_summary\".\n"
        f"Просто напиши 3-5 предложений подряд."
    )
    user = (
        f"Новости дня:\n{digest_text}\n\n"
        "Напиши ИТОГ ДНЯ (обычный текст, 3-5 предложений):"
    )

    try:
        # json_mode=False — модель вернёт чистый текст
        summary = await _gemini(system, user, max_tokens=600, json_mode=False)
        # На случай если модель всё равно вернула JSON — вытащим текст
        summary = _extract_plain_text(summary)
        return summary
    except RuntimeError as e:
        logger.warning("day_summary skipped: %s", e)
        return ""
    except Exception as e:
        logger.error("day_summary error: %s", e)
        return ""


# ── 4. Форматирование для Telegram ───────────────────────────────

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
        # Никаких <i>...</i> вокруг — просто аккуратный абзац
        parts.append(f"\n\n📊 <b>Итог дня</b>\n{_he(day_summary)}")

    if api_error:
        parts.append(f"\n\n⚠️ <i>{_he(api_error)}</i>")

    model_short = settings.GEMINI_MODEL.replace("gemini-", "Gemini ")
    parts.append(f"\n🤖 <i>{_he(model_short)}</i>")

    return "\n".join(parts)
