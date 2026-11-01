"""
summarizer.py — Мультипровайдерный клиент для LLM (Gemini, DeepSeek, Qwen, OpenRouter).

Все провайдеры вызываются через единый OpenAI-совместимый SDK.
Выбор провайдера и модели осуществляется через переменные окружения.
"""
import asyncio
import json
import logging
import random
import re
from dataclasses import dataclass
from typing import Optional

from openai import AsyncOpenAI
from config import settings

logger = logging.getLogger(__name__)


# ── Инициализация единого клиента ────────────────────────────────
_client = AsyncOpenAI(
    api_key=settings.get_api_key(),
    base_url=settings.get_base_url(),
    default_headers={"HTTP-Referer": "https://github.com/your-bot", "X-Title": "News Digest Bot"} 
    if settings.LLM_PROVIDER == "openrouter" else None,
)


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
    400: "❌ LLM отклонил запрос (400). Сокращаю промпт...",
    401: "❌ Неверный API ключ для выбранного провайдера.",
    402: "❌ Недостаточно средств на балансе провайдера.",
    429: "⚠️ Лимит запросов LLM. Попробуй позже.",
    503: "⚠️ LLM временно перегружен. Повторяю запрос...",
}

_RETRYABLE_STATUS = {429, 503, 500, 502, 504}


# ── Робастная очистка текста ─────────────────────────────────────

_CONTROL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_ZEROWIDTH_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2060-\u206f\ufeff]")
_VARIATION_RE = re.compile(r"[\ufe00-\ufe0f]")
_EMOJI_RE = re.compile(r"[\U00010000-\U0010ffff]")
_SURROGATE_RE = re.compile(r"[\ud800-\udfff]")

def _clean_text(text: str) -> str:
    if not text:
        return ""
    text = _CONTROL_RE.sub("", text)
    text = _ZEROWIDTH_RE.sub("", text)
    text = _VARIATION_RE.sub("", text)
    text = _SURROGATE_RE.sub("", text)
    text = _EMOJI_RE.sub("", text)
    text = text.replace('"', "'").replace("\\", "").replace("\u00a0", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()

def _is_clean_enough(text: str) -> bool:
    if not text or len(text) < 30:
        return False
    printable = sum(1 for c in text if c.isprintable() or c.isspace())
    return printable / len(text) > 0.8


# ── Устойчивый парсинг JSON ──────────────────────────────────────

def _extract_json_array(raw: str) -> list[dict]:
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
                        pass
                start = None

    if items:
        logger.info("Recovered %d objects from partial JSON", len(items))
    return items

def _repair_json_chunk(chunk: str) -> str:
    fixed = re.sub(r"(?<![\\])'", '"', chunk)
    fixed = re.sub(r",\s*}", "}", fixed)
    return fixed


# ── Вызов LLM ───────────────────────────────────────────────────

async def _llm_call(
    model: str,
    system: str,
    user: str,
    max_tokens: int,
    json_mode: bool = True,
) -> str:
    """Один вызов LLM через OpenAI-совместимый SDK."""
    kwargs = {
        "model": model,
        "messages": [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ],
        "temperature": 0.3,
        "max_tokens": max_tokens,
    }
    if json_mode:
        kwargs["response_format"] = {"type": "json_object"}

    response = await _client.chat.completions.create(**kwargs)
    return (response.choices[0].message.content or "").strip()


async def _llm(
    system: str,
    user: str,
    max_tokens: int = 2000,
    json_mode: bool = True,
) -> str:
    """Вызов LLM с retry + fallback."""
    models = [settings.LLM_MODEL, settings.LLM_FALLBACK_MODEL]
    last_error = None
    user_for_attempt = user
    json_mode_for_attempt = json_mode

    for model_idx, model in enumerate(models):
        if model_idx > 0 and settings.LLM_FALLBACK_MODEL == settings.LLM_MODEL:
            break

        if model_idx > 0:
            logger.info("Falling back to model: %s", model)

        for attempt in range(settings.LLM_MAX_RETRIES):
            try:
                result = await _llm_call(
                    model, system, user_for_attempt,
                    max_tokens, json_mode_for_attempt,
                )
                if model_idx > 0:
                    logger.info("Fallback model %s succeeded", model)
                return result

            except Exception as e:
                status = getattr(e, "status_code", None) or getattr(e, "code", None)
                last_error = e

                if status == 400:
                    if attempt == 0 and len(user_for_attempt) > 5000:
                        user_for_attempt = user_for_attempt[: len(user_for_attempt) // 2]
                        logger.warning("400 error → truncated prompt by 50%")
                        continue
                    if json_mode_for_attempt:
                        json_mode_for_attempt = False
                        logger.warning("400 error → retrying without JSON mode")
                        continue
                    logger.error("400 persists after all mitigations")
                    break

                if status and status not in _RETRYABLE_STATUS:
                    hint = _ERROR_HINTS.get(status, f"HTTP {status}")
                    logger.error("Non-retryable error %s: %s", status, hint)
                    break

                if attempt < settings.LLM_MAX_RETRIES - 1:
                    delay = settings.LLM_RETRY_DELAY * (2 ** attempt)
                    delay += random.uniform(0, 1)
                    logger.warning(
                        "LLM %s error (attempt %d/%d), retrying in %.1fs: %s",
                        status, attempt + 1, settings.LLM_MAX_RETRIES, delay, e
                    )
                    await asyncio.sleep(delay)
                else:
                    logger.error("Model %s failed after %d attempts: %s", model, settings.LLM_MAX_RETRIES, e)

    status = getattr(last_error, "status_code", None) or getattr(last_error, "code", None)
    if status and status in _ERROR_HINTS:
        raise RuntimeError(_ERROR_HINTS[status]) from last_error
    raise RuntimeError(f"⚠️ LLM error after all attempts: {last_error}") from last_error


# ── Форматирование постов ────────────────────────────────────────

def _fmt_posts(posts) -> str:
    limit = settings.POST_TEXT_LIMIT
    lines = []
    for i, p in enumerate(posts, 1):
        date_str = p.date.strftime("%d.%m %H:%M")
        text = _clean_text(p.text[:limit]).replace(chr(10), " ")
        if not text:
            continue
        lines.append(f"[{i}] @{p.channel} | {date_str} | {text} | URL: {p.url}")
    return "\n".join(lines)


# ── Основной промпт ──────────────────────────────────────────────

_SYSTEM_PROMPT = (
    "You are an editor of a personal news digest from Telegram channels.\n"
    "Your task: select important posts from the PROVIDED LIST and compress them "
    "into short, informative summaries.\n\n"
    "1. SOURCE OF TRUTH — only the post texts in the list. Do NOT add facts.\n"
    "2. DEDUPLICATION. Merge same events. Note framing differences.\n"
    "3. FILTERING. Discard ads, memes, weather, sports without context.\n"
    "4. PRIORITIES: AI/Tech, Business, Economics, Politics, Regulations, Analysis.\n"
    "5. LANGUAGE: ALL text field VALUES must be in RUSSIAN.\n"
    "6. FORMAT: STRICTLY a JSON array of objects.\n\n"
    "Schema:\n"
    "{\n"
    '  "title": "up to 80 chars, RUSSIAN",\n'
    '  "summary": "1-2 sentences, RUSSIAN",\n'
    '  "why": "1 sentence, RUSSIAN",\n'
    '  "framing": "differences, RUSSIAN or empty",\n'
    '  "importance": 1-10,\n'
    '  "channel": "@channel1, @channel2",\n'
    '  "url": "link or empty"\n'
    "}\n\n"
    "Return no more than the requested number of items. If all garbage, return []."
)


async def summarize_posts(posts) -> tuple[list[DigestItem], Optional[str]]:
    if not posts:
        return [], None

    posts = sorted(posts, key=lambda p: p.date, reverse=True)
    posts = posts[: settings.MAX_POSTS_TO_AI]
    posts = [p for p in posts if _is_clean_enough(_clean_text(p.text))]

    if not posts:
        return [], None

    user = (
        f"Analyze {len(posts)} posts. Select up to {settings.MAX_NEWS_IN_DIGEST} most important.\n"
        "REMINDER: all text values in Russian.\n\n"
        f"Posts:\n{_fmt_posts(posts)}\n\nJSON:"
    )

    max_tokens = 700 + settings.MAX_NEWS_IN_DIGEST * 350

    try:
        raw = await _llm(_SYSTEM_PROMPT, user, max_tokens=max_tokens, json_mode=True)
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
            except (ValueError, TypeError):
                pass

        items.sort(key=lambda x: x.importance, reverse=True)
        logger.info("TG digest: %d items (from %d posts)", len(items), len(posts))
        return items, None

    except RuntimeError as e:
        logger.error("summarize_posts: %s", e)
        return [], str(e)
    except Exception as e:
        logger.error("summarize_posts unexpected: %s", e, exc_info=True)
        return [], f"⚠️ Error: {e}"


# ── Форматирование для Telegram ─────────────────────────────────

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

    provider = settings.LLM_PROVIDER.upper()
    model_short = settings.LLM_MODEL.split("/")[-1]
    parts.append(f"\n🤖 <i>{provider} · {_he(model_short)}</i>")
    return "\n".join(parts)
