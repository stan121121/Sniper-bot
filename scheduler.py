"""
scheduler.py — тик каждую минуту + резервный интервальный запуск
"""
import logging
from datetime import datetime, timezone

from aiogram import Bot
from aiogram.exceptions import TelegramForbiddenError

from channel_reader import fetch_all_user_channels
from config import settings
from database import Database
from summarizer import (
    summarize_posts, fetch_web_news,
    generate_day_summary, format_digest_message,
)

logger = logging.getLogger(__name__)


async def tick(bot: Bot, db: Database):
    """Каждую минуту: проверяем у кого сейчас время дайджеста."""
    now = datetime.now(timezone.utc)
    user_ids = await db.get_users_for_time(now.hour, now.minute)
    if not user_ids:
        return
    logger.info("Tick %02d:%02d — %d users", now.hour, now.minute, len(user_ids))
    for uid in user_ids:
        channels = await db.get_user_channels(uid)
        try:
            await _send_user_digest(bot, db, None, uid, channels,
                                    settings.DEFAULT_DIGEST_INTERVAL_HOURS)
        except TelegramForbiddenError:
            logger.warning("User %d blocked bot", uid)
        except Exception as e:
            logger.error("Digest error uid=%d: %s", uid, e)


async def run_digest(bot: Bot, db: Database):
    """Резервный запуск для пользователей без расписания."""
    for user in await db.get_all_active_users():
        uid = user["user_id"]
        if await db.get_user_schedules(uid):
            continue  # есть расписание — пропускаем
        channels = await db.get_user_channels(uid)
        if not channels:
            continue
        try:
            await _send_user_digest(bot, db, None, uid, channels,
                                    settings.DEFAULT_DIGEST_INTERVAL_HOURS)
        except Exception as e:
            logger.error("Interval digest uid=%d: %s", uid, e)


async def _send_user_digest(bot, db, client, user_id, channels, since_hours):
    # 1. Telegram-каналы
    tg_items, api_error = [], None
    if channels:
        posts = await fetch_all_user_channels(
            channels,
            limit_per_channel=settings.POSTS_PER_CHANNEL,
            since_hours=since_hours,
        )
        new_posts = []
        for post in posts:
            if await db.filter_new_posts(user_id, post.channel, [post.id]):
                new_posts.append(post)

        if new_posts:
            tg_items = await summarize_posts(new_posts)
            for post in new_posts:
                await db.mark_seen(user_id, post.channel, [post.id])

    # 2. Веб-новости
    web_items = []
    if settings.INCLUDE_WEB_NEWS:
        web_items, api_error = await fetch_web_news(
            topic=settings.WEB_NEWS_TOPIC,
            lang=settings.DIGEST_LANGUAGE,
        )

    all_items = tg_items + web_items
    if not all_items and not api_error:
        logger.info("User %d: nothing to send.", user_id)
        return

    # 3. Итог дня (только если есть новости)
    day_summary = ""
    if all_items:
        day_summary = await generate_day_summary(all_items, lang=settings.DIGEST_LANGUAGE)

    # 4. Отправка
    msg = format_digest_message(
        tg_items=tg_items,
        web_items=web_items,
        day_summary=day_summary,
        api_error=api_error,
        lang=settings.DIGEST_LANGUAGE,
    )

    # Разбиваем длинные сообщения (лимит Telegram 4096 символов)
    for chunk in _split_message(msg):
        await bot.send_message(
            chat_id=user_id, text=chunk,
            parse_mode="HTML", disable_web_page_preview=True,
        )

    # 5. Кеш для оффлайн
    for item in all_items:
        if item.url:
            await db.cache_article(
                user_id=user_id, url=item.url,
                title=item.title, full_text=item.summary,
                source=item.channel,
            )

    await db.log_digest(user_id, len(all_items), day_summary)
    logger.info("User %d: sent %d items (tg=%d web=%d).",
                user_id, len(all_items), len(tg_items), len(web_items))


def _split_message(text: str, limit: int = 4000) -> list[str]:
    """Разбить длинное сообщение на части по границам абзацев."""
    if len(text) <= limit:
        return [text]
    parts, buf = [], ""
    for line in text.split("\n"):
        if len(buf) + len(line) + 1 > limit:
            if buf:
                parts.append(buf.rstrip())
            buf = line + "\n"
        else:
            buf += line + "\n"
    if buf.strip():
        parts.append(buf.rstrip())
    return parts or [text[:limit]]
