"""
scheduler.py — планировщик дайджестов по времени + разовый запуск
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
    """
    Вызывается каждую минуту планировщиком.
    Проверяет, у кого из пользователей сейчас время дайджеста.
    """
    now = datetime.now(timezone.utc)
    users = await db.get_users_for_time(now.hour, now.minute)
    if not users:
        return
    logger.info("Tick %02d:%02d — %d users", now.hour, now.minute, len(users))
    for uid in users:
        channels = await db.get_user_channels(uid)
        try:
            await _send_user_digest(bot=bot, db=db, client=None,
                                    user_id=uid, channels=channels,
                                    since_hours=settings.DEFAULT_DIGEST_INTERVAL_HOURS)
        except TelegramForbiddenError:
            logger.warning("User %d blocked bot", uid)
        except Exception as e:
            logger.error("Digest error uid=%d: %s", uid, e)


async def run_digest(bot: Bot, db: Database):
    """Резервный интервальный запуск (если у пользователя нет расписания)."""
    users = await db.get_all_active_users()
    for user in users:
        uid = user["user_id"]
        schedules = await db.get_user_schedules(uid)
        if schedules:
            continue   # у этого юзера есть расписание — пропускаем
        channels = await db.get_user_channels(uid)
        if not channels:
            continue
        try:
            await _send_user_digest(bot=bot, db=db, client=None,
                                    user_id=uid, channels=channels,
                                    since_hours=settings.DEFAULT_DIGEST_INTERVAL_HOURS)
        except Exception as e:
            logger.error("Interval digest uid=%d: %s", uid, e)


async def _send_user_digest(bot, db, client, user_id, channels, since_hours):
    # 1. Telegram-каналы
    tg_items = []
    if channels:
        posts = await fetch_all_user_channels(
            channels,
            limit_per_channel=settings.POSTS_PER_CHANNEL,
            since_hours=since_hours,
        )
        # Фильтр уже виденных
        new_posts = []
        for post in posts:
            new_ids = await db.filter_new_posts(user_id, post.channel, [post.id])
            if new_ids:
                new_posts.append(post)

        if new_posts:
            tg_items = await summarize_posts(new_posts)
            for post in new_posts:
                await db.mark_seen(user_id, post.channel, [post.id])

    # 2. Веб-новости (всегда, если включено)
    web_items = []
    if settings.INCLUDE_WEB_NEWS:
        web_items = await fetch_web_news(
            topic=settings.WEB_NEWS_TOPIC,
            lang=settings.DIGEST_LANGUAGE,
        )

    all_items = tg_items + web_items
    if not all_items:
        logger.info("User %d: nothing to send.", user_id)
        return

    # 3. Итог дня
    day_summary = await generate_day_summary(all_items, lang=settings.DIGEST_LANGUAGE)

    # 4. Форматируем и отправляем
    msg = format_digest_message(
        tg_items=tg_items,
        web_items=web_items,
        day_summary=day_summary,
        lang=settings.DIGEST_LANGUAGE,
    )
    await bot.send_message(
        chat_id=user_id,
        text=msg,
        parse_mode="HTML",
        disable_web_page_preview=True,
    )

    # 5. Кешируем статьи для оффлайн-доступа
    for item in all_items:
        if item.url:
            await db.cache_article(
                user_id=user_id,
                url=item.url,
                title=item.title,
                full_text=item.summary,  # summary как preview; full fetch — через /read
                source=item.channel,
            )

    await db.log_digest(user_id, len(all_items), day_summary)
    logger.info("User %d: sent %d items (tg=%d web=%d).",
                user_id, len(all_items), len(tg_items), len(web_items))
