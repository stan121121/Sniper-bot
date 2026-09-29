"""
main.py — запуск бота + планировщик.
С логированием в файл (для Railway).
"""
import asyncio
import logging
import logging.handlers
import os

from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import settings
from database import Database
from handlers import router
from scheduler import tick, run_digest

# ── Логирование ──────────────────────────────────────────────────
LOG_LEVEL = logging.INFO
LOG_FORMAT = "%(asctime)s [%(levelname)s] %(name)s: %(message)s"

logging.basicConfig(level=LOG_LEVEL, format=LOG_FORMAT)
logger = logging.getLogger(__name__)

# Дополнительно пишем в файл (Railway сохраняет logs/)
if os.path.isdir("/app/logs"):
    file_handler = logging.handlers.RotatingFileHandler(
        "/app/logs/bot.log", maxBytes=5 * 1024 * 1024, backupCount=3
    )
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT))
    logging.getLogger().addHandler(file_handler)


async def main():
    bot = Bot(token=settings.BOT_TOKEN)
    dp = Dispatcher(storage=MemoryStorage())
    dp.include_router(router)

    db = Database()
    await db.init()

    scheduler = AsyncIOScheduler(timezone="UTC")

    # Каждую минуту — проверяем расписание пользователей
    scheduler.add_job(tick, "cron", minute="*",
                      args=[bot, db], id="tick")

    # Резервный интервальный дайджест
    scheduler.add_job(run_digest, "interval",
                      hours=settings.DEFAULT_DIGEST_INTERVAL_HOURS,
                      args=[bot, db], id="interval_digest")

    scheduler.start()
    logger.info("Bot started. Model: %s", settings.DEEPSEEK_MODEL)

    try:
        await dp.start_polling(bot, db=db, scheduler=scheduler)
    finally:
        scheduler.shutdown()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
