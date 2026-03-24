"""
main.py — запуск бота + планировщик (каждую минуту проверяем расписание)
"""
import asyncio
import logging
from aiogram import Bot, Dispatcher
from aiogram.fsm.storage.memory import MemoryStorage
from apscheduler.schedulers.asyncio import AsyncIOScheduler

from config import settings
from database import Database
from handlers import router
from scheduler import tick, run_digest

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)


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

    # Резервный интервальный дайджест для тех, у кого нет расписания
    scheduler.add_job(run_digest, "interval",
                      hours=settings.DEFAULT_DIGEST_INTERVAL_HOURS,
                      args=[bot, db], id="interval_digest")

    scheduler.start()
    logger.info("Bot started.")

    try:
        await dp.start_polling(bot, db=db, scheduler=scheduler)
    finally:
        scheduler.shutdown()
        await bot.session.close()


if __name__ == "__main__":
    asyncio.run(main())
