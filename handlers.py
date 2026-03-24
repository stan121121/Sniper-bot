"""
handlers.py — все команды бота
"""
import logging
import re

from aiogram import Router, F
from aiogram.filters import Command, CommandStart
from aiogram.fsm.context import FSMContext
from aiogram.fsm.state import State, StatesGroup
from aiogram.types import Message, ReplyKeyboardMarkup, KeyboardButton, ReplyKeyboardRemove

from config import settings
from scheduler import _send_user_digest
from channel_reader import get_telethon_client

router = Router()
logger = logging.getLogger(__name__)


# ── Утилиты ──────────────────────────────────────────────────────
def he(t: str) -> str:
    return t.replace("&","&amp;").replace("<","&lt;").replace(">","&gt;")

def parse_channel_input(raw: str) -> str | None:
    raw = raw.strip()
    m = re.match(r"(?:https?://)?t(?:elegram)?\.me/([A-Za-z0-9_]{3,})", raw, re.I)
    if m:
        return m.group(1).lower()
    if "joinchat" in raw or "+" in raw:
        return None
    username = raw.lstrip("@").strip().lower()
    return username if re.match(r"^[A-Za-z0-9_]{3,}$", username) else None

def parse_time(s: str) -> tuple[int,int] | None:
    """Парсит '09:00' или '9' → (9, 0)."""
    s = s.strip()
    if ":" in s:
        parts = s.split(":")
        try:
            h, m = int(parts[0]), int(parts[1])
            if 0 <= h <= 23 and 0 <= m <= 59:
                return h, m
        except ValueError:
            pass
    elif s.isdigit():
        h = int(s)
        if 0 <= h <= 23:
            return h, 0
    return None


# ── FSM ──────────────────────────────────────────────────────────
class AddChannel(StatesGroup):
    waiting = State()

class AddSchedule(StatesGroup):
    waiting = State()


# ── Клавиатура ───────────────────────────────────────────────────
def main_kb():
    return ReplyKeyboardMarkup(keyboard=[
        [KeyboardButton(text="📋 Каналы"),    KeyboardButton(text="➕ Добавить канал")],
        [KeyboardButton(text="⏰ Расписание"), KeyboardButton(text="📰 Дайджест сейчас")],
        [KeyboardButton(text="📚 Кеш статей"),KeyboardButton(text="ℹ️ Помощь")],
    ], resize_keyboard=True)


# ── /start ────────────────────────────────────────────────────────
@router.message(CommandStart())
async def cmd_start(message: Message, db):
    await db.upsert_user(message.from_user.id, message.from_user.username)
    await message.answer(
        "👋 Привет! Я новостной дайджест-бот.\n\n"
        "<b>Что умею:</b>\n"
        "📱 Читать твои Telegram-каналы\n"
        "🌐 Добавлять важные новости из интернета\n"
        "📊 Делать <b>Итог дня</b>\n"
        "📚 Кешировать статьи для оффлайн-чтения\n"
        "⏰ Присылать дайджест в нужное время (09:00, 21:00 и т.д.)\n\n"
        "Начни с /add чтобы добавить каналы,\n"
        "затем /schedule чтобы задать время дайджеста.",
        parse_mode="HTML", reply_markup=main_kb(),
    )


# ── /help ─────────────────────────────────────────────────────────
@router.message(Command("help"))
@router.message(F.text == "ℹ️ Помощь")
async def cmd_help(message: Message):
    await message.answer(
        "📖 <b>Команды:</b>\n\n"
        "<b>Каналы:</b>\n"
        "/add <code>@channel</code> — добавить канал\n"
        "/remove <code>@channel</code> — удалить канал\n"
        "/channels — список каналов\n\n"
        "<b>Расписание:</b>\n"
        "/schedule <code>09:00</code> — добавить время дайджеста\n"
        "/unschedule <code>09:00</code> — удалить время\n"
        "/schedules — показать расписание\n\n"
        "<b>Дайджест:</b>\n"
        "/digest — получить дайджест прямо сейчас\n"
        "/cache — показать кешированные статьи\n\n"
        "Время указывается в UTC. Примеры: <code>09:00</code>, <code>21:30</code>",
        parse_mode="HTML",
    )


# ── Каналы ────────────────────────────────────────────────────────
@router.message(Command("channels"))
@router.message(F.text == "📋 Каналы")
async def cmd_channels(message: Message, db):
    channels = await db.get_user_channels(message.from_user.id)
    if not channels:
        await message.answer("Каналов нет. Добавь через /add", reply_markup=main_kb())
        return
    lines = "\n".join(f"• @{he(ch)}" for ch in channels)
    await message.answer(f"📋 <b>Твои каналы:</b>\n\n{lines}",
                         parse_mode="HTML", reply_markup=main_kb())


@router.message(Command("add"))
@router.message(F.text == "➕ Добавить канал")
async def cmd_add_start(message: Message, state: FSMContext, db):
    parts = message.text.split(maxsplit=1)
    if len(parts) == 2 and parts[0] == "/add":
        await _do_add(message, parts[1], db=db)
        return
    await state.set_state(AddChannel.waiting)
    await message.answer(
        "Отправь username или ссылку:\n"
        "<code>@rbc_news</code>  или  <code>https://t.me/rbc_news</code>",
        parse_mode="HTML", reply_markup=ReplyKeyboardRemove(),
    )

@router.message(AddChannel.waiting)
async def cmd_add_input(message: Message, state: FSMContext, db):
    await state.clear()
    await _do_add(message, message.text.strip(), db=db)

async def _do_add(message: Message, raw: str, db=None):
    if db is None: return
    username = parse_channel_input(raw)
    if not username:
        await message.answer(
            "❌ Не распознан канал.\n"
            "Формат: <code>@username</code> или <code>https://t.me/username</code>",
            parse_mode="HTML", reply_markup=main_kb())
        return
    added = await db.add_channel(message.from_user.id, username)
    if added:
        await message.answer(f"✅ <code>@{he(username)}</code> добавлен!",
                             parse_mode="HTML", reply_markup=main_kb())
    else:
        await message.answer(f"⚠️ <code>@{he(username)}</code> уже в списке.",
                             parse_mode="HTML", reply_markup=main_kb())


@router.message(Command("remove"))
async def cmd_remove(message: Message, db):
    parts = message.text.split(maxsplit=1)
    if len(parts) == 2:
        username = parse_channel_input(parts[1])
        if username:
            removed = await db.remove_channel(message.from_user.id, username)
            if removed:
                await message.answer(f"✅ <code>@{he(username)}</code> удалён.",
                                     parse_mode="HTML", reply_markup=main_kb())
            else:
                await message.answer(f"❌ <code>@{he(username)}</code> не найден.",
                                     parse_mode="HTML", reply_markup=main_kb())
            return
    channels = await db.get_user_channels(message.from_user.id)
    lines = "\n".join(f"• @{he(ch)}" for ch in channels) if channels else "—"
    await message.answer(
        f"Напиши: <code>/remove @username</code>\n\nТвои каналы:\n{lines}",
        parse_mode="HTML", reply_markup=main_kb())


# ── Расписание ────────────────────────────────────────────────────
@router.message(Command("schedules"))
@router.message(F.text == "⏰ Расписание")
async def cmd_schedules(message: Message, db):
    schedules = await db.get_user_schedules(message.from_user.id)
    if not schedules:
        await message.answer(
            "Расписание не задано.\n\n"
            "Добавь: <code>/schedule 09:00</code>\n"
            "Можно несколько: <code>/schedule 21:00</code>",
            parse_mode="HTML", reply_markup=main_kb())
        return
    lines = "\n".join(f"• {s['hour']:02d}:{s['minute']:02d} UTC" for s in schedules)
    await message.answer(
        f"⏰ <b>Расписание дайджестов:</b>\n\n{lines}\n\n"
        "Удалить: <code>/unschedule 09:00</code>",
        parse_mode="HTML", reply_markup=main_kb())


@router.message(Command("schedule"))
async def cmd_schedule_add(message: Message, state: FSMContext, db):
    parts = message.text.split(maxsplit=1)
    if len(parts) == 2:
        t = parse_time(parts[1])
        if t:
            added = await db.add_schedule(message.from_user.id, t[0], t[1])
            if added:
                await message.answer(
                    f"✅ Дайджест будет приходить в <b>{t[0]:02d}:{t[1]:02d} UTC</b>",
                    parse_mode="HTML", reply_markup=main_kb())
            else:
                await message.answer(
                    f"⚠️ Время {t[0]:02d}:{t[1]:02d} уже задано.",
                    reply_markup=main_kb())
            return
    await state.set_state(AddSchedule.waiting)
    await message.answer(
        "Укажи время в формате <code>ЧЧ:ММ</code> (UTC):\n"
        "Например: <code>09:00</code> или <code>21:30</code>",
        parse_mode="HTML", reply_markup=ReplyKeyboardRemove())

@router.message(AddSchedule.waiting)
async def cmd_schedule_input(message: Message, state: FSMContext, db):
    await state.clear()
    t = parse_time(message.text.strip())
    if not t:
        await message.answer("❌ Неверный формат. Пример: <code>09:00</code>",
                             parse_mode="HTML", reply_markup=main_kb())
        return
    added = await db.add_schedule(message.from_user.id, t[0], t[1])
    if added:
        await message.answer(
            f"✅ Дайджест в <b>{t[0]:02d}:{t[1]:02d} UTC</b> добавлен!",
            parse_mode="HTML", reply_markup=main_kb())
    else:
        await message.answer(f"⚠️ Время уже задано.", reply_markup=main_kb())


@router.message(Command("unschedule"))
async def cmd_unschedule(message: Message, db):
    parts = message.text.split(maxsplit=1)
    if len(parts) == 2:
        t = parse_time(parts[1])
        if t:
            removed = await db.remove_schedule(message.from_user.id, t[0], t[1])
            if removed:
                await message.answer(
                    f"✅ Время {t[0]:02d}:{t[1]:02d} удалено из расписания.",
                    reply_markup=main_kb())
            else:
                await message.answer(
                    f"❌ Время {t[0]:02d}:{t[1]:02d} не найдено.",
                    reply_markup=main_kb())
            return
    await message.answer(
        "Формат: <code>/unschedule 09:00</code>",
        parse_mode="HTML", reply_markup=main_kb())


# ── Дайджест сейчас ───────────────────────────────────────────────
@router.message(Command("digest"))
@router.message(F.text == "📰 Дайджест сейчас")
async def cmd_digest_now(message: Message, db):
    channels = await db.get_user_channels(message.from_user.id)
    await message.answer("⏳ Собираю новости (10–30 сек)...")
    try:
        await _send_user_digest(
            bot=message.bot, db=db, client=None,
            user_id=message.from_user.id,
            channels=channels,
            since_hours=settings.DEFAULT_DIGEST_INTERVAL_HOURS,
        )
    except Exception as e:
        logger.error("Manual digest: %s", e)
        await message.answer(
            f"❌ Ошибка:\n<code>{he(str(e))}</code>",
            parse_mode="HTML", reply_markup=main_kb())


# ── Кеш статей ────────────────────────────────────────────────────
@router.message(Command("cache"))
@router.message(F.text == "📚 Кеш статей")
async def cmd_cache(message: Message, db):
    articles = await db.get_user_cache(message.from_user.id, limit=10)
    if not articles:
        await message.answer(
            "📭 Кеш пуст.\n\nСтатьи сохраняются автоматически после каждого дайджеста.",
            reply_markup=main_kb())
        return
    lines = []
    for a in articles:
        date_str = a["cached_at"][:16]
        url_part = f' <a href="{a["url"]}">↗</a>' if a["url"] else ""
        lines.append(f'• <b>{he(a["title"][:60])}</b>{url_part}\n'
                     f'  <i>{he(a["source"])} · {date_str}</i>')
    await message.answer(
        f"📚 <b>Кеш статей</b> (последние 10):\n\n" + "\n\n".join(lines),
        parse_mode="HTML",
        disable_web_page_preview=True,
        reply_markup=main_kb())
