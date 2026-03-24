"""
instapaper.py — интеграция с Instapaper API
https://www.instapaper.com/api/simple

Instapaper Simple API (бесплатный):
  - Сохранить URL в закладки
  - Не требует OAuth — только логин/пароль аккаунта

Instapaper Full API (для партнёров):
  - Чтение, папки, полный текст статей
  - Требует OAuth 1.0a + одобрение Instapaper

КАК ИСПОЛЬЗОВАТЬ В БОТЕ:
  1. Пользователь регистрируется на instapaper.com
  2. Указывает логин/пароль боту (/instapaper login@mail.com password)
  3. Бот сохраняет учётные данные зашифровано в БД
  4. При каждом дайджесте важные статьи автоматически → Instapaper
  5. Пользователь читает в приложении Instapaper (оффлайн!)

ПОЧЕМУ INSTAPAPER ПОЛЕЗЕН ДЛЯ ЭТОГО БОТА:
  - Instapaper скачивает и хранит полный текст статьи (не только URL)
  - Приложение работает полностью ОФФЛАЙН
  - Синхронизируется через все устройства
  - Встроенный читалка без рекламы
"""
import logging
import httpx
from typing import Optional

logger = logging.getLogger(__name__)

INSTAPAPER_SIMPLE_API = "https://www.instapaper.com/api/add"
INSTAPAPER_AUTH_URL   = "https://www.instapaper.com/api/authenticate"


class InstapaperClient:
    def __init__(self, username: str, password: str = ""):
        self.username = username
        self.password = password

    async def authenticate(self) -> bool:
        """Проверить учётные данные (Simple API)."""
        try:
            async with httpx.AsyncClient(timeout=10.0) as http:
                resp = await http.post(INSTAPAPER_AUTH_URL, data={
                    "username": self.username,
                    "password": self.password,
                })
            # 200 = ok, 403 = неверный пароль, 500 = нет аккаунта
            return resp.status_code == 200
        except Exception as e:
            logger.error("Instapaper auth error: %s", e)
            return False

    async def save_url(self, url: str, title: str = "", description: str = "") -> bool:
        """
        Сохранить URL в Instapaper.
        Instapaper сам скачает полный текст статьи.
        """
        if not url:
            return False
        try:
            async with httpx.AsyncClient(timeout=15.0) as http:
                resp = await http.post(INSTAPAPER_SIMPLE_API, data={
                    "username": self.username,
                    "password": self.password,
                    "url": url,
                    "title": title[:250] if title else "",
                    "description": description[:500] if description else "",
                })
            if resp.status_code == 201:
                logger.info("Instapaper saved: %s", url)
                return True
            else:
                logger.warning("Instapaper save failed %d: %s", resp.status_code, url)
                return False
        except Exception as e:
            logger.error("Instapaper save error: %s", e)
            return False

    async def save_digest_items(self, items) -> int:
        """Сохранить все статьи из дайджеста. Возвращает кол-во сохранённых."""
        saved = 0
        for item in items:
            if item.url:
                ok = await self.save_url(
                    url=item.url,
                    title=item.title,
                    description=item.summary,
                )
                if ok:
                    saved += 1
        return saved


# ── Как добавить Instapaper в бота ───────────────────────────────
"""
1. В database.py добавить таблицу:

    CREATE TABLE IF NOT EXISTS instapaper_accounts (
        user_id   INTEGER PRIMARY KEY,
        username  TEXT NOT NULL,
        password  TEXT          -- хранить зашифровано!
    );

2. В handlers.py добавить команды:

    @router.message(Command("instapaper"))
    async def cmd_instapaper(message, state, db):
        # /instapaper connect login@mail.com mypassword
        parts = message.text.split()
        if len(parts) == 4 and parts[1] == "connect":
            client = InstapaperClient(parts[2], parts[3])
            ok = await client.authenticate()
            if ok:
                await db.save_instapaper(message.from_user.id, parts[2], parts[3])
                await message.answer("✅ Instapaper подключён!")
            else:
                await message.answer("❌ Неверный логин/пароль Instapaper")

3. В scheduler._send_user_digest добавить после формирования all_items:

    ip_account = await db.get_instapaper(user_id)
    if ip_account:
        client = InstapaperClient(ip_account["username"], ip_account["password"])
        saved = await client.save_digest_items(all_items)
        logger.info("Instapaper: saved %d articles for user %d", saved, user_id)

ВАЖНО:
- Instapaper Simple API бесплатный и не требует ключей, только аккаунт
- Full API (OAuth) нужен только для чтения статей обратно из Instapaper
- Для Full API нужно подать заявку: https://www.instapaper.com/main/request_oauth_consumer_token
- Альтернативы: Pocket (getpocket.com/developer), Readability API
"""
