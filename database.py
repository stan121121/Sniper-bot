"""
database.py — SQLite: пользователи, каналы, расписания, кеш статей
"""
import aiosqlite
import logging
from typing import Optional
from config import settings

logger = logging.getLogger(__name__)


class Database:
    def __init__(self, path: str = None):
        self.path = path or settings.DB_PATH

    async def init(self):
        async with aiosqlite.connect(self.path) as db:
            await db.executescript("""
                CREATE TABLE IF NOT EXISTS users (
                    user_id     INTEGER PRIMARY KEY,
                    username    TEXT,
                    created_at  TEXT DEFAULT (datetime('now')),
                    active      INTEGER DEFAULT 1
                );

                CREATE TABLE IF NOT EXISTS channels (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    username    TEXT NOT NULL,
                    title       TEXT,
                    added_at    TEXT DEFAULT (datetime('now')),
                    UNIQUE(user_id, username)
                );

                -- Расписание: несколько временных точек на пользователя
                CREATE TABLE IF NOT EXISTS schedules (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    hour        INTEGER NOT NULL,
                    minute      INTEGER NOT NULL DEFAULT 0,
                    UNIQUE(user_id, hour, minute)
                );

                CREATE TABLE IF NOT EXISTS digest_log (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    sent_at     TEXT DEFAULT (datetime('now')),
                    news_count  INTEGER,
                    summary     TEXT   -- "Итог дня"
                );

                CREATE TABLE IF NOT EXISTS seen_posts (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    channel     TEXT NOT NULL,
                    post_id     INTEGER NOT NULL,
                    UNIQUE(user_id, channel, post_id)
                );

                -- Кеш статей для оффлайн-доступа
                CREATE TABLE IF NOT EXISTS article_cache (
                    id          INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id     INTEGER NOT NULL,
                    url         TEXT NOT NULL,
                    title       TEXT,
                    full_text   TEXT,
                    source      TEXT,
                    cached_at   TEXT DEFAULT (datetime('now')),
                    UNIQUE(user_id, url)
                );
            """)
            await db.commit()
        logger.info("Database initialized: %s", self.path)

    # ── Пользователи ──────────────────────────────────────────────
    async def upsert_user(self, user_id: int, username: Optional[str] = None):
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """INSERT INTO users(user_id, username) VALUES(?,?)
                   ON CONFLICT(user_id) DO UPDATE SET username=excluded.username""",
                (user_id, username),
            )
            await db.commit()

    async def get_all_active_users(self) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM users WHERE active=1")
            return [dict(r) for r in await cur.fetchall()]

    async def get_user(self, user_id: int) -> Optional[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute("SELECT * FROM users WHERE user_id=?", (user_id,))
            row = await cur.fetchone()
            return dict(row) if row else None

    # ── Каналы ────────────────────────────────────────────────────
    async def add_channel(self, user_id: int, username: str, title: str = "") -> bool:
        try:
            async with aiosqlite.connect(self.path) as db:
                await db.execute(
                    "INSERT INTO channels(user_id, username, title) VALUES(?,?,?)",
                    (user_id, username.lower().lstrip("@"), title),
                )
                await db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def remove_channel(self, user_id: int, username: str) -> bool:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "DELETE FROM channels WHERE user_id=? AND username=?",
                (user_id, username.lower().lstrip("@")),
            )
            await db.commit()
            return cur.rowcount > 0

    async def get_user_channels(self, user_id: int) -> list[str]:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "SELECT username FROM channels WHERE user_id=?", (user_id,)
            )
            return [r[0] for r in await cur.fetchall()]

    # ── Расписание ────────────────────────────────────────────────
    async def add_schedule(self, user_id: int, hour: int, minute: int = 0) -> bool:
        try:
            async with aiosqlite.connect(self.path) as db:
                await db.execute(
                    "INSERT INTO schedules(user_id, hour, minute) VALUES(?,?,?)",
                    (user_id, hour, minute),
                )
                await db.commit()
            return True
        except aiosqlite.IntegrityError:
            return False

    async def remove_schedule(self, user_id: int, hour: int, minute: int = 0) -> bool:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "DELETE FROM schedules WHERE user_id=? AND hour=? AND minute=?",
                (user_id, hour, minute),
            )
            await db.commit()
            return cur.rowcount > 0

    async def get_user_schedules(self, user_id: int) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT hour, minute FROM schedules WHERE user_id=? ORDER BY hour, minute",
                (user_id,),
            )
            return [dict(r) for r in await cur.fetchall()]

    async def get_users_for_time(self, hour: int, minute: int) -> list[int]:
        """Вернуть user_id всех пользователей, у которых расписание совпадает с H:MM."""
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                """SELECT DISTINCT s.user_id FROM schedules s
                   JOIN users u ON u.user_id = s.user_id
                   WHERE s.hour=? AND s.minute=? AND u.active=1""",
                (hour, minute),
            )
            return [r[0] for r in await cur.fetchall()]

    # ── Seen posts ────────────────────────────────────────────────
    async def mark_seen(self, user_id: int, channel: str, post_ids: list[int]):
        async with aiosqlite.connect(self.path) as db:
            await db.executemany(
                "INSERT OR IGNORE INTO seen_posts(user_id,channel,post_id) VALUES(?,?,?)",
                [(user_id, channel, pid) for pid in post_ids],
            )
            await db.commit()

    async def filter_new_posts(self, user_id: int, channel: str, post_ids: list[int]) -> list[int]:
        if not post_ids:
            return []
        ph = ",".join("?" * len(post_ids))
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                f"SELECT post_id FROM seen_posts WHERE user_id=? AND channel=? AND post_id IN ({ph})",
                [user_id, channel, *post_ids],
            )
            seen = {r[0] for r in await cur.fetchall()}
        return [pid for pid in post_ids if pid not in seen]

    # ── Кеш статей ───────────────────────────────────────────────
    async def cache_article(self, user_id: int, url: str, title: str,
                             full_text: str, source: str):
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                """INSERT INTO article_cache(user_id,url,title,full_text,source)
                   VALUES(?,?,?,?,?)
                   ON CONFLICT(user_id,url) DO UPDATE SET
                     full_text=excluded.full_text,
                     title=excluded.title,
                     cached_at=datetime('now')""",
                (user_id, url, title, full_text, source),
            )
            await db.commit()

    async def get_cached_article(self, user_id: int, url: str) -> Optional[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                "SELECT * FROM article_cache WHERE user_id=? AND url=?",
                (user_id, url),
            )
            row = await cur.fetchone()
            return dict(row) if row else None

    async def get_user_cache(self, user_id: int, limit: int = 20) -> list[dict]:
        async with aiosqlite.connect(self.path) as db:
            db.row_factory = aiosqlite.Row
            cur = await db.execute(
                """SELECT id, title, url, source, cached_at
                   FROM article_cache WHERE user_id=?
                   ORDER BY cached_at DESC LIMIT ?""",
                (user_id, limit),
            )
            return [dict(r) for r in await cur.fetchall()]

    async def delete_cached_article(self, user_id: int, article_id: int) -> bool:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                "DELETE FROM article_cache WHERE user_id=? AND id=?",
                (user_id, article_id),
            )
            await db.commit()
            return cur.rowcount > 0

    # ── Лог дайджестов ───────────────────────────────────────────
    async def log_digest(self, user_id: int, news_count: int, summary: str = ""):
        async with aiosqlite.connect(self.path) as db:
            await db.execute(
                "INSERT INTO digest_log(user_id,news_count,summary) VALUES(?,?,?)",
                (user_id, news_count, summary),
            )
            await db.commit()

    async def get_last_digest_summary(self, user_id: int) -> Optional[str]:
        async with aiosqlite.connect(self.path) as db:
            cur = await db.execute(
                """SELECT summary FROM digest_log
                   WHERE user_id=? AND summary!=''
                   ORDER BY sent_at DESC LIMIT 1""",
                (user_id,),
            )
            row = await cur.fetchone()
            return row[0] if row else None
