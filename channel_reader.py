"""
channel_reader.py — scraping t.me/s/{channel}
Только stdlib: html.parser + re (NO beautifulsoup4, NO telethon)
"""
import asyncio
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from html.parser import HTMLParser

import httpx
from config import settings

logger = logging.getLogger(__name__)

TG_PREVIEW_URL = "https://t.me/s/{username}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/120.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "ru-RU,ru;q=0.9,en;q=0.8",
}


@dataclass
class Post:
    id: int
    channel: str
    channel_title: str
    text: str
    date: datetime
    url: str


# ── stdlib HTML parser ────────────────────────────────────────────
class _TgParser(HTMLParser):
    """
    Парсит HTML t.me/s/{channel}.
    Собирает посты: текст, дата, url.
    """
    def __init__(self, username: str):
        super().__init__()
        self.username = username
        self.channel_title = username
        self.posts: list[dict] = []

        # State
        self._in_title = False
        self._in_msg = False
        self._in_text = False
        self._depth_msg = 0
        self._depth_text = 0
        self._cur: dict = {}
        self._text_parts: list[str] = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        classes = attrs.get("class", "")

        # Заголовок канала
        if "tgme_channel_info_header_title" in classes:
            self._in_title = True

        # Начало сообщения
        if "tgme_widget_message" in classes and "tgme_widget_message_wrap" not in classes:
            self._in_msg = True
            self._depth_msg = 0
            self._cur = {}
            self._text_parts = []

        if self._in_msg:
            self._depth_msg += 1

            # Ссылка с датой
            if "tgme_widget_message_date" in classes and "href" in attrs:
                self._cur["url"] = attrs["href"]
                m = re.search(r"/(\d+)$", attrs["href"])
                if m:
                    self._cur["id"] = int(m.group(1))

            # Время публикации
            if tag == "time" and "datetime" in attrs:
                self._cur["datetime"] = attrs["datetime"]

            # Начало текста поста
            if "tgme_widget_message_text" in classes:
                self._in_text = True
                self._depth_text = 0

        if self._in_text:
            self._depth_text += 1
            if tag == "br":
                self._text_parts.append("\n")

    def handle_endtag(self, tag):
        if self._in_title:
            self._in_title = False

        if self._in_text:
            self._depth_text -= 1
            if self._depth_text <= 0:
                self._in_text = False
                self._cur["text"] = "".join(self._text_parts).strip()

        if self._in_msg:
            self._depth_msg -= 1
            if self._depth_msg <= 0:
                self._in_msg = False
                if self._cur.get("id") and self._cur.get("text"):
                    self.posts.append(dict(self._cur))

    def handle_data(self, data):
        if self._in_title:
            self.channel_title += data  # накапливаем (сбросим ниже)
        if self._in_text:
            self._text_parts.append(data)

    def handle_entityref(self, name):
        refs = {"amp": "&", "lt": "<", "gt": ">", "quot": '"', "nbsp": " "}
        if self._in_text:
            self._text_parts.append(refs.get(name, ""))

    def handle_charref(self, name):
        if self._in_text:
            try:
                ch = chr(int(name[1:], 16) if name.startswith("x") else int(name))
                self._text_parts.append(ch)
            except Exception:
                pass


def _parse_dt(iso: str) -> datetime:
    try:
        return datetime.fromisoformat(iso).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _scrape(html: str, username: str) -> list[Post]:
    parser = _TgParser(username)
    # Убираем двойное накопление заголовка
    parser.channel_title = ""
    parser.feed(html)
    title = parser.channel_title.strip() or username

    posts = []
    for p in parser.posts:
        posts.append(Post(
            id=p["id"],
            channel=username,
            channel_title=title,
            text=p.get("text", ""),
            date=_parse_dt(p.get("datetime", "")),
            url=p.get("url", f"https://t.me/{username}/{p['id']}"),
        ))
    return posts


# ── Public API ────────────────────────────────────────────────────
async def fetch_channel_posts(
    channel_username: str,
    limit: int = 20,
    since_hours: int = None,
    http_client: httpx.AsyncClient = None,
) -> list[Post]:
    url = TG_PREVIEW_URL.format(username=channel_username)
    min_date = (
        datetime.now(timezone.utc) - timedelta(hours=since_hours)
        if since_hours else None
    )
    own = http_client is None
    if own:
        http_client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)
    try:
        resp = await http_client.get(url, headers=HEADERS)
        if resp.status_code != 200:
            logger.warning("@%s HTTP %d", channel_username, resp.status_code)
            return []
        posts = _scrape(resp.text, channel_username)
    except httpx.RequestError as e:
        logger.warning("@%s network error: %s", channel_username, e)
        return []
    finally:
        if own:
            await http_client.aclose()

    result = [p for p in posts if not min_date or p.date >= min_date]
    result.sort(key=lambda p: p.date, reverse=True)
    return result[:limit]


async def fetch_all_user_channels(
    channels: list[str],
    limit_per_channel: int = None,
    since_hours: int = None,
    client=None,   # совместимость, игнорируется
) -> list[Post]:
    limit = limit_per_channel or settings.POSTS_PER_CHANNEL
    all_posts: list[Post] = []
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http:
        for ch in channels:
            posts = await fetch_channel_posts(ch, limit=limit,
                                              since_hours=since_hours,
                                              http_client=http)
            all_posts.extend(posts)
            logger.info("Fetched %d posts from @%s", len(posts), ch)
            await asyncio.sleep(0.3)
    all_posts.sort(key=lambda p: p.date, reverse=True)
    return all_posts


# ── Заглушка Telethon (совместимость) ────────────────────────────
async def get_telethon_client():
    return _DummyClient()

class _DummyClient:
    async def disconnect(self): pass
