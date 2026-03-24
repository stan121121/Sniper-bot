"""
channel_reader.py — scraping https://t.me/s/{channel}
Только stdlib: re + html (NO beautifulsoup, NO telethon)

Реальная структура HTML t.me/s/:
  <div class="tgme_widget_message_wrap ...">
    <div class="tgme_widget_message ... " data-post="channel/12345">
      <div class="tgme_widget_message_text js-message_text">текст поста</div>
      <a class="tgme_widget_message_date" href="https://t.me/channel/12345">
        <time datetime="2024-01-01T12:00:00+00:00">...</time>
      </a>
    </div>
  </div>
"""
import asyncio
import html as html_stdlib
import logging
import re
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta

import httpx
from config import settings

logger = logging.getLogger(__name__)

TG_URL = "https://t.me/s/{username}"
HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (X11; Linux x86_64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/121.0.0.0 Safari/537.36"
    ),
    "Accept": "text/html,application/xhtml+xml",
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


# ── Regex-парсер (надёжнее stateful HTMLParser для этой страницы) ─

def _strip_tags(s: str) -> str:
    """Убрать HTML-теги, заменить <br> на \n."""
    s = re.sub(r"<br\s*/?>", "\n", s, flags=re.I)
    s = re.sub(r"<[^>]+>", "", s)
    return html_stdlib.unescape(s).strip()


def _parse_dt(iso: str) -> datetime:
    try:
        return datetime.fromisoformat(iso).astimezone(timezone.utc)
    except Exception:
        return datetime.now(timezone.utc)


def _scrape(html: str, username: str) -> list[Post]:
    """
    Парсим страницу t.me/s/{username} через регулярки.
    Telegram генерирует стабильную структуру, regex надёжнее stateful-парсера.
    """
    # 1. Название канала
    title_m = re.search(
        r'class="tgme_channel_info_header_title[^"]*"[^>]*>\s*<span[^>]*>([^<]+)</span>',
        html
    )
    if not title_m:
        title_m = re.search(r'<title>([^<]+)</title>', html)
    channel_title = html_stdlib.unescape(title_m.group(1).strip()) if title_m else username

    posts: list[Post] = []

    # 2. Каждый пост — блок между data-post="channel/ID"
    # Ищем все data-post атрибуты с ID
    post_ids_urls = re.findall(
        r'href="(https://t\.me/' + re.escape(username) + r'/(\d+))"',
        html, re.I
    )
    if not post_ids_urls:
        logger.debug("@%s: no post links found in HTML (len=%d)", username, len(html))
        return []

    # Дедупликация: одна ссылка может встречаться несколько раз (превью и кнопка)
    seen_ids: set[int] = set()
    unique_posts = []
    for url, pid_str in post_ids_urls:
        pid = int(pid_str)
        if pid not in seen_ids:
            seen_ids.add(pid)
            unique_posts.append((url, pid))

    # 3. Для каждого ID ищем соответствующий datetime и текст
    for post_url, post_id in unique_posts:
        # datetime рядом с этой ссылкой
        # Ищем: href=".../<post_id>"><time datetime="...">
        dt_pattern = re.escape(post_url) + r'"[^>]*>\s*<time[^>]+datetime="([^"]+)"'
        dt_m = re.search(dt_pattern, html)
        if not dt_m:
            # fallback: любой datetime поблизости
            idx = html.find(post_url)
            snippet = html[max(0, idx-50):idx+300]
            dt_m2 = re.search(r'datetime="([^"]+)"', snippet)
            post_date = _parse_dt(dt_m2.group(1)) if dt_m2 else datetime.now(timezone.utc)
        else:
            post_date = _parse_dt(dt_m.group(1))

        # Текст поста: блок tgme_widget_message_text перед этой ссылкой
        idx = html.find(post_url)
        if idx == -1:
            continue
        # Берём HTML от начала поста (data-post="channel/ID") до ссылки с датой
        start_marker = f'data-post="{username}/{post_id}"'
        start_idx = html.rfind(start_marker, 0, idx)
        if start_idx == -1:
            continue
        block = html[start_idx:idx]

        # Ищем текст внутри tgme_widget_message_text
        text_m = re.search(
            r'class="tgme_widget_message_text[^"]*"[^>]*>(.*?)</div>',
            block, re.S | re.I
        )
        if not text_m:
            continue  # пост без текста (только медиа)

        text = _strip_tags(text_m.group(1))
        if not text:
            continue

        posts.append(Post(
            id=post_id,
            channel=username,
            channel_title=channel_title,
            text=text[:1000],
            date=post_date,
            url=post_url,
        ))

    logger.debug("@%s: scraped %d posts", username, len(posts))
    return posts


# ── Public API ────────────────────────────────────────────────────

async def fetch_channel_posts(
    channel_username: str,
    limit: int = 20,
    since_hours: int = None,
    http_client: httpx.AsyncClient = None,
) -> list[Post]:
    url = TG_URL.format(username=channel_username)
    min_date = (
        datetime.now(timezone.utc) - timedelta(hours=since_hours)
        if since_hours else None
    )
    own = http_client is None
    if own:
        http_client = httpx.AsyncClient(timeout=20.0, follow_redirects=True)
    try:
        resp = await http_client.get(url, headers=HEADERS)
        if resp.status_code == 200:
            posts = _scrape(resp.text, channel_username)
        elif resp.status_code == 404:
            logger.warning("@%s: канал не найден (404)", channel_username)
            return []
        else:
            logger.warning("@%s: HTTP %d", channel_username, resp.status_code)
            return []
    except httpx.RequestError as e:
        logger.warning("@%s: сетевая ошибка: %s", channel_username, e)
        return []
    finally:
        if own:
            await http_client.aclose()

    if min_date:
        posts = [p for p in posts if p.date >= min_date]
    posts.sort(key=lambda p: p.date, reverse=True)
    return posts[:limit]


async def fetch_all_user_channels(
    channels: list[str],
    limit_per_channel: int = None,
    since_hours: int = None,
    client=None,  # совместимость, игнорируется
) -> list[Post]:
    limit = limit_per_channel or settings.POSTS_PER_CHANNEL
    all_posts: list[Post] = []
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as http:
        for ch in channels:
            posts = await fetch_channel_posts(
                ch, limit=limit, since_hours=since_hours, http_client=http
            )
            all_posts.extend(posts)
            logger.info("Fetched %d posts from @%s", len(posts), ch)
            await asyncio.sleep(0.4)
    all_posts.sort(key=lambda p: p.date, reverse=True)
    return all_posts


# ── Заглушки Telethon (совместимость) ────────────────────────────
async def get_telethon_client():
    return _DummyClient()

class _DummyClient:
    async def disconnect(self): pass
