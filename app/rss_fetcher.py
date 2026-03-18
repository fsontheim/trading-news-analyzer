"""
RSS Fetcher
-----------
Parses one or more RSS/Atom feeds and returns normalised entry dicts.
Portable to C# via SyndicationFeed / XmlReader.
"""

import logging
from datetime import datetime
from typing import List, Dict, Any

import feedparser

logger = logging.getLogger(__name__)

MAX_ENTRIES_PER_FEED = 30   # Only process the latest N entries per poll


def fetch_feed(url: str) -> List[Dict[str, Any]]:
    """
    Fetch and parse an RSS feed.

    Returns a list of dicts:
        {title, url, published, summary}
    Empty list on error.
    """
    try:
        feed = feedparser.parse(url, request_headers={"User-Agent": "TradingNewsBot/1.0"})

        if feed.bozo and feed.bozo_exception:
            logger.warning(f"Feed parse warning for {url}: {feed.bozo_exception}")

        entries = []
        for entry in feed.entries[:MAX_ENTRIES_PER_FEED]:
            title = entry.get("title", "").strip()
            if not title:
                continue

            link = entry.get("link") or entry.get("id") or ""

            # Try to parse published date
            published: datetime = datetime.utcnow()
            if hasattr(entry, "published_parsed") and entry.published_parsed:
                try:
                    published = datetime(*entry.published_parsed[:6])
                except Exception:
                    pass
            elif hasattr(entry, "updated_parsed") and entry.updated_parsed:
                try:
                    published = datetime(*entry.updated_parsed[:6])
                except Exception:
                    pass

            entries.append(
                {
                    "title": title,
                    "url": link or f"no-url-{hash(title)}",
                    "published": published,
                    "summary": entry.get("summary", ""),
                }
            )

        logger.debug(f"Fetched {len(entries)} entries from {url}")
        return entries

    except Exception as exc:
        logger.error(f"Error fetching {url}: {exc}")
        return []
