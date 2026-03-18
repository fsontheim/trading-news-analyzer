"""
JSON Storage Layer
------------------
Replaces SQLite/SQLAlchemy with plain JSON files.

Files (in /app/data/):
  feeds.json         – RSS feed configurations
  news.json          – Processed headlines (capped at MAX_NEWS)
  score_history.json – Score snapshots (capped at MAX_HISTORY)

Thread-safe via threading.Lock per file.
Easily portable to C#: System.Text.Json + file I/O.
"""

import json
import os
import threading
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

DATA_DIR = Path(os.getenv("DATA_DIR", "/app/data"))
DATA_DIR.mkdir(parents=True, exist_ok=True)

FEEDS_FILE         = DATA_DIR / "feeds.json"
NEWS_FILE          = DATA_DIR / "news.json"
SCORE_HISTORY_FILE = DATA_DIR / "score_history.json"

MAX_NEWS    = 2000   # Keep latest N headlines
MAX_HISTORY = 5000   # Keep latest N score snapshots

_locks: Dict[str, threading.Lock] = {
    "feeds":   threading.Lock(),
    "news":    threading.Lock(),
    "history": threading.Lock(),
}

# ---------------------------------------------------------------------------
# Low-level helpers
# ---------------------------------------------------------------------------
def _read(path: Path) -> Any:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)   # atomic on POSIX & NTFS


def _now() -> str:
    return datetime.utcnow().isoformat()

# ---------------------------------------------------------------------------
# Feed CRUD
# ---------------------------------------------------------------------------
def get_feeds() -> List[Dict]:
    with _locks["feeds"]:
        return _read(FEEDS_FILE)


def get_feed(feed_id: int) -> Optional[Dict]:
    return next((f for f in get_feeds() if f["id"] == feed_id), None)


def add_feed(name: str, url: str, weight: float = 1.0) -> Dict:
    with _locks["feeds"]:
        feeds = _read(FEEDS_FILE)
        new_id = max((f["id"] for f in feeds), default=0) + 1
        feed = {
            "id":            new_id,
            "name":          name.strip(),
            "url":           url.strip(),
            "weight":        round(min(max(weight, 0.1), 1.0), 2),
            "enabled":       True,
            "poll_interval": 30,
            "created_at":    _now(),
        }
        feeds.append(feed)
        _write(FEEDS_FILE, feeds)
        return feed


def update_feed(feed_id: int, name: str, url: str, weight: float) -> bool:
    with _locks["feeds"]:
        feeds = _read(FEEDS_FILE)
        for f in feeds:
            if f["id"] == feed_id:
                f["name"]   = name.strip()
                f["url"]    = url.strip()
                f["weight"] = round(min(max(weight, 0.1), 1.0), 2)
                _write(FEEDS_FILE, feeds)
                return True
        return False


def toggle_feed(feed_id: int) -> bool:
    with _locks["feeds"]:
        feeds = _read(FEEDS_FILE)
        for f in feeds:
            if f["id"] == feed_id:
                f["enabled"] = not f["enabled"]
                _write(FEEDS_FILE, feeds)
                return True
        return False


def delete_feed(feed_id: int) -> bool:
    with _locks["feeds"]:
        feeds = _read(FEEDS_FILE)
        new_feeds = [f for f in feeds if f["id"] != feed_id]
        if len(new_feeds) == len(feeds):
            return False
        _write(FEEDS_FILE, new_feeds)

    # Remove associated news items
    with _locks["news"]:
        news = _read(NEWS_FILE)
        _write(NEWS_FILE, [n for n in news if n.get("feed_id") != feed_id])

    return True


# ---------------------------------------------------------------------------
# News CRUD
# ---------------------------------------------------------------------------
def get_news(limit: int = 100) -> List[Dict]:
    with _locks["news"]:
        items = _read(NEWS_FILE)
    # Already sorted newest-first; just slice
    return items[:limit]


def news_url_exists(url: str) -> bool:
    with _locks["news"]:
        items = _read(NEWS_FILE)
    return any(n["url"] == url for n in items)


def add_news_item(item: Dict) -> Dict:
    """Prepend a new item and trim to MAX_NEWS."""
    with _locks["news"]:
        items = _read(NEWS_FILE)
        items.insert(0, item)
        if len(items) > MAX_NEWS:
            items = items[:MAX_NEWS]
        _write(NEWS_FILE, items)
    return item


def update_news_sentiment(item_url: str, sentiment: Dict) -> bool:
    """Write FinBERT results back to an existing item."""
    with _locks["news"]:
        items = _read(NEWS_FILE)
        for n in items:
            if n["url"] == item_url:
                n["positive"]  = sentiment["positive"]
                n["negative"]  = sentiment["negative"]
                n["neutral"]   = sentiment["neutral"]
                n["raw_score"] = sentiment["score"]
                n["analyzed"]  = True
                # Recalculate weighted score
                n["weighted_score"] = (
                    sentiment["score"] * n.get("relevance", 1.0) * n.get("feed_weight", 1.0)
                )
                _write(NEWS_FILE, items)
                return True
        return False


def get_pending_news(limit: int = 50) -> List[Dict]:
    """Items that are relevant but not yet analysed."""
    with _locks["news"]:
        items = _read(NEWS_FILE)
    return [n for n in items if n.get("is_relevant") and not n.get("analyzed")][:limit]


def get_analyzed_news(limit: int = 300) -> List[Dict]:
    with _locks["news"]:
        items = _read(NEWS_FILE)
    return [n for n in items if n.get("analyzed")][:limit]


def get_stats() -> Dict:
    with _locks["news"]:
        items = _read(NEWS_FILE)
    total    = len(items)
    analyzed = sum(1 for n in items if n.get("analyzed"))
    pending  = sum(1 for n in items if n.get("is_relevant") and not n.get("analyzed"))
    return {"total_news": total, "analyzed": analyzed, "pending": pending}


# ---------------------------------------------------------------------------
# Score History
# ---------------------------------------------------------------------------
def append_score_history(score: float, news_count: int) -> None:
    with _locks["history"]:
        history = _read(SCORE_HISTORY_FILE)
        history.append({"timestamp": _now(), "score": score, "news_count": news_count})
        if len(history) > MAX_HISTORY:
            history = history[-MAX_HISTORY:]
        _write(SCORE_HISTORY_FILE, history)


def get_score_history(hours: int = 2) -> List[Dict]:
    cutoff = datetime.utcnow().timestamp() - hours * 3600
    with _locks["history"]:
        history = _read(SCORE_HISTORY_FILE)
    return [
        h for h in history
        if datetime.fromisoformat(h["timestamp"]).timestamp() >= cutoff
    ]


# ---------------------------------------------------------------------------
# Seed default data
# ---------------------------------------------------------------------------
def seed_defaults() -> None:
    if not get_feeds():
        add_feed(
            name="FinancialJuice",
            url="https://www.financialjuice.com/feed.ashx?xy=1",
            weight=0.8,
        )
