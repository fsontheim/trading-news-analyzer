"""
JSON Storage Layer
------------------
Files in /app/data/:
  feeds.json         – RSS feed configurations
  news.json          – Processed headlines (capped at MAX_NEWS)
  score_history.json – Score snapshots (capped at MAX_HISTORY)
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
SETTINGS_FILE      = DATA_DIR / "settings.json"

MAX_NEWS    = 2000
MAX_HISTORY = 5000

_locks: Dict[str, threading.Lock] = {
    "feeds":    threading.Lock(),
    "news":     threading.Lock(),
    "history":  threading.Lock(),
    "settings": threading.Lock(),
}

def _read(path: Path) -> Any:
    if not path.exists():
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def _write(path: Path, data: Any) -> None:
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    tmp.replace(path)

def _now() -> str:
    return datetime.utcnow().isoformat()

# ---------------------------------------------------------------------------
# Feed CRUD
# ---------------------------------------------------------------------------
def get_feeds() -> List[Dict]:
    with _locks["feeds"]:
        feeds = _read(FEEDS_FILE)
    # Migration: ensure all feeds have required fields
    changed = False
    for f in feeds:
        if "poll_interval" not in f:
            f["poll_interval"] = 30
            changed = True
        if "last_polled" not in f:
            f["last_polled"] = None
            changed = True
    if changed:
        with _locks["feeds"]:
            _write(FEEDS_FILE, feeds)
    return feeds

def get_feed(feed_id: int) -> Optional[Dict]:
    return next((f for f in get_feeds() if f["id"] == feed_id), None)

def add_feed(name: str, url: str, weight: float = 1.0, poll_interval: int = 30) -> Dict:
    with _locks["feeds"]:
        feeds = _read(FEEDS_FILE)
        new_id = max((f["id"] for f in feeds), default=0) + 1
        feed = {
            "id":            new_id,
            "name":          name.strip(),
            "url":           url.strip(),
            "weight":        round(min(max(weight, 0.1), 1.0), 2),
            "enabled":       True,
            "poll_interval": max(10, int(poll_interval)),  # seconds, minimum 10
            "last_polled":   None,
            "created_at":    _now(),
        }
        feeds.append(feed)
        _write(FEEDS_FILE, feeds)
        return feed

def update_feed(feed_id: int, name: str, url: str, weight: float, poll_interval: int) -> bool:
    with _locks["feeds"]:
        feeds = _read(FEEDS_FILE)
        for f in feeds:
            if f["id"] == feed_id:
                f["name"]          = name.strip()
                f["url"]           = url.strip()
                f["weight"]        = round(min(max(weight, 0.1), 1.0), 2)
                f["poll_interval"] = max(10, int(poll_interval))
                _write(FEEDS_FILE, feeds)
                return True
        return False

def update_feed_last_polled(feed_id: int) -> None:
    with _locks["feeds"]:
        feeds = _read(FEEDS_FILE)
        for f in feeds:
            if f["id"] == feed_id:
                f["last_polled"] = _now()
                break
        _write(FEEDS_FILE, feeds)

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
    return items[:limit]

def news_url_exists(url: str) -> bool:
    with _locks["news"]:
        items = _read(NEWS_FILE)
    return any(n["url"] == url for n in items)

def add_news_item(item: Dict) -> Dict:
    with _locks["news"]:
        items = _read(NEWS_FILE)
        items.insert(0, item)
        if len(items) > MAX_NEWS:
            items = items[:MAX_NEWS]
        _write(NEWS_FILE, items)
    return item

def update_news_sentiment(item_url: str, sentiment: Dict) -> bool:
    with _locks["news"]:
        items = _read(NEWS_FILE)
        for n in items:
            if n["url"] == item_url:
                n["positive"]       = sentiment["positive"]
                n["negative"]       = sentiment["negative"]
                n["neutral"]        = sentiment["neutral"]
                n["raw_score"]      = sentiment["score"]
                n["analyzed"]       = True
                n["weighted_score"] = (
                    sentiment["score"] * n.get("relevance", 1.0) * n.get("feed_weight", 1.0)
                )
                _write(NEWS_FILE, items)
                return True
        return False

def get_pending_news(limit: int = 50) -> List[Dict]:
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
# Engine Settings & Profiles
# ---------------------------------------------------------------------------

BUILTIN_PROFILES = {
    "default": {
        "label":               "Default",
        "description":         "Balanced settings. Good starting point for any market.",
        "bull_threshold":      0.35,
        "bear_threshold":     -0.35,
        "window_minutes":      10,
        "relevance_threshold": 0.30,
    },
    "0dte": {
        "label":               "0DTE SPX/SPY",
        "description":         "Optimised for BullPut/BearCall Spreads & Iron Condors on SPX/SPY. Fast window, tight thresholds, strict relevance filter.",
        "bull_threshold":      0.20,
        "bear_threshold":     -0.20,
        "window_minutes":      2,
        "relevance_threshold": 0.50,
    },
    "swing": {
        "label":               "Swing Trading",
        "description":         "Multi-day positions. Slower, smoother signals. Less noise, more stability.",
        "bull_threshold":      0.35,
        "bear_threshold":     -0.35,
        "window_minutes":      30,
        "relevance_threshold": 0.40,
    },
    "event_day": {
        "label":               "Event Day (FOMC/CPI/NFP)",
        "description":         "High-impact macro days. Ultra-sensitive, 1-min window, hard macro-only filter. Switch manually on known event days.",
        "bull_threshold":      0.15,
        "bear_threshold":     -0.15,
        "window_minutes":      1,
        "relevance_threshold": 0.70,
    },
}

SETTINGS_DEFAULTS = {
    "active_profile":      "0dte",
    "bull_threshold":       0.20,
    "bear_threshold":      -0.20,
    "window_minutes":       2,
    "relevance_threshold":  0.50,
    "dedup_enabled":        True,   # fuzzy title deduplication
    "dedup_threshold":      0.80,   # similarity threshold (0.0–1.0)
    "dedup_window":         100,    # compare against last N headlines
}


def get_engine_settings() -> Dict:
    with _locks["settings"]:
        data = _read(SETTINGS_FILE) if SETTINGS_FILE.exists() else {}
    if not isinstance(data, dict):
        data = {}
    result = {**SETTINGS_DEFAULTS, **data}
    result["profiles"] = BUILTIN_PROFILES
    return result


def save_engine_settings(
    bull_threshold:      float,
    bear_threshold:      float,
    window_minutes:      int,
    relevance_threshold: float,
    active_profile:      str = "custom",
    dedup_enabled:       bool = True,
    dedup_threshold:     float = 0.80,
    dedup_window:        int = 100,
) -> Dict:
    settings = {
        "active_profile":      active_profile,
        "bull_threshold":      round(min(max(float(bull_threshold),   0.05),  0.95), 2),
        "bear_threshold":      round(max(min(float(bear_threshold),  -0.05), -0.95), 2),
        "window_minutes":      max(1, min(int(window_minutes), 120)),
        "relevance_threshold": round(min(max(float(relevance_threshold), 0.05), 0.95), 2),
        "dedup_enabled":       bool(dedup_enabled),
        "dedup_threshold":     round(min(max(float(dedup_threshold), 0.50), 0.99), 2),
        "dedup_window":        max(10, min(int(dedup_window), 500)),
    }
    with _locks["settings"]:
        _write(SETTINGS_FILE, settings)
    settings["profiles"] = BUILTIN_PROFILES
    return settings


def activate_profile(profile_key: str) -> Dict:
    """Switch to a built-in profile, preserving dedup settings."""
    profile = BUILTIN_PROFILES.get(profile_key)
    if not profile:
        raise ValueError(f"Unknown profile: {profile_key}")
    # Preserve current dedup settings when switching profiles
    current = get_engine_settings()
    return save_engine_settings(
        bull_threshold=      profile["bull_threshold"],
        bear_threshold=      profile["bear_threshold"],
        window_minutes=      profile["window_minutes"],
        relevance_threshold= profile["relevance_threshold"],
        active_profile=      profile_key,
        dedup_enabled=       current.get("dedup_enabled", True),
        dedup_threshold=     current.get("dedup_threshold", 0.80),
        dedup_window=        current.get("dedup_window", 100),
    )


# ---------------------------------------------------------------------------
# Fuzzy Deduplication
# ---------------------------------------------------------------------------
def is_fuzzy_duplicate(title: str) -> tuple[bool, float, str]:
    """
    Check if a headline is too similar to a recently stored one.

    Returns:
        (is_duplicate, similarity_score, matched_title)

    Uses difflib.SequenceMatcher — no external dependencies.
    Portable to C#: String.Compare or Levenshtein via NuGet.
    """
    import difflib

    cfg     = get_engine_settings()
    if not cfg.get("dedup_enabled", True):
        return False, 0.0, ""

    threshold = cfg.get("dedup_threshold", 0.80)
    window    = cfg.get("dedup_window",    100)

    title_lower = title.lower().strip()

    with _locks["news"]:
        recent = _read(NEWS_FILE)[:window]

    for item in recent:
        existing = item.get("title", "").lower().strip()
        if not existing:
            continue
        ratio = difflib.SequenceMatcher(None, title_lower, existing).ratio()
        if ratio >= threshold:
            return True, round(ratio, 3), item.get("title", "")

    return False, 0.0, ""


# ---------------------------------------------------------------------------
# Seed defaults
# ---------------------------------------------------------------------------
def seed_defaults() -> None:
    if not get_feeds():
        add_feed(
            name="FinancialJuice",
            url="https://www.financialjuice.com/feed.ashx?xy=1",
            weight=0.8,
            poll_interval=30,
        )
