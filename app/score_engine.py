"""
Score Engine - works with plain dicts (JSON storage)
Thresholds and window are read from settings.json at runtime.
"""
from datetime import datetime, timedelta
from typing import List, Dict

RELEVANCE_KEYWORDS = {
    1.0: [
        "federal reserve", "fed rate", "fomc", "interest rate", "rate hike", "rate cut",
        "inflation", "cpi", "pce", "gdp", "nonfarm payroll", "unemployment rate",
        "recession", "default", "debt ceiling", "bankruptcy", "bank failure",
        "sanctions", "war", "military strike", "nuclear",
    ],
    0.85: [
        "earnings beat", "earnings miss", "beats expectations", "misses expectations",
        "guidance raised", "guidance lowered", "revenue", "profit warning",
        "layoffs", "job cuts", "merger", "acquisition", "ipo",
        "rate decision", "central bank", "ecb", "boe",
    ],
    0.65: [
        "oil", "energy crisis", "opec", "trade war", "tariff", "china", "bitcoin",
        "crypto", "nasdaq", "dow jones", "s&p", "market crash", "market rally",
        "bank", "liquidity", "yield curve",
    ],
    0.45: [
        "analyst", "upgrade", "downgrade", "forecast", "outlook", "stocks",
        "market", "trading", "bonds", "fed", "trump", "president",
    ],
}

RELEVANCE_THRESHOLD = 0.3   # fallback default only


def compute_relevance(text: str) -> float:
    lower = text.lower()
    best = 0.1
    for level, keywords in RELEVANCE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                if level > best:
                    best = level
                break
    return best


def _recency_weight(created_at_str: str, window_minutes: int) -> float:
    try:
        created_at = datetime.fromisoformat(created_at_str)
    except Exception:
        return 0.05
    age_minutes = (datetime.utcnow() - created_at).total_seconds() / 60
    if age_minutes >= window_minutes:
        return 0.05
    return max(0.05, 1.0 - (age_minutes / window_minutes) * 0.95)


def compute_aggregate_score(news_items: List[Dict], window_minutes: int = None) -> Dict:
    # Pull live settings — import here to avoid circular imports
    from .storage import get_engine_settings
    cfg = get_engine_settings()

    if window_minutes is None:
        window_minutes = cfg["window_minutes"]

    bull_threshold = cfg["bull_threshold"]
    bear_threshold = cfg["bear_threshold"]
    rel_threshold  = cfg["relevance_threshold"]

    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)
    relevant = [
        n for n in news_items
        if n.get("analyzed") and n.get("is_relevant")
        and n.get("relevance", 0) >= rel_threshold
        and datetime.fromisoformat(n["created_at"]) >= cutoff
    ]
    if not relevant:
        return {"score": 0.0, "direction": "neutral", "news_count": 0, "window_minutes": window_minutes}

    total_weight = 0.0
    weighted_sum = 0.0
    for item in relevant:
        rw       = _recency_weight(item["created_at"], window_minutes)
        source_w = item.get("feed_weight", 1.0)
        rel      = item.get("relevance", 1.0)
        w        = rw * rel * source_w
        weighted_sum += item.get("raw_score", 0.0) * w
        total_weight += w

    raw   = weighted_sum / total_weight if total_weight else 0.0
    score = max(-1.0, min(1.0, round(raw, 4)))

    if score >= bull_threshold:
        direction = "bullish"
    elif score <= bear_threshold:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "score":         score,
        "direction":     direction,
        "news_count":    len(relevant),
        "window_minutes": window_minutes,
        "bull_threshold": bull_threshold,
        "bear_threshold": bear_threshold,
    }
