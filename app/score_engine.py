"""
Score Engine
------------
Computes:
  1. Relevance score per headline  (keyword dictionary approach)
  2. Aggregate market score        (weighted average over a time window)

Formula (per headline):
  weighted_score = raw_score × relevance × source_weight

Aggregate score (rolling window):
  score = Σ(weighted_score × recency_weight) / Σ(recency_weight)
  clamped to [-1, +1]

Portable to C#: pure business-logic, no framework dependencies.
"""

from datetime import datetime, timedelta
from typing import List, Optional

# ---------------------------------------------------------------------------
# Keyword relevance dictionary
# Relevance level → list of lowercase phrases
# ---------------------------------------------------------------------------
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
        "market", "trading", "bonds", "ipo", "fed", "trump", "president",
    ],
}

# Items below this threshold are stored but NOT sent through FinBERT
RELEVANCE_THRESHOLD = 0.3


def compute_relevance(text: str) -> float:
    """
    Score how market-relevant a headline is (0.0 – 1.0).
    Returns the highest matched relevance level, or 0.1 if no match.
    """
    lower = text.lower()
    best = 0.1
    for level, keywords in RELEVANCE_KEYWORDS.items():
        for kw in keywords:
            if kw in lower:
                if level > best:
                    best = level
                break  # only one match needed per level
    return best


def recency_weight(created_at: datetime, window_minutes: int) -> float:
    """
    Linear decay:  1.0 at age=0,  0.05 at age=window_minutes
    """
    age_seconds = (datetime.utcnow() - created_at).total_seconds()
    age_minutes = age_seconds / 60
    if age_minutes >= window_minutes:
        return 0.05
    return max(0.05, 1.0 - (age_minutes / window_minutes) * 0.95)


def compute_aggregate_score(
    news_items,          # List[NewsItem] – SQLAlchemy objects or dicts
    window_minutes: int = 10,
) -> dict:
    """
    Compute the aggregate market score from recent analysed news.

    Returns:
        {score, direction, news_count, window_minutes}
    """
    cutoff = datetime.utcnow() - timedelta(minutes=window_minutes)

    # Filter: within window, analysed, relevant
    relevant = [
        n for n in news_items
        if n.analyzed and n.is_relevant and n.created_at >= cutoff
    ]

    if not relevant:
        return {
            "score": 0.0,
            "direction": "neutral",
            "news_count": 0,
            "window_minutes": window_minutes,
        }

    total_weight = 0.0
    weighted_sum = 0.0

    for item in relevant:
        rw = recency_weight(item.created_at, window_minutes)
        # source weight comes from item.weighted_score already incorporating feed.weight
        # but we recompute for the aggregate with recency on top
        source_w = getattr(item.feed, "weight", 1.0) if item.feed else 1.0
        w = rw * item.relevance * source_w
        weighted_sum += item.raw_score * w
        total_weight += w

    if total_weight == 0:
        raw = 0.0
    else:
        raw = weighted_sum / total_weight

    score = max(-1.0, min(1.0, round(raw, 4)))

    if score >= 0.35:
        direction = "bullish"
    elif score <= -0.35:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "score": score,
        "direction": direction,
        "news_count": len(relevant),
        "window_minutes": window_minutes,
    }
