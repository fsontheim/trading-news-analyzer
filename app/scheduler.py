"""
Scheduler - updated for JSON storage
"""
import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

from .storage import (
    get_feeds, news_url_exists, add_news_item,
    update_news_sentiment, get_pending_news, append_score_history,
    get_analyzed_news,
)
from .rss_fetcher import fetch_feed
from .sentiment import analyze_sentiment, is_model_ready
from .score_engine import compute_relevance, RELEVANCE_THRESHOLD, compute_aggregate_score

logger = logging.getLogger(__name__)

_scheduler = BackgroundScheduler(
    executors={"default": ThreadPoolExecutor(max_workers=4)},
    job_defaults={"coalesce": True, "max_instances": 1},
)


def poll_feeds() -> None:
    feeds = [f for f in get_feeds() if f.get("enabled", True)]
    for feed in feeds:
        try:
            entries = fetch_feed(feed["url"])
            new_count = 0
            for entry in entries:
                if news_url_exists(entry["url"]):
                    continue
                relevance   = compute_relevance(entry["title"])
                is_relevant = relevance >= RELEVANCE_THRESHOLD
                if is_relevant and is_model_ready():
                    sentiment = analyze_sentiment(entry["title"])
                    raw_score = sentiment["score"]
                    positive  = sentiment["positive"]
                    negative  = sentiment["negative"]
                    neutral   = sentiment["neutral"]
                    analyzed  = True
                else:
                    raw_score = positive = negative = neutral = 0.0
                    analyzed  = False
                fw = feed.get("weight", 1.0)
                item = {
                    "feed_id":        feed["id"],
                    "feed_name":      feed["name"],
                    "feed_weight":    fw,
                    "title":          entry["title"],
                    "url":            entry["url"],
                    "published":      entry["published"].isoformat() if entry["published"] else None,
                    "positive":       positive,
                    "neutral":        neutral,
                    "negative":       negative,
                    "raw_score":      raw_score,
                    "relevance":      relevance,
                    "weighted_score": raw_score * relevance * fw,
                    "is_relevant":    is_relevant,
                    "analyzed":       analyzed,
                    "created_at":     datetime.utcnow().isoformat(),
                }
                add_news_item(item)
                new_count += 1
            if new_count:
                logger.info(f"[{feed['name']}] +{new_count} headlines")
        except Exception as exc:
            logger.error(f"Feed error '{feed['name']}': {exc}")
    _analyse_pending()


def _analyse_pending() -> None:
    if not is_model_ready():
        return
    pending = get_pending_news(limit=50)
    if not pending:
        return
    for item in pending:
        try:
            sentiment = analyze_sentiment(item["title"])
            update_news_sentiment(item["url"], sentiment)
        except Exception as exc:
            logger.warning(f"Retroactive analysis failed: {exc}")
    if pending:
        logger.info(f"Retroactively analysed {len(pending)} items")


def snapshot_score() -> None:
    news   = get_analyzed_news(limit=300)
    result = compute_aggregate_score(news, window_minutes=10)
    append_score_history(result["score"], result["news_count"])


def start_scheduler() -> None:
    _scheduler.add_job(poll_feeds,     "interval", seconds=30, id="poll_feeds",     replace_existing=True)
    _scheduler.add_job(snapshot_score, "interval", seconds=60, id="snapshot_score", replace_existing=True)
    _scheduler.start()
    logger.info("Scheduler started")


def stop_scheduler() -> None:
    if _scheduler.running:
        _scheduler.shutdown(wait=False)


def trigger_poll_now() -> None:
    _scheduler.modify_job("poll_feeds", next_run_time=datetime.now())
