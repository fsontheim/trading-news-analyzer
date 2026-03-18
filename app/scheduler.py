"""
Scheduler
---------
Two background jobs:
  1. poll_feeds()       – every 30 s  – fetches RSS, runs FinBERT, stores results
  2. snapshot_score()   – every 60 s  – writes current aggregate score to history

Uses APScheduler with a ThreadPool so blocking calls don't block the event loop.
"""

import logging
from datetime import datetime

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.executors.pool import ThreadPoolExecutor

from .database import SessionLocal
from .models import Feed, NewsItem, ScoreHistory
from .rss_fetcher import fetch_feed
from .sentiment import analyze_sentiment, is_model_ready
from .score_engine import compute_relevance, RELEVANCE_THRESHOLD, compute_aggregate_score

logger = logging.getLogger(__name__)

_scheduler = BackgroundScheduler(
    executors={"default": ThreadPoolExecutor(max_workers=4)},
    job_defaults={"coalesce": True, "max_instances": 1},
)


# ---------------------------------------------------------------------------
# Job 1 – Poll RSS feeds
# ---------------------------------------------------------------------------
def poll_feeds() -> None:
    db = SessionLocal()
    try:
        feeds = db.query(Feed).filter(Feed.enabled == True).all()

        for feed in feeds:
            try:
                entries = fetch_feed(feed.url)
                new_count = 0

                for entry in entries:
                    # Skip if already stored (URL is unique key)
                    if db.query(NewsItem).filter(NewsItem.url == entry["url"]).first():
                        continue

                    relevance = compute_relevance(entry["title"])
                    is_relevant = relevance >= RELEVANCE_THRESHOLD

                    # Run FinBERT only if model ready AND headline is relevant
                    if is_relevant and is_model_ready():
                        sentiment = analyze_sentiment(entry["title"])
                        raw_score = sentiment["score"]
                        positive = sentiment["positive"]
                        negative = sentiment["negative"]
                        neutral = sentiment["neutral"]
                        analyzed = True
                    else:
                        raw_score = positive = negative = neutral = 0.0
                        analyzed = False

                    weighted_score = raw_score * relevance * feed.weight

                    item = NewsItem(
                        feed_id=feed.id,
                        title=entry["title"],
                        url=entry["url"],
                        published=entry["published"],
                        positive=positive,
                        neutral=neutral,
                        negative=negative,
                        raw_score=raw_score,
                        relevance=relevance,
                        weighted_score=weighted_score,
                        is_relevant=is_relevant,
                        analyzed=analyzed,
                    )
                    db.add(item)
                    new_count += 1

                if new_count:
                    db.commit()
                    logger.info(f"[{feed.name}] Added {new_count} new headlines")

            except Exception as exc:
                logger.error(f"Error processing feed '{feed.name}': {exc}")
                db.rollback()

        # Retroactively analyse pending items if model just became ready
        _analyse_pending(db)

    finally:
        db.close()


def _analyse_pending(db) -> None:
    """Process headlines that arrived before FinBERT was ready."""
    if not is_model_ready():
        return

    pending = (
        db.query(NewsItem)
        .filter(NewsItem.analyzed == False, NewsItem.is_relevant == True)
        .limit(50)
        .all()
    )
    if not pending:
        return

    for item in pending:
        try:
            sentiment = analyze_sentiment(item.title)
            item.positive = sentiment["positive"]
            item.negative = sentiment["negative"]
            item.neutral = sentiment["neutral"]
            item.raw_score = sentiment["score"]
            item.analyzed = True
            # Recalculate weighted score using feed weight
            source_w = item.feed.weight if item.feed else 1.0
            item.weighted_score = item.raw_score * item.relevance * source_w
        except Exception as exc:
            logger.warning(f"Failed retroactive analysis for item {item.id}: {exc}")

    db.commit()
    logger.info(f"Retroactively analysed {len(pending)} pending headlines")


# ---------------------------------------------------------------------------
# Job 2 – Score snapshot for history chart
# ---------------------------------------------------------------------------
def snapshot_score() -> None:
    db = SessionLocal()
    try:
        news = (
            db.query(NewsItem)
            .filter(NewsItem.analyzed == True)
            .order_by(NewsItem.created_at.desc())
            .limit(200)
            .all()
        )
        result = compute_aggregate_score(news, window_minutes=10)
        snap = ScoreHistory(
            score=result["score"],
            news_count=result["news_count"],
        )
        db.add(snap)
        db.commit()
    except Exception as exc:
        logger.error(f"Score snapshot error: {exc}")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def start_scheduler() -> None:
    _scheduler.add_job(poll_feeds, "interval", seconds=30, id="poll_feeds", replace_existing=True)
    _scheduler.add_job(snapshot_score, "interval", seconds=60, id="snapshot_score", replace_existing=True)
    _scheduler.start()
    logger.info("Scheduler started")


def stop_scheduler() -> None:
    if _scheduler.running:
        _scheduler.shutdown(wait=False)
        logger.info("Scheduler stopped")


def trigger_poll_now() -> None:
    """Trigger an immediate feed poll (called from API endpoint)."""
    _scheduler.modify_job("poll_feeds", next_run_time=datetime.now())
