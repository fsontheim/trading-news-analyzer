"""
Trading News Analyzer – FastAPI Application
-------------------------------------------
Routes:
  GET  /                    Dashboard (HTML)
  GET  /settings            Settings page (HTML)

  GET  /api/status          Model & system status (JSON)
  GET  /api/news            Recent news with scores (JSON)
  GET  /api/score           Current aggregate score (JSON)
  GET  /api/score/history   Score history for chart (JSON)

  POST /api/feeds/add       Add RSS feed
  POST /api/feeds/{id}/toggle   Enable / disable feed
  POST /api/feeds/{id}/delete   Delete feed
  POST /api/feeds/{id}/update   Update feed name/URL/weight
  POST /api/feeds/refresh   Trigger immediate poll
"""

import threading
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Form, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates
from sqlalchemy.orm import Session

from .database import engine, Base, get_db, SessionLocal
from .models import Feed, NewsItem, ScoreHistory
from .sentiment import load_model, get_status as model_status
from .score_engine import compute_aggregate_score, RELEVANCE_THRESHOLD
from .scheduler import start_scheduler, stop_scheduler, trigger_poll_now

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App lifespan
# ---------------------------------------------------------------------------
@asynccontextmanager
async def lifespan(app: FastAPI):
    # Create DB tables
    Base.metadata.create_all(bind=engine)
    _seed_default_feeds()
    # Load FinBERT in background (non-blocking)
    threading.Thread(target=load_model, daemon=True, name="finbert-loader").start()
    # Start RSS polling scheduler
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="Trading News Analyzer", version="1.0.0", lifespan=lifespan)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


# ---------------------------------------------------------------------------
# Seed default feed
# ---------------------------------------------------------------------------
def _seed_default_feeds():
    db = SessionLocal()
    try:
        if db.query(Feed).count() == 0:
            db.add(
                Feed(
                    name="FinancialJuice",
                    # NOTE: Verify this URL – update in Settings if needed
                    url="https://www.financialjuice.com/feed.ashx?xy=1",
                    weight=0.8,
                    enabled=True,
                )
            )
            db.commit()
            logger.info("Seeded default FinancialJuice feed")
    finally:
        db.close()


# ---------------------------------------------------------------------------
# Helper: format relative time for templates
# ---------------------------------------------------------------------------
def _rel_time(dt: datetime) -> str:
    if not dt:
        return "—"
    diff = datetime.utcnow() - dt
    s = int(diff.total_seconds())
    if s < 60:
        return f"{s}s ago"
    m = s // 60
    if m < 60:
        return f"{m}m ago"
    h = m // 60
    return f"{h}h ago"


# ---------------------------------------------------------------------------
# HTML Pages
# ---------------------------------------------------------------------------
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request, db: Session = Depends(get_db)):
    news = (
        db.query(NewsItem)
        .order_by(NewsItem.created_at.desc())
        .limit(100)
        .all()
    )
    score_data = compute_aggregate_score(news, window_minutes=10)

    return templates.TemplateResponse(
        "dashboard.html",
        {
            "request": request,
            "active_page": "dashboard",
            "score": score_data,
            "news": news,
            "rel_time": _rel_time,
            "model": model_status(),
        },
    )


@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request, db: Session = Depends(get_db)):
    feeds = db.query(Feed).order_by(Feed.created_at).all()
    stats = {
        "total_news": db.query(NewsItem).count(),
        "analyzed": db.query(NewsItem).filter(NewsItem.analyzed == True).count(),
        "pending": db.query(NewsItem).filter(NewsItem.analyzed == False, NewsItem.is_relevant == True).count(),
    }
    return templates.TemplateResponse(
        "settings.html",
        {
            "request": request,
            "active_page": "settings",
            "feeds": feeds,
            "stats": stats,
            "model": model_status(),
        },
    )


# ---------------------------------------------------------------------------
# JSON API – Status & Data
# ---------------------------------------------------------------------------
@app.get("/api/status")
async def api_status(db: Session = Depends(get_db)):
    return {
        "model": model_status(),
        "feeds": db.query(Feed).filter(Feed.enabled == True).count(),
        "total_news": db.query(NewsItem).count(),
        "analyzed": db.query(NewsItem).filter(NewsItem.analyzed == True).count(),
        "pending": db.query(NewsItem).filter(NewsItem.analyzed == False, NewsItem.is_relevant == True).count(),
        "server_time": datetime.utcnow().isoformat(),
    }


@app.get("/api/news")
async def api_news(limit: int = 50, db: Session = Depends(get_db)):
    news = (
        db.query(NewsItem)
        .order_by(NewsItem.created_at.desc())
        .limit(limit)
        .all()
    )
    return [
        {
            "id": n.id,
            "title": n.title,
            "url": n.url,
            "feed": n.feed.name if n.feed else "—",
            "published": n.published.isoformat() if n.published else None,
            "created_at": n.created_at.isoformat(),
            "age": _rel_time(n.created_at),
            "positive": round(n.positive, 3),
            "negative": round(n.negative, 3),
            "neutral": round(n.neutral, 3),
            "raw_score": round(n.raw_score, 3),
            "relevance": round(n.relevance, 2),
            "analyzed": n.analyzed,
            "is_relevant": n.is_relevant,
        }
        for n in news
    ]


@app.get("/api/score")
async def api_score(window: int = 10, db: Session = Depends(get_db)):
    news = (
        db.query(NewsItem)
        .filter(NewsItem.analyzed == True)
        .order_by(NewsItem.created_at.desc())
        .limit(300)
        .all()
    )
    return compute_aggregate_score(news, window_minutes=window)


@app.get("/api/score/history")
async def api_score_history(hours: int = 2, db: Session = Depends(get_db)):
    since = datetime.utcnow() - timedelta(hours=hours)
    history = (
        db.query(ScoreHistory)
        .filter(ScoreHistory.timestamp >= since)
        .order_by(ScoreHistory.timestamp.asc())
        .all()
    )
    return [
        {"timestamp": h.timestamp.isoformat(), "score": h.score, "news_count": h.news_count}
        for h in history
    ]


# ---------------------------------------------------------------------------
# JSON API – Feed Management
# ---------------------------------------------------------------------------
@app.post("/api/feeds/add")
async def add_feed(
    name: str = Form(...),
    url: str = Form(...),
    weight: float = Form(1.0),
    db: Session = Depends(get_db),
):
    # Validate URL (basic)
    if not url.startswith("http"):
        raise HTTPException(status_code=400, detail="URL must start with http")

    # Duplicate check
    if db.query(Feed).filter(Feed.url == url).first():
        raise HTTPException(status_code=409, detail="Feed URL already exists")

    feed = Feed(name=name.strip(), url=url.strip(), weight=min(max(weight, 0.1), 1.0))
    db.add(feed)
    db.commit()
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/api/feeds/{feed_id}/toggle")
async def toggle_feed(feed_id: int, db: Session = Depends(get_db)):
    feed = db.query(Feed).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404)
    feed.enabled = not feed.enabled
    db.commit()
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/api/feeds/{feed_id}/delete")
async def delete_feed(feed_id: int, db: Session = Depends(get_db)):
    feed = db.query(Feed).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404)
    # Also delete associated news items
    db.query(NewsItem).filter(NewsItem.feed_id == feed_id).delete()
    db.delete(feed)
    db.commit()
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/api/feeds/{feed_id}/update")
async def update_feed(
    feed_id: int,
    name: str = Form(...),
    url: str = Form(...),
    weight: float = Form(1.0),
    db: Session = Depends(get_db),
):
    feed = db.query(Feed).filter(Feed.id == feed_id).first()
    if not feed:
        raise HTTPException(status_code=404)
    feed.name = name.strip()
    feed.url = url.strip()
    feed.weight = min(max(weight, 0.1), 1.0)
    db.commit()
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/api/feeds/refresh")
async def refresh_feeds():
    """Trigger an immediate RSS poll."""
    trigger_poll_now()
    return {"status": "ok", "message": "Refresh triggered"}
