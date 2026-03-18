"""
Trading News Analyzer – FastAPI Application (JSON storage edition)
"""

import threading
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timedelta
from pathlib import Path

from fastapi import FastAPI, Request, Form
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from .storage import (
    seed_defaults, get_feeds, add_feed, update_feed, toggle_feed, delete_feed,
    get_news, get_analyzed_news, get_score_history, get_stats,
)
from .sentiment import load_model, get_status as model_status, analyze_sentiment, is_model_ready
from .score_engine import compute_aggregate_score, compute_relevance, RELEVANCE_THRESHOLD
from .scheduler import start_scheduler, stop_scheduler, trigger_poll_now

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
)
logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    seed_defaults()
    threading.Thread(target=load_model, daemon=True, name="finbert-loader").start()
    start_scheduler()
    yield
    stop_scheduler()


app = FastAPI(title="Trading News Analyzer", version="2.0.0", lifespan=lifespan)

TEMPLATES_DIR = Path(__file__).parent / "templates"
templates = Jinja2Templates(directory=str(TEMPLATES_DIR))


def _rel_time(dt_str: str) -> str:
    if not dt_str:
        return "—"
    try:
        dt   = datetime.fromisoformat(dt_str)
        diff = datetime.utcnow() - dt
        s    = int(diff.total_seconds())
        if s < 60:
            return f"{s}s ago"
        m = s // 60
        if m < 60:
            return f"{m}m ago"
        return f"{m // 60}h ago"
    except Exception:
        return "—"


# ── HTML Pages ────────────────────────────────────────────────────
@app.get("/", response_class=HTMLResponse)
async def dashboard(request: Request):
    news       = get_news(limit=100)
    score_data = compute_aggregate_score(get_analyzed_news(300), window_minutes=10)
    return templates.TemplateResponse("dashboard.html", {
        "request":    request,
        "active_page": "dashboard",
        "score":      score_data,
        "news":       news,
        "rel_time":   _rel_time,
        "model":      model_status(),
    })


@app.get("/settings", response_class=HTMLResponse)
async def settings(request: Request):
    return templates.TemplateResponse("settings.html", {
        "request":    request,
        "active_page": "settings",
        "feeds":      get_feeds(),
        "stats":      get_stats(),
        "model":      model_status(),
    })


# ── JSON API ──────────────────────────────────────────────────────
@app.get("/api/status")
async def api_status():
    stats = get_stats()
    return {
        "model":       model_status(),
        "feeds":       sum(1 for f in get_feeds() if f.get("enabled")),
        "total_news":  stats["total_news"],
        "analyzed":    stats["analyzed"],
        "pending":     stats["pending"],
        "server_time": datetime.utcnow().isoformat(),
    }


@app.get("/api/news")
async def api_news(limit: int = 50):
    news = get_news(limit=limit)
    return [
        {
            **n,
            "age":      _rel_time(n.get("created_at", "")),
            "feed":     n.get("feed_name", "—"),
            "raw_score": round(n.get("raw_score", 0), 3),
            "positive":  round(n.get("positive", 0), 3),
            "negative":  round(n.get("negative", 0), 3),
            "neutral":   round(n.get("neutral", 0), 3),
            "relevance": round(n.get("relevance", 0), 2),
        }
        for n in news
    ]


@app.get("/api/score")
async def api_score(window: int = 10):
    return compute_aggregate_score(get_analyzed_news(300), window_minutes=window)


@app.get("/api/score/history")
async def api_score_history(hours: int = 2):
    return get_score_history(hours=hours)


# ── Test Analyzer Endpoint ────────────────────────────────────────
@app.post("/api/analyze")
async def api_analyze(text: str = Form(...)):
    """
    Analyze an arbitrary headline on-demand.
    Returns FinBERT scores + relevance + final score.
    """
    if not text.strip():
        return JSONResponse({"error": "Empty text"}, status_code=400)

    relevance   = compute_relevance(text)
    is_relevant = relevance >= RELEVANCE_THRESHOLD

    if not is_model_ready():
        return {
            "text":        text,
            "model_ready": False,
            "relevance":   round(relevance, 3),
            "is_relevant": is_relevant,
            "positive":    None,
            "negative":    None,
            "neutral":     None,
            "score":       None,
            "direction":   "model loading…",
        }

    sentiment = analyze_sentiment(text)
    score     = sentiment["score"]

    if score >= 0.35:
        direction = "bullish"
    elif score <= -0.35:
        direction = "bearish"
    else:
        direction = "neutral"

    return {
        "text":        text,
        "model_ready": True,
        "relevance":   round(relevance, 3),
        "is_relevant": is_relevant,
        "positive":    round(sentiment["positive"], 4),
        "negative":    round(sentiment["negative"], 4),
        "neutral":     round(sentiment["neutral"],  4),
        "score":       round(score, 4),
        "direction":   direction,
    }


# ── Feed Management ───────────────────────────────────────────────
@app.post("/api/feeds/add")
async def api_add_feed(name: str = Form(...), url: str = Form(...), weight: float = Form(1.0)):
    if not url.startswith("http"):
        return JSONResponse({"error": "Invalid URL"}, status_code=400)
    feeds = get_feeds()
    if any(f["url"] == url.strip() for f in feeds):
        return JSONResponse({"error": "Feed URL already exists"}, status_code=409)
    add_feed(name=name, url=url, weight=weight)
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/api/feeds/{feed_id}/toggle")
async def api_toggle_feed(feed_id: int):
    toggle_feed(feed_id)
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/api/feeds/{feed_id}/delete")
async def api_delete_feed(feed_id: int):
    delete_feed(feed_id)
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/api/feeds/{feed_id}/update")
async def api_update_feed(feed_id: int, name: str = Form(...), url: str = Form(...), weight: float = Form(1.0)):
    update_feed(feed_id, name=name, url=url, weight=weight)
    return RedirectResponse(url="/settings", status_code=303)


@app.post("/api/feeds/refresh")
async def api_refresh():
    trigger_poll_now()
    return {"status": "ok"}
