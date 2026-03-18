# Trading News Analyzer

News-driven sentiment module for a trading bot.
Fetches RSS feeds → FinBERT sentiment → rolling score −1…+1.

## Quick Start

```bash
# 1. Clone / extract project
cd trading-news-analyzer

# 2. Create data directory
mkdir -p data

# 3. Build & start
docker compose up --build

# 4. Open browser
open http://localhost:8003
```

**First start:** FinBERT (~430 MB) downloads automatically and is cached.
Expect 2–5 min before the first scores appear. Watch the navbar status dot.

---

## Architecture

```
RSS Feeds (30 s poll)
        │
        ▼
  Keyword Filter  ──── relevance < 0.3 ──→ stored, not analysed
        │
        ▼
  FinBERT ONNX  (ProsusAI/finbert)
   positive / negative / neutral
        │
        ▼
  Score Engine
   raw_score   = positive − negative
   weighted    = raw_score × relevance × recency × feed_weight
   aggregate   = Σ weighted / Σ weights  (10-min window)
        │
        ▼
  Bot Signal  −1.0 … +1.0
   ≥ +0.35  →  bullish
   ≤ −0.35  →  bearish
   else     →  neutral
```

## File Structure

```
app/
  main.py          ← FastAPI routes (HTML + JSON API)
  database.py      ← SQLAlchemy engine / session
  models.py        ← Feed, NewsItem, ScoreHistory ORM models
  rss_fetcher.py   ← feedparser wrapper
  sentiment.py     ← FinBERT wrapper (lazy-loading, thread-safe)
  score_engine.py  ← relevance scoring + aggregate calculation
  scheduler.py     ← APScheduler background jobs
  templates/
    base.html
    dashboard.html
    settings.html
```

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| GET | `/` | Dashboard (HTML) |
| GET | `/settings` | Feed settings (HTML) |
| GET | `/api/status` | Model & system status |
| GET | `/api/news?limit=50` | Recent headlines (JSON) |
| GET | `/api/score?window=10` | Current aggregate score |
| GET | `/api/score/history?hours=2` | Score history for chart |
| POST | `/api/feeds/add` | Add RSS feed |
| POST | `/api/feeds/{id}/toggle` | Enable / disable feed |
| POST | `/api/feeds/{id}/update` | Update feed |
| POST | `/api/feeds/{id}/delete` | Delete feed |
| POST | `/api/feeds/refresh` | Trigger immediate poll |

## Score API Response

```json
{
  "score": -0.42,
  "direction": "bearish",
  "news_count": 8,
  "window_minutes": 10
}
```

Consume this from C# with `HttpClient` → JSON deserialization.

## Porting to C#

Each module maps cleanly:

| Python module | C# equivalent |
|--------------|---------------|
| `rss_fetcher.py` | `SyndicationFeed` + `XmlReader` |
| `sentiment.py` | `Microsoft.ML.OnnxRuntime` + finbert.onnx |
| `score_engine.py` | Plain class with same math |
| `scheduler.py` | `System.Timers.Timer` or BackgroundService |
| `main.py` | ASP.NET Core Minimal API |

For ONNX conversion: `optimum-cli export onnx --model ProsusAI/finbert finbert_onnx/`

## Adding More Feeds (Settings Page)

Good free RSS sources:
- Reuters: `https://feeds.reuters.com/reuters/businessNews`
- CNBC: `https://search.cnbc.com/rs/search/combinedcms/view.xml?partnerId=wrss01&id=10000664`
- Investing.com: `https://www.investing.com/rss/news.rss`

**FinancialJuice URL:** Verify the exact URL at financialjuice.com – update in Settings if the default doesn't work.

## Notes

- **Latency:** RSS poll 100 ms + FinBERT CPU ~25 ms + scoring <1 ms ≈ **~130 ms total**
- **Noise reduction:** Only headlines matching market keywords are sent to FinBERT
- **Score alone is not a trading signal** – combine 30% news + 70% technical signal
