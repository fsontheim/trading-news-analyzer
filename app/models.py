from datetime import datetime
from sqlalchemy import Column, Integer, String, Float, Boolean, DateTime, ForeignKey, Text
from sqlalchemy.orm import relationship
from .database import Base


class Feed(Base):
    """RSS Feed source configuration."""
    __tablename__ = "feeds"

    id = Column(Integer, primary_key=True, index=True)
    name = Column(String(100), nullable=False)
    url = Column(String(500), unique=True, nullable=False)
    weight = Column(Float, default=1.0)        # Source credibility weight (0.1 – 1.0)
    enabled = Column(Boolean, default=True)
    poll_interval = Column(Integer, default=30) # Seconds between polls
    created_at = Column(DateTime, default=datetime.utcnow)

    news_items = relationship("NewsItem", back_populates="feed")


class NewsItem(Base):
    """Processed news headline with sentiment scores."""
    __tablename__ = "news_items"

    id = Column(Integer, primary_key=True, index=True)
    feed_id = Column(Integer, ForeignKey("feeds.id"), nullable=False)
    title = Column(String(500), nullable=False)
    url = Column(String(500), unique=True, nullable=False, index=True)
    published = Column(DateTime, nullable=True)

    # FinBERT output (0.0 – 1.0 each, sum ≈ 1.0)
    positive = Column(Float, default=0.0)
    neutral = Column(Float, default=0.0)
    negative = Column(Float, default=0.0)

    # Derived scores
    raw_score = Column(Float, default=0.0)       # positive – negative  (-1 to +1)
    relevance = Column(Float, default=0.0)       # keyword relevance score (0 – 1)
    weighted_score = Column(Float, default=0.0)  # raw_score * relevance * feed.weight

    is_relevant = Column(Boolean, default=False)
    analyzed = Column(Boolean, default=False)    # False = FinBERT not yet run

    created_at = Column(DateTime, default=datetime.utcnow)

    feed = relationship("Feed", back_populates="news_items")


class ScoreHistory(Base):
    """Periodic snapshot of the aggregate market score (for chart)."""
    __tablename__ = "score_history"

    id = Column(Integer, primary_key=True, index=True)
    score = Column(Float, default=0.0)
    news_count = Column(Integer, default=0)
    timestamp = Column(DateTime, default=datetime.utcnow, index=True)
