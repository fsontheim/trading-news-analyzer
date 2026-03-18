# ── Trading News Analyzer ─────────────────────────────────────────
# Multi-stage build: separate dependency install for faster rebuilds
# CPU-only PyTorch keeps the image manageable (~2 GB)
# ──────────────────────────────────────────────────────────────────
FROM python:3.11-slim AS base

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential \
        curl \
    && rm -rf /var/lib/apt/lists/*

# ── Dependencies ──────────────────────────────────────────────────
FROM base AS deps

COPY requirements.txt .

# CPU-only torch first (much smaller than CUDA version)
RUN pip install --no-cache-dir \
        torch==2.1.0 \
        --index-url https://download.pytorch.org/whl/cpu

# Remaining dependencies
RUN pip install --no-cache-dir -r requirements.txt

# ── Final image ───────────────────────────────────────────────────
FROM base AS final

# Copy installed packages from deps stage
COPY --from=deps /usr/local/lib/python3.11 /usr/local/lib/python3.11
COPY --from=deps /usr/local/bin /usr/local/bin

# Copy application code
COPY app/ ./app/

# Persistent data directory (SQLite DB lives here)
RUN mkdir -p /app/data

# HuggingFace model cache will be stored in a named volume
ENV HF_HOME=/root/.cache/huggingface
ENV DATABASE_URL=sqlite:////app/data/trading_news.db
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

EXPOSE 8003

HEALTHCHECK --interval=30s --timeout=10s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:8003/api/status || exit 1

CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8003", "--workers", "1"]
