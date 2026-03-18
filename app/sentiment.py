"""
FinBERT Sentiment Analyser
--------------------------
Model: ProsusAI/finbert  (positive / negative / neutral)
Label order: positive=0, negative=1, neutral=2

Thread-safe lazy loading – model is downloaded once and cached in HF_HOME.
Easily portable to C#: replace with ONNX Runtime + same tokenizer.
"""

import threading
import logging
from typing import Dict

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_tokenizer = None
_model = None
_model_ready = False
_model_loading = False
_model_error: str = ""

MODEL_NAME = "ProsusAI/finbert"


def load_model() -> None:
    """Download and load FinBERT. Call once in a background thread."""
    global _tokenizer, _model, _model_ready, _model_loading, _model_error

    with _lock:
        if _model_ready or _model_loading:
            return
        _model_loading = True

    logger.info("Loading FinBERT model – this may take a few minutes on first run...")

    try:
        # Import here so the app starts even if torch is still installing
        from transformers import BertTokenizer, BertForSequenceClassification
        import torch

        tok = BertTokenizer.from_pretrained(MODEL_NAME)
        mdl = BertForSequenceClassification.from_pretrained(MODEL_NAME)
        mdl.eval()

        with _lock:
            _tokenizer = tok
            _model = mdl
            _model_ready = True
            _model_loading = False

        logger.info("FinBERT model loaded successfully.")

    except Exception as exc:
        with _lock:
            _model_error = str(exc)
            _model_loading = False
        logger.error(f"Failed to load FinBERT: {exc}")


def is_model_ready() -> bool:
    return _model_ready


def get_status() -> Dict:
    return {
        "ready": _model_ready,
        "loading": _model_loading,
        "error": _model_error,
        "model": MODEL_NAME,
    }


def analyze_sentiment(text: str) -> Dict[str, float]:
    """
    Run FinBERT inference on a single headline.

    Returns:
        {positive, negative, neutral, score}
        score = positive – negative  ∈ [-1, +1]
    """
    if not _model_ready:
        return {"positive": 0.33, "neutral": 0.34, "negative": 0.33, "score": 0.0}

    import torch
    import torch.nn.functional as F

    inputs = _tokenizer(
        text,
        return_tensors="pt",
        max_length=128,
        truncation=True,
        padding=True,
    )

    with torch.no_grad():
        outputs = _model(**inputs)

    probs = F.softmax(outputs.logits, dim=1)[0]

    positive = round(probs[0].item(), 4)
    negative = round(probs[1].item(), 4)
    neutral = round(probs[2].item(), 4)
    score = round(positive - negative, 4)

    return {
        "positive": positive,
        "negative": negative,
        "neutral": neutral,
        "score": score,
    }
