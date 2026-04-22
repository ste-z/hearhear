from functools import lru_cache


NEGATIVE_SENTIMENT_THRESHOLD = -0.05
POSITIVE_SENTIMENT_THRESHOLD = 0.05


@lru_cache(maxsize=1)
def _vader_analyzer():
    try:
        from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    except ImportError as exc:
        raise RuntimeError(
            "VADER sentiment analysis requires the vaderSentiment package. "
            "Install backend requirements before requesting article sentiment."
        ) from exc

    return SentimentIntensityAnalyzer()


def _sentiment_label(compound):
    if compound >= POSITIVE_SENTIMENT_THRESHOLD:
        return "positive"
    if compound <= NEGATIVE_SENTIMENT_THRESHOLD:
        return "negative"
    return "neutral"


def vader_sentiment_scores(text):
    resolved_text = str(text or "").strip()
    if not resolved_text:
        return None

    scores = _vader_analyzer().polarity_scores(resolved_text)
    compound = float(scores.get("compound", 0.0))

    return {
        "compound": compound,
        "negative": float(scores.get("neg", 0.0)),
        "neutral": float(scores.get("neu", 0.0)),
        "positive": float(scores.get("pos", 0.0)),
        "label": _sentiment_label(compound),
        "method": "VADER",
    }
