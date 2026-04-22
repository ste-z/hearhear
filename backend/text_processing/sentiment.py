from functools import lru_cache
import re


NEGATIVE_SENTIMENT_THRESHOLD = -0.05
POSITIVE_SENTIMENT_THRESHOLD = 0.05
MODERATE_SENTIMENT_THRESHOLD = 0.33
STRONG_SENTIMENT_THRESHOLD = 0.66
DEFAULT_SENTIMENT_SNIPPET_LIMIT = 3
MAX_SENTIMENT_SENTENCES = 180
SENTENCE_BOUNDARY_PATTERN = re.compile(r"(?<=[.!?])\s+(?=[\"'(\[]?[A-Z0-9])")


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


def _tone_strength(compound):
    magnitude = abs(float(compound))
    if magnitude >= STRONG_SENTIMENT_THRESHOLD:
        return "strong"
    if magnitude >= MODERATE_SENTIMENT_THRESHOLD:
        return "moderate"
    return "mild"


def _normalize_text(text):
    return re.sub(r"\s+", " ", str(text or "")).strip()


def _split_sentences(text):
    resolved_text = _normalize_text(text)
    if not resolved_text:
        return []

    sentences = []
    for sentence in SENTENCE_BOUNDARY_PATTERN.split(resolved_text):
        cleaned = _normalize_text(sentence)
        if len(cleaned) < 24:
            continue
        sentences.append(cleaned)
        if len(sentences) >= MAX_SENTIMENT_SENTENCES:
            break
    return sentences


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


def _sentence_sentiment_snippets(text, limit=DEFAULT_SENTIMENT_SNIPPET_LIMIT):
    scored_sentences = []
    for sentence in _split_sentences(text):
        scores = vader_sentiment_scores(sentence)
        if not scores:
            continue
        scored_sentences.append({
            "text": sentence,
            "compound": scores["compound"],
            "label": scores["label"],
        })

    positive = [
        row for row in scored_sentences
        if row["compound"] >= POSITIVE_SENTIMENT_THRESHOLD
    ]
    negative = [
        row for row in scored_sentences
        if row["compound"] <= NEGATIVE_SENTIMENT_THRESHOLD
    ]

    return {
        "positive": sorted(
            positive,
            key=lambda row: row["compound"],
            reverse=True,
        )[:limit],
        "negative": sorted(
            negative,
            key=lambda row: row["compound"],
        )[:limit],
    }


def vader_article_sentiment(article_text, title=None, summary=None):
    resolved_article_text = _normalize_text(article_text)
    if not resolved_article_text:
        resolved_article_text = _normalize_text(" ".join(
            part for part in (_normalize_text(title), _normalize_text(summary)) if part
        ))

    article_scores = vader_sentiment_scores(resolved_article_text)
    if article_scores is None:
        return None

    sentiment = dict(article_scores)
    sentiment["tone_strength"] = _tone_strength(sentiment["compound"])
    sentiment["text_scores"] = {
        "title": vader_sentiment_scores(title),
        "summary": vader_sentiment_scores(summary),
        "article": article_scores,
    }
    sentiment["snippets"] = _sentence_sentiment_snippets(resolved_article_text)
    sentiment["ranking_role"] = "display_only"
    return sentiment
