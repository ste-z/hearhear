import re

from backend.db.models import GuardianArticle


WORD_COUNT_PATTERN = re.compile(r"\b[\w'-]+\b")


def normalize_avoid_words(value):
    if value is None:
        return []

    if isinstance(value, str):
        raw_values = [value]
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]

    normalized = []
    seen = set()
    for raw_value in raw_values:
        for token in WORD_COUNT_PATTERN.findall(str(raw_value or "")):
            word = token.strip("_-'").casefold()
            if not word or word in seen:
                continue
            normalized.append(word)
            seen.add(word)
    return normalized


def _word_token_variants(value):
    tokens = set()
    if value is None:
        return tokens

    for token in WORD_COUNT_PATTERN.findall(str(value)):
        normalized = token.strip("_-'").casefold()
        if not normalized:
            continue
        tokens.add(normalized)
        for part in re.split(r"[-']", normalized):
            part = part.strip("_")
            if part:
                tokens.add(part)
    return tokens


def _article_searchable_token_set(article):
    if isinstance(article, dict):
        fields = [
            article.get("title"),
            article.get("summary"),
            article.get("body_text"),
            article.get("central_claim_summary"),
        ]
        keywords = article.get("keywords")
    else:
        fields = [
            getattr(article, "title", None),
            getattr(article, "summary", None),
            getattr(article, "body_text", None),
        ]
        keywords = getattr(article, "keywords", None)

    if isinstance(keywords, (list, tuple, set)):
        fields.extend(keywords)
    elif keywords:
        fields.append(keywords)

    tokens = set()
    for field in fields:
        tokens.update(_word_token_variants(field))
    return tokens


def filter_ranked_articles_by_avoid_words(ranked_articles, words_to_avoid=None):
    resolved_avoid_words = normalize_avoid_words(words_to_avoid)
    if not resolved_avoid_words:
        return list(ranked_articles)

    avoid_word_set = set(resolved_avoid_words)
    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    token_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(
                GuardianArticle.id,
                GuardianArticle.title,
                GuardianArticle.summary,
                GuardianArticle.body_text,
                GuardianArticle.keywords,
            )
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        token_lookup = {
            article_id: _article_searchable_token_set(
                {
                    "title": title,
                    "summary": summary,
                    "body_text": body_text,
                    "keywords": keywords,
                }
            )
            for article_id, title, summary, body_text, keywords in rows
        }

    filtered = []
    for article, score in ranked_articles:
        if isinstance(article, str):
            article_tokens = token_lookup.get(article.strip())
            if article_tokens is None:
                filtered.append((article, score))
                continue
        else:
            article_tokens = _article_searchable_token_set(article)

        if article_tokens.isdisjoint(avoid_word_set):
            filtered.append((article, score))

    return filtered

