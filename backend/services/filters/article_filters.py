from functools import lru_cache

from sqlalchemy import func

from backend.db.models import GuardianArticle
from backend.services.filters.text_filters import WORD_COUNT_PATTERN
from backend.services.rocchio_feedback import normalize_article_id_list


READING_TIME_WORDS_PER_MINUTE = 250


def coerce_year(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def coerce_length_count(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def article_character_count_expression():
    return GuardianArticle.body_character_count


def article_word_count_expression():
    return GuardianArticle.body_word_count


@lru_cache(maxsize=1)
def _available_article_bounds():
    character_count = article_character_count_expression()
    word_count = article_word_count_expression()
    row = (
        GuardianArticle.query.with_entities(
            func.min(GuardianArticle.year),
            func.max(GuardianArticle.year),
            func.min(character_count),
            func.max(character_count),
            func.min(word_count),
            func.max(word_count),
        )
        .first()
    )
    if row is None:
        return (None, None, None, None, None, None)
    return tuple(None if value is None else int(value) for value in row)


def clear_available_article_range_cache():
    _available_article_bounds.cache_clear()


def _reading_minutes_from_word_count(word_count):
    resolved_count = coerce_length_count(word_count)
    if resolved_count is None or resolved_count <= 0:
        return 0
    return max(1, ((resolved_count - 1) // READING_TIME_WORDS_PER_MINUTE) + 1)


def _reading_time_min_word_count(reading_minutes):
    resolved_minutes = coerce_length_count(reading_minutes)
    if resolved_minutes is None:
        return None
    if resolved_minutes <= 1:
        return 1
    return ((resolved_minutes - 1) * READING_TIME_WORDS_PER_MINUTE) + 1


def _reading_time_max_word_count(reading_minutes):
    resolved_minutes = coerce_length_count(reading_minutes)
    if resolved_minutes is None:
        return None
    return max(0, resolved_minutes * READING_TIME_WORDS_PER_MINUTE)


def available_article_year_range():
    min_year, max_year, _min_characters, _max_characters, _min_words, _max_words = (
        _available_article_bounds()
    )
    if min_year is None or max_year is None:
        return None, None
    return min_year, max_year


def available_article_character_range():
    _min_year, _max_year, min_characters, max_characters, _min_words, _max_words = (
        _available_article_bounds()
    )
    if min_characters is None or max_characters is None:
        return None, None
    return min_characters, max_characters


def available_article_word_range():
    _min_year, _max_year, _min_characters, _max_characters, min_words, max_words = (
        _available_article_bounds()
    )
    if min_words is None or max_words is None:
        return None, None
    return min_words, max_words


def available_article_reading_time_range():
    min_words, max_words = available_article_word_range()
    if min_words is None or max_words is None:
        return None, None
    return (
        _reading_minutes_from_word_count(min_words),
        _reading_minutes_from_word_count(max_words),
    )


def normalize_article_year_range(year_start=None, year_end=None):
    available_start, available_end = available_article_year_range()
    if available_start is None or available_end is None:
        return None, None
    if year_start is None and year_end is None:
        return None, None

    resolved_start = coerce_year(year_start)
    resolved_end = coerce_year(year_end)
    if resolved_start is None:
        resolved_start = available_start
    if resolved_end is None:
        resolved_end = available_end

    resolved_start = max(available_start, min(available_end, resolved_start))
    resolved_end = max(available_start, min(available_end, resolved_end))

    if resolved_start > resolved_end:
        raise ValueError("Start year must be less than or equal to end year.")

    if resolved_start == available_start and resolved_end == available_end:
        return None, None

    return resolved_start, resolved_end


def normalize_article_character_range(character_start=None, character_end=None):
    available_start, available_end = available_article_character_range()
    if available_start is None or available_end is None:
        return None, None
    if character_start is None and character_end is None:
        return None, None

    resolved_start = coerce_length_count(character_start)
    resolved_end = coerce_length_count(character_end)
    if resolved_start is None:
        resolved_start = available_start
    if resolved_end is None:
        resolved_end = available_end

    resolved_start = max(available_start, min(available_end, resolved_start))
    resolved_end = max(available_start, min(available_end, resolved_end))

    if resolved_start > resolved_end:
        raise ValueError(
            "Minimum article length must be less than or equal to maximum article length."
        )

    if resolved_start == available_start and resolved_end == available_end:
        return None, None

    return resolved_start, resolved_end


def normalize_article_word_range(word_start=None, word_end=None):
    available_start, available_end = available_article_word_range()
    if available_start is None or available_end is None:
        return None, None
    if word_start is None and word_end is None:
        return None, None

    resolved_start = coerce_length_count(word_start)
    resolved_end = coerce_length_count(word_end)
    if resolved_start is None:
        resolved_start = available_start
    if resolved_end is None:
        resolved_end = available_end

    resolved_start = max(available_start, min(available_end, resolved_start))
    resolved_end = max(available_start, min(available_end, resolved_end))

    if resolved_start > resolved_end:
        raise ValueError(
            "Minimum article word count must be less than or equal to maximum article word count."
        )

    if resolved_start == available_start and resolved_end == available_end:
        return None, None

    return resolved_start, resolved_end


def normalize_article_reading_time_range(reading_time_start=None, reading_time_end=None):
    available_start, available_end = available_article_reading_time_range()
    if available_start is None or available_end is None:
        return None, None
    if reading_time_start is None and reading_time_end is None:
        return None, None

    resolved_start = coerce_length_count(reading_time_start)
    resolved_end = coerce_length_count(reading_time_end)
    if resolved_start is None:
        resolved_start = available_start
    if resolved_end is None:
        resolved_end = available_end

    resolved_start = max(available_start, min(available_end, resolved_start))
    resolved_end = max(available_start, min(available_end, resolved_end))

    if resolved_start > resolved_end:
        raise ValueError(
            "Minimum article reading time must be less than or equal to maximum article reading time."
        )

    if resolved_start == available_start and resolved_end == available_end:
        return None, None

    return resolved_start, resolved_end


def word_range_for_reading_time_range(reading_time_start=None, reading_time_end=None):
    if reading_time_start is None and reading_time_end is None:
        return None, None
    return (
        _reading_time_min_word_count(reading_time_start),
        _reading_time_max_word_count(reading_time_end),
    )


def ranked_article_id(article):
    if isinstance(article, str):
        return article.strip()
    if isinstance(article, dict):
        value = article.get("id")
    else:
        value = getattr(article, "id", None)
    article_id = str(value or "").strip()
    return article_id or None


def _ranked_article_year(article, year_lookup):
    if isinstance(article, str):
        return coerce_year(year_lookup.get(article))
    if isinstance(article, dict):
        return coerce_year(article.get("year"))
    return coerce_year(getattr(article, "year", None))


def _body_text_character_count(value):
    if value is None:
        return None
    return len(str(value))


def _body_text_word_count(value):
    if value is None:
        return None
    return len(WORD_COUNT_PATTERN.findall(str(value)))


def _ranked_article_character_count(article, character_lookup):
    if isinstance(article, str):
        return coerce_length_count(character_lookup.get(article))
    if isinstance(article, dict):
        for count_key in (
            "character_count",
            "body_character_count",
            "article_character_count",
        ):
            explicit_count = coerce_length_count(article.get(count_key))
            if explicit_count is not None:
                return explicit_count
        if "body_text" in article:
            return _body_text_character_count(article.get("body_text"))
        article_id = ranked_article_id(article)
        return coerce_length_count(character_lookup.get(article_id))

    for count_attr in ("character_count", "body_character_count", "article_character_count"):
        explicit_count = coerce_length_count(getattr(article, count_attr, None))
        if explicit_count is not None:
            return explicit_count
    if hasattr(article, "body_text"):
        return _body_text_character_count(getattr(article, "body_text", None))
    article_id = ranked_article_id(article)
    return coerce_length_count(character_lookup.get(article_id))


def _ranked_article_word_count(article, word_lookup):
    if isinstance(article, str):
        return coerce_length_count(word_lookup.get(article))
    if isinstance(article, dict):
        for count_key in (
            "word_count",
            "body_word_count",
            "article_word_count",
        ):
            explicit_count = coerce_length_count(article.get(count_key))
            if explicit_count is not None:
                return explicit_count
        if "body_text" in article:
            return _body_text_word_count(article.get("body_text"))
        article_id = ranked_article_id(article)
        return coerce_length_count(word_lookup.get(article_id))

    explicit_count = coerce_length_count(getattr(article, "body_word_count", None))
    if explicit_count is not None:
        return explicit_count
    if hasattr(article, "body_text"):
        return _body_text_word_count(getattr(article, "body_text", None))
    article_id = ranked_article_id(article)
    return coerce_length_count(word_lookup.get(article_id))


def filter_query_by_article_ranges(
    query,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_word_start=None,
    reading_time_word_end=None,
):
    if year_start is not None:
        query = query.filter(GuardianArticle.year >= year_start)
    if year_end is not None:
        query = query.filter(GuardianArticle.year <= year_end)

    character_count = article_character_count_expression()
    if character_start is not None:
        query = query.filter(character_count >= character_start)
    if character_end is not None:
        query = query.filter(character_count <= character_end)

    word_count = article_word_count_expression()
    if word_start is not None:
        query = query.filter(word_count >= word_start)
    if word_end is not None:
        query = query.filter(word_count <= word_end)
    if reading_time_word_start is not None:
        query = query.filter(word_count >= reading_time_word_start)
    if reading_time_word_end is not None:
        query = query.filter(word_count <= reading_time_word_end)

    return query


def filter_ranked_articles_by_year_range(ranked_articles, year_start=None, year_end=None):
    if year_start is None and year_end is None:
        return list(ranked_articles)

    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    year_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(GuardianArticle.id, GuardianArticle.year)
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        year_lookup = {
            article_id: coerce_year(article_year)
            for article_id, article_year in rows
        }

    filtered = []
    for article, score in ranked_articles:
        article_year = _ranked_article_year(article, year_lookup)
        if article_year is None:
            continue
        if year_start is not None and article_year < year_start:
            continue
        if year_end is not None and article_year > year_end:
            continue
        filtered.append((article, score))

    return filtered


def filter_ranked_articles_by_character_range(
    ranked_articles,
    character_start=None,
    character_end=None,
):
    if character_start is None and character_end is None:
        return list(ranked_articles)

    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    character_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(
                GuardianArticle.id,
                article_character_count_expression(),
            )
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        character_lookup = {
            article_id: coerce_length_count(character_count)
            for article_id, character_count in rows
        }

    filtered = []
    for article, score in ranked_articles:
        character_count = _ranked_article_character_count(article, character_lookup)
        if character_count is None:
            continue
        if character_start is not None and character_count < character_start:
            continue
        if character_end is not None and character_count > character_end:
            continue
        filtered.append((article, score))

    return filtered


def filter_ranked_articles_by_word_range(ranked_articles, word_start=None, word_end=None):
    if word_start is None and word_end is None:
        return list(ranked_articles)

    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    word_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(
                GuardianArticle.id,
                article_word_count_expression(),
            )
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        word_lookup = {
            article_id: coerce_length_count(word_count)
            for article_id, word_count in rows
        }

    filtered = []
    for article, score in ranked_articles:
        word_count = _ranked_article_word_count(article, word_lookup)
        if word_count is None:
            continue
        if word_start is not None and word_count < word_start:
            continue
        if word_end is not None and word_count > word_end:
            continue
        filtered.append((article, score))

    return filtered


def filter_ranked_articles_by_metadata_ranges(
    ranked_articles,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_word_start=None,
    reading_time_word_end=None,
):
    if all(
        value is None
        for value in (
            year_start,
            year_end,
            character_start,
            character_end,
            word_start,
            word_end,
            reading_time_word_start,
            reading_time_word_end,
        )
    ):
        return list(ranked_articles)

    has_year_filter = year_start is not None or year_end is not None
    has_character_filter = character_start is not None or character_end is not None
    has_word_filter = any(
        value is not None
        for value in (
            word_start,
            word_end,
            reading_time_word_start,
            reading_time_word_end,
        )
    )

    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    year_lookup = {}
    character_lookup = {}
    word_lookup = {}
    if doc_ids_to_lookup:
        rows = (
            GuardianArticle.query.with_entities(
                GuardianArticle.id,
                GuardianArticle.year,
                article_character_count_expression(),
                article_word_count_expression(),
            )
            .filter(GuardianArticle.id.in_(doc_ids_to_lookup))
            .all()
        )
        for article_id, article_year, character_count, word_count in rows:
            year_lookup[article_id] = coerce_year(article_year)
            character_lookup[article_id] = coerce_length_count(character_count)
            word_lookup[article_id] = coerce_length_count(word_count)

    filtered = []
    for article, score in ranked_articles:
        if has_year_filter:
            article_year = _ranked_article_year(article, year_lookup)
            if article_year is None:
                continue
            if year_start is not None and article_year < year_start:
                continue
            if year_end is not None and article_year > year_end:
                continue

        if has_character_filter:
            character_count = _ranked_article_character_count(article, character_lookup)
            if character_count is None:
                continue
            if character_start is not None and character_count < character_start:
                continue
            if character_end is not None and character_count > character_end:
                continue

        if has_word_filter:
            word_count = _ranked_article_word_count(article, word_lookup)
            if word_count is None:
                continue
            if word_start is not None and word_count < word_start:
                continue
            if word_end is not None and word_count > word_end:
                continue
            if reading_time_word_start is not None and word_count < reading_time_word_start:
                continue
            if reading_time_word_end is not None and word_count > reading_time_word_end:
                continue

        filtered.append((article, score))

    return filtered


def filter_ranked_articles_by_excluded_ids(ranked_articles, excluded_article_ids):
    excluded_ids = set(normalize_article_id_list(excluded_article_ids))
    if not excluded_ids:
        return list(ranked_articles)

    return [
        (article, score)
        for article, score in ranked_articles
        if ranked_article_id(article) not in excluded_ids
    ]
