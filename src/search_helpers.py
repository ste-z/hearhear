from pathlib import Path
from threading import Lock

from backend.runtime_debug import log_runtime_event
from flask import current_app, has_app_context

from models import GuardianArticle


_vector_index = None
_vector_index_doc_count = -1
_vector_index_lock = Lock()


def _resolve_db_path():
    if has_app_context():
        return Path(current_app.instance_path) / "data.db"
    return Path(__file__).resolve().parent.parent / "instance" / "data.db"


def _get_value(article, key, default=None):
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def serialize_article(article, score=None):
    authors_raw = _get_value(article, "contributors", [])
    authors = authors_raw if isinstance(authors_raw, list) else []
    author_raw = _get_value(article, "author_raw", "") or ""
    author_display = ", ".join(authors) if authors else author_raw

    date_val = _get_value(article, "date")
    if hasattr(date_val, "isoformat"):
        date_iso = date_val.isoformat()
    elif isinstance(date_val, str) and date_val.strip():
        date_iso = date_val
    else:
        date_iso = None

    n_contributors = _get_value(article, "n_contributors")
    if n_contributors is None:
        n_contributors = len(authors)

    payload = {
        "id": _get_value(article, "id"),
        "title": _get_value(article, "title"),
        "summary": _get_value(article, "summary"),
        "date": date_iso,
        "url": _get_value(article, "url"),
        "authors": authors,
        "author_display": author_display,
        "author_raw": author_raw,
        "n_contributors": int(n_contributors),
        "keywords": _get_value(article, "keywords", []) or [],
        "year": _get_value(article, "year"),
    }
    if score is not None:
        payload["score"] = float(score)
    return payload


def build_matches(ranked_articles):
    matches = []
    for article, score in ranked_articles:
        matches.append(serialize_article(article, score=score))
    return matches


def build_vector_processor(force_rebuild=False):
    global _vector_index, _vector_index_doc_count

    current_doc_count = GuardianArticle.query.count()
    cache_ok = (
        not force_rebuild
        and _vector_index is not None
        and _vector_index_doc_count == current_doc_count
    )
    if cache_ok:
        log_runtime_event(
            "vector_processor.cache_hit",
            doc_count=current_doc_count,
        )
        return _vector_index

    with _vector_index_lock:
        current_doc_count = GuardianArticle.query.count()
        cache_ok = (
            not force_rebuild
            and _vector_index is not None
            and _vector_index_doc_count == current_doc_count
        )
        if cache_ok:
            log_runtime_event(
                "vector_processor.cache_hit_after_lock",
                doc_count=current_doc_count,
            )
            return _vector_index

        from backend.text_preprocess import (
            DEFAULT_INDEX_DIR,
            DEFAULT_INDEX_NAME,
            preprocess_tfidf_index,
        )
        from backend.text_processor import VectorizedText

        log_runtime_event(
            "vector_processor.build_start",
            doc_count=current_doc_count,
            force_rebuild=bool(force_rebuild),
        )
        preprocess_tfidf_index(
            db_path=_resolve_db_path(),
            index_dir=DEFAULT_INDEX_DIR,
            index_name=DEFAULT_INDEX_NAME,
            force_rebuild=force_rebuild,
        )
        log_runtime_event("vector_processor.load_start", index_name=DEFAULT_INDEX_NAME)
        vector_index, _meta = VectorizedText.load(
            index_dir=DEFAULT_INDEX_DIR,
            index_name=DEFAULT_INDEX_NAME,
        )

        _vector_index = vector_index
        _vector_index_doc_count = current_doc_count
        log_runtime_event(
            "vector_processor.load_done",
            doc_count=current_doc_count,
            n_docs=getattr(vector_index, "n_docs", None),
            n_terms=getattr(vector_index, "n_terms", None),
        )
        return _vector_index
