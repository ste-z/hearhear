from pathlib import Path
from threading import Lock

from backend.db.models import GuardianArticle
from backend.runtime.runtime_debug import log_runtime_event
from flask import current_app, has_app_context


DEFAULT_RETRIEVAL_MODEL = "tfidf"
SUPPORTED_RETRIEVAL_MODELS = ("tfidf", "svd")

_RETRIEVAL_MODEL_ALIASES = {
    "tfidf": "tfidf",
    "svd": "svd",
    "truncated_svd": "svd",
    "truncated-svd": "svd",
    "lsa": "svd",
}

_vector_processors = {}
_vector_processor_doc_counts = {}
_vector_index_lock = Lock()


def _resolve_db_path():
    if has_app_context():
        return Path(current_app.instance_path) / "data.db"
    return Path(__file__).resolve().parent.parent.parent / "instance" / "data.db"


def _get_value(article, key, default=None):
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def normalize_retrieval_model(value, default=DEFAULT_RETRIEVAL_MODEL):
    if value is None:
        return default

    normalized = str(value).strip().lower()
    if not normalized:
        return default

    normalized = normalized.replace(" ", "_")
    resolved = _RETRIEVAL_MODEL_ALIASES.get(normalized)
    if resolved is None:
        supported = ", ".join(SUPPORTED_RETRIEVAL_MODELS)
        raise ValueError(
            f"Unsupported retrieval_model {value!r}. Supported models: {supported}."
        )
    return resolved


def _retrieval_model_config(retrieval_model):
    resolved_model = normalize_retrieval_model(retrieval_model)

    if resolved_model == "tfidf":
        from backend.text_processing.text_preprocess import (
            DEFAULT_INDEX_DIR,
            DEFAULT_INDEX_NAME,
            preprocess_tfidf_index,
        )
        from backend.text_processing.text_processor import (
            TfidfPostingsIndex,
            load_search_index,
        )

        return {
            "retrieval_model": resolved_model,
            "index_dir": DEFAULT_INDEX_DIR,
            "index_name": DEFAULT_INDEX_NAME,
            "preprocess": preprocess_tfidf_index,
            "load": load_search_index,
            "has_artifacts": TfidfPostingsIndex.has_artifacts,
            "load_kwargs": {
                "load_articles": False,
                "allow_matrix_fallback": False,
            },
        }

    from backend.text_processing.svd_processor import (
        DEFAULT_INDEX_DIR,
        DEFAULT_SVD_INDEX_NAME,
        TruncatedSvdIndex,
        load_svd_index,
        preprocess_svd_index,
    )

    return {
        "retrieval_model": resolved_model,
        "index_dir": DEFAULT_INDEX_DIR,
        "index_name": DEFAULT_SVD_INDEX_NAME,
        "preprocess": preprocess_svd_index,
        "load": load_svd_index,
        "has_artifacts": TruncatedSvdIndex.has_artifacts,
        "load_kwargs": {
            "load_articles": False,
        },
    }


def _artifacts_available(config):
    checker = config.get("has_artifacts")
    if checker is None:
        return True
    return bool(
        checker(
            index_dir=config["index_dir"],
            index_name=config["index_name"],
        )
    )


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
    doc_ids_to_lookup = [
        article
        for article, _score in ranked_articles
        if isinstance(article, str) and article.strip()
    ]
    article_map = {}
    if doc_ids_to_lookup:
        log_runtime_event(
            "search_matches.db_lookup_start",
            doc_id_count=len(doc_ids_to_lookup),
        )
        rows = GuardianArticle.query.filter(GuardianArticle.id.in_(doc_ids_to_lookup)).all()
        article_map = {row.id: row for row in rows}
        log_runtime_event(
            "search_matches.db_lookup_done",
            fetched_count=len(article_map),
        )

    matches = []
    for article, score in ranked_articles:
        resolved_article = article_map.get(article) if isinstance(article, str) else article
        if resolved_article is None:
            resolved_article = {"id": article}
        matches.append(serialize_article(resolved_article, score=score))
    return matches


def build_retrieval_processor(
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    force_rebuild=False,
    ensure_preprocessed=True,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    config = _retrieval_model_config(resolved_model)

    current_doc_count = GuardianArticle.query.count()
    artifacts_available = (
        _artifacts_available(config) if ensure_preprocessed else True
    )
    if (
        ensure_preprocessed
        and _vector_processors.get(resolved_model) is not None
        and not artifacts_available
    ):
        log_runtime_event(
            "retrieval_processor.cache_invalidated",
            retrieval_model=resolved_model,
            reason="missing_artifacts",
        )
        _vector_processors.pop(resolved_model, None)
        _vector_processor_doc_counts.pop(resolved_model, None)
    cache_ok = (
        not force_rebuild
        and _vector_processors.get(resolved_model) is not None
        and _vector_processor_doc_counts.get(resolved_model) == current_doc_count
        and artifacts_available
    )
    if cache_ok:
        log_runtime_event(
            "retrieval_processor.cache_hit",
            retrieval_model=resolved_model,
            doc_count=current_doc_count,
        )
        return _vector_processors[resolved_model]

    with _vector_index_lock:
        current_doc_count = GuardianArticle.query.count()
        artifacts_available = (
            _artifacts_available(config) if ensure_preprocessed else True
        )
        if (
            ensure_preprocessed
            and _vector_processors.get(resolved_model) is not None
            and not artifacts_available
        ):
            log_runtime_event(
                "retrieval_processor.cache_invalidated_after_lock",
                retrieval_model=resolved_model,
                reason="missing_artifacts",
            )
            _vector_processors.pop(resolved_model, None)
            _vector_processor_doc_counts.pop(resolved_model, None)
        cache_ok = (
            not force_rebuild
            and _vector_processors.get(resolved_model) is not None
            and _vector_processor_doc_counts.get(resolved_model) == current_doc_count
            and artifacts_available
        )
        if cache_ok:
            log_runtime_event(
                "retrieval_processor.cache_hit_after_lock",
                retrieval_model=resolved_model,
                doc_count=current_doc_count,
            )
            return _vector_processors[resolved_model]

        log_runtime_event(
            "retrieval_processor.build_start",
            retrieval_model=resolved_model,
            index_name=config["index_name"],
            doc_count=current_doc_count,
            force_rebuild=bool(force_rebuild),
        )
        if ensure_preprocessed:
            config["preprocess"](
                db_path=_resolve_db_path(),
                index_dir=config["index_dir"],
                index_name=config["index_name"],
                force_rebuild=force_rebuild,
            )
        log_runtime_event(
            "retrieval_processor.load_start",
            retrieval_model=resolved_model,
            index_name=config["index_name"],
        )
        try:
            vector_index, _meta = config["load"](
                index_dir=config["index_dir"],
                index_name=config["index_name"],
                **config["load_kwargs"],
            )
        except FileNotFoundError:
            if not ensure_preprocessed or force_rebuild:
                raise
            log_runtime_event(
                "retrieval_processor.load_missing_artifacts_rebuild",
                retrieval_model=resolved_model,
                index_name=config["index_name"],
            )
            config["preprocess"](
                db_path=_resolve_db_path(),
                index_dir=config["index_dir"],
                index_name=config["index_name"],
                force_rebuild=True,
            )
            vector_index, _meta = config["load"](
                index_dir=config["index_dir"],
                index_name=config["index_name"],
                **config["load_kwargs"],
            )

        _vector_processors[resolved_model] = vector_index
        _vector_processor_doc_counts[resolved_model] = current_doc_count
        log_runtime_event(
            "retrieval_processor.load_done",
            retrieval_model=resolved_model,
            doc_count=current_doc_count,
            n_docs=getattr(vector_index, "n_docs", None),
            n_terms=getattr(vector_index, "n_terms", None),
        )
        return _vector_processors[resolved_model]


def build_vector_processor(force_rebuild=False, ensure_preprocessed=True):
    return build_retrieval_processor(
        retrieval_model=DEFAULT_RETRIEVAL_MODEL,
        force_rebuild=force_rebuild,
        ensure_preprocessed=ensure_preprocessed,
    )
