from pathlib import Path
import re
from threading import Lock

from backend.db.models import GuardianArticle
from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.sentiment import vader_article_sentiment
from flask import current_app, has_app_context


DEFAULT_RETRIEVAL_MODEL = "svd"
SUPPORTED_RETRIEVAL_MODELS = ("tfidf", "svd", "minilm")
DEFAULT_SVD_EXPLAINABILITY_TOP_N = 15
DEFAULT_SVD_CHART_TOP_N = 10
DEFAULT_SVD_POLE_TOP_N = 5
DEFAULT_SVD_DIMENSION_LABEL_TOP_N = 5
WORD_COUNT_PATTERN = re.compile(r"\b[\w'-]+\b")

_RETRIEVAL_MODEL_ALIASES = {
    "tfidf": "tfidf",
    "svd": "svd",
    "truncated_svd": "svd",
    "truncated-svd": "svd",
    "lsa": "svd",
    "minilm": "minilm",
    "mini_lm": "minilm",
    "mini-lm": "minilm",
    "enhanced_semantic": "minilm",
    "enhanced-semantic": "minilm",
    "dense": "minilm",
}

_vector_processors = {}
_vector_processor_doc_counts = {}
_vector_index_lock = Lock()
_svd_dimension_labels_by_index = None


def _resolve_db_path():
    if has_app_context():
        return Path(current_app.instance_path) / "data.db"
    return Path(__file__).resolve().parent.parent.parent / "instance" / "data.db"


def _get_value(article, key, default=None):
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def _cached_svd_dimension_labels_by_index():
    global _svd_dimension_labels_by_index
    if _svd_dimension_labels_by_index is None:
        try:
            from backend.text_processing.svd_dimension_labels import (
                cached_svd_dimension_label_map,
            )

            _svd_dimension_labels_by_index = cached_svd_dimension_label_map()
        except Exception:
            log_runtime_event("svd_dimension_labels.load_failed")
            _svd_dimension_labels_by_index = {}
    return _svd_dimension_labels_by_index


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

    if resolved_model == "minilm":
        from backend.text_processing.minilm_processor import (
            DEFAULT_MINILM_ARTICLE_INDEX_NAME,
            DEFAULT_MINILM_CHUNK_INDEX_NAME,
            MiniLmEmbeddingIndex,
            load_minilm_article_index,
            preprocess_minilm_indexes,
        )
        from backend.text_processing.indexing.corpus import DEFAULT_INDEX_DIR

        def preprocess_minilm_article_index(**kwargs):
            resolved_kwargs = dict(kwargs)
            resolved_kwargs.pop("index_name", None)
            return preprocess_minilm_indexes(
                article_index_name=DEFAULT_MINILM_ARTICLE_INDEX_NAME,
                chunk_index_name=DEFAULT_MINILM_CHUNK_INDEX_NAME,
                **resolved_kwargs,
            )

        return {
            "retrieval_model": resolved_model,
            "index_dir": DEFAULT_INDEX_DIR,
            "index_name": DEFAULT_MINILM_ARTICLE_INDEX_NAME,
            "preprocess": preprocess_minilm_article_index,
            "load": load_minilm_article_index,
            "has_artifacts": MiniLmEmbeddingIndex.has_artifacts,
            "load_kwargs": {
                "load_articles": False,
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

    character_count = None
    for count_key in (
        "character_count",
        "body_character_count",
        "article_character_count",
    ):
        character_count = _get_value(article, count_key)
        if character_count is not None:
            break
    if character_count is None:
        body_text = _get_value(article, "body_text")
        character_count = len(str(body_text)) if body_text is not None else None
    try:
        character_count = int(character_count) if character_count is not None else None
    except (TypeError, ValueError):
        character_count = None

    word_count = None
    for count_key in (
        "word_count",
        "body_word_count",
        "article_word_count",
    ):
        word_count = _get_value(article, count_key)
        if word_count is not None:
            break
    if word_count is None:
        body_text = _get_value(article, "body_text")
        word_count = (
            len(WORD_COUNT_PATTERN.findall(str(body_text)))
            if body_text is not None
            else None
        )
    try:
        word_count = int(word_count) if word_count is not None else None
    except (TypeError, ValueError):
        word_count = None

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
        "character_count": character_count,
        "word_count": word_count,
    }
    sentiment = vader_article_sentiment(
        article_text=_get_value(article, "body_text"),
        title=_get_value(article, "title"),
        summary=_get_value(article, "summary"),
    )
    if sentiment is not None:
        payload["vader_sentiment"] = sentiment
    if score is not None:
        payload["score"] = float(score)
    return payload


def _article_doc_id(article):
    doc_id = _get_value(article, "id")
    if doc_id is None:
        return None
    normalized = str(doc_id).strip()
    return normalized or None


def _get_svd_dimension_summary(
    processor,
    dimension,
    dimension_cache,
    top_n_terms=DEFAULT_SVD_DIMENSION_LABEL_TOP_N,
):
    dim = int(dimension)
    summary = dimension_cache.get(dim)
    if summary is None:
        summary = processor.dimension_summary_record(
            dim,
            top_n=top_n_terms,
            format_terms=False,
        )
        dimension_cache[dim] = summary
    return summary


def _svd_dimension_entry(dimension, raw_value, label_terms):
    dim = int(dimension)
    value = float(raw_value)
    pole = "positive" if value >= 0 else "negative"
    resolved_label_terms = [str(term) for term in label_terms if str(term).strip()]
    entry = {
        "dimension_index": dim,
        "dimension_label": dim + 1,
        "value": value,
        "magnitude": abs(value),
        "pole": pole,
        "label_terms": resolved_label_terms,
        "label_text": ", ".join(resolved_label_terms),
    }
    display_label = _cached_svd_dimension_labels_by_index().get(dim)
    if display_label:
        entry["display_label"] = display_label
    return entry


def _svd_chart_dimension_payload(
    processor,
    doc_id,
    dimension_cache,
    top_n_dimensions=DEFAULT_SVD_CHART_TOP_N,
    top_n_terms=DEFAULT_SVD_DIMENSION_LABEL_TOP_N,
):
    if processor is None or not doc_id:
        return []
    if not hasattr(processor, "get_doc_vector"):
        return []

    try:
        doc_vector = processor.get_doc_vector(doc_id, normalize=True)
    except Exception:
        return []

    total_dimensions = min(
        max(0, int(top_n_dimensions)),
        int(getattr(processor, "n_components", 0)),
    )
    if total_dimensions <= 0:
        return []

    payload = []
    for dim in range(total_dimensions):
        summary = _get_svd_dimension_summary(
            processor=processor,
            dimension=dim,
            dimension_cache=dimension_cache,
            top_n_terms=top_n_terms,
        )
        label_terms = [
            str(term)
            for term, _weight in list(summary["absolute_terms"])[:top_n_terms]
        ]
        payload.append(
            _svd_dimension_entry(
                dimension=dim,
                raw_value=doc_vector[dim],
                label_terms=label_terms,
            )
        )

    return payload


def _svd_dimension_payload(
    processor,
    doc_id,
    dimension_cache,
    top_n_dimensions=DEFAULT_SVD_EXPLAINABILITY_TOP_N,
    top_n_terms=DEFAULT_SVD_DIMENSION_LABEL_TOP_N,
):
    if processor is None or not doc_id:
        return []
    if not hasattr(processor, "top_dimensions_for_doc"):
        return []

    try:
        top_dimensions = processor.top_dimensions_for_doc(
            doc_id,
            top_n=top_n_dimensions,
            normalize=True,
        )
    except Exception:
        return []

    payload = []
    for dimension, raw_value in top_dimensions:
        dim = int(dimension)
        summary = _get_svd_dimension_summary(
            processor=processor,
            dimension=dim,
            dimension_cache=dimension_cache,
            top_n_terms=top_n_terms,
        )
        pole = "positive" if float(raw_value) >= 0 else "negative"
        term_weights = summary[f"{pole}_terms"]
        label_terms = [str(term) for term, _weight in list(term_weights)[:top_n_terms]]
        payload.append(_svd_dimension_entry(dim, raw_value, label_terms))

    return payload


def _svd_pole_dimension_payload(
    processor,
    doc_id,
    dimension_cache,
    pole,
    top_n_dimensions=DEFAULT_SVD_POLE_TOP_N,
    top_n_terms=DEFAULT_SVD_DIMENSION_LABEL_TOP_N,
):
    if processor is None or not doc_id:
        return []
    if not hasattr(processor, "get_doc_vector"):
        return []

    try:
        doc_vector = processor.get_doc_vector(doc_id, normalize=True)
    except Exception:
        return []

    resolved_top_n = max(0, int(top_n_dimensions))
    if resolved_top_n <= 0:
        return []

    indexed_values = [
        (idx, float(raw_value))
        for idx, raw_value in enumerate(doc_vector)
    ]
    if pole == "positive":
        ranked_dimensions = sorted(
            ((idx, value) for idx, value in indexed_values if value > 0),
            key=lambda item: item[1],
            reverse=True,
        )
    elif pole == "negative":
        ranked_dimensions = sorted(
            ((idx, value) for idx, value in indexed_values if value < 0),
            key=lambda item: item[1],
        )
    else:
        raise ValueError("pole must be 'positive' or 'negative'.")

    payload = []
    for dim, raw_value in ranked_dimensions[:resolved_top_n]:
        summary = _get_svd_dimension_summary(
            processor=processor,
            dimension=dim,
            dimension_cache=dimension_cache,
            top_n_terms=top_n_terms,
        )
        term_weights = summary[f"{pole}_terms"]
        label_terms = [str(term) for term, _weight in list(term_weights)[:top_n_terms]]
        payload.append(_svd_dimension_entry(dim, raw_value, label_terms))

    return payload


def query_svd_dimensions(
    query,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    processor=None,
    top_n_dimensions=DEFAULT_SVD_EXPLAINABILITY_TOP_N,
    top_n_terms=DEFAULT_SVD_DIMENSION_LABEL_TOP_N,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    if resolved_model != "svd":
        return []

    resolved_query = str(query or "").strip()
    if not resolved_query:
        return []

    resolved_processor = processor
    if resolved_processor is None:
        resolved_processor = build_retrieval_processor(
            retrieval_model=resolved_model,
            force_rebuild=False,
            ensure_preprocessed=True,
        )
    if resolved_processor is None or not hasattr(resolved_processor, "top_dimensions_for_query"):
        return []

    try:
        top_dimensions = resolved_processor.top_dimensions_for_query(
            resolved_query,
            top_n=top_n_dimensions,
            normalize=True,
        )
    except Exception:
        return []

    dimension_cache = {}
    payload = []
    for dimension, raw_value in top_dimensions:
        dim = int(dimension)
        summary = _get_svd_dimension_summary(
            processor=resolved_processor,
            dimension=dim,
            dimension_cache=dimension_cache,
            top_n_terms=top_n_terms,
        )
        pole = "positive" if float(raw_value) >= 0 else "negative"
        term_weights = summary[f"{pole}_terms"]
        label_terms = [str(term) for term, _weight in list(term_weights)[:top_n_terms]]
        payload.append(_svd_dimension_entry(dim, raw_value, label_terms))

    return payload


def query_svd_corpus_chart_dimensions(
    query,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    processor=None,
    top_n_dimensions=DEFAULT_SVD_CHART_TOP_N,
    top_n_terms=DEFAULT_SVD_DIMENSION_LABEL_TOP_N,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    if resolved_model != "svd":
        return []

    resolved_query = str(query or "").strip()
    if not resolved_query:
        return []

    resolved_processor = processor
    if resolved_processor is None:
        resolved_processor = build_retrieval_processor(
            retrieval_model=resolved_model,
            force_rebuild=False,
            ensure_preprocessed=True,
        )
    if resolved_processor is None or not hasattr(resolved_processor, "project_query"):
        return []

    try:
        query_vector = resolved_processor.project_query(
            resolved_query,
            normalize=True,
        )
    except Exception:
        return []
    if query_vector is None:
        return []

    total_dimensions = min(
        max(0, int(top_n_dimensions)),
        int(getattr(resolved_processor, "n_components", 0)),
    )
    if total_dimensions <= 0:
        return []

    dimension_cache = {}
    payload = []
    for dim in range(total_dimensions):
        summary = _get_svd_dimension_summary(
            processor=resolved_processor,
            dimension=dim,
            dimension_cache=dimension_cache,
            top_n_terms=top_n_terms,
        )
        label_terms = [
            str(term)
            for term, _weight in list(summary["absolute_terms"])[:top_n_terms]
        ]
        payload.append(
            _svd_dimension_entry(
                dimension=dim,
                raw_value=query_vector[dim],
                label_terms=label_terms,
            )
        )

    return payload


def _svd_query_chart_dimension_payload(
    processor,
    doc_id,
    query_dimensions,
):
    if processor is None or not doc_id:
        return []
    if not query_dimensions or not hasattr(processor, "get_doc_vector"):
        return []

    try:
        doc_vector = processor.get_doc_vector(doc_id, normalize=True)
    except Exception:
        return []

    payload = []
    for query_dimension in list(query_dimensions)[:DEFAULT_SVD_CHART_TOP_N]:
        try:
            dim = int(query_dimension.get("dimension_index"))
        except (AttributeError, TypeError, ValueError):
            continue

        label_terms = [
            str(term)
            for term in list(query_dimension.get("label_terms") or [])
            if str(term).strip()
        ]
        entry = _svd_dimension_entry(
            dimension=dim,
            raw_value=doc_vector[dim],
            label_terms=label_terms,
        )
        try:
            entry["dimension_label"] = int(
                query_dimension.get("dimension_label", dim + 1)
            )
        except (TypeError, ValueError):
            entry["dimension_label"] = dim + 1
        payload.append(entry)

    return payload


def _svd_query_dimension_payload_for_axes(query_vector, axis_dimensions):
    if query_vector is None or not axis_dimensions:
        return []

    payload = []
    vector_length = len(query_vector)
    for axis_dimension in axis_dimensions:
        try:
            dim = int(axis_dimension.get("dimension_index"))
        except (AttributeError, TypeError, ValueError):
            continue
        if dim < 0 or dim >= vector_length:
            continue

        label_terms = [
            str(term)
            for term in list(axis_dimension.get("label_terms") or [])
            if str(term).strip()
        ]
        entry = _svd_dimension_entry(
            dimension=dim,
            raw_value=query_vector[dim],
            label_terms=label_terms,
        )
        try:
            entry["dimension_label"] = int(
                axis_dimension.get("dimension_label", dim + 1)
            )
        except (TypeError, ValueError):
            entry["dimension_label"] = dim + 1
        payload.append(entry)

    return payload


def attach_query_svd_chart_dimensions(
    matches,
    query_dimensions,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    processor=None,
    query=None,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    if resolved_model != "svd":
        return matches
    if not matches or not query_dimensions:
        return matches

    resolved_processor = processor
    if resolved_processor is None:
        resolved_processor = build_retrieval_processor(
            retrieval_model=resolved_model,
            force_rebuild=False,
            ensure_preprocessed=True,
        )
    if resolved_processor is None:
        return matches

    query_vector = None
    resolved_query = str(query or "").strip()
    if resolved_query and hasattr(resolved_processor, "project_query"):
        try:
            query_vector = resolved_processor.project_query(
                resolved_query,
                normalize=True,
            )
        except Exception:
            query_vector = None

    for match in matches:
        if not isinstance(match, dict):
            continue
        doc_id = _article_doc_id(match)
        query_chart_dimensions = _svd_query_chart_dimension_payload(
            processor=resolved_processor,
            doc_id=doc_id,
            query_dimensions=query_dimensions,
        )
        if query_chart_dimensions:
            match["svd_query_chart_dimensions"] = query_chart_dimensions
        article_dimensions = match.get("svd_dimensions")
        if isinstance(article_dimensions, list) and query_vector is not None:
            article_query_dimensions = _svd_query_dimension_payload_for_axes(
                query_vector=query_vector,
                axis_dimensions=article_dimensions,
            )
            if article_query_dimensions:
                match["svd_article_query_dimensions"] = article_query_dimensions

    return matches


def build_matches(
    ranked_articles,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    processor=None,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
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
    svd_dimension_cache = {} if resolved_model == "svd" else None
    for article, score in ranked_articles:
        resolved_article = article_map.get(article) if isinstance(article, str) else article
        if resolved_article is None:
            resolved_article = {"id": article}

        payload = serialize_article(resolved_article, score=score)
        if resolved_model == "svd":
            doc_id = article if isinstance(article, str) else _article_doc_id(resolved_article)
            svd_chart_dimensions = _svd_chart_dimension_payload(
                processor=processor,
                doc_id=doc_id,
                dimension_cache=svd_dimension_cache,
            )
            svd_dimensions = _svd_dimension_payload(
                processor=processor,
                doc_id=doc_id,
                dimension_cache=svd_dimension_cache,
            )
            svd_positive_dimensions = _svd_pole_dimension_payload(
                processor=processor,
                doc_id=doc_id,
                dimension_cache=svd_dimension_cache,
                pole="positive",
            )
            svd_negative_dimensions = _svd_pole_dimension_payload(
                processor=processor,
                doc_id=doc_id,
                dimension_cache=svd_dimension_cache,
                pole="negative",
            )
            if svd_chart_dimensions:
                payload["svd_chart_dimensions"] = svd_chart_dimensions
            if svd_dimensions:
                payload["svd_dimensions"] = svd_dimensions
            if svd_positive_dimensions:
                payload["svd_positive_dimensions"] = svd_positive_dimensions
            if svd_negative_dimensions:
                payload["svd_negative_dimensions"] = svd_negative_dimensions

        matches.append(payload)
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


def unload_retrieval_processors(keep_models=None):
    keep = {
        normalize_retrieval_model(model)
        for model in list(keep_models or [])
        if str(model or "").strip()
    }
    unloaded = []
    with _vector_index_lock:
        for model in list(_vector_processors.keys()):
            if model in keep:
                continue
            _vector_processors.pop(model, None)
            _vector_processor_doc_counts.pop(model, None)
            unloaded.append(model)

    if unloaded:
        log_runtime_event(
            "retrieval_processor.cache_unloaded",
            unloaded_models=unloaded,
            kept_models=sorted(keep),
        )
    return unloaded
