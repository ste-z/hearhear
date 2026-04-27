import numpy as np
from scipy import sparse

from backend.db.models import GuardianArticle
from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.indexing.dense_search import top_positive_dot_candidates
from backend.text_processing.text_normalization import normalize_text_for_vectorization


ROCCHIO_ALPHA = 1.0
ROCCHIO_GAMMA = 0.5


def normalize_article_id_list(value):
    if value is None:
        return []

    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]

    normalized = []
    seen = set()
    for raw_value in raw_values:
        article_id = str(raw_value or "").strip()
        if not article_id or article_id in seen:
            continue
        normalized.append(article_id)
        seen.add(article_id)
    return normalized


def _feedback_article_text(article):
    keywords = article.keywords if isinstance(article.keywords, list) else []
    return " ".join(
        part
        for part in [
            article.title or "",
            article.summary or "",
            " ".join(str(keyword) for keyword in keywords),
            article.body_text or "",
        ]
        if str(part or "").strip()
    )


def _feedback_article_texts(article_ids):
    normalized_ids = normalize_article_id_list(article_ids)
    if not normalized_ids:
        return []

    rows = GuardianArticle.query.filter(GuardianArticle.id.in_(normalized_ids)).all()
    row_by_id = {row.id: row for row in rows}

    texts = []
    for article_id in normalized_ids:
        article = row_by_id.get(article_id)
        if article is None:
            continue
        text = _feedback_article_text(article)
        if text.strip():
            texts.append(text)
    return texts


def _normalize_sparse_row(row):
    row = row.tocsr()
    magnitude = float(np.sqrt(row.multiply(row).sum()))
    if magnitude <= 0:
        return row
    normalized = row * (1.0 / magnitude)
    normalized.eliminate_zeros()
    return normalized.tocsr()


def _build_rocchio_tfidf_query_vector(query, processor, irrelevant_article_ids):
    normalized_query = normalize_text_for_vectorization(query)
    if not normalized_query:
        return None

    query_vector = processor.vectorizer.transform([normalized_query]).tocsr()
    if int(getattr(query_vector, "nnz", 0)) <= 0:
        return None

    feedback_texts = [
        normalize_text_for_vectorization(text)
        for text in _feedback_article_texts(irrelevant_article_ids)
    ]
    feedback_texts = [text for text in feedback_texts if text]
    if not feedback_texts:
        return query_vector

    feedback_matrix = processor.vectorizer.transform(feedback_texts).tocsr()
    if int(getattr(feedback_matrix, "nnz", 0)) <= 0:
        return query_vector

    nonrelevant_centroid = sparse.csr_matrix(
        feedback_matrix.sum(axis=0) / max(1, feedback_matrix.shape[0])
    )
    adjusted_vector = (
        (query_vector * ROCCHIO_ALPHA)
        - (nonrelevant_centroid * ROCCHIO_GAMMA)
    )
    adjusted_vector = adjusted_vector.tocsr()
    adjusted_vector.eliminate_zeros()
    if int(getattr(adjusted_vector, "nnz", 0)) <= 0:
        return query_vector
    return _normalize_sparse_row(adjusted_vector)


def _search_tfidf_matrix_by_query_vector(processor, query_vector, top_n):
    score_matrix = query_vector @ processor.term_doc_matrix.T
    score_coo = score_matrix.tocoo()
    if int(score_coo.nnz) <= 0:
        return []

    positive_positions = np.flatnonzero(np.asarray(score_coo.data) > 0)
    if positive_positions.size == 0:
        return []

    score_values = np.asarray(score_coo.data[positive_positions], dtype=np.float32)
    doc_indices = np.asarray(score_coo.col[positive_positions], dtype=np.intp)
    resolved_top_n = min(max(1, int(top_n)), int(score_values.shape[0]))
    if score_values.shape[0] > resolved_top_n:
        top_positions = np.argpartition(score_values, -resolved_top_n)[-resolved_top_n:]
        sorted_positions = top_positions[np.argsort(score_values[top_positions])[::-1]]
    else:
        sorted_positions = np.argsort(score_values)[::-1]

    return [
        (processor.doc_ids[int(doc_indices[pos])], float(score_values[pos]))
        for pos in sorted_positions
    ]


def _search_tfidf_postings_by_query_vector(processor, query_vector, top_n):
    query_vector = query_vector.tocsr()
    query_term_indices = np.asarray(query_vector.indices, dtype=np.int32)
    query_term_weights = np.asarray(query_vector.data, dtype=np.float32)
    if query_term_indices.size == 0:
        return []

    scores = np.zeros(processor.n_docs, dtype=np.float32)
    for term_idx, query_weight in zip(query_term_indices, query_term_weights):
        start = int(processor.postings_indptr[int(term_idx)])
        end = int(processor.postings_indptr[int(term_idx) + 1])
        if end <= start:
            continue

        doc_slice = np.asarray(
            processor.postings_doc_indices[start:end],
            dtype=np.intp,
        )
        weight_slice = np.asarray(
            processor.postings_data[start:end],
            dtype=np.float32,
        )
        scores[doc_slice] += np.float32(query_weight) * weight_slice

    candidate_doc_indices = np.flatnonzero(scores > 0)
    if candidate_doc_indices.size == 0:
        return []

    candidate_scores = scores[candidate_doc_indices]
    resolved_top_n = min(max(1, int(top_n)), int(candidate_doc_indices.size))
    if candidate_scores.size > resolved_top_n:
        top_positions = np.argpartition(candidate_scores, -resolved_top_n)[-resolved_top_n:]
        sorted_positions = top_positions[np.argsort(candidate_scores[top_positions])[::-1]]
    else:
        sorted_positions = np.argsort(candidate_scores)[::-1]

    return [
        (processor.doc_ids[int(candidate_doc_indices[pos])], float(candidate_scores[pos]))
        for pos in sorted_positions
    ]


def _search_tfidf_by_query_vector(processor, query_vector, top_n):
    if hasattr(processor, "term_doc_matrix"):
        return _search_tfidf_matrix_by_query_vector(processor, query_vector, top_n)
    if all(
        hasattr(processor, attr)
        for attr in ("postings_data", "postings_doc_indices", "postings_indptr")
    ):
        return _search_tfidf_postings_by_query_vector(processor, query_vector, top_n)
    return []


def _normalize_dense_vector(vector):
    resolved = np.asarray(vector, dtype=np.float32).reshape(-1)
    magnitude = float(np.linalg.norm(resolved))
    if magnitude <= 0:
        return None
    return resolved / magnitude


def _build_rocchio_dense_query_vector(query, processor, irrelevant_article_ids):
    if not hasattr(processor, "project_query") or not hasattr(processor, "get_doc_vector"):
        return None

    query_vector = processor.project_query(query, normalize=True)
    if query_vector is None:
        return None

    feedback_vectors = []
    for article_id in normalize_article_id_list(irrelevant_article_ids):
        try:
            feedback_vectors.append(processor.get_doc_vector(article_id, normalize=True))
        except Exception:
            continue

    if not feedback_vectors:
        return np.asarray(query_vector, dtype=np.float32)

    nonrelevant_centroid = np.mean(
        np.asarray(feedback_vectors, dtype=np.float32),
        axis=0,
    )
    adjusted_vector = (
        (np.asarray(query_vector, dtype=np.float32) * ROCCHIO_ALPHA)
        - (nonrelevant_centroid * ROCCHIO_GAMMA)
    )
    normalized = _normalize_dense_vector(adjusted_vector)
    if normalized is None:
        return np.asarray(query_vector, dtype=np.float32)
    return normalized


def _search_dense_by_query_vector(processor, query_vector, top_n):
    candidate_doc_indices, candidate_scores = top_positive_dot_candidates(
        processor.normalized_doc_embeddings,
        query_vector,
        top_n=max(1, int(top_n)),
    )
    if candidate_doc_indices.size == 0:
        return []

    return [
        (processor.doc_ids[int(idx)], float(score))
        for idx, score in zip(candidate_doc_indices, candidate_scores)
    ]


def build_rocchio_processor_searcher(
    query,
    processor,
    retrieval_model,
    irrelevant_article_ids,
):
    feedback_ids = normalize_article_id_list(irrelevant_article_ids)
    if not feedback_ids:
        return lambda top_n: processor.search(query, top_n=top_n)

    if retrieval_model in {"svd", "minilm"}:
        query_vector = _build_rocchio_dense_query_vector(query, processor, feedback_ids)
        if query_vector is not None:
            log_runtime_event(
                "rocchio_search.dense_ready",
                retrieval_model=retrieval_model,
                irrelevant_count=len(feedback_ids),
            )
            return lambda top_n: _search_dense_by_query_vector(
                processor,
                query_vector,
                top_n,
            )

    query_vector = _build_rocchio_tfidf_query_vector(query, processor, feedback_ids)
    if query_vector is not None:
        log_runtime_event(
            "rocchio_search.tfidf_ready",
            irrelevant_count=len(feedback_ids),
            query_term_count=int(getattr(query_vector, "nnz", 0)),
        )
        return lambda top_n: _search_tfidf_by_query_vector(
            processor,
            query_vector,
            top_n,
        )

    log_runtime_event(
        "rocchio_search.fallback_standard",
        retrieval_model=retrieval_model,
        irrelevant_count=len(feedback_ids),
    )
    return lambda top_n: processor.search(query, top_n=top_n)
