import argparse
import gc
import json
import re
import sqlite3
from pathlib import Path
from threading import Lock

import numpy as np
import pandas as pd

from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.minilm_processor import (
    DEFAULT_MINILM_CHUNK_INDEX_NAME,
    MiniLmEmbeddingIndex,
    load_minilm_chunk_index,
    preprocess_minilm_indexes,
)
from backend.text_processing.paragraph_splitter import (
    normalize_paragraph_text,
    split_into_paragraphs,
)
from backend.text_processing.search_helpers import (
    DEFAULT_RETRIEVAL_MODEL,
    build_matches,
    normalize_retrieval_model,
)
from backend.text_processing.semantic_chunker import (
    DEFAULT_SEMANTIC_BREAK_SIMILARITY_THRESHOLD,
    semantic_chunk_rows_from_text,
)
from backend.text_processing.indexing.artifacts import (
    _artifact_exists,
    _artifact_within_size_limit,
    _materialized_artifact_path,
)
from backend.text_processing.indexing.corpus import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_DIR,
    _current_db_row_count,
    _db_has_complete_body_text,
    _db_years,
    _filter_articles_to_years,
    _load_guardian_articles_from_raw,
    _load_guardian_articles_from_sqlite,
    _normalized_years,
    _relative_db_path_for_meta,
)
from backend.text_processing.svd_processor import (
    DEFAULT_SVD_INDEX_NAME,
    DEFAULT_SVD_N_COMPONENTS,
    TruncatedSvdIndex,
    ensure_normalized_doc_embeddings_artifact,
    _load_index_meta,
    _resolved_svd_params,
    _resolved_vectorizer_params,
    load_svd_index,
    preprocess_svd_index,
)
from backend.text_processing.text_normalization import (
    TEXT_NORMALIZATION_VERSION,
    normalize_text_for_vectorization,
)


DEFAULT_CHUNK_CANDIDATE_TOP_K = 100
MAX_CHUNK_CANDIDATE_TOP_K = 500
DEFAULT_CHUNK_ARTICLE_TOP_K = 5
DEFAULT_CHUNK_AGGREGATION_TOP_K = 3
DEFAULT_CHUNK_TARGET_CHARS = 1800
DEFAULT_CHUNK_MAX_CHARS = 2500
DEFAULT_CHUNK_MIN_CHARS = 500
DEFAULT_CHUNK_PARAGRAPH_MIN_CHARS = 20
DEFAULT_CHUNK_BUILD_PROGRESS_INTERVAL = 1000
DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE = "semantic"
DEFAULT_CHUNK_SVD_INDEX_NAME = "guardian_chunk_svd_semantic"
CHUNK_INDEX_SCHEMA_VERSION = 2
CHUNK_ROW_STORE_SCHEMA_VERSION = 1
DEFAULT_CHUNK_AUTO_THRESHOLDS = {
    "tfidf": 0.12,
    "svd": 0.35,
    "minilm": 0.45,
}
DEFAULT_CHUNK_DENSE_STORAGE_DTYPE = np.float16
CHUNK_SCORE_WEIGHTS = {
    "max": 0.5,
    "top_k_mean": 0.4,
    "coverage": 0.1,
}
WORD_COUNT_PATTERN = re.compile(r"\b[\w'-]+\b")

SUPPORTED_CHUNK_RETRIEVAL_MODELS = ("tfidf", "svd", "minilm")
SUPPORTED_CHUNK_INDEX_MODES = ("paragraph", "semantic")

_chunk_indexes = {}
_chunk_index_lock = Lock()


class ChunkRowMetadata:
    __slots__ = (
        "row_indices",
        "chunk_ids",
        "article_ids",
        "chunk_indices",
        "sources",
        "years",
        "character_counts",
        "word_counts",
        "sentence_start_indices",
        "sentence_end_indices",
    )

    def __init__(
        self,
        row_indices,
        chunk_ids,
        article_ids,
        chunk_indices,
        sources,
        years,
        character_counts,
        word_counts,
        sentence_start_indices,
        sentence_end_indices,
    ):
        self.row_indices = np.asarray(row_indices, dtype=np.int32)
        self.chunk_ids = np.asarray(chunk_ids, dtype=object)
        self.article_ids = np.asarray(article_ids, dtype=object)
        self.chunk_indices = np.asarray(chunk_indices, dtype=np.int32)
        self.sources = np.asarray(sources, dtype=object)
        self.years = np.asarray(years, dtype=np.int32)
        self.character_counts = np.asarray(character_counts, dtype=np.int32)
        self.word_counts = np.asarray(word_counts, dtype=np.int32)
        self.sentence_start_indices = np.asarray(sentence_start_indices, dtype=np.int32)
        self.sentence_end_indices = np.asarray(sentence_end_indices, dtype=np.int32)

    def __len__(self):
        return int(self.row_indices.shape[0])

    def __bool__(self):
        return len(self) > 0

    def row_dict(self, index):
        idx = int(index)
        sentence_start = int(self.sentence_start_indices[idx])
        sentence_end = int(self.sentence_end_indices[idx])
        return {
            "row_index": int(self.row_indices[idx]),
            "chunk_id": str(self.chunk_ids[idx]),
            "article_id": str(self.article_ids[idx]),
            "chunk_index": int(self.chunk_indices[idx]),
            "source": self.sources[idx],
            "year": int(self.years[idx]),
            "character_count": int(self.character_counts[idx]),
            "word_count": int(self.word_counts[idx]),
            "sentence_start_index": None if sentence_start < 0 else sentence_start,
            "sentence_end_index": None if sentence_end < 0 else sentence_end,
        }


def normalize_chunk_candidate_top_k(value, default=DEFAULT_CHUNK_CANDIDATE_TOP_K):
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(25, min(MAX_CHUNK_CANDIDATE_TOP_K, resolved))


def normalize_chunk_article_top_k(value, default=DEFAULT_CHUNK_ARTICLE_TOP_K):
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(1, min(10, resolved))


def default_chunk_auto_threshold(retrieval_model=DEFAULT_RETRIEVAL_MODEL):
    resolved_model = normalize_retrieval_model(retrieval_model)
    return float(DEFAULT_CHUNK_AUTO_THRESHOLDS[resolved_model])


def normalize_chunk_index_mode(value, default=DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE):
    normalized = str(value or default).strip().lower().replace("-", "_").replace(" ", "_")
    if normalized in {
        "semantic",
        "semantic_chunk",
        "semantic_chunks",
        "semantic_chunking",
    }:
        return "semantic"
    if normalized in {
        "paragraph",
        "paragraphs",
        "paragraph_chunk",
        "paragraph_chunks",
        "paragraph_chunking",
        "chunking",
        "chunked",
        "true",
        "1",
        "yes",
        "on",
    }:
        return "paragraph"
    if normalized in {"", "none", "off", "false", "0", "no"}:
        return default
    return default


def _normalize_dense_rows(matrix, storage_dtype=DEFAULT_CHUNK_DENSE_STORAGE_DTYPE):
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32, copy=False)
    normalized = np.empty(array.shape, dtype=storage_dtype)
    np.divide(array, norms, out=normalized, casting="unsafe")
    return normalized


def _normalize_dense_vector(vector):
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return None
    return (array / norm).astype(np.float32, copy=False)


def _article_value(article, key, default=None):
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def _safe_int(value, default=0):
    try:
        if value is None:
            return int(default)
        return int(value)
    except (TypeError, ValueError):
        return int(default)


def _split_long_text(text, max_chars=DEFAULT_CHUNK_MAX_CHARS):
    normalized = normalize_paragraph_text(text)
    if not normalized:
        return []
    if not max_chars or max_chars <= 0 or len(normalized) <= max_chars:
        return [normalized]

    chunks = []
    remaining = normalized
    while len(remaining) > max_chars:
        split_at = remaining.rfind(" ", 0, max_chars)
        if split_at < int(max_chars * 0.55):
            split_at = max_chars
        chunk = remaining[:split_at].strip()
        if chunk:
            chunks.append(chunk)
        remaining = remaining[split_at:].strip()

    if remaining:
        chunks.append(remaining)
    return chunks


def _candidate_text(candidate):
    return normalize_paragraph_text(candidate.get("text") or "")


def _merge_candidates(candidates, source):
    texts = [_candidate_text(candidate) for candidate in candidates]
    text = "\n\n".join(part for part in texts if part).strip()
    row = {
        "text": text,
        "source": source,
    }
    starts = [
        candidate.get("sentence_start_index")
        for candidate in candidates
        if candidate.get("sentence_start_index") is not None
    ]
    ends = [
        candidate.get("sentence_end_index")
        for candidate in candidates
        if candidate.get("sentence_end_index") is not None
    ]
    if starts:
        row["sentence_start_index"] = min(starts)
    if ends:
        row["sentence_end_index"] = max(ends)
    return row


def _pack_body_candidates(candidates, source):
    packed = []
    current = []

    def flush_current():
        nonlocal current
        if current:
            packed.append(_merge_candidates(current, source=source))
            current = []

    for candidate in candidates:
        text = _candidate_text(candidate)
        if not text:
            continue

        if len(text) > DEFAULT_CHUNK_MAX_CHARS:
            flush_current()
            for split_text in _split_long_text(text):
                split_candidate = {
                    "text": split_text,
                    "sentence_start_index": candidate.get("sentence_start_index"),
                    "sentence_end_index": candidate.get("sentence_end_index"),
                }
                packed.append(_merge_candidates([split_candidate], source=source))
            continue

        current_text = "\n\n".join(_candidate_text(row) for row in current).strip()
        projected = f"{current_text}\n\n{text}".strip() if current_text else text
        if current and (
            len(projected) > DEFAULT_CHUNK_MAX_CHARS
            or len(current_text) >= DEFAULT_CHUNK_TARGET_CHARS
        ):
            flush_current()

        current.append({
            **candidate,
            "text": text,
        })

    flush_current()

    if len(packed) >= 2 and len(packed[-1]["text"]) < DEFAULT_CHUNK_MIN_CHARS:
        merged_tail = _merge_candidates([packed[-2], packed[-1]], source=source)
        if len(merged_tail["text"]) <= DEFAULT_CHUNK_MAX_CHARS:
            packed[-2] = merged_tail
            packed.pop()

    return packed


def _paragraph_body_sections(text):
    paragraphs = split_into_paragraphs(
        text,
        min_chars=DEFAULT_CHUNK_PARAGRAPH_MIN_CHARS,
        max_chars=None,
    )
    candidates = [{"text": paragraph} for paragraph in paragraphs]
    if not candidates:
        candidates = [{"text": split_text} for split_text in _split_long_text(text)]
    return _pack_body_candidates(candidates, source="body")


def _semantic_body_sections(article_id, text, svd_processor):
    if svd_processor is None:
        return _paragraph_body_sections(text)

    semantic_rows = semantic_chunk_rows_from_text(
        text,
        article_id=article_id,
        prefix="semantic_chunk_",
        svd_processor=svd_processor,
        similarity_threshold=DEFAULT_SEMANTIC_BREAK_SIMILARITY_THRESHOLD,
        max_chars=DEFAULT_CHUNK_MAX_CHARS,
    )
    candidates = [
        {
            "text": row.get("paragraph"),
            "sentence_start_index": row.get("sentence_start_index"),
            "sentence_end_index": row.get("sentence_end_index"),
        }
        for row in semantic_rows
    ]
    if not candidates:
        return _paragraph_body_sections(text)
    return _pack_body_candidates(candidates, source="semantic_body")


def _chunk_rows_for_article(
    article,
    chunking_mode=DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE,
    svd_processor=None,
):
    article_id = str(_article_value(article, "id") or "").strip()
    rows = []
    body_text = _article_value(article, "body_text") or ""
    resolved_chunking_mode = normalize_chunk_index_mode(chunking_mode)
    if resolved_chunking_mode == "semantic":
        body_sections = _semantic_body_sections(article_id, body_text, svd_processor)
    else:
        body_sections = _paragraph_body_sections(body_text)

    if not body_sections:
        return rows

    for section_index, section in enumerate(body_sections):
        text = normalize_paragraph_text(section.get("text"))
        if not text:
            continue
        rows.append({
            "chunk_id": f"{article_id}::section::{section_index}",
            "article_id": article_id,
            "chunk_index": len(rows),
            "text": text,
            "source": section.get("source") or resolved_chunking_mode,
            "sentence_start_index": section.get("sentence_start_index"),
            "sentence_end_index": section.get("sentence_end_index"),
        })

    return rows


def _chunk_svd_required_paths(index_dir, index_name):
    paths = TruncatedSvdIndex.artifact_paths(index_dir, index_name)
    return [
        paths["vectorizer"],
        paths["terms"],
        paths["doc_ids"],
        paths["articles"],
        paths["doc_embeddings"],
        paths["components"],
        paths["singular_values"],
        paths["explained_variance_ratio"],
        paths["meta"],
    ]


def _is_existing_chunk_svd_index_fresh(
    index_dir,
    index_name,
    db_row_count,
    expected_years=None,
    expected_chunking_mode=DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE,
    expected_n_components=DEFAULT_SVD_N_COMPONENTS,
    expected_vectorizer_params=None,
    expected_svd_params=None,
):
    paths = TruncatedSvdIndex.artifact_paths(index_dir, index_name)
    meta = _load_index_meta(paths["meta"])
    if not meta:
        return False

    try:
        stored_count = int(meta.get("db_row_count"))
    except (TypeError, ValueError):
        return False

    if stored_count != int(db_row_count):
        return False
    if meta.get("search_backend") != "svd":
        return False
    if meta.get("index_kind") != "chunk_svd":
        return False
    if int(meta.get("chunk_index_schema_version") or -1) != CHUNK_INDEX_SCHEMA_VERSION:
        return False
    if meta.get("chunking_mode") != normalize_chunk_index_mode(expected_chunking_mode):
        return False
    if meta.get("text_normalization_version") != TEXT_NORMALIZATION_VERSION:
        return False
    if meta.get("vectorizer_params") != _resolved_vectorizer_params(expected_vectorizer_params):
        return False

    expected_svd = _resolved_svd_params(
        n_components=expected_n_components,
        svd_params=expected_svd_params,
    )
    stored_svd = dict(meta.get("svd_params") or {})
    stored_svd.pop("n_components", None)
    expected_svd.pop("n_components", None)
    for key, expected_value in expected_svd.items():
        if stored_svd.get(key) != expected_value:
            return False

    try:
        stored_requested_n_components = int(meta.get("requested_n_components"))
    except (TypeError, ValueError):
        return False
    if stored_requested_n_components != int(expected_n_components):
        return False

    normalized_expected_years = _normalized_years(expected_years)
    if normalized_expected_years is not None:
        try:
            stored_source_years = _normalized_years(meta.get("source_years"))
        except (TypeError, ValueError):
            return False
        if stored_source_years != normalized_expected_years:
            return False

    required_paths = _chunk_svd_required_paths(index_dir, index_name)
    if meta.get("explained_variance_files"):
        required_paths.append(paths["explained_variance"])
    if not all(_artifact_exists(path) for path in required_paths):
        return False
    if not all(_artifact_within_size_limit(path) for path in required_paths):
        return False

    return True


def _load_chunk_source_articles(db_path, years=None):
    db_path = Path(db_path)
    normalized_years = _normalized_years(years)
    source_kind = "sqlite"
    if _db_has_complete_body_text(db_path):
        articles = _load_guardian_articles_from_sqlite(db_path)
        articles = _filter_articles_to_years(articles, years=normalized_years)
    else:
        source_years = normalized_years or _db_years(db_path)
        articles = _load_guardian_articles_from_raw(years=source_years)
        source_kind = "raw_csv"

    if articles is None or articles.empty:
        return pd.DataFrame(), source_kind, []

    source_years = normalized_years or sorted(
        {
            int(year)
            for year in pd.to_numeric(articles.get("year"), errors="coerce").dropna().tolist()
        }
    )
    return articles.sort_values("id").reset_index(drop=True), source_kind, source_years


def _chunk_dataframe_from_articles(
    articles,
    chunking_mode=DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE,
    svd_processor=None,
    progress_interval=DEFAULT_CHUNK_BUILD_PROGRESS_INTERVAL,
):
    rows = []
    article_records = articles.to_dict(orient="records")
    total_articles = int(len(article_records))
    resolved_chunking_mode = normalize_chunk_index_mode(chunking_mode)
    log_runtime_event(
        "chunk_svd_preprocess.chunk_build_start",
        article_count=total_articles,
        chunking_mode=resolved_chunking_mode,
    )
    for article_offset, article in enumerate(article_records, start=1):
        article_id = str(article.get("id") or "").strip()
        if not article_id:
            continue
        body_text = str(article.get("body_text") or "")
        article_chunks = _chunk_rows_for_article(
            article,
            chunking_mode=resolved_chunking_mode,
            svd_processor=svd_processor,
        )
        character_count = _safe_int(
            article.get("body_character_count"),
            default=len(body_text),
        )
        word_count = _safe_int(
            article.get("body_word_count"),
            default=len(WORD_COUNT_PATTERN.findall(body_text)),
        )
        for chunk in article_chunks:
            chunk_text = normalize_paragraph_text(chunk.get("text"))
            if not chunk_text:
                continue
            rows.append({
                "id": chunk["chunk_id"],
                "article_id": article_id,
                "chunk_index": int(chunk.get("chunk_index") or 0),
                "text": chunk_text,
                "source": chunk.get("source"),
                "year": _safe_int(article.get("year")),
                "character_count": character_count,
                "word_count": word_count,
                "sentence_start_index": chunk.get("sentence_start_index"),
                "sentence_end_index": chunk.get("sentence_end_index"),
            })
        if (
            progress_interval
            and (
                article_offset % int(progress_interval) == 0
                or article_offset == total_articles
            )
        ):
            log_runtime_event(
                "chunk_svd_preprocess.chunk_build_progress",
                processed_article_count=article_offset,
                article_count=total_articles,
                chunk_count=len(rows),
                chunking_mode=resolved_chunking_mode,
            )
    log_runtime_event(
        "chunk_svd_preprocess.chunk_build_done",
        article_count=total_articles,
        chunk_count=len(rows),
        chunking_mode=resolved_chunking_mode,
    )
    return pd.DataFrame(rows)


def preprocess_chunk_svd_index(
    db_path=DEFAULT_DB_PATH,
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_CHUNK_SVD_INDEX_NAME,
    force_rebuild=False,
    years=None,
    chunking_mode=DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE,
    n_components=DEFAULT_SVD_N_COMPONENTS,
    vectorizer_params=None,
    svd_params=None,
):
    db_path = Path(db_path)
    index_dir = Path(index_dir)
    resolved_chunking_mode = normalize_chunk_index_mode(chunking_mode)
    normalized_years = _normalized_years(years)
    resolved_vectorizer_params = _resolved_vectorizer_params(vectorizer_params)
    resolved_svd_params = _resolved_svd_params(
        n_components=n_components,
        svd_params=svd_params,
    )

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    db_row_count = _current_db_row_count(db_path)
    log_runtime_event(
        "chunk_svd_preprocess.start",
        db_path=str(db_path),
        index_name=index_name,
        db_row_count=db_row_count,
        force_rebuild=bool(force_rebuild),
        chunking_mode=resolved_chunking_mode,
        requested_years=normalized_years,
    )
    if not force_rebuild and _is_existing_chunk_svd_index_fresh(
        index_dir=index_dir,
        index_name=index_name,
        db_row_count=db_row_count,
        expected_years=normalized_years,
        expected_chunking_mode=resolved_chunking_mode,
        expected_n_components=int(resolved_svd_params["n_components"]),
        expected_vectorizer_params=resolved_vectorizer_params,
        expected_svd_params=resolved_svd_params,
    ):
        log_runtime_event(
            "chunk_svd_preprocess.up_to_date",
            index_name=index_name,
            db_row_count=db_row_count,
        )
        normalized_path = ensure_normalized_doc_embeddings_artifact(
            index_dir=index_dir,
            index_name=index_name,
        )
        row_store_path = ensure_chunk_row_store(
            index_dir=index_dir,
            index_name=index_name,
            force_rebuild=False,
        )
        return {
            "built": False,
            "reason": "up_to_date",
            "db_row_count": db_row_count,
            "index_dir": str(index_dir),
            "index_name": index_name,
            "row_store_path": str(row_store_path),
            "normalized_doc_embeddings_path": str(normalized_path),
        }

    semantic_processor = None
    if resolved_chunking_mode == "semantic":
        log_runtime_event(
            "chunk_svd_preprocess.boundary_svd_start",
            index_name=DEFAULT_SVD_INDEX_NAME,
            requested_years=normalized_years,
        )
        preprocess_svd_index(
            db_path=db_path,
            index_dir=index_dir,
            index_name=DEFAULT_SVD_INDEX_NAME,
            force_rebuild=False,
            years=normalized_years,
        )
        semantic_processor, semantic_meta = load_svd_index(
            index_dir=index_dir,
            index_name=DEFAULT_SVD_INDEX_NAME,
            load_articles=False,
        )
        log_runtime_event(
            "chunk_svd_preprocess.boundary_svd_ready",
            index_name=DEFAULT_SVD_INDEX_NAME,
            n_docs=_safe_int(getattr(semantic_processor, "n_docs", None)),
            n_components=_safe_int(getattr(semantic_processor, "n_components", None)),
            meta_db_row_count=semantic_meta.get("db_row_count") if isinstance(semantic_meta, dict) else None,
        )

    log_runtime_event(
        "chunk_svd_preprocess.source_articles_start",
        db_path=str(db_path),
        requested_years=normalized_years,
    )
    articles, source_kind, source_years = _load_chunk_source_articles(
        db_path=db_path,
        years=normalized_years,
    )
    if articles.empty:
        raise ValueError("No source articles found; cannot build chunk SVD index.")
    log_runtime_event(
        "chunk_svd_preprocess.source_articles_ready",
        source_kind=source_kind,
        article_count=int(len(articles)),
        source_years=source_years,
    )

    chunk_frame = _chunk_dataframe_from_articles(
        articles=articles,
        chunking_mode=resolved_chunking_mode,
        svd_processor=semantic_processor,
    )
    if chunk_frame.empty:
        raise ValueError("No article chunks found; cannot build chunk SVD index.")

    log_runtime_event(
        "chunk_svd_preprocess.chunks_ready",
        source_kind=source_kind,
        article_count=int(len(articles)),
        chunk_count=int(len(chunk_frame)),
        chunking_mode=resolved_chunking_mode,
        source_years=source_years,
    )
    log_runtime_event(
        "chunk_svd_preprocess.fit_start",
        index_name=index_name,
        chunk_count=int(len(chunk_frame)),
        requested_n_components=int(resolved_svd_params["n_components"]),
    )
    chunk_svd_index = TruncatedSvdIndex.from_articles(
        articles=chunk_frame,
        n_components=int(resolved_svd_params["n_components"]),
        vectorizer_params=resolved_vectorizer_params,
        svd_params=resolved_svd_params,
        text_column="text",
        id_column="id",
        include_text_in_articles=True,
    )
    log_runtime_event(
        "chunk_svd_preprocess.save_start",
        index_name=index_name,
        chunk_count=int(len(chunk_frame)),
        n_components=int(chunk_svd_index.n_components),
    )
    paths = chunk_svd_index.save(
        index_dir=index_dir,
        index_name=index_name,
        extra_meta={
            "index_kind": "chunk_svd",
            "chunk_index_schema_version": CHUNK_INDEX_SCHEMA_VERSION,
            "chunking_mode": resolved_chunking_mode,
            "chunk_target_chars": DEFAULT_CHUNK_TARGET_CHARS,
            "chunk_max_chars": DEFAULT_CHUNK_MAX_CHARS,
            "chunk_min_chars": DEFAULT_CHUNK_MIN_CHARS,
            "semantic_break_threshold": DEFAULT_SEMANTIC_BREAK_SIMILARITY_THRESHOLD,
            "db_row_count": int(db_row_count),
            "source_db_path": _relative_db_path_for_meta(db_path),
            "text_source": source_kind,
            "source_years": source_years,
            "source_article_count": int(len(articles)),
            "chunk_count": int(len(chunk_frame)),
            "boundary_index_name": DEFAULT_SVD_INDEX_NAME if resolved_chunking_mode == "semantic" else None,
            "text_normalization_version": TEXT_NORMALIZATION_VERSION,
            "vectorizer_params": resolved_vectorizer_params,
        },
    )
    log_runtime_event(
        "chunk_svd_preprocess.done",
        index_name=index_name,
        chunk_count=int(len(chunk_frame)),
        n_components=int(chunk_svd_index.n_components),
        requested_n_components=int(chunk_svd_index.requested_n_components),
    )

    return {
        "built": True,
        "db_row_count": db_row_count,
        "index_dir": str(index_dir),
        "index_name": index_name,
        "chunk_count": int(len(chunk_frame)),
        "n_components": int(chunk_svd_index.n_components),
        "requested_n_components": int(chunk_svd_index.requested_n_components),
        "paths": {
            key: [str(path) for path in value]
            if key.endswith("_files")
            else str(value)
            for key, value in paths.items()
        },
    }


def load_chunk_svd_index(
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_CHUNK_SVD_INDEX_NAME,
    load_articles=True,
):
    return TruncatedSvdIndex.load(
        index_dir=index_dir,
        index_name=index_name,
        load_articles=load_articles,
    )


def _chunk_row_store_path(index_dir=DEFAULT_INDEX_DIR, index_name=DEFAULT_CHUNK_SVD_INDEX_NAME):
    return Path(index_dir) / f"{index_name}_chunk_rows.sqlite"


def _is_chunk_row_store_fresh(store_path, expected_row_count=None, expected_saved_at_utc=None):
    path = Path(store_path)
    if not path.exists():
        return False

    try:
        with sqlite3.connect(path) as conn:
            schema_version = conn.execute(
                "SELECT value FROM meta WHERE key = 'schema_version'"
            ).fetchone()
            row_count = conn.execute(
                "SELECT value FROM meta WHERE key = 'row_count'"
            ).fetchone()
            saved_at = conn.execute(
                "SELECT value FROM meta WHERE key = 'index_saved_at_utc'"
            ).fetchone()
    except sqlite3.Error:
        return False

    try:
        if int(schema_version[0]) != CHUNK_ROW_STORE_SCHEMA_VERSION:
            return False
        if expected_row_count is not None and int(row_count[0]) != int(expected_row_count):
            return False
        if expected_saved_at_utc and str(saved_at[0]) != str(expected_saved_at_utc):
            return False
    except (TypeError, ValueError):
        return False

    return True


def _chunk_rows_from_articles_frame(articles, include_text=True):
    if articles is None or not isinstance(articles, pd.DataFrame):
        raise RuntimeError("Chunk retrieval index is missing its precomputed chunk metadata.")

    columns = {column_name: idx for idx, column_name in enumerate(articles.columns)}

    def row_value(row, column_name, default=None):
        idx = columns.get(column_name)
        if idx is None:
            return default
        return row[idx]

    def string_value(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    rows = []
    for row in articles.itertuples(index=False, name=None):
        chunk_id = string_value(row_value(row, "id"))
        if not chunk_id:
            continue
        payload = {
            "chunk_id": chunk_id,
            "article_id": string_value(row_value(row, "article_id")),
            "chunk_index": _safe_int(row_value(row, "chunk_index")),
            "source": row_value(row, "source"),
            "year": _safe_int(row_value(row, "year")),
            "character_count": _safe_int(row_value(row, "character_count")),
            "word_count": _safe_int(row_value(row, "word_count")),
            "sentence_start_index": row_value(row, "sentence_start_index"),
            "sentence_end_index": row_value(row, "sentence_end_index"),
        }
        if include_text:
            payload["text"] = normalize_paragraph_text(row_value(row, "text"))
        rows.append(payload)
    return rows


def _chunk_rows_from_svd_processor(processor, include_text=True):
    return _chunk_rows_from_articles_frame(
        getattr(processor, "articles", None),
        include_text=include_text,
    )


def _iter_chunk_row_store_records(articles):
    if articles is None or not isinstance(articles, pd.DataFrame):
        raise RuntimeError("Chunk retrieval index is missing its precomputed chunk metadata.")

    columns = {column_name: idx for idx, column_name in enumerate(articles.columns)}

    def row_value(row, column_name, default=None):
        idx = columns.get(column_name)
        if idx is None:
            return default
        return row[idx]

    def string_value(value):
        if value is None:
            return ""
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value).strip()

    for row_index, row in enumerate(articles.itertuples(index=False, name=None)):
        chunk_id = string_value(row_value(row, "id"))
        if not chunk_id:
            continue
        yield (
            row_index,
            chunk_id,
            string_value(row_value(row, "article_id")),
            _safe_int(row_value(row, "chunk_index")),
            normalize_paragraph_text(row_value(row, "text")),
            string_value(row_value(row, "source")),
            _safe_int(row_value(row, "year")),
            _safe_int(row_value(row, "character_count")),
            _safe_int(row_value(row, "word_count")),
            _safe_int(row_value(row, "sentence_start_index"), default=-1),
            _safe_int(row_value(row, "sentence_end_index"), default=-1),
        )


def _build_chunk_row_store(index_dir, index_name, meta):
    paths = TruncatedSvdIndex.artifact_paths(index_dir, index_name)
    if not _artifact_exists(paths["articles"]):
        raise FileNotFoundError(
            f"Missing chunk metadata artifact for row store: {paths['articles']}"
        )

    store_path = _chunk_row_store_path(index_dir=index_dir, index_name=index_name)
    temp_path = store_path.with_suffix(f"{store_path.suffix}.tmp")
    if temp_path.exists():
        temp_path.unlink()

    log_runtime_event(
        "chunk_row_store.build_start",
        index_name=index_name,
        store_path=str(store_path),
    )
    with _materialized_artifact_path(paths["articles"]) as articles_path:
        articles = pd.read_pickle(articles_path)

    try:
        with sqlite3.connect(temp_path) as conn:
            conn.execute("PRAGMA journal_mode = OFF")
            conn.execute("PRAGMA synchronous = OFF")
            conn.execute("PRAGMA temp_store = MEMORY")
            conn.execute(
                "CREATE TABLE chunk_rows ("
                "row_index INTEGER PRIMARY KEY, "
                "chunk_id TEXT NOT NULL, "
                "article_id TEXT NOT NULL, "
                "chunk_index INTEGER NOT NULL, "
                "text TEXT NOT NULL, "
                "source TEXT, "
                "year INTEGER NOT NULL, "
                "character_count INTEGER NOT NULL, "
                "word_count INTEGER NOT NULL, "
                "sentence_start_index INTEGER NOT NULL, "
                "sentence_end_index INTEGER NOT NULL"
                ")"
            )
            conn.execute("CREATE INDEX ix_chunk_rows_article_id ON chunk_rows (article_id)")
            conn.execute("CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            conn.executemany(
                "INSERT INTO chunk_rows VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                _iter_chunk_row_store_records(articles),
            )
            row_count = conn.execute("SELECT COUNT(*) FROM chunk_rows").fetchone()[0]
            conn.executemany(
                "INSERT INTO meta (key, value) VALUES (?, ?)",
                [
                    ("schema_version", str(CHUNK_ROW_STORE_SCHEMA_VERSION)),
                    ("index_name", str(index_name)),
                    ("row_count", str(row_count)),
                    ("index_saved_at_utc", str(meta.get("saved_at_utc") or "")),
                ],
            )
            conn.commit()
    finally:
        del articles
        gc.collect()

    temp_path.replace(store_path)
    log_runtime_event(
        "chunk_row_store.build_done",
        index_name=index_name,
        store_path=str(store_path),
        row_count=row_count,
    )
    return store_path


def ensure_chunk_row_store(index_dir=DEFAULT_INDEX_DIR, index_name=DEFAULT_CHUNK_SVD_INDEX_NAME, force_rebuild=False):
    paths = TruncatedSvdIndex.artifact_paths(index_dir, index_name)
    meta = _load_index_meta(paths["meta"])
    if not meta:
        raise FileNotFoundError(f"Missing chunk SVD metadata artifact: {paths['meta']}")

    expected_row_count = meta.get("n_docs") or meta.get("chunk_count")
    store_path = _chunk_row_store_path(index_dir=index_dir, index_name=index_name)
    if (
        not force_rebuild
        and _is_chunk_row_store_fresh(
            store_path,
            expected_row_count=expected_row_count,
            expected_saved_at_utc=meta.get("saved_at_utc"),
        )
    ):
        log_runtime_event(
            "chunk_row_store.up_to_date",
            index_name=index_name,
            store_path=str(store_path),
            row_count=_safe_int(expected_row_count),
        )
        return store_path

    return _build_chunk_row_store(index_dir=index_dir, index_name=index_name, meta=meta)


def _chunk_row_metadata_from_store(store_path):
    path = Path(store_path)
    if not path.exists():
        raise FileNotFoundError(f"Chunk row store not found: {path}")

    with sqlite3.connect(path) as conn:
        conn.row_factory = sqlite3.Row
        row_count = int(conn.execute("SELECT COUNT(*) FROM chunk_rows").fetchone()[0])
        row_indices = np.empty(row_count, dtype=np.int32)
        chunk_ids = np.empty(row_count, dtype=object)
        article_ids = np.empty(row_count, dtype=object)
        chunk_indices = np.empty(row_count, dtype=np.int32)
        sources = np.empty(row_count, dtype=object)
        years = np.empty(row_count, dtype=np.int32)
        character_counts = np.empty(row_count, dtype=np.int32)
        word_counts = np.empty(row_count, dtype=np.int32)
        sentence_start_indices = np.empty(row_count, dtype=np.int32)
        sentence_end_indices = np.empty(row_count, dtype=np.int32)

        cursor = conn.execute(
            "SELECT row_index, chunk_id, article_id, chunk_index, source, year, "
            "character_count, word_count, sentence_start_index, sentence_end_index "
            "FROM chunk_rows ORDER BY row_index"
        )
        filled_count = 0
        for offset, row in enumerate(cursor):
            row_indices[offset] = int(row["row_index"])
            chunk_ids[offset] = row["chunk_id"]
            article_ids[offset] = row["article_id"]
            chunk_indices[offset] = int(row["chunk_index"] or 0)
            sources[offset] = row["source"]
            years[offset] = int(row["year"] or 0)
            character_counts[offset] = int(row["character_count"] or 0)
            word_counts[offset] = int(row["word_count"] or 0)
            sentence_start_indices[offset] = int(row["sentence_start_index"])
            sentence_end_indices[offset] = int(row["sentence_end_index"])
            filled_count = offset + 1
        if filled_count != row_count:
            row_indices = row_indices[:filled_count]
            chunk_ids = chunk_ids[:filled_count]
            article_ids = article_ids[:filled_count]
            chunk_indices = chunk_indices[:filled_count]
            sources = sources[:filled_count]
            years = years[:filled_count]
            character_counts = character_counts[:filled_count]
            word_counts = word_counts[:filled_count]
            sentence_start_indices = sentence_start_indices[:filled_count]
            sentence_end_indices = sentence_end_indices[:filled_count]
    return ChunkRowMetadata(
        row_indices=row_indices,
        chunk_ids=chunk_ids,
        article_ids=article_ids,
        chunk_indices=chunk_indices,
        sources=sources,
        years=years,
        character_counts=character_counts,
        word_counts=word_counts,
        sentence_start_indices=sentence_start_indices,
        sentence_end_indices=sentence_end_indices,
    )


def _chunk_rows_by_indices_from_store(store_path, row_indices):
    indices = sorted({int(index) for index in row_indices})
    if not indices:
        return {}

    placeholders = ", ".join("?" for _index in indices)
    with sqlite3.connect(Path(store_path)) as conn:
        conn.row_factory = sqlite3.Row
        cursor = conn.execute(
            f"SELECT row_index, text FROM chunk_rows WHERE row_index IN ({placeholders})",
            indices,
        )
        return {
            int(row["row_index"]): {"text": normalize_paragraph_text(row["text"])}
            for row in cursor
        }


class ChunkRetrievalIndex:
    def __init__(
        self,
        retrieval_model,
        chunking_mode,
        processor,
        chunk_rows,
        chunk_matrix=None,
        chunk_row_store_path=None,
    ):
        self.retrieval_model = normalize_retrieval_model(retrieval_model)
        if self.retrieval_model not in SUPPORTED_CHUNK_RETRIEVAL_MODELS:
            raise ValueError(f"Chunk retrieval does not support {retrieval_model!r}.")
        self.chunking_mode = normalize_chunk_index_mode(chunking_mode)
        self.processor = processor
        self.vectorizer = getattr(processor, "vectorizer", None)
        if isinstance(chunk_rows, ChunkRowMetadata):
            self.chunk_rows = chunk_rows
        else:
            self.chunk_rows = list(chunk_rows)
        self.chunk_matrix = chunk_matrix
        self.chunk_row_store_path = (
            Path(chunk_row_store_path) if chunk_row_store_path is not None else None
        )
        self.n_chunks = len(self.chunk_rows)
        self.svd_components = (
            np.asarray(processor.components, dtype=np.float32)
            if self.retrieval_model == "svd" and hasattr(processor, "components")
            else None
        )
        self._normalized_dense_chunk_embeddings = None
        if (
            self.retrieval_model == "minilm"
            and hasattr(processor, "normalized_doc_embeddings")
            and int(getattr(processor, "n_docs", self.n_chunks)) == self.n_chunks
        ):
            self._normalized_dense_chunk_embeddings = np.asarray(
                processor.normalized_doc_embeddings
            )
        if isinstance(self.chunk_rows, ChunkRowMetadata):
            self.article_ids = self.chunk_rows.article_ids
            self.article_years = self.chunk_rows.years
            self.article_character_counts = self.chunk_rows.character_counts
            self.article_word_counts = self.chunk_rows.word_counts
        else:
            self.article_ids = np.asarray(
                [row["article_id"] for row in self.chunk_rows],
                dtype=object,
            )
            self.article_years = np.asarray(
                [int(row.get("year") or 0) for row in self.chunk_rows],
                dtype=np.int32,
            )
            self.article_character_counts = np.asarray(
                [int(row.get("character_count") or 0) for row in self.chunk_rows],
                dtype=np.int32,
            )
            self.article_word_counts = np.asarray(
                [int(row.get("word_count") or 0) for row in self.chunk_rows],
                dtype=np.int32,
            )

    def _stored_rows_for_indices(self, chunk_indices):
        if self.chunk_row_store_path is None:
            return {}
        return _chunk_rows_by_indices_from_store(
            self.chunk_row_store_path,
            chunk_indices,
        )

    def _chunk_row_for_index(self, chunk_idx):
        if isinstance(self.chunk_rows, ChunkRowMetadata):
            return self.chunk_rows.row_dict(chunk_idx)
        return dict(self.chunk_rows[chunk_idx])

    def _dense_chunk_embeddings(self):
        if self.retrieval_model == "minilm":
            return self._normalized_dense_chunk_embeddings
        if self.svd_components is None:
            return None
        if self._normalized_dense_chunk_embeddings is None:
            if int(getattr(self.processor, "n_docs", self.n_chunks)) == self.n_chunks:
                processor_embeddings = getattr(
                    self.processor,
                    "normalized_doc_embeddings",
                    None,
                )
            else:
                processor_embeddings = None

            if processor_embeddings is not None:
                self._normalized_dense_chunk_embeddings = np.asarray(
                    processor_embeddings,
                    dtype=DEFAULT_CHUNK_DENSE_STORAGE_DTYPE,
                )
            else:
                if self.chunk_matrix is None:
                    return None
                log_runtime_event(
                    "chunk_index.svd_project_start",
                    chunk_count=self.n_chunks,
                )
                embeddings = self.chunk_matrix @ self.svd_components.T
                self._normalized_dense_chunk_embeddings = _normalize_dense_rows(
                    embeddings
                )
                log_runtime_event(
                    "chunk_index.svd_project_done",
                    chunk_count=self.n_chunks,
                    n_components=int(self._normalized_dense_chunk_embeddings.shape[1]),
                    dtype=str(self._normalized_dense_chunk_embeddings.dtype),
                )
        return self._normalized_dense_chunk_embeddings

    def _candidate_filter_mask(
        self,
        candidate_indices,
        year_start=None,
        year_end=None,
        character_start=None,
        character_end=None,
        word_start=None,
        word_end=None,
        reading_time_word_start=None,
        reading_time_word_end=None,
        exclude_article_ids=None,
    ):
        indices = np.asarray(candidate_indices, dtype=np.intp)
        if indices.size == 0:
            return np.asarray([], dtype=bool)

        mask = np.ones(indices.size, dtype=bool)
        if year_start is not None:
            mask &= self.article_years[indices] >= int(year_start)
        if year_end is not None:
            mask &= self.article_years[indices] <= int(year_end)
        if character_start is not None:
            mask &= self.article_character_counts[indices] >= int(character_start)
        if character_end is not None:
            mask &= self.article_character_counts[indices] <= int(character_end)
        if word_start is not None:
            mask &= self.article_word_counts[indices] >= int(word_start)
        if word_end is not None:
            mask &= self.article_word_counts[indices] <= int(word_end)
        if reading_time_word_start is not None:
            mask &= self.article_word_counts[indices] >= int(reading_time_word_start)
        if reading_time_word_end is not None:
            mask &= self.article_word_counts[indices] <= int(reading_time_word_end)

        excluded = {str(article_id) for article_id in (exclude_article_ids or [])}
        if excluded:
            mask &= np.asarray(
                [str(article_id) not in excluded for article_id in self.article_ids[indices]],
                dtype=bool,
            )

        return mask

    def search_chunks(
        self,
        query,
        top_n=DEFAULT_CHUNK_CANDIDATE_TOP_K,
        threshold=None,
        year_start=None,
        year_end=None,
        character_start=None,
        character_end=None,
        word_start=None,
        word_end=None,
        reading_time_word_start=None,
        reading_time_word_end=None,
        exclude_article_ids=None,
    ):
        resolved_query = normalize_text_for_vectorization(query)
        if not resolved_query:
            return []

        resolved_top_n = normalize_chunk_candidate_top_k(top_n)
        if self.retrieval_model in {"svd", "minilm"}:
            embeddings = self._dense_chunk_embeddings()
            if embeddings is None:
                return []
            if not hasattr(self.processor, "project_query"):
                return []
            query_embedding = self.processor.project_query(query, normalize=True)
            if query_embedding is None:
                return []
            scores = np.asarray(
                embeddings @ np.asarray(query_embedding, dtype=np.float32),
                dtype=np.float32,
            )
            candidate_indices = np.flatnonzero(scores > 0)
            candidate_scores = scores[candidate_indices]
        else:
            query_vec = self.vectorizer.transform([resolved_query])
            if int(getattr(query_vec, "nnz", 0)) <= 0:
                return []
            score_matrix = query_vec @ self.chunk_matrix.T
            if int(getattr(score_matrix, "nnz", 0)) <= 0:
                return []
            score_coo = score_matrix.tocoo()
            candidate_indices = np.asarray(score_coo.col, dtype=np.intp)
            candidate_scores = np.asarray(score_coo.data, dtype=np.float32)
            positive_mask = candidate_scores > 0
            candidate_indices = candidate_indices[positive_mask]
            candidate_scores = candidate_scores[positive_mask]

        filter_mask = self._candidate_filter_mask(
            candidate_indices,
            year_start=year_start,
            year_end=year_end,
            character_start=character_start,
            character_end=character_end,
            word_start=word_start,
            word_end=word_end,
            reading_time_word_start=reading_time_word_start,
            reading_time_word_end=reading_time_word_end,
            exclude_article_ids=exclude_article_ids,
        )
        candidate_indices = candidate_indices[filter_mask]
        candidate_scores = candidate_scores[filter_mask]
        if candidate_indices.size == 0:
            return []

        if self.retrieval_model == "svd":
            filtered_scores = np.asarray(candidate_scores, dtype=np.float32)
        else:
            # TF-IDF candidates can contain duplicate columns only in unusual sparse
            # formats; keep the strongest score per chunk before ranking.
            strongest = {}
            for idx, score in zip(candidate_indices, candidate_scores):
                strongest[int(idx)] = max(float(score), strongest.get(int(idx), 0.0))
            candidate_indices = np.asarray(list(strongest.keys()), dtype=np.intp)
            filtered_scores = np.asarray(
                [strongest[int(idx)] for idx in candidate_indices],
                dtype=np.float32,
            )

        if threshold is not None:
            threshold_mask = filtered_scores >= float(threshold)
            candidate_indices = candidate_indices[threshold_mask]
            filtered_scores = filtered_scores[threshold_mask]
        if candidate_indices.size == 0:
            return []

        selected_count = min(resolved_top_n, int(filtered_scores.size))
        if filtered_scores.size > selected_count:
            top_positions = np.argpartition(filtered_scores, -selected_count)[-selected_count:]
            sorted_positions = top_positions[np.argsort(filtered_scores[top_positions])[::-1]]
        else:
            sorted_positions = np.argsort(filtered_scores)[::-1]

        selected = [
            (int(candidate_indices[int(pos)]), int(pos))
            for pos in sorted_positions
        ]
        selected_chunk_indices = [chunk_idx for chunk_idx, _pos in selected]
        stored_rows = self._stored_rows_for_indices(selected_chunk_indices)

        results = []
        for rank, (chunk_idx, pos) in enumerate(selected, start=1):
            row = self._chunk_row_for_index(chunk_idx)
            row.update(stored_rows.get(chunk_idx, {}))
            row["score"] = float(filtered_scores[int(pos)])
            row["chunk_rank"] = int(rank)
            row["retrieval_model"] = self.retrieval_model
            results.append(row)
        return results


def build_chunk_retrieval_index(
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    chunking_mode=DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE,
):
    resolved_model = normalize_retrieval_model(retrieval_model)
    if resolved_model not in {"svd", "minilm"}:
        raise ValueError("Chunk retrieval uses a precomputed semantic chunk index.")
    resolved_chunking_mode = normalize_chunk_index_mode(chunking_mode)
    cache_key = (resolved_model, resolved_chunking_mode)
    index_name = (
        DEFAULT_CHUNK_SVD_INDEX_NAME
        if resolved_model == "svd"
        else DEFAULT_MINILM_CHUNK_INDEX_NAME
    )

    cached = _chunk_indexes.get(cache_key)
    if cached is not None:
        if resolved_model == "svd":
            current_meta = _load_index_meta(
                TruncatedSvdIndex.artifact_paths(DEFAULT_INDEX_DIR, index_name)["meta"]
            )
        else:
            current_meta = _load_index_meta(
                MiniLmEmbeddingIndex.artifact_paths(DEFAULT_INDEX_DIR, index_name)["meta"]
            )
        if current_meta.get("saved_at_utc") == getattr(cached, "index_saved_at_utc", None):
            return cached
        _chunk_indexes.pop(cache_key, None)

    with _chunk_index_lock:
        cached = _chunk_indexes.get(cache_key)
        if cached is not None:
            if resolved_model == "svd":
                current_meta = _load_index_meta(
                    TruncatedSvdIndex.artifact_paths(DEFAULT_INDEX_DIR, index_name)["meta"]
                )
            else:
                current_meta = _load_index_meta(
                    MiniLmEmbeddingIndex.artifact_paths(DEFAULT_INDEX_DIR, index_name)["meta"]
                )
            if current_meta.get("saved_at_utc") == getattr(cached, "index_saved_at_utc", None):
                return cached
            _chunk_indexes.pop(cache_key, None)

        log_runtime_event(
            "chunk_index.load_start",
            retrieval_model=resolved_model,
            chunking_mode=resolved_chunking_mode,
            index_name=index_name,
        )
        if resolved_model == "svd":
            preprocess_chunk_svd_index(
                index_dir=DEFAULT_INDEX_DIR,
                index_name=DEFAULT_CHUNK_SVD_INDEX_NAME,
                force_rebuild=False,
                chunking_mode=resolved_chunking_mode,
            )
            row_store_path = ensure_chunk_row_store(
                index_dir=DEFAULT_INDEX_DIR,
                index_name=DEFAULT_CHUNK_SVD_INDEX_NAME,
                force_rebuild=False,
            )
            processor, meta = load_chunk_svd_index(
                index_name=DEFAULT_CHUNK_SVD_INDEX_NAME,
                load_articles=False,
            )
        else:
            preprocess_minilm_indexes(
                index_dir=DEFAULT_INDEX_DIR,
                force_rebuild=False,
            )
            row_store_path = ensure_chunk_row_store(
                index_dir=DEFAULT_INDEX_DIR,
                index_name=DEFAULT_CHUNK_SVD_INDEX_NAME,
                force_rebuild=False,
            )
            processor, meta = load_minilm_chunk_index(
                index_name=DEFAULT_MINILM_CHUNK_INDEX_NAME,
                load_articles=False,
            )
        chunk_rows = _chunk_row_metadata_from_store(row_store_path)
        if not chunk_rows:
            raise RuntimeError("The precomputed chunk retrieval index has no chunk rows.")

        index = ChunkRetrievalIndex(
            retrieval_model=resolved_model,
            chunking_mode=resolved_chunking_mode,
            processor=processor,
            chunk_rows=chunk_rows,
            chunk_row_store_path=row_store_path,
        )
        index.index_saved_at_utc = meta.get("saved_at_utc")
        _chunk_indexes[cache_key] = index
        log_runtime_event(
            "chunk_index.load_done",
            retrieval_model=resolved_model,
            chunking_mode=resolved_chunking_mode,
            index_name=index_name,
            meta_chunking_mode=meta.get("chunking_mode"),
            chunk_count=index.n_chunks,
        )
        return index


def unload_chunk_retrieval_indexes(keep_indexes=None):
    keep = set()
    for item in list(keep_indexes or []):
        if isinstance(item, dict):
            raw_model = item.get("retrieval_model")
            raw_chunking_mode = item.get("chunking_mode")
        else:
            try:
                raw_model, raw_chunking_mode = item
            except (TypeError, ValueError):
                continue

        try:
            model = normalize_retrieval_model(raw_model)
        except ValueError:
            continue
        if model not in {"svd", "minilm"}:
            continue
        keep.add((model, normalize_chunk_index_mode(raw_chunking_mode)))

    unloaded = []
    with _chunk_index_lock:
        for cache_key in list(_chunk_indexes.keys()):
            if cache_key in keep:
                continue
            _chunk_indexes.pop(cache_key, None)
            unloaded.append(cache_key)

    if unloaded:
        log_runtime_event(
            "chunk_index.cache_unloaded",
            unloaded_indexes=[
                {"retrieval_model": model, "chunking_mode": chunking_mode}
                for model, chunking_mode in unloaded
            ],
            kept_indexes=[
                {"retrieval_model": model, "chunking_mode": chunking_mode}
                for model, chunking_mode in sorted(keep)
            ],
        )
    return unloaded


def _aggregate_article_chunks(
    chunks,
    article_top_k=DEFAULT_CHUNK_ARTICLE_TOP_K,
    aggregation_top_k=DEFAULT_CHUNK_AGGREGATION_TOP_K,
):
    grouped = {}
    for chunk in chunks:
        grouped.setdefault(str(chunk["article_id"]), []).append(chunk)

    rows = []
    for article_id, article_chunks in grouped.items():
        ranked_chunks = sorted(
            article_chunks,
            key=lambda chunk: float(chunk.get("score") or 0.0),
            reverse=True,
        )
        top_mean_chunks = ranked_chunks[:max(1, int(aggregation_top_k))]
        kept_chunks = ranked_chunks[:normalize_chunk_article_top_k(article_top_k)]
        max_score = float(ranked_chunks[0].get("score") or 0.0)
        top_k_mean = (
            sum(float(chunk.get("score") or 0.0) for chunk in top_mean_chunks)
            / len(top_mean_chunks)
        )
        coverage_score = min(
            len(top_mean_chunks) / float(max(1, int(aggregation_top_k))),
            1.0,
        )
        topic_score = (
            (CHUNK_SCORE_WEIGHTS["max"] * max_score)
            + (CHUNK_SCORE_WEIGHTS["top_k_mean"] * top_k_mean)
            + (CHUNK_SCORE_WEIGHTS["coverage"] * coverage_score)
        )
        rows.append({
            "article_id": article_id,
            "topic_score": float(topic_score),
            "max_chunk_score": max_score,
            "top_k_mean_chunk_score": float(top_k_mean),
            "coverage_score": float(coverage_score),
            "chunks": kept_chunks,
            "matched_chunk_count": len(ranked_chunks),
        })

    return sorted(rows, key=lambda row: row["topic_score"], reverse=True)


def chunk_retrieval_search(
    query,
    top_n=20,
    retrieval_model=DEFAULT_RETRIEVAL_MODEL,
    chunking_mode=DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE,
    rerank_selection_mode="manual",
    rerank_threshold=None,
    chunk_candidate_top_k=DEFAULT_CHUNK_CANDIDATE_TOP_K,
    chunk_article_top_k=DEFAULT_CHUNK_ARTICLE_TOP_K,
    year_start=None,
    year_end=None,
    character_start=None,
    character_end=None,
    word_start=None,
    word_end=None,
    reading_time_word_start=None,
    reading_time_word_end=None,
    words_to_avoid=None,
    topic_feedback_irrelevant_article_ids=None,
):
    from backend.services.filters.text_filters import (
        filter_ranked_articles_by_avoid_words,
        normalize_avoid_words,
    )

    resolved_model = normalize_retrieval_model(retrieval_model)
    resolved_chunking_mode = normalize_chunk_index_mode(chunking_mode)
    resolved_top_n = max(1, int(top_n))
    resolved_chunk_candidate_top_k = normalize_chunk_candidate_top_k(chunk_candidate_top_k)
    resolved_chunk_article_top_k = normalize_chunk_article_top_k(chunk_article_top_k)
    threshold = (
        float(rerank_threshold)
        if rerank_selection_mode == "automatic" and rerank_threshold is not None
        else None
    )
    log_runtime_event(
        "chunk_search.start",
        retrieval_model=resolved_model,
        chunking_mode=resolved_chunking_mode,
        query_chars=len(str(query or "").strip()),
        requested_top_n=resolved_top_n,
        chunk_candidate_top_k=resolved_chunk_candidate_top_k,
        chunk_article_top_k=resolved_chunk_article_top_k,
        selection_mode=rerank_selection_mode,
        threshold=threshold,
        has_filters=any(
            value is not None
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
        ),
        excluded_article_count=len(topic_feedback_irrelevant_article_ids or []),
    )
    index = build_chunk_retrieval_index(
        retrieval_model=resolved_model,
        chunking_mode=resolved_chunking_mode,
    )

    chunks = index.search_chunks(
        query=query,
        top_n=resolved_chunk_candidate_top_k,
        threshold=threshold,
        year_start=year_start,
        year_end=year_end,
        character_start=character_start,
        character_end=character_end,
        word_start=word_start,
        word_end=word_end,
        reading_time_word_start=reading_time_word_start,
        reading_time_word_end=reading_time_word_end,
        exclude_article_ids=topic_feedback_irrelevant_article_ids,
    )
    log_runtime_event(
        "chunk_search.chunks_done",
        retrieval_model=resolved_model,
        chunking_mode=resolved_chunking_mode,
        matched_chunk_count=len(chunks),
        chunk_candidate_top_k=resolved_chunk_candidate_top_k,
        threshold=threshold,
    )
    if not chunks:
        retrieval_label = (
            "SVD" if resolved_model == "svd"
            else ("Enhanced Semantic" if resolved_model == "minilm" else "TF-IDF")
        )
        threshold_copy = (
            f" above the {threshold:.2f} chunk relevance threshold"
            if threshold is not None
            else ""
        )
        log_runtime_event(
            "chunk_search.empty",
            retrieval_model=resolved_model,
            chunking_mode=resolved_chunking_mode,
            threshold=threshold,
        )
        return {
            "matches": [],
            "selection_mode": rerank_selection_mode,
            "candidate_count": 0,
            "rerank_threshold": threshold,
            "empty_results_message": (
                f"No relevant chunks found{threshold_copy} for {retrieval_label}."
            ),
        }

    article_rows = _aggregate_article_chunks(
        chunks,
        article_top_k=resolved_chunk_article_top_k,
    )
    log_runtime_event(
        "chunk_search.aggregate_done",
        article_candidate_count=len(article_rows),
        matched_chunk_count=len(chunks),
        top_article_score=article_rows[0]["topic_score"] if article_rows else None,
    )
    ranked_articles = [
        (row["article_id"], row["topic_score"])
        for row in article_rows
    ]
    resolved_avoid_words = normalize_avoid_words(words_to_avoid)
    if resolved_model == "tfidf" and resolved_avoid_words:
        article_count_before = len(article_rows)
        ranked_articles = filter_ranked_articles_by_avoid_words(
            ranked_articles,
            resolved_avoid_words,
        )
        allowed_article_ids = {article_id for article_id, _score in ranked_articles}
        article_rows = [
            row for row in article_rows
            if row["article_id"] in allowed_article_ids
        ]
        log_runtime_event(
            "chunk_search.avoid_words_done",
            avoid_word_count=len(resolved_avoid_words),
            article_count_before=article_count_before,
            article_count_after=len(article_rows),
        )

    selected_article_rows = article_rows[:resolved_top_n]
    selected_ranked_articles = [
        (row["article_id"], row["topic_score"])
        for row in selected_article_rows
    ]
    log_runtime_event(
        "chunk_search.matches_start",
        selected_article_count=len(selected_article_rows),
        article_candidate_count=len(article_rows),
    )
    matches = build_matches(
        selected_ranked_articles,
        retrieval_model=resolved_model,
        processor=index.processor,
    )
    row_by_article_id = {
        row["article_id"]: row for row in selected_article_rows
    }
    for match in matches:
        article_id = str(match.get("id") or "").strip()
        row = row_by_article_id.get(article_id)
        if not row:
            continue
        match["chunk_retrieval_enabled"] = True
        match["chunk_retrieval_model"] = resolved_model
        match["chunk_retrieval_chunking_mode"] = resolved_chunking_mode
        match["chunk_retrieval_candidate_count"] = len(chunks)
        match["chunk_retrieval_matched_chunk_count"] = row["matched_chunk_count"]
        match["chunk_topic_score_max"] = row["max_chunk_score"]
        match["chunk_topic_score_top_k_mean"] = row["top_k_mean_chunk_score"]
        match["chunk_topic_score_coverage"] = row["coverage_score"]
        match["topic_relevant_chunks"] = [
            {
                "chunk_id": chunk["chunk_id"],
                "chunk_index": chunk["chunk_index"],
                "text": chunk["text"],
                "topic_score": float(chunk["score"]),
                "chunk_rank": int(chunk["chunk_rank"]),
                "source": chunk.get("source"),
                "retrieval_model": chunk.get("retrieval_model"),
                "sentence_start_index": chunk.get("sentence_start_index"),
                "sentence_end_index": chunk.get("sentence_end_index"),
            }
            for chunk in row["chunks"]
        ]

    log_runtime_event(
        "chunk_search.done",
        match_count=len(matches),
        article_candidate_count=len(article_rows),
        selected_article_count=len(selected_article_rows),
        chunk_candidate_count=len(chunks),
    )
    return {
        "matches": matches,
        "selection_mode": rerank_selection_mode,
        "candidate_count": len(matches),
        "chunk_candidate_count": len(chunks),
        "rerank_threshold": threshold,
        "empty_results_message": None,
    }


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute semantic chunk SVD retrieval artifacts."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--index-name", default=DEFAULT_CHUNK_SVD_INDEX_NAME)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument(
        "--chunking-mode",
        default=DEFAULT_CHUNK_RETRIEVAL_CHUNKING_MODE,
        choices=SUPPORTED_CHUNK_INDEX_MODES,
    )
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_SVD_N_COMPONENTS,
    )
    parser.add_argument(
        "--ensure-row-store",
        action="store_true",
        help="Build or refresh the disk-backed chunk row store for the existing index.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ensure_row_store:
        row_store_path = ensure_chunk_row_store(
            index_dir=args.index_dir,
            index_name=args.index_name,
            force_rebuild=args.force_rebuild,
        )
        result = {
            "row_store_path": str(row_store_path),
        }
    else:
        result = preprocess_chunk_svd_index(
            db_path=args.db_path,
            index_dir=args.index_dir,
            index_name=args.index_name,
            force_rebuild=args.force_rebuild,
            chunking_mode=args.chunking_mode,
            n_components=args.n_components,
        )
    print(json.dumps(result, indent=2))
