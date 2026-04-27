import argparse
import json
import os
import sys
import weakref
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.indexing.artifacts import (
    _artifact_exists,
    _artifact_within_size_limit,
    _cleanup_temp_paths,
    _materialized_artifact_path,
    _materialized_artifact_path_for_mmap,
    _write_dataframe_pickle_artifact,
    _write_json_artifact,
    _write_npy_artifact,
)
from backend.text_processing.indexing.corpus import DEFAULT_DB_PATH, DEFAULT_INDEX_DIR
from backend.text_processing.indexing.dense_search import (
    top_positive_dot_candidates,
)
from backend.text_processing.indexing.normalization import (
    _normalize_articles_for_doc_ids,
    _normalize_doc_ids,
)
from backend.text_processing.svd_processor import _load_index_meta, TruncatedSvdIndex


DEFAULT_MINILM_MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"
DEFAULT_MINILM_BATCH_SIZE = 64
DEFAULT_MINILM_MAX_LENGTH = 256
DEFAULT_MINILM_ARTICLE_INDEX_NAME = "guardian_article_minilm_semantic"
DEFAULT_MINILM_CHUNK_INDEX_NAME = "guardian_chunk_minilm_semantic"
DEFAULT_SOURCE_CHUNK_INDEX_NAME = "guardian_chunk_svd_semantic"
MINILM_INDEX_SCHEMA_VERSION = 1
DEFAULT_MINILM_ARTICLE_POOL_WEIGHTS = (0.5, 0.3, 0.2)
DEFAULT_MINILM_ARTICLE_POOL_TOP_K = len(DEFAULT_MINILM_ARTICLE_POOL_WEIGHTS)
DEFAULT_MINILM_STORAGE_DTYPE = np.float16
DEFAULT_MINILM_QUERY_DTYPE = np.float32
DEFAULT_MINILM_BUILD_PROGRESS_INTERVAL = 1000


def _default_device(torch):
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def _normalize_dense_rows(matrix):
    array = np.asarray(matrix, dtype=np.float32)
    norms = np.linalg.norm(array, axis=1, keepdims=True)
    norms = np.where(norms > 0, norms, 1.0).astype(np.float32, copy=False)
    return (array / norms).astype(np.float32, copy=False)


def _normalize_dense_vector(vector):
    array = np.asarray(vector, dtype=np.float32).reshape(-1)
    norm = float(np.linalg.norm(array))
    if norm <= 0.0:
        return None
    return (array / norm).astype(np.float32, copy=False)


def _normalize_pool_weights(weights):
    resolved = [
        float(weight)
        for weight in list(weights or DEFAULT_MINILM_ARTICLE_POOL_WEIGHTS)
        if float(weight) > 0
    ]
    if not resolved:
        resolved = list(DEFAULT_MINILM_ARTICLE_POOL_WEIGHTS)
    total = sum(resolved)
    return [float(weight / total) for weight in resolved]


@lru_cache(maxsize=1)
def load_minilm_bundle(
    model_name=DEFAULT_MINILM_MODEL_NAME,
):
    log_runtime_event(
        "minilm_bundle.load_start",
        model_name=model_name,
    )

    try:
        import torch
        from transformers import AutoModel, AutoTokenizer
    except ModuleNotFoundError as exc:
        raise RuntimeError(
            "The MiniLM dependencies are not available. Install torch and transformers "
            "to enable Enhanced Semantic retrieval."
        ) from exc

    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        model = AutoModel.from_pretrained(model_name)
    except OSError as exc:
        cache_dir = os.getenv("HF_HOME") or os.getenv("TRANSFORMERS_CACHE")
        cache_note = f" Cache directory: {cache_dir}." if cache_dir else ""
        raise RuntimeError(
            "The MiniLM model could not be loaded. Ensure the Hugging Face files are "
            f"available in the runtime image or that the environment can download them.{cache_note}"
        ) from exc

    device = _default_device(torch)
    model.to(device)
    model.eval()
    log_runtime_event(
        "minilm_bundle.load_done",
        model_name=model_name,
        device=str(device),
    )
    return {
        "torch": torch,
        "tokenizer": tokenizer,
        "model": model,
        "device": device,
        "model_name": model_name,
    }


def unload_minilm_bundle():
    had_bundle = load_minilm_bundle.cache_info().currsize > 0
    load_minilm_bundle.cache_clear()
    if had_bundle:
        torch = sys.modules.get("torch")
        if torch is not None:
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            if (
                hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
                and hasattr(torch, "mps")
            ):
                torch.mps.empty_cache()
        log_runtime_event("minilm_bundle.cache_unloaded")
    return had_bundle


def encode_minilm_texts(
    texts,
    model_name=DEFAULT_MINILM_MODEL_NAME,
    batch_size=DEFAULT_MINILM_BATCH_SIZE,
    max_length=DEFAULT_MINILM_MAX_LENGTH,
    normalize=True,
):
    cleaned_texts = [str(text or "").strip() for text in list(texts or [])]
    if not cleaned_texts:
        return np.zeros((0, 0), dtype=np.float32)

    bundle = load_minilm_bundle(model_name=model_name)
    torch = bundle["torch"]
    tokenizer = bundle["tokenizer"]
    model = bundle["model"]
    device = bundle["device"]
    resolved_batch_size = max(1, int(batch_size or DEFAULT_MINILM_BATCH_SIZE))
    resolved_max_length = max(8, int(max_length or DEFAULT_MINILM_MAX_LENGTH))

    log_runtime_event(
        "minilm_encode.start",
        text_count=len(cleaned_texts),
        batch_size=resolved_batch_size,
        max_length=resolved_max_length,
        model_name=model_name,
    )

    embeddings = None
    total_batches = (len(cleaned_texts) + resolved_batch_size - 1) // resolved_batch_size
    for batch_index, start_index in enumerate(
        range(0, len(cleaned_texts), resolved_batch_size),
        start=1,
    ):
        batch_texts = cleaned_texts[start_index:start_index + resolved_batch_size]
        log_runtime_event(
            "minilm_encode.batch_start",
            batch_index=batch_index,
            batch_total=total_batches,
            batch_size=len(batch_texts),
        )
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            max_length=resolved_max_length,
            return_tensors="pt",
        )
        encoded = {
            key: value.to(device)
            for key, value in encoded.items()
        }

        with torch.inference_mode():
            model_output = model(**encoded)
            token_embeddings = model_output.last_hidden_state
            attention_mask = encoded["attention_mask"].unsqueeze(-1).to(token_embeddings.dtype)
            summed = torch.sum(token_embeddings * attention_mask, dim=1)
            counts = torch.clamp(attention_mask.sum(dim=1), min=1e-9)
            pooled = summed / counts
            if normalize:
                pooled = torch.nn.functional.normalize(pooled, p=2, dim=1)

        batch_embeddings = np.asarray(pooled.cpu(), dtype=np.float32)
        if embeddings is None:
            embeddings = np.empty(
                (len(cleaned_texts), int(batch_embeddings.shape[1])),
                dtype=np.float32,
            )
        embeddings[start_index:start_index + len(batch_texts)] = batch_embeddings
        del encoded, model_output, token_embeddings, attention_mask, summed, counts, pooled, batch_embeddings
        log_runtime_event(
            "minilm_encode.batch_done",
            batch_index=batch_index,
            batch_total=total_batches,
        )

    if embeddings is None:
        embeddings = np.zeros((0, 0), dtype=np.float32)
    log_runtime_event(
        "minilm_encode.done",
        text_count=len(cleaned_texts),
        embedding_dim=int(embeddings.shape[1]) if embeddings.ndim == 2 and embeddings.size else 0,
    )
    return embeddings


class MiniLmEmbeddingIndex:
    def __init__(
        self,
        normalized_doc_embeddings,
        doc_ids,
        articles=None,
        id_column="id",
        model_name=DEFAULT_MINILM_MODEL_NAME,
        max_length=DEFAULT_MINILM_MAX_LENGTH,
        temp_artifact_paths=None,
    ):
        self.doc_ids = _normalize_doc_ids(doc_ids)
        self.doc_to_idx = {
            doc_id: idx
            for idx, doc_id in enumerate(self.doc_ids)
        }
        self.n_docs = len(self.doc_ids)

        embeddings = np.asarray(normalized_doc_embeddings)
        if embeddings.ndim != 2:
            raise ValueError("normalized_doc_embeddings must be a 2-dimensional array.")
        if embeddings.shape[0] != self.n_docs:
            raise ValueError(
                "Embedding row count must match doc_ids count."
            )
        self.normalized_doc_embeddings = np.asarray(
            embeddings,
            dtype=DEFAULT_MINILM_STORAGE_DTYPE,
        )
        self.embedding_dim = int(self.normalized_doc_embeddings.shape[1])
        self.model_name = str(model_name or DEFAULT_MINILM_MODEL_NAME)
        self.max_length = int(max_length or DEFAULT_MINILM_MAX_LENGTH)
        self.id_column = id_column
        self._temp_artifact_paths = [Path(path) for path in list(temp_artifact_paths or [])]
        self._temp_artifact_finalizer = weakref.finalize(
            self,
            _cleanup_temp_paths,
            list(self._temp_artifact_paths),
        )

        self.articles = None
        if articles is not None:
            self.articles = _normalize_articles_for_doc_ids(
                articles=articles,
                doc_ids=self.doc_ids,
                id_column=id_column,
            )

    def get_doc_idx_by_id(self, doc_id):
        key = str(doc_id).strip()
        idx = self.doc_to_idx.get(key)
        if idx is None:
            raise ValueError(f"Document ID {doc_id!r} not found.")
        return idx

    def get_doc_vector(self, doc_id, normalize=True):
        _ = normalize  # Embeddings are stored normalized already.
        idx = self.get_doc_idx_by_id(doc_id)
        return np.asarray(
            self.normalized_doc_embeddings[idx],
            dtype=DEFAULT_MINILM_QUERY_DTYPE,
        )

    def project_query(self, query, normalize=True):
        resolved_query = str(query or "").strip()
        if not resolved_query:
            return None
        embeddings = encode_minilm_texts(
            [resolved_query],
            model_name=self.model_name,
            max_length=self.max_length,
            normalize=normalize,
        )
        if embeddings.size <= 0:
            return None
        vector = np.asarray(embeddings[0], dtype=DEFAULT_MINILM_QUERY_DTYPE)
        if normalize:
            return _normalize_dense_vector(vector)
        return vector

    def search(self, query, top_n=100, return_articles=True):
        if top_n is None:
            top_n = self.n_docs
        resolved_top_n = max(1, int(top_n))
        query_embedding = self.project_query(query, normalize=True)
        if query_embedding is None:
            return []

        log_runtime_event(
            "minilm_search.start",
            query_chars=len(str(query or "").strip()),
            top_n=resolved_top_n,
            n_docs=self.n_docs,
            embedding_dim=self.embedding_dim,
        )
        candidate_doc_indices, candidate_scores = top_positive_dot_candidates(
            self.normalized_doc_embeddings,
            query_embedding,
            top_n=resolved_top_n,
        )
        if candidate_doc_indices.size == 0:
            log_runtime_event("minilm_search.empty")
            return []

        results = []
        for idx, score in zip(candidate_doc_indices, candidate_scores):
            idx = int(idx)
            payload = self.doc_ids[idx]
            if return_articles and self.articles is not None:
                payload = self.articles.iloc[idx].to_dict()
            results.append((payload, float(score)))

        log_runtime_event(
            "minilm_search.done",
            result_count=len(results),
        )
        return results

    @staticmethod
    def artifact_paths(index_dir, index_name):
        directory = Path(index_dir)
        return {
            "embeddings": directory / f"{index_name}_embeddings.npy",
            "doc_ids": directory / f"{index_name}_doc_ids.json",
            "articles": directory / f"{index_name}_articles.pkl",
            "meta": directory / f"{index_name}_meta.json",
        }

    @classmethod
    def has_artifacts(cls, index_dir, index_name):
        paths = cls.artifact_paths(index_dir, index_name)
        return all(
            _artifact_exists(path)
            for path in (
                paths["embeddings"],
                paths["doc_ids"],
                paths["meta"],
            )
        )

    @classmethod
    def artifacts_within_size_limit(cls, index_dir, index_name):
        paths = cls.artifact_paths(index_dir, index_name)
        required_paths = [
            paths["embeddings"],
            paths["doc_ids"],
            paths["meta"],
        ]
        if _artifact_exists(paths["articles"]):
            required_paths.append(paths["articles"])
        return all(_artifact_within_size_limit(path) for path in required_paths)

    def save(self, index_dir, index_name, extra_meta=None):
        paths = self.artifact_paths(index_dir, index_name)
        Path(index_dir).mkdir(parents=True, exist_ok=True)

        embeddings_artifact = _write_npy_artifact(
            paths["embeddings"],
            np.asarray(self.normalized_doc_embeddings, dtype=DEFAULT_MINILM_STORAGE_DTYPE),
        )
        doc_ids_artifact = _write_json_artifact(
            paths["doc_ids"],
            list(self.doc_ids),
        )
        articles_artifact = None
        if self.articles is not None:
            articles_artifact = _write_dataframe_pickle_artifact(
                paths["articles"],
                self.articles,
            )

        meta_payload = {
            "saved_at_utc": pd.Timestamp.utcnow().isoformat(),
            "search_backend": "minilm_dense",
            "model_name": self.model_name,
            "max_length": int(self.max_length),
            "n_docs": int(self.n_docs),
            "embedding_dim": int(self.embedding_dim),
            "has_articles": bool(self.articles is not None),
            "id_column": self.id_column,
        }
        if extra_meta:
            meta_payload.update(dict(extra_meta))
        meta_artifact = _write_json_artifact(paths["meta"], meta_payload)

        return {
            "embeddings": embeddings_artifact["path"],
            "doc_ids": doc_ids_artifact["path"],
            "articles": None if articles_artifact is None else articles_artifact["path"],
            "meta": meta_artifact["path"],
            "embeddings_files": embeddings_artifact["files"],
            "doc_ids_files": doc_ids_artifact["files"],
            "articles_files": [] if articles_artifact is None else articles_artifact["files"],
            "meta_files": meta_artifact["files"],
        }

    @classmethod
    def load(
        cls,
        index_dir,
        index_name,
        load_articles=True,
    ):
        paths = cls.artifact_paths(index_dir, index_name)
        temp_artifact_paths = []
        try:
            embeddings_path, temp_embeddings = _materialized_artifact_path_for_mmap(
                paths["embeddings"]
            )
            if temp_embeddings is not None:
                temp_artifact_paths.append(temp_embeddings)
            embeddings = np.load(
                embeddings_path,
                mmap_mode="r",
                allow_pickle=False,
            )
            with _materialized_artifact_path(paths["doc_ids"]) as doc_ids_path:
                with open(doc_ids_path, "r", encoding="utf-8") as f:
                    doc_ids = json.load(f) or []
            meta = _load_index_meta(paths["meta"])

            articles = None
            if load_articles and _artifact_exists(paths["articles"]):
                with _materialized_artifact_path(paths["articles"]) as articles_path:
                    articles = pd.read_pickle(articles_path)

            instance = cls(
                normalized_doc_embeddings=embeddings,
                doc_ids=doc_ids,
                articles=articles,
                id_column=meta.get("id_column") or "id",
                model_name=meta.get("model_name") or DEFAULT_MINILM_MODEL_NAME,
                max_length=meta.get("max_length") or DEFAULT_MINILM_MAX_LENGTH,
                temp_artifact_paths=temp_artifact_paths,
            )
        except Exception:
            _cleanup_temp_paths(temp_artifact_paths)
            raise
        return instance, meta


def _minilm_source_chunk_meta(
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_SOURCE_CHUNK_INDEX_NAME,
):
    paths = TruncatedSvdIndex.artifact_paths(index_dir, index_name)
    return _load_index_meta(paths["meta"])


def _minilm_index_fresh(
    index_dir,
    index_name,
    expected_index_kind,
    source_chunk_meta,
    model_name=DEFAULT_MINILM_MODEL_NAME,
    max_length=DEFAULT_MINILM_MAX_LENGTH,
    pool_weights=None,
):
    if not MiniLmEmbeddingIndex.has_artifacts(index_dir=index_dir, index_name=index_name):
        return False
    if not MiniLmEmbeddingIndex.artifacts_within_size_limit(index_dir=index_dir, index_name=index_name):
        return False

    paths = MiniLmEmbeddingIndex.artifact_paths(index_dir, index_name)
    meta = _load_index_meta(paths["meta"])
    if not meta:
        return False
    if meta.get("search_backend") != "minilm_dense":
        return False
    if meta.get("index_kind") != expected_index_kind:
        return False
    if int(meta.get("minilm_index_schema_version") or -1) != MINILM_INDEX_SCHEMA_VERSION:
        return False
    if meta.get("model_name") != str(model_name):
        return False
    if int(meta.get("max_length") or -1) != int(max_length):
        return False
    if meta.get("source_chunk_index_name") != DEFAULT_SOURCE_CHUNK_INDEX_NAME:
        return False
    if meta.get("source_chunk_saved_at_utc") != source_chunk_meta.get("saved_at_utc"):
        return False
    if int(meta.get("source_chunk_count") or -1) != int(source_chunk_meta.get("chunk_count") or -1):
        return False
    if int(meta.get("source_chunk_schema_version") or -1) != int(
        source_chunk_meta.get("chunk_index_schema_version") or -1
    ):
        return False
    expected_pool_weights = _normalize_pool_weights(pool_weights)
    if meta.get("article_pool_weights") != expected_pool_weights:
        return False
    return True


def _article_pool_row(article_id, article_chunks, article_chunk_embeddings, pool_weights):
    centroid = np.mean(
        np.asarray(article_chunk_embeddings, dtype=np.float32),
        axis=0,
    )
    normalized_centroid = _normalize_dense_vector(centroid)
    if normalized_centroid is None:
        normalized_centroid = np.asarray(article_chunk_embeddings[0], dtype=np.float32)

    similarities = np.asarray(
        np.asarray(article_chunk_embeddings, dtype=np.float32) @ normalized_centroid,
        dtype=np.float32,
    )
    ranked_positions = np.argsort(similarities)[::-1]
    selected_positions = ranked_positions[:min(len(pool_weights), len(ranked_positions))]
    selected_embeddings = np.asarray(article_chunk_embeddings[selected_positions], dtype=np.float32)
    selected_chunks = article_chunks.iloc[selected_positions]

    resolved_weights = np.asarray(pool_weights[:len(selected_positions)], dtype=np.float32)
    resolved_weights = resolved_weights / resolved_weights.sum()
    pooled_embedding = np.sum(
        selected_embeddings * resolved_weights[:, None],
        axis=0,
    )
    pooled_embedding = _normalize_dense_vector(pooled_embedding)
    if pooled_embedding is None:
        pooled_embedding = np.asarray(selected_embeddings[0], dtype=np.float32)

    representative_chunk_ids = selected_chunks["id"].astype(str).tolist()
    representative_chunk_indices = [
        int(value)
        for value in selected_chunks["chunk_index"].tolist()
    ]
    article_row = selected_chunks.iloc[0].to_dict()
    return {
        "id": str(article_id),
        "article_id": str(article_id),
        "year": article_row.get("year"),
        "character_count": article_row.get("character_count"),
        "word_count": article_row.get("word_count"),
        "chunk_count": int(len(article_chunks)),
        "representative_chunk_ids": representative_chunk_ids,
        "representative_chunk_indices": representative_chunk_indices,
        "representative_chunk_weights": [float(weight) for weight in resolved_weights.tolist()],
    }, pooled_embedding


def preprocess_minilm_indexes(
    db_path=DEFAULT_DB_PATH,
    index_dir=DEFAULT_INDEX_DIR,
    chunk_index_name=DEFAULT_MINILM_CHUNK_INDEX_NAME,
    article_index_name=DEFAULT_MINILM_ARTICLE_INDEX_NAME,
    source_chunk_index_name=DEFAULT_SOURCE_CHUNK_INDEX_NAME,
    force_rebuild=False,
    model_name=DEFAULT_MINILM_MODEL_NAME,
    batch_size=DEFAULT_MINILM_BATCH_SIZE,
    max_length=DEFAULT_MINILM_MAX_LENGTH,
    article_pool_weights=DEFAULT_MINILM_ARTICLE_POOL_WEIGHTS,
):
    resolved_index_dir = Path(index_dir)
    resolved_db_path = Path(db_path)
    resolved_pool_weights = _normalize_pool_weights(article_pool_weights)

    log_runtime_event(
        "minilm_preprocess.start",
        db_path=str(resolved_db_path),
        chunk_index_name=chunk_index_name,
        article_index_name=article_index_name,
        force_rebuild=bool(force_rebuild),
        model_name=model_name,
        max_length=int(max_length),
    )

    from backend.services.chunk_retrieval_service import (
        load_chunk_svd_index,
        preprocess_chunk_svd_index,
    )

    preprocess_chunk_svd_index(
        db_path=resolved_db_path,
        index_dir=resolved_index_dir,
        index_name=source_chunk_index_name,
        force_rebuild=False,
    )
    source_chunk_meta = _minilm_source_chunk_meta(
        index_dir=resolved_index_dir,
        index_name=source_chunk_index_name,
    )
    if not source_chunk_meta:
        raise RuntimeError("The semantic chunk SVD artifact metadata is missing.")

    chunk_fresh = (
        not force_rebuild
        and _minilm_index_fresh(
            index_dir=resolved_index_dir,
            index_name=chunk_index_name,
            expected_index_kind="chunk_minilm",
            source_chunk_meta=source_chunk_meta,
            model_name=model_name,
            max_length=max_length,
            pool_weights=resolved_pool_weights,
        )
    )
    article_fresh = (
        not force_rebuild
        and _minilm_index_fresh(
            index_dir=resolved_index_dir,
            index_name=article_index_name,
            expected_index_kind="article_minilm",
            source_chunk_meta=source_chunk_meta,
            model_name=model_name,
            max_length=max_length,
            pool_weights=resolved_pool_weights,
        )
    )
    if chunk_fresh and article_fresh:
        log_runtime_event(
            "minilm_preprocess.up_to_date",
            chunk_index_name=chunk_index_name,
            article_index_name=article_index_name,
        )
        return {
            "built": False,
            "reason": "up_to_date",
            "chunk_index_name": chunk_index_name,
            "article_index_name": article_index_name,
        }

    log_runtime_event(
        "minilm_preprocess.source_chunk_start",
        source_chunk_index_name=source_chunk_index_name,
    )
    chunk_processor, chunk_meta = load_chunk_svd_index(
        index_dir=resolved_index_dir,
        index_name=source_chunk_index_name,
        load_articles=True,
    )
    chunk_frame = getattr(chunk_processor, "articles", None)
    if chunk_frame is None or not isinstance(chunk_frame, pd.DataFrame) or chunk_frame.empty:
        raise RuntimeError("The semantic chunk source artifact has no chunk rows.")
    log_runtime_event(
        "minilm_preprocess.source_chunk_ready",
        source_chunk_index_name=source_chunk_index_name,
        chunk_count=int(len(chunk_frame)),
        article_count=int(chunk_frame["article_id"].nunique()),
    )

    chunk_texts = chunk_frame["text"].astype(str).tolist()
    chunk_embeddings = encode_minilm_texts(
        chunk_texts,
        model_name=model_name,
        batch_size=batch_size,
        max_length=max_length,
        normalize=True,
    )
    if chunk_embeddings.shape[0] != len(chunk_frame):
        raise RuntimeError("MiniLM chunk embedding count does not match chunk metadata.")
    log_runtime_event(
        "minilm_preprocess.chunk_embeddings_ready",
        chunk_count=int(chunk_embeddings.shape[0]),
        embedding_dim=int(chunk_embeddings.shape[1]),
    )

    article_rows = []
    article_embeddings = []
    grouped = chunk_frame.groupby("article_id", sort=True, dropna=False)
    total_articles = int(grouped.ngroups)
    for article_offset, (article_id, article_chunks) in enumerate(grouped, start=1):
        indices = article_chunks.index.to_numpy(dtype=np.intp)
        article_chunk_embeddings = chunk_embeddings[indices]
        article_row, article_embedding = _article_pool_row(
            article_id=article_id,
            article_chunks=article_chunks.reset_index(drop=True),
            article_chunk_embeddings=article_chunk_embeddings,
            pool_weights=resolved_pool_weights,
        )
        article_rows.append(article_row)
        article_embeddings.append(article_embedding)
        if (
            article_offset % DEFAULT_MINILM_BUILD_PROGRESS_INTERVAL == 0
            or article_offset == total_articles
        ):
            log_runtime_event(
                "minilm_preprocess.article_pool_progress",
                processed_article_count=article_offset,
                article_count=total_articles,
            )

    article_frame = pd.DataFrame(article_rows)
    article_embedding_matrix = np.vstack(article_embeddings).astype(np.float32, copy=False)
    log_runtime_event(
        "minilm_preprocess.article_pool_done",
        article_count=int(len(article_frame)),
        embedding_dim=int(article_embedding_matrix.shape[1]),
    )

    chunk_index = MiniLmEmbeddingIndex(
        normalized_doc_embeddings=chunk_embeddings,
        doc_ids=chunk_frame["id"].astype(str).tolist(),
        articles=chunk_frame,
        id_column="id",
        model_name=model_name,
        max_length=max_length,
    )
    article_index = MiniLmEmbeddingIndex(
        normalized_doc_embeddings=article_embedding_matrix,
        doc_ids=article_frame["id"].astype(str).tolist(),
        articles=article_frame,
        id_column="id",
        model_name=model_name,
        max_length=max_length,
    )

    log_runtime_event(
        "minilm_preprocess.save_start",
        chunk_index_name=chunk_index_name,
        article_index_name=article_index_name,
    )
    chunk_paths = chunk_index.save(
        index_dir=resolved_index_dir,
        index_name=chunk_index_name,
        extra_meta={
            "index_kind": "chunk_minilm",
            "minilm_index_schema_version": MINILM_INDEX_SCHEMA_VERSION,
            "source_chunk_index_name": source_chunk_index_name,
            "source_chunk_saved_at_utc": chunk_meta.get("saved_at_utc"),
            "source_chunk_count": int(chunk_meta.get("chunk_count") or len(chunk_frame)),
            "source_chunk_schema_version": int(chunk_meta.get("chunk_index_schema_version") or -1),
            "chunking_mode": chunk_meta.get("chunking_mode"),
            "article_pool_weights": resolved_pool_weights,
            "batch_size": int(batch_size),
        },
    )
    article_paths = article_index.save(
        index_dir=resolved_index_dir,
        index_name=article_index_name,
        extra_meta={
            "index_kind": "article_minilm",
            "minilm_index_schema_version": MINILM_INDEX_SCHEMA_VERSION,
            "source_chunk_index_name": source_chunk_index_name,
            "source_chunk_saved_at_utc": chunk_meta.get("saved_at_utc"),
            "source_chunk_count": int(chunk_meta.get("chunk_count") or len(chunk_frame)),
            "source_chunk_schema_version": int(chunk_meta.get("chunk_index_schema_version") or -1),
            "chunking_mode": chunk_meta.get("chunking_mode"),
            "article_pool_weights": resolved_pool_weights,
            "article_pool_top_k": int(len(resolved_pool_weights)),
            "batch_size": int(batch_size),
        },
    )
    log_runtime_event(
        "minilm_preprocess.done",
        chunk_index_name=chunk_index_name,
        article_index_name=article_index_name,
        chunk_count=int(len(chunk_frame)),
        article_count=int(len(article_frame)),
    )
    return {
        "built": True,
        "chunk_index_name": chunk_index_name,
        "article_index_name": article_index_name,
        "chunk_count": int(len(chunk_frame)),
        "article_count": int(len(article_frame)),
        "chunk_paths": {
            key: [str(path) for path in value]
            if key.endswith("_files")
            else (None if value is None else str(value))
            for key, value in chunk_paths.items()
        },
        "article_paths": {
            key: [str(path) for path in value]
            if key.endswith("_files")
            else (None if value is None else str(value))
            for key, value in article_paths.items()
        },
    }


def load_minilm_article_index(
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_MINILM_ARTICLE_INDEX_NAME,
    load_articles=True,
):
    return MiniLmEmbeddingIndex.load(
        index_dir=index_dir,
        index_name=index_name,
        load_articles=load_articles,
    )


def load_minilm_chunk_index(
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_MINILM_CHUNK_INDEX_NAME,
    load_articles=True,
):
    return MiniLmEmbeddingIndex.load(
        index_dir=index_dir,
        index_name=index_name,
        load_articles=load_articles,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute MiniLM chunk and pooled article retrieval artifacts.",
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--chunk-index-name", default=DEFAULT_MINILM_CHUNK_INDEX_NAME)
    parser.add_argument("--article-index-name", default=DEFAULT_MINILM_ARTICLE_INDEX_NAME)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_MINILM_BATCH_SIZE)
    parser.add_argument("--max-length", type=int, default=DEFAULT_MINILM_MAX_LENGTH)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = preprocess_minilm_indexes(
        db_path=args.db_path,
        index_dir=args.index_dir,
        chunk_index_name=args.chunk_index_name,
        article_index_name=args.article_index_name,
        force_rebuild=args.force_rebuild,
        batch_size=args.batch_size,
        max_length=args.max_length,
    )
    print(json.dumps(result, indent=2))
