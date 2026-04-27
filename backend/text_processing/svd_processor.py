import argparse
import json
import pickle
import weakref
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.indexing.artifacts import (
    _artifact_exists,
    _artifact_files,
    _artifact_within_size_limit,
    _cleanup_temp_paths,
    _materialized_artifact_path,
    _materialized_artifact_path_for_mmap,
    _write_dataframe_pickle_artifact,
    _write_json_artifact,
    _write_npy_artifact,
    _write_pickle_artifact,
)
from backend.text_processing.indexing.normalization import (
    _normalize_articles_for_doc_ids,
    _normalize_doc_ids,
    _normalize_id_series,
    _normalize_terms,
)
from backend.text_processing.indexing.settings import DEFAULT_TFIDF_PARAMS
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
from backend.text_processing.indexing.dense_search import (
    top_positive_dot_candidates,
)
from backend.text_processing.text_normalization import (
    TEXT_NORMALIZATION_VERSION,
    normalize_text_for_vectorization,
)


DEFAULT_SVD_INDEX_NAME = "guardian_tfidf_svd"
DEFAULT_SVD_N_COMPONENTS = 100
DEFAULT_SVD_PARAMS = {
    "algorithm": "randomized",
    "n_iter": 7,
    "random_state": 0,
}
DEFAULT_SVD_NORMALIZED_DTYPE = np.float16


def _unlink_paths(paths):
    for path in list(paths or []):
        resolved = Path(path)
        if resolved.exists():
            resolved.unlink()


def _load_index_meta(path):
    resolved_path = Path(path)
    if not _artifact_exists(resolved_path):
        return {}

    try:
        with _materialized_artifact_path(resolved_path) as meta_path:
            with open(meta_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        return {}


def _row_normalize_dense(matrix, storage_dtype=DEFAULT_SVD_NORMALIZED_DTYPE):
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


def _normalized_doc_embeddings_fresh(path, expected_shape=None):
    if not _artifact_exists(path):
        return False
    if not _artifact_within_size_limit(path):
        return False

    temp_path = None
    try:
        materialized_path, temp_path = _materialized_artifact_path_for_mmap(path)
        array = np.load(materialized_path, mmap_mode="r", allow_pickle=False)
        if array.dtype != DEFAULT_SVD_NORMALIZED_DTYPE:
            return False
        if expected_shape is not None and tuple(array.shape) != tuple(expected_shape):
            return False
        return True
    except Exception:
        return False
    finally:
        if temp_path is not None and Path(temp_path).exists():
            Path(temp_path).unlink()


def _update_normalized_doc_embeddings_meta(paths, meta, artifact):
    next_meta = dict(meta or {})
    next_meta["normalized_doc_embeddings_dtype"] = str(
        np.dtype(DEFAULT_SVD_NORMALIZED_DTYPE)
    )
    next_meta["normalized_doc_embeddings_files"] = [
        path.name
        for path in artifact["files"]
    ]
    _write_json_artifact(paths["meta"], next_meta)
    return next_meta


def ensure_normalized_doc_embeddings_artifact(
    index_dir,
    index_name,
    meta=None,
    force_rebuild=False,
):
    paths = TruncatedSvdIndex.artifact_paths(index_dir, index_name)
    resolved_meta = dict(meta or _load_index_meta(paths["meta"]) or {})
    expected_shape = None
    try:
        n_docs = int(resolved_meta.get("n_docs"))
        n_components = int(resolved_meta.get("n_components"))
        expected_shape = (n_docs, n_components)
    except (TypeError, ValueError):
        expected_shape = None

    if (
        not force_rebuild
        and _normalized_doc_embeddings_fresh(
            paths["normalized_doc_embeddings"],
            expected_shape=expected_shape,
        )
    ):
        artifact_files = _artifact_files(paths["normalized_doc_embeddings"])
        artifact_file_names = [path.name for path in artifact_files]
        if (
            resolved_meta.get("normalized_doc_embeddings_dtype")
            != str(np.dtype(DEFAULT_SVD_NORMALIZED_DTYPE))
            or resolved_meta.get("normalized_doc_embeddings_files")
            != artifact_file_names
        ):
            _update_normalized_doc_embeddings_meta(
                paths,
                resolved_meta,
                {"files": artifact_files},
            )
        return paths["normalized_doc_embeddings"]

    log_runtime_event(
        "svd_index.normalized_embeddings_build_start",
        index_name=index_name,
        dtype=str(np.dtype(DEFAULT_SVD_NORMALIZED_DTYPE)),
    )
    doc_embeddings_path, temp_doc_embeddings = _materialized_artifact_path_for_mmap(
        paths["doc_embeddings"]
    )
    try:
        doc_embeddings = np.load(
            doc_embeddings_path,
            mmap_mode="r",
            allow_pickle=False,
        )
        normalized = _row_normalize_dense(doc_embeddings)
        artifact = _write_npy_artifact(
            paths["normalized_doc_embeddings"],
            normalized,
        )
        _update_normalized_doc_embeddings_meta(paths, resolved_meta, artifact)
    finally:
        if temp_doc_embeddings is not None and Path(temp_doc_embeddings).exists():
            Path(temp_doc_embeddings).unlink()

    log_runtime_event(
        "svd_index.normalized_embeddings_build_done",
        index_name=index_name,
        shape=list(normalized.shape),
        dtype=str(normalized.dtype),
    )
    return paths["normalized_doc_embeddings"]


def _resolved_vectorizer_params(vectorizer_params=None):
    params = dict(DEFAULT_TFIDF_PARAMS)
    if vectorizer_params:
        params.update(dict(vectorizer_params))
    return params


def _resolved_svd_params(n_components=DEFAULT_SVD_N_COMPONENTS, svd_params=None):
    try:
        resolved_n_components = int(n_components)
    except (TypeError, ValueError) as exc:
        raise ValueError("n_components must be an integer.") from exc

    if resolved_n_components <= 0:
        raise ValueError("n_components must be positive.")

    params = dict(DEFAULT_SVD_PARAMS)
    if svd_params:
        params.update(dict(svd_params))
    params["n_components"] = resolved_n_components
    return params


def _resolve_effective_n_components(requested_n_components, matrix_shape):
    if len(matrix_shape) != 2:
        raise ValueError("matrix_shape must be 2-dimensional.")

    n_rows, n_cols = [int(value) for value in matrix_shape]
    max_rank = min(n_rows, n_cols) - 1
    if max_rank < 1:
        raise ValueError(
            "Truncated SVD requires at least 2 documents and 2 terms after vectorization."
        )

    resolved = min(int(requested_n_components), int(max_rank))
    return max(1, resolved)


class TruncatedSvdIndex:
    def __init__(
        self,
        doc_embeddings,
        components,
        singular_values,
        explained_variance_ratio,
        vectorizer,
        terms,
        doc_ids,
        articles=None,
        id_column="id",
        text_column="body_text",
        explained_variance=None,
        normalized_doc_embeddings=None,
        temp_artifact_paths=None,
        svd_params=None,
        requested_n_components=None,
    ):
        if vectorizer is None:
            raise ValueError("vectorizer cannot be None.")

        self.terms = _normalize_terms(terms)
        self._term_to_idx = None
        self.n_terms = len(self.terms)

        self.doc_ids = _normalize_doc_ids(doc_ids)
        self.doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        self.n_docs = len(self.doc_ids)

        self.doc_embeddings = np.asarray(doc_embeddings, dtype=np.float32)
        self.components = np.asarray(components, dtype=np.float32)
        self.singular_values = np.asarray(singular_values, dtype=np.float32).reshape(-1)
        self.explained_variance_ratio = np.asarray(
            explained_variance_ratio,
            dtype=np.float32,
        ).reshape(-1)
        self.explained_variance = None
        if explained_variance is not None:
            self.explained_variance = np.asarray(
                explained_variance,
                dtype=np.float32,
            ).reshape(-1)

        if self.doc_embeddings.ndim != 2:
            raise ValueError("doc_embeddings must be 2-dimensional.")
        if self.components.ndim != 2:
            raise ValueError("components must be 2-dimensional.")

        n_docs, n_components = self.doc_embeddings.shape
        component_rows, n_terms = self.components.shape
        if n_docs != self.n_docs:
            raise ValueError(
                f"doc_embeddings row count ({n_docs}) does not match doc_ids count ({self.n_docs})."
            )
        if component_rows != n_components:
            raise ValueError(
                "components row count must match doc_embeddings column count."
            )
        if n_terms != self.n_terms:
            raise ValueError(
                f"components column count ({n_terms}) does not match terms count ({self.n_terms})."
            )
        if len(self.singular_values) != n_components:
            raise ValueError(
                "singular_values length must match embedding dimension count."
            )
        if len(self.explained_variance_ratio) != n_components:
            raise ValueError(
                "explained_variance_ratio length must match embedding dimension count."
            )
        if (
            self.explained_variance is not None
            and len(self.explained_variance) != n_components
        ):
            raise ValueError(
                "explained_variance length must match embedding dimension count."
            )
        if normalized_doc_embeddings is not None:
            normalized = np.asarray(normalized_doc_embeddings)
            if normalized.shape != self.doc_embeddings.shape:
                raise ValueError(
                    "normalized_doc_embeddings shape must match doc_embeddings shape."
                )
            self._normalized_doc_embeddings = (
                normalized
                if normalized.dtype == DEFAULT_SVD_NORMALIZED_DTYPE
                else normalized.astype(DEFAULT_SVD_NORMALIZED_DTYPE, copy=False)
            )
        else:
            self._normalized_doc_embeddings = None

        if hasattr(vectorizer, "get_feature_names_out"):
            vectorizer_terms = vectorizer.get_feature_names_out()
            if len(vectorizer_terms) != self.n_terms:
                raise ValueError(
                    "Vectorizer vocabulary size does not match provided terms."
                )

        self.vectorizer = vectorizer
        self.n_components = int(n_components)
        self.requested_n_components = int(requested_n_components or self.n_components)
        self.svd_params = dict(svd_params or {})
        self.id_column = id_column
        self.text_column = text_column

        self._temp_artifact_paths = [Path(path) for path in list(temp_artifact_paths or [])]
        self._temp_artifact_finalizer = weakref.finalize(
            self,
            _cleanup_temp_paths,
            list(self._temp_artifact_paths),
        )

        self._top_term_indices_cache = {}

        self.articles = None
        if articles is not None:
            self.articles = _normalize_articles_for_doc_ids(
                articles=articles,
                doc_ids=self.doc_ids,
                id_column=id_column,
            )

    @property
    def term_to_idx(self):
        if self._term_to_idx is None:
            self._term_to_idx = {
                term: idx
                for idx, term in enumerate(self.terms)
            }
        return self._term_to_idx

    @property
    def normalized_doc_embeddings(self):
        if self._normalized_doc_embeddings is None:
            log_runtime_event(
                "svd_index.normalize_doc_embeddings_start",
                n_docs=self.n_docs,
                n_components=self.n_components,
            )
            self._normalized_doc_embeddings = _row_normalize_dense(self.doc_embeddings)
            log_runtime_event(
                "svd_index.normalize_doc_embeddings_done",
                n_docs=self.n_docs,
                n_components=self.n_components,
                dtype=str(self._normalized_doc_embeddings.dtype),
            )
        return self._normalized_doc_embeddings

    @classmethod
    def from_articles(
        cls,
        articles,
        n_components=DEFAULT_SVD_N_COMPONENTS,
        vectorizer=None,
        vectorizer_params=None,
        svd_params=None,
        text_column="body_text",
        id_column="id",
        include_text_in_articles=False,
    ):
        if not isinstance(articles, pd.DataFrame):
            raise TypeError("articles must be a pandas DataFrame.")
        if text_column not in articles.columns:
            raise ValueError(f"Column '{text_column}' not found in articles.")
        if id_column not in articles.columns:
            raise ValueError(f"Column '{id_column}' not found in articles.")

        normalized = articles.reset_index(drop=True).copy()
        text_series = normalized[text_column].astype("string").fillna("")
        text_values = [
            normalize_text_for_vectorization(value)
            for value in text_series.tolist()
        ]
        if not any(text_values):
            raise ValueError(f"Column '{text_column}' contains no non-empty text.")

        id_series = _normalize_id_series(normalized[id_column], id_column)
        normalized[id_column] = id_series

        resolved_vectorizer_params = _resolved_vectorizer_params(vectorizer_params)
        if vectorizer is None:
            vectorizer = TfidfVectorizer(**resolved_vectorizer_params)

        term_doc_matrix = vectorizer.fit_transform(text_values)
        terms = vectorizer.get_feature_names_out().tolist()

        resolved_svd_params = _resolved_svd_params(
            n_components=n_components,
            svd_params=svd_params,
        )
        requested_n_components = int(resolved_svd_params["n_components"])
        effective_n_components = _resolve_effective_n_components(
            requested_n_components,
            term_doc_matrix.shape,
        )
        if effective_n_components != requested_n_components:
            log_runtime_event(
                "svd_index.n_components_adjusted",
                requested_n_components=requested_n_components,
                effective_n_components=effective_n_components,
                n_docs=int(term_doc_matrix.shape[0]),
                n_terms=int(term_doc_matrix.shape[1]),
            )
            resolved_svd_params["n_components"] = effective_n_components

        reducer = TruncatedSVD(**resolved_svd_params)
        log_runtime_event(
            "svd_index.fit_start",
            requested_n_components=requested_n_components,
            effective_n_components=effective_n_components,
            n_docs=int(term_doc_matrix.shape[0]),
            n_terms=int(term_doc_matrix.shape[1]),
        )
        doc_embeddings = reducer.fit_transform(term_doc_matrix)
        log_runtime_event(
            "svd_index.fit_done",
            effective_n_components=effective_n_components,
            explained_variance_ratio_sum=float(
                np.asarray(reducer.explained_variance_ratio_).sum()
            ),
        )

        if include_text_in_articles:
            articles_payload = normalized.copy()
        else:
            articles_payload = normalized.drop(columns=[text_column], errors="ignore")

        resolved_explained_variance = getattr(reducer, "explained_variance_", None)

        return cls(
            doc_embeddings=np.asarray(doc_embeddings, dtype=np.float32),
            components=np.asarray(reducer.components_, dtype=np.float32),
            singular_values=np.asarray(reducer.singular_values_, dtype=np.float32),
            explained_variance_ratio=np.asarray(
                reducer.explained_variance_ratio_,
                dtype=np.float32,
            ),
            explained_variance=None
            if resolved_explained_variance is None
            else np.asarray(resolved_explained_variance, dtype=np.float32),
            vectorizer=vectorizer,
            terms=terms,
            doc_ids=id_series.tolist(),
            articles=articles_payload,
            id_column=id_column,
            text_column=text_column,
            svd_params=dict(resolved_svd_params),
            requested_n_components=requested_n_components,
        )

    def get_doc_idx_by_id(self, doc_id):
        key = str(doc_id).strip()
        idx = self.doc_to_idx.get(key)
        if idx is None:
            raise ValueError(f"Document ID {doc_id} not found.")
        return idx

    def get_doc_vector(self, doc_id, normalize=False):
        idx = self.get_doc_idx_by_id(doc_id)
        if normalize:
            return self.normalized_doc_embeddings[idx]
        return self.doc_embeddings[idx]

    def project_query(self, query, normalize=False):
        query_text = normalize_text_for_vectorization(query)
        if not query_text:
            return None

        query_vec = self.vectorizer.transform([query_text])
        if int(getattr(query_vec, "nnz", 0)) <= 0:
            return None

        projected = query_vec @ self.components.T
        projected = np.asarray(projected, dtype=np.float32).reshape(-1)
        if normalize:
            return _normalize_dense_vector(projected)
        return projected

    def search(self, query, top_n=100, return_articles=True):
        if top_n is None:
            top_n = self.n_docs
        top_n = int(top_n)
        if top_n <= 0:
            return []

        log_runtime_event(
            "svd_search.query_start",
            query_chars=len(str(query or "")),
            top_n=top_n,
            n_docs=self.n_docs,
            n_terms=self.n_terms,
            n_components=self.n_components,
        )
        query_embedding = self.project_query(query, normalize=True)
        if query_embedding is None:
            log_runtime_event("svd_search.empty_query_projection")
            return []

        log_runtime_event("svd_search.topk_select_start", top_n=top_n)
        candidate_doc_indices, candidate_scores = top_positive_dot_candidates(
            self.normalized_doc_embeddings,
            query_embedding,
            top_n=top_n,
        )
        if candidate_doc_indices.size == 0:
            log_runtime_event("svd_search.no_candidates")
            return []

        log_runtime_event(
            "svd_search.topk_select_done",
            selected_count=int(candidate_doc_indices.size),
        )

        results = []
        for idx, score in zip(candidate_doc_indices, candidate_scores):
            idx = int(idx)
            doc_id = self.doc_ids[idx]
            if return_articles and self.articles is not None:
                payload = self.articles.iloc[idx].to_dict()
            else:
                payload = doc_id
            results.append((payload, float(score)))

        log_runtime_event("svd_search.results_done", result_count=len(results))
        return results

    def diagnostics(self):
        explained = np.asarray(self.explained_variance_ratio, dtype=np.float64)
        cumulative = np.cumsum(explained)
        return {
            "n_docs": int(self.n_docs),
            "n_terms": int(self.n_terms),
            "n_components": int(self.n_components),
            "requested_n_components": int(self.requested_n_components),
            "singular_values": [float(value) for value in self.singular_values.tolist()],
            "explained_variance_ratio": [float(value) for value in explained.tolist()],
            "cumulative_explained_variance_ratio": [
                float(value) for value in cumulative.tolist()
            ],
            "explained_variance_ratio_sum": float(explained.sum()),
        }

    def top_terms_for_dimension(self, dimension, top_n=10, order="positive"):
        dim = int(dimension)
        if dim < 0 or dim >= self.n_components:
            raise ValueError(
                f"dimension must be between 0 and {self.n_components - 1}."
            )

        top_n = max(1, int(top_n))
        weights = np.asarray(self.components[dim], dtype=np.float32)
        selected = self._top_term_indices(
            dim,
            top_n=top_n,
            order=order,
            weights=weights,
        )

        return [(self.terms[int(idx)], float(weights[int(idx)])) for idx in selected]

    def _top_term_indices(self, dimension, top_n, order, weights):
        if order not in {"positive", "negative", "absolute"}:
            raise ValueError("order must be 'positive', 'negative', or 'absolute'.")

        dim = int(dimension)
        resolved_top_n = min(max(1, int(top_n)), int(weights.size))
        cache_key = (dim, resolved_top_n, order)
        cached = self._top_term_indices_cache.get(cache_key)
        if cached is not None:
            return cached

        if resolved_top_n >= weights.size:
            if order == "positive":
                selected = np.argsort(weights)[::-1]
            elif order == "negative":
                selected = np.argsort(weights)
            else:
                selected = np.argsort(np.abs(weights))[::-1]
        elif order == "positive":
            selected = np.argpartition(weights, -resolved_top_n)[-resolved_top_n:]
            selected = selected[np.argsort(weights[selected])[::-1]]
        elif order == "negative":
            selected = np.argpartition(weights, resolved_top_n - 1)[:resolved_top_n]
            selected = selected[np.argsort(weights[selected])]
        else:
            abs_weights = np.abs(weights)
            selected = np.argpartition(abs_weights, -resolved_top_n)[-resolved_top_n:]
            selected = selected[np.argsort(abs_weights[selected])[::-1]]

        cached_selected = tuple(int(idx) for idx in selected)
        self._top_term_indices_cache[cache_key] = cached_selected
        return cached_selected

    def dimension_summary(self, dimension, top_n=10):
        return {
            "dimension": int(dimension),
            "positive_terms": self.top_terms_for_dimension(
                dimension=dimension,
                top_n=top_n,
                order="positive",
            ),
            "negative_terms": self.top_terms_for_dimension(
                dimension=dimension,
                top_n=top_n,
                order="negative",
            ),
            "absolute_terms": self.top_terms_for_dimension(
                dimension=dimension,
                top_n=top_n,
                order="absolute",
            ),
        }

    @staticmethod
    def format_term_weights(term_weights, precision=3):
        resolved_precision = max(0, int(precision))
        return ", ".join(
            f"{term} ({weight:.{resolved_precision}f})"
            for term, weight in term_weights
        )

    def dimension_summary_record(
        self,
        dimension,
        top_n=10,
        format_terms=True,
        precision=3,
    ):
        summary = self.dimension_summary(dimension=dimension, top_n=top_n)
        record = {
            "dimension_index": int(summary["dimension"]),
            "dimension_label": int(summary["dimension"]) + 1,
        }
        for key in ("positive_terms", "negative_terms", "absolute_terms"):
            values = summary[key]
            record[key] = (
                self.format_term_weights(values, precision=precision)
                if format_terms
                else values
            )
        return record

    def dimension_summary_frame(
        self,
        dimensions=None,
        top_n=10,
        format_terms=True,
        precision=3,
    ):
        if dimensions is None:
            dimensions = range(self.n_components)
        rows = [
            self.dimension_summary_record(
                dimension=dim,
                top_n=top_n,
                format_terms=format_terms,
                precision=precision,
            )
            for dim in dimensions
        ]
        return pd.DataFrame(rows)

    def export_dimension_summaries(
        self,
        output_path,
        dimensions=None,
        top_n=10,
        format_terms=True,
        precision=3,
    ):
        resolved_output_path = Path(output_path)
        resolved_output_path.parent.mkdir(parents=True, exist_ok=True)
        df = self.dimension_summary_frame(
            dimensions=dimensions,
            top_n=top_n,
            format_terms=format_terms,
            precision=precision,
        )

        suffix = resolved_output_path.suffix.lower()
        if suffix == ".csv":
            df.to_csv(resolved_output_path, index=False)
        elif suffix == ".json":
            df.to_json(resolved_output_path, orient="records", indent=2)
        elif suffix in {".jsonl", ".ndjson"}:
            df.to_json(resolved_output_path, orient="records", lines=True)
        else:
            raise ValueError(
                "output_path must end in .csv, .json, .jsonl, or .ndjson"
            )

        return resolved_output_path, df

    def top_dimensions_for_query(self, query, top_n=5, normalize=False):
        vector = self.project_query(query, normalize=normalize)
        if vector is None:
            return []
        return self._top_dimensions_for_vector(vector, top_n=top_n)

    def top_dimensions_for_doc(self, doc_id, top_n=5, normalize=False):
        vector = self.get_doc_vector(doc_id, normalize=normalize)
        return self._top_dimensions_for_vector(vector, top_n=top_n)

    @staticmethod
    def _top_dimensions_for_vector(vector, top_n=5):
        values = np.asarray(vector, dtype=np.float32).reshape(-1)
        if values.size == 0:
            return []
        top_n = min(max(1, int(top_n)), int(values.size))
        selected = np.argsort(np.abs(values))[::-1][:top_n]
        return [(int(idx), float(values[int(idx)])) for idx in selected]

    @staticmethod
    def artifact_paths(index_dir, index_name):
        directory = Path(index_dir)
        return {
            "vectorizer": directory / f"{index_name}_vectorizer.pkl",
            "terms": directory / f"{index_name}_terms.json",
            "doc_ids": directory / f"{index_name}_doc_ids.json",
            "articles": directory / f"{index_name}_articles.pkl",
            "doc_embeddings": directory / f"{index_name}_doc_embeddings.npy",
            "normalized_doc_embeddings": directory
            / f"{index_name}_normalized_doc_embeddings.npy",
            "components": directory / f"{index_name}_components.npy",
            "singular_values": directory / f"{index_name}_singular_values.npy",
            "explained_variance_ratio": directory
            / f"{index_name}_explained_variance_ratio.npy",
            "explained_variance": directory / f"{index_name}_explained_variance.npy",
            "meta": directory / f"{index_name}_meta.json",
        }

    @classmethod
    def has_artifacts(cls, index_dir, index_name):
        paths = cls.artifact_paths(index_dir, index_name)
        required = [
            paths["vectorizer"],
            paths["terms"],
            paths["doc_ids"],
            paths["doc_embeddings"],
            paths["normalized_doc_embeddings"],
            paths["components"],
            paths["singular_values"],
            paths["explained_variance_ratio"],
            paths["meta"],
        ]
        return all(_artifact_exists(path) for path in required)

    @classmethod
    def artifacts_within_size_limit(cls, index_dir, index_name):
        paths = cls.artifact_paths(index_dir, index_name)
        required = [
            paths["vectorizer"],
            paths["terms"],
            paths["doc_ids"],
            paths["doc_embeddings"],
            paths["components"],
            paths["singular_values"],
            paths["explained_variance_ratio"],
            paths["meta"],
        ]
        if _artifact_exists(paths["articles"]):
            required.append(paths["articles"])
        if _artifact_exists(paths["normalized_doc_embeddings"]):
            required.append(paths["normalized_doc_embeddings"])
        if _artifact_exists(paths["explained_variance"]):
            required.append(paths["explained_variance"])
        return all(_artifact_within_size_limit(path) for path in required)

    @classmethod
    def artifact_infos(cls, index_dir, index_name):
        paths = cls.artifact_paths(index_dir, index_name)
        infos = {}
        for key, path in paths.items():
            files = _artifact_files(path)
            infos[key] = {
                "path": path,
                "files": files,
                "storage": "chunked" if len(files) > 1 else "single_file",
            }
        return infos

    def save(self, index_dir, index_name, extra_meta=None):
        paths = self.artifact_paths(index_dir, index_name)
        Path(index_dir).mkdir(parents=True, exist_ok=True)

        vectorizer_artifact = _write_pickle_artifact(paths["vectorizer"], self.vectorizer)
        terms_artifact = _write_json_artifact(paths["terms"], self.terms)
        doc_ids_artifact = _write_json_artifact(paths["doc_ids"], self.doc_ids)
        doc_embeddings_artifact = _write_npy_artifact(
            paths["doc_embeddings"],
            np.asarray(self.doc_embeddings, dtype=np.float32),
        )
        normalized_doc_embeddings_artifact = _write_npy_artifact(
            paths["normalized_doc_embeddings"],
            np.asarray(
                self.normalized_doc_embeddings,
                dtype=DEFAULT_SVD_NORMALIZED_DTYPE,
            ),
        )
        components_artifact = _write_npy_artifact(
            paths["components"],
            np.asarray(self.components, dtype=np.float32),
        )
        singular_values_artifact = _write_npy_artifact(
            paths["singular_values"],
            np.asarray(self.singular_values, dtype=np.float32),
        )
        explained_variance_ratio_artifact = _write_npy_artifact(
            paths["explained_variance_ratio"],
            np.asarray(self.explained_variance_ratio, dtype=np.float32),
        )

        explained_variance_artifact = None
        if self.explained_variance is not None:
            explained_variance_artifact = _write_npy_artifact(
                paths["explained_variance"],
                np.asarray(self.explained_variance, dtype=np.float32),
            )
        else:
            _unlink_paths(_artifact_files(paths["explained_variance"]))

        articles_artifact = None
        if self.articles is not None:
            articles_artifact = _write_dataframe_pickle_artifact(
                paths["articles"],
                self.articles,
            )
        else:
            _unlink_paths(_artifact_files(paths["articles"]))

        meta = {
            "index_name": index_name,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "search_backend": "svd",
            "n_docs": int(self.n_docs),
            "n_terms": int(self.n_terms),
            "n_components": int(self.n_components),
            "requested_n_components": int(self.requested_n_components),
            "id_column": self.id_column,
            "text_column": self.text_column,
            "has_articles": self.articles is not None,
            "vectorizer_class": self.vectorizer.__class__.__name__,
            "reducer_class": "TruncatedSVD",
            "svd_params": dict(self.svd_params),
            "vectorizer_files": [path.name for path in vectorizer_artifact["files"]],
            "terms_files": [path.name for path in terms_artifact["files"]],
            "doc_ids_files": [path.name for path in doc_ids_artifact["files"]],
            "doc_embeddings_files": [
                path.name for path in doc_embeddings_artifact["files"]
            ],
            "normalized_doc_embeddings_dtype": str(
                np.dtype(DEFAULT_SVD_NORMALIZED_DTYPE)
            ),
            "normalized_doc_embeddings_files": [
                path.name for path in normalized_doc_embeddings_artifact["files"]
            ],
            "components_files": [path.name for path in components_artifact["files"]],
            "singular_values_files": [
                path.name for path in singular_values_artifact["files"]
            ],
            "explained_variance_ratio_files": [
                path.name for path in explained_variance_ratio_artifact["files"]
            ],
            "explained_variance_files": []
            if explained_variance_artifact is None
            else [path.name for path in explained_variance_artifact["files"]],
            "articles_files": []
            if articles_artifact is None
            else [path.name for path in articles_artifact["files"]],
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_artifact = _write_json_artifact(paths["meta"], meta)

        returned_paths = dict(paths)
        returned_paths["vectorizer_files"] = vectorizer_artifact["files"]
        returned_paths["terms_files"] = terms_artifact["files"]
        returned_paths["doc_ids_files"] = doc_ids_artifact["files"]
        returned_paths["doc_embeddings_files"] = doc_embeddings_artifact["files"]
        returned_paths["normalized_doc_embeddings_files"] = (
            normalized_doc_embeddings_artifact["files"]
        )
        returned_paths["components_files"] = components_artifact["files"]
        returned_paths["singular_values_files"] = singular_values_artifact["files"]
        returned_paths["explained_variance_ratio_files"] = (
            explained_variance_ratio_artifact["files"]
        )
        returned_paths["explained_variance_files"] = (
            []
            if explained_variance_artifact is None
            else explained_variance_artifact["files"]
        )
        returned_paths["articles_files"] = (
            [] if articles_artifact is None else articles_artifact["files"]
        )
        returned_paths["meta_files"] = meta_artifact["files"]
        return returned_paths

    @classmethod
    def load(cls, index_dir, index_name, load_articles=True):
        paths = cls.artifact_paths(index_dir, index_name)
        required = [
            paths["vectorizer"],
            paths["terms"],
            paths["doc_ids"],
            paths["doc_embeddings"],
            paths["components"],
            paths["singular_values"],
            paths["explained_variance_ratio"],
            paths["meta"],
        ]
        missing = [str(path) for path in required if not _artifact_exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing required SVD index artifacts: {', '.join(missing)}"
            )

        log_runtime_event("svd_index.load_start", index_name=index_name)
        with _materialized_artifact_path(paths["vectorizer"]) as vectorizer_path:
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)

        with _materialized_artifact_path(paths["terms"]) as terms_path:
            with open(terms_path, "r", encoding="utf-8") as f:
                terms = json.load(f)

        with _materialized_artifact_path(paths["doc_ids"]) as doc_ids_path:
            with open(doc_ids_path, "r", encoding="utf-8") as f:
                doc_ids = json.load(f)

        temp_artifact_paths = []
        try:
            doc_embeddings_path, temp_doc_embeddings = _materialized_artifact_path_for_mmap(
                paths["doc_embeddings"]
            )
            if temp_doc_embeddings is not None:
                temp_artifact_paths.append(temp_doc_embeddings)
            doc_embeddings = np.load(
                doc_embeddings_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            normalized_doc_embeddings = None
            if _normalized_doc_embeddings_fresh(
                paths["normalized_doc_embeddings"],
                expected_shape=doc_embeddings.shape,
            ):
                (
                    normalized_doc_embeddings_path,
                    temp_normalized_doc_embeddings,
                ) = _materialized_artifact_path_for_mmap(
                    paths["normalized_doc_embeddings"]
                )
                if temp_normalized_doc_embeddings is not None:
                    temp_artifact_paths.append(temp_normalized_doc_embeddings)
                normalized_doc_embeddings = np.load(
                    normalized_doc_embeddings_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )

            components_path, temp_components = _materialized_artifact_path_for_mmap(
                paths["components"]
            )
            if temp_components is not None:
                temp_artifact_paths.append(temp_components)
            components = np.load(
                components_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            singular_values_path, temp_singular_values = _materialized_artifact_path_for_mmap(
                paths["singular_values"]
            )
            if temp_singular_values is not None:
                temp_artifact_paths.append(temp_singular_values)
            singular_values = np.load(
                singular_values_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            explained_variance_ratio_path, temp_explained_variance_ratio = (
                _materialized_artifact_path_for_mmap(paths["explained_variance_ratio"])
            )
            if temp_explained_variance_ratio is not None:
                temp_artifact_paths.append(temp_explained_variance_ratio)
            explained_variance_ratio = np.load(
                explained_variance_ratio_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            explained_variance = None
            if _artifact_exists(paths["explained_variance"]):
                explained_variance_path, temp_explained_variance = (
                    _materialized_artifact_path_for_mmap(paths["explained_variance"])
                )
                if temp_explained_variance is not None:
                    temp_artifact_paths.append(temp_explained_variance)
                explained_variance = np.load(
                    explained_variance_path,
                    mmap_mode="r",
                    allow_pickle=False,
                )

            with _materialized_artifact_path(paths["meta"]) as meta_path:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}

            articles = None
            if load_articles and _artifact_exists(paths["articles"]):
                with _materialized_artifact_path(paths["articles"]) as articles_path:
                    articles = pd.read_pickle(articles_path)

            instance = cls(
                doc_embeddings=doc_embeddings,
                components=components,
                singular_values=singular_values,
                explained_variance_ratio=explained_variance_ratio,
                explained_variance=explained_variance,
                normalized_doc_embeddings=normalized_doc_embeddings,
                vectorizer=vectorizer,
                terms=terms,
                doc_ids=doc_ids,
                articles=articles,
                id_column=meta.get("id_column", "id"),
                text_column=meta.get("text_column", "body_text"),
                temp_artifact_paths=temp_artifact_paths,
                svd_params=meta.get("svd_params"),
                requested_n_components=meta.get("requested_n_components"),
            )
        except Exception:
            _cleanup_temp_paths(temp_artifact_paths)
            raise

        log_runtime_event(
            "svd_index.load_done",
            index_name=index_name,
            n_docs=instance.n_docs,
            n_terms=instance.n_terms,
            n_components=instance.n_components,
            has_normalized_doc_embeddings=(
                instance._normalized_doc_embeddings is not None
            ),
            normalized_doc_embeddings_dtype=(
                str(instance._normalized_doc_embeddings.dtype)
                if instance._normalized_doc_embeddings is not None
                else None
            ),
        )
        return instance, meta


def _is_existing_svd_index_fresh(
    index_dir,
    index_name,
    db_row_count,
    expected_years=None,
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

    if meta.get("search_backend") != "svd":
        return False

    if meta.get("vectorizer_params") != _resolved_vectorizer_params(expected_vectorizer_params):
        return False
    if meta.get("text_normalization_version") != TEXT_NORMALIZATION_VERSION:
        return False

    expected_svd = _resolved_svd_params(
        n_components=expected_n_components,
        svd_params=expected_svd_params,
    )
    stored_svd = dict(meta.get("svd_params") or {})
    stored_effective_n_components = stored_svd.pop("n_components", None)
    if stored_effective_n_components is None:
        stored_effective_n_components = meta.get("n_components", -1)
    if int(stored_effective_n_components) != int(meta.get("n_components", -1)):
        return False
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

    required_paths = [
        paths["vectorizer"],
        paths["terms"],
        paths["doc_ids"],
        paths["doc_embeddings"],
        paths["components"],
        paths["singular_values"],
        paths["explained_variance_ratio"],
        paths["meta"],
    ]
    if meta.get("has_articles"):
        required_paths.append(paths["articles"])
    if meta.get("explained_variance_files"):
        required_paths.append(paths["explained_variance"])
    if not all(_artifact_exists(path) for path in required_paths):
        return False

    if not all(_artifact_within_size_limit(path) for path in required_paths):
        return False

    return stored_count == int(db_row_count)


def preprocess_svd_index(
    db_path=DEFAULT_DB_PATH,
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_SVD_INDEX_NAME,
    force_rebuild=False,
    years=None,
    n_components=DEFAULT_SVD_N_COMPONENTS,
    vectorizer_params=None,
    svd_params=None,
):
    db_path = Path(db_path)
    index_dir = Path(index_dir)
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
        "svd_preprocess.start",
        db_path=str(db_path),
        index_name=index_name,
        db_row_count=db_row_count,
        force_rebuild=bool(force_rebuild),
        requested_years=normalized_years,
        requested_n_components=int(resolved_svd_params["n_components"]),
    )
    if not force_rebuild and _is_existing_svd_index_fresh(
        index_dir=index_dir,
        index_name=index_name,
        db_row_count=db_row_count,
        expected_years=normalized_years,
        expected_n_components=int(resolved_svd_params["n_components"]),
        expected_vectorizer_params=resolved_vectorizer_params,
        expected_svd_params=resolved_svd_params,
    ):
        log_runtime_event(
            "svd_preprocess.up_to_date",
            index_name=index_name,
            db_row_count=db_row_count,
        )
        normalized_path = ensure_normalized_doc_embeddings_artifact(
            index_dir=index_dir,
            index_name=index_name,
        )
        return {
            "built": False,
            "reason": "up_to_date",
            "db_row_count": db_row_count,
            "index_dir": str(index_dir),
            "index_name": index_name,
            "normalized_doc_embeddings_path": str(normalized_path),
        }

    source_kind = "sqlite"
    if _db_has_complete_body_text(db_path):
        articles = _load_guardian_articles_from_sqlite(db_path)
        articles = _filter_articles_to_years(articles, years=normalized_years)
    else:
        source_years = normalized_years or _db_years(db_path)
        articles = _load_guardian_articles_from_raw(years=source_years)
        source_kind = "raw_csv"
    source_years = normalized_years or sorted(
        {
            int(year)
            for year in pd.to_numeric(articles.get("year"), errors="coerce").dropna().tolist()
        }
    )
    log_runtime_event(
        "svd_preprocess.source_ready",
        source_kind=source_kind,
        article_count=int(len(articles)),
        source_years=source_years,
    )

    if articles.empty:
        raise ValueError("No guardian_articles rows found; cannot build SVD index.")

    vectorizer = TfidfVectorizer(**resolved_vectorizer_params)
    svd_index = TruncatedSvdIndex.from_articles(
        articles=articles,
        n_components=int(resolved_svd_params["n_components"]),
        vectorizer=vectorizer,
        vectorizer_params=resolved_vectorizer_params,
        svd_params=resolved_svd_params,
        text_column="body_text",
        id_column="id",
        include_text_in_articles=False,
    )
    paths = svd_index.save(
        index_dir=index_dir,
        index_name=index_name,
        extra_meta={
            "db_row_count": int(db_row_count),
            "source_db_path": _relative_db_path_for_meta(db_path),
            "text_source": source_kind,
            "source_years": source_years,
            "text_normalization_version": TEXT_NORMALIZATION_VERSION,
            "vectorizer_params": resolved_vectorizer_params,
        },
    )

    return {
        "built": True,
        "db_row_count": db_row_count,
        "index_dir": str(index_dir),
        "index_name": index_name,
        "n_components": int(svd_index.n_components),
        "requested_n_components": int(svd_index.requested_n_components),
        "paths": {
            key: [str(path) for path in value]
            if key.endswith("_files")
            else str(value)
            for key, value in paths.items()
        },
    }


def load_svd_index(
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_SVD_INDEX_NAME,
    load_articles=False,
):
    return TruncatedSvdIndex.load(
        index_dir=index_dir,
        index_name=index_name,
        load_articles=load_articles,
    )


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute truncated-SVD index artifacts for Guardian article search."
    )
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--index-name", default=DEFAULT_SVD_INDEX_NAME)
    parser.add_argument(
        "--n-components",
        type=int,
        default=DEFAULT_SVD_N_COMPONENTS,
    )
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = preprocess_svd_index(
        db_path=args.db_path,
        index_dir=args.index_dir,
        index_name=args.index_name,
        n_components=args.n_components,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, indent=2))
