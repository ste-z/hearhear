import json
import pickle
import tempfile
import weakref
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.indexing.artifacts import (
    _artifact_chunk_paths,
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
    _repartition_existing_artifact,
)
from backend.text_processing.indexing.normalization import (
    _normalize_articles_for_doc_ids,
    _normalize_doc_ids,
    _normalize_id_series,
    _normalize_terms,
)
from backend.text_processing.indexing.settings import (
    DEFAULT_TFIDF_PARAMS,
    MAX_TERM_DOC_MATRIX_CHUNK_BYTES,
    MAX_VECTOR_INDEX_ARTIFACT_BYTES,
)


class TfidfMatrixIndex:
    def __init__(
        self,
        term_doc_matrix,
        terms,
        doc_ids,
        vectorizer,
        articles=None,
        id_column="id",
        text_column="body_text",
    ):
        if vectorizer is None:
            raise ValueError("vectorizer cannot be None.")
        if term_doc_matrix is None or not hasattr(term_doc_matrix, "shape"):
            raise ValueError("term_doc_matrix must be matrix-like with a shape.")
        if len(term_doc_matrix.shape) != 2:
            raise ValueError("term_doc_matrix must be 2-dimensional.")

        normalized_terms = self._normalize_terms(terms)
        normalized_doc_ids = self._normalize_doc_ids(doc_ids)

        n_docs, n_terms = term_doc_matrix.shape
        if n_docs != len(normalized_doc_ids):
            raise ValueError(
                f"Matrix row count ({n_docs}) does not match doc_ids count ({len(normalized_doc_ids)})."
            )
        if n_terms != len(normalized_terms):
            raise ValueError(
                f"Matrix column count ({n_terms}) does not match terms count ({len(normalized_terms)})."
            )

        if hasattr(vectorizer, "get_feature_names_out"):
            vectorizer_terms = vectorizer.get_feature_names_out()
            if len(vectorizer_terms) != len(normalized_terms):
                raise ValueError(
                    "Vectorizer vocabulary size does not match provided terms."
                )

        self.term_doc_matrix = term_doc_matrix
        self.terms = normalized_terms
        self.term_to_idx = {term: idx for idx, term in enumerate(self.terms)}
        self.n_terms = len(self.terms)

        self.doc_ids = normalized_doc_ids
        self.doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        self.n_docs = len(self.doc_ids)

        self.vectorizer = vectorizer
        self.id_column = id_column
        self.text_column = text_column

        self.articles = None
        if articles is not None:
            self.articles = self._normalize_articles(articles)

    @staticmethod
    def _normalize_terms(terms):
        return _normalize_terms(terms)

    @staticmethod
    def _normalize_doc_ids(doc_ids):
        return _normalize_doc_ids(doc_ids)

    @staticmethod
    def _normalize_id_series(id_values, field_name):
        return _normalize_id_series(id_values, field_name)

    def _normalize_articles(self, articles):
        return _normalize_articles_for_doc_ids(
            articles=articles,
            doc_ids=self.doc_ids,
            id_column=self.id_column,
        )

    @classmethod
    def from_articles(
        cls,
        articles,
        vectorizer=None,
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

        text_series = normalized[text_column].astype("string").fillna("").str.strip()
        if (text_series == "").all():
            raise ValueError(f"Column '{text_column}' contains no non-empty text.")

        id_series = cls._normalize_id_series(normalized[id_column], id_column)
        normalized[id_column] = id_series

        if vectorizer is None:
            vectorizer = TfidfVectorizer(**DEFAULT_TFIDF_PARAMS)
        term_doc_matrix = vectorizer.fit_transform(text_series.tolist())
        terms = vectorizer.get_feature_names_out().tolist()
        doc_ids = id_series.tolist()

        if include_text_in_articles:
            articles_payload = normalized.copy()
        else:
            articles_payload = normalized.drop(columns=[text_column], errors="ignore")

        return cls(
            term_doc_matrix=term_doc_matrix,
            terms=terms,
            doc_ids=doc_ids,
            vectorizer=vectorizer,
            articles=articles_payload,
            id_column=id_column,
            text_column=text_column,
        )

    def get_term_doc_matrix(self):
        return self.term_doc_matrix

    def get_doc_idx_by_id(self, doc_id):
        key = str(doc_id).strip()
        idx = self.doc_to_idx.get(key)
        if idx is None:
            raise ValueError(f"Document ID {doc_id} not found.")
        return idx

    def get_doc_vector(self, doc_id):
        idx = self.get_doc_idx_by_id(doc_id)
        return self.term_doc_matrix[idx]

    def get_term_vector(self, term):
        if term is None:
            raise ValueError("Term must be non-empty.")
        term_key = str(term).strip()
        if getattr(self.vectorizer, "lowercase", False):
            term_key = term_key.lower()
        term_idx = self.term_to_idx.get(term_key)
        if term_idx is None:
            raise ValueError(f"Term '{term}' not found in vocabulary.")
        return self.term_doc_matrix[:, term_idx]

    def search(self, query, top_n=100, return_articles=True):
        if top_n is None:
            top_n = self.n_docs
        top_n = int(top_n)
        if top_n <= 0:
            return []

        log_runtime_event(
            "vector_search.query_vector_start",
            query_chars=len(str(query or "")),
            top_n=top_n,
            n_docs=self.n_docs,
            n_terms=self.n_terms,
        )
        query_vec = self.vectorizer.transform([str(query)])
        log_runtime_event("vector_search.query_vector_done")
        log_runtime_event("vector_search.sparse_dot_start")
        score_matrix = query_vec @ self.term_doc_matrix.T
        nonzero_count = int(getattr(score_matrix, "nnz", 0))
        log_runtime_event(
            "vector_search.sparse_dot_done",
            nonzero_count=nonzero_count,
        )
        if nonzero_count <= 0:
            log_runtime_event("vector_search.no_nonzero_scores")
            return []

        score_coo = score_matrix.tocoo()
        score_values = score_coo.data
        doc_indices = score_coo.col

        top_n = min(top_n, int(score_values.shape[0]))
        log_runtime_event("vector_search.topk_select_start", candidate_count=int(score_values.shape[0]))
        if score_values.shape[0] > top_n:
            top_positions = np.argpartition(score_values, -top_n)[-top_n:]
            sorted_positions = top_positions[np.argsort(score_values[top_positions])[::-1]]
        else:
            sorted_positions = np.argsort(score_values)[::-1]
        log_runtime_event("vector_search.topk_select_done", selected_count=int(len(sorted_positions)))

        results = []
        for pos in sorted_positions:
            idx = int(doc_indices[pos])
            score = float(score_values[pos])
            if score <= 0:
                continue

            doc_id = self.doc_ids[idx]
            if return_articles and self.articles is not None:
                payload = self.articles.iloc[idx].to_dict()
            else:
                payload = doc_id
            results.append((payload, score))

        log_runtime_event("vector_search.results_done", result_count=len(results))
        return results

    @staticmethod
    def artifact_paths(index_dir, index_name):
        directory = Path(index_dir)
        return {
            "vectorizer": directory / f"{index_name}_vectorizer.pkl",
            "term_doc_matrix": directory / f"{index_name}_term_doc_matrix.npz",
            "terms": directory / f"{index_name}_terms.json",
            "doc_ids": directory / f"{index_name}_doc_ids.json",
            "articles": directory / f"{index_name}_articles.pkl",
            "meta": directory / f"{index_name}_meta.json",
            "postings_data": directory / f"{index_name}_postings_data.npy",
            "postings_doc_indices": directory / f"{index_name}_postings_doc_indices.npy",
            "postings_indptr": directory / f"{index_name}_postings_indptr.npy",
        }

    @classmethod
    def has_postings_artifacts(cls, index_dir, index_name):
        paths = cls.artifact_paths(index_dir, index_name)
        required = [
            paths["postings_data"],
            paths["postings_doc_indices"],
            paths["postings_indptr"],
        ]
        return all(_artifact_exists(path) for path in required)

    @classmethod
    def postings_artifacts_within_size_limit(cls, index_dir, index_name):
        paths = cls.artifact_paths(index_dir, index_name)
        required = [
            paths["postings_data"],
            paths["postings_doc_indices"],
            paths["postings_indptr"],
        ]
        return all(_artifact_within_size_limit(path) for path in required)

    @classmethod
    def posting_artifact_infos(cls, index_dir, index_name):
        paths = cls.artifact_paths(index_dir, index_name)
        infos = {}
        for key in ("postings_data", "postings_doc_indices", "postings_indptr"):
            files = _artifact_files(paths[key])
            infos[key] = {
                "path": paths[key],
                "files": files,
                "storage": "chunked" if len(files) > 1 else "single_file",
            }
        return infos

    @classmethod
    def term_doc_matrix_chunk_paths(cls, index_dir, index_name):
        matrix_path = cls.artifact_paths(index_dir, index_name)["term_doc_matrix"]
        return _artifact_chunk_paths(matrix_path)

    @staticmethod
    def _unlink_paths(paths):
        for path in list(paths):
            resolved = Path(path)
            if resolved.exists():
                resolved.unlink()

    @classmethod
    def _write_term_doc_matrix_chunks(cls, source_path, index_dir, index_name):
        directory = Path(index_dir)
        temp_chunk_paths = []
        final_chunk_paths = []
        chunk_prefix = f"{index_name}_term_doc_matrix.npz.part-"

        with open(source_path, "rb") as source:
            chunk_idx = 0
            while True:
                chunk = source.read(MAX_TERM_DOC_MATRIX_CHUNK_BYTES)
                if not chunk:
                    break

                final_path = directory / f"{chunk_prefix}{chunk_idx:03d}"
                temp_path = directory / f"{chunk_prefix}{chunk_idx:03d}.tmp"
                with open(temp_path, "wb") as f:
                    f.write(chunk)

                temp_chunk_paths.append(temp_path)
                final_chunk_paths.append(final_path)
                chunk_idx += 1

        if not final_chunk_paths:
            raise ValueError("Serialized term document matrix was empty.")

        for temp_path, final_path in zip(temp_chunk_paths, final_chunk_paths):
            temp_path.replace(final_path)

        return final_chunk_paths

    @classmethod
    def _load_term_doc_matrix(cls, matrix_path, chunk_paths):
        if chunk_paths:
            log_runtime_event(
                "vector_index.matrix_chunked_load_start",
                chunk_count=len(chunk_paths),
            )
            temp_matrix_path = None
            try:
                with tempfile.NamedTemporaryFile(suffix=".npz", delete=False) as temp_file:
                    temp_matrix_path = Path(temp_file.name)
                    for idx, chunk_path in enumerate(chunk_paths, start=1):
                        log_runtime_event(
                            "vector_index.matrix_chunk_read",
                            chunk_index=idx,
                            chunk_total=len(chunk_paths),
                            chunk_name=chunk_path.name,
                        )
                        with open(chunk_path, "rb") as source:
                            while True:
                                block = source.read(1024 * 1024)
                                if not block:
                                    break
                                temp_file.write(block)

                log_runtime_event(
                    "vector_index.matrix_chunked_load_npz_start",
                    temp_matrix_name=temp_matrix_path.name,
                )
                matrix = load_npz(temp_matrix_path)
                log_runtime_event(
                    "vector_index.matrix_chunked_load_done",
                    n_rows=int(matrix.shape[0]),
                    n_cols=int(matrix.shape[1]),
                )
                return matrix
            finally:
                if temp_matrix_path is not None and temp_matrix_path.exists():
                    temp_matrix_path.unlink()

        log_runtime_event(
            "vector_index.matrix_single_load_start",
            matrix_name=Path(matrix_path).name,
        )
        matrix = load_npz(matrix_path)
        log_runtime_event(
            "vector_index.matrix_single_load_done",
            n_rows=int(matrix.shape[0]),
            n_cols=int(matrix.shape[1]),
        )
        return matrix

    def save_postings_artifacts(self, index_dir, index_name):
        paths = self.artifact_paths(index_dir, index_name)
        log_runtime_event(
            "postings_index.save_start",
            index_name=index_name,
            n_docs=self.n_docs,
            n_terms=self.n_terms,
        )
        postings = self.term_doc_matrix.tocsc()
        posting_infos = {
            "postings_data": _write_npy_artifact(
                paths["postings_data"],
                np.asarray(postings.data, dtype=np.float32),
            ),
            "postings_doc_indices": _write_npy_artifact(
                paths["postings_doc_indices"],
                np.asarray(postings.indices),
            ),
            "postings_indptr": _write_npy_artifact(
                paths["postings_indptr"],
                np.asarray(postings.indptr),
            ),
        }
        log_runtime_event(
            "postings_index.save_done",
            index_name=index_name,
            posting_count=int(postings.nnz),
        )
        return posting_infos

    def save(
        self,
        index_dir,
        index_name,
        extra_meta=None,
        include_matrix_artifacts=False,
    ):
        paths = self.artifact_paths(index_dir, index_name)
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        existing_chunk_paths = self.term_doc_matrix_chunk_paths(index_dir, index_name)

        vectorizer_artifact = _write_pickle_artifact(paths["vectorizer"], self.vectorizer)

        matrix_storage = "omitted"
        matrix_artifact_paths = []
        if include_matrix_artifacts:
            temp_matrix_path = None
            matrix_storage = "single_file"
            matrix_artifact_paths = [paths["term_doc_matrix"]]
            try:
                with tempfile.NamedTemporaryFile(
                    suffix=".npz",
                    dir=index_dir,
                    delete=False,
                ) as temp_file:
                    temp_matrix_path = Path(temp_file.name)

                save_npz(temp_matrix_path, self.term_doc_matrix)
                serialized_matrix_size = temp_matrix_path.stat().st_size

                if serialized_matrix_size > MAX_TERM_DOC_MATRIX_CHUNK_BYTES:
                    matrix_storage = "chunked"
                    matrix_artifact_paths = self._write_term_doc_matrix_chunks(
                        source_path=temp_matrix_path,
                        index_dir=index_dir,
                        index_name=index_name,
                    )
                    stale_chunk_paths = [
                        path for path in existing_chunk_paths if path not in matrix_artifact_paths
                    ]
                    self._unlink_paths(stale_chunk_paths)
                    if paths["term_doc_matrix"].exists():
                        paths["term_doc_matrix"].unlink()
                    for chunk_path in matrix_artifact_paths:
                        if not chunk_path.exists():
                            raise FileNotFoundError(f"Missing expected matrix chunk: {chunk_path}")
                else:
                    self._unlink_paths(existing_chunk_paths)
                    temp_matrix_path.replace(paths["term_doc_matrix"])
                    temp_matrix_path = None
            finally:
                if temp_matrix_path is not None and temp_matrix_path.exists():
                    temp_matrix_path.unlink()
        else:
            self._unlink_paths(existing_chunk_paths + [paths["term_doc_matrix"]])

        posting_infos = self.save_postings_artifacts(index_dir=index_dir, index_name=index_name)

        terms_artifact = _write_json_artifact(paths["terms"], self.terms)
        doc_ids_artifact = _write_json_artifact(paths["doc_ids"], self.doc_ids)

        articles_artifact = None
        if self.articles is not None:
            articles_artifact = _write_dataframe_pickle_artifact(paths["articles"], self.articles)
        else:
            self._unlink_paths(_artifact_files(paths["articles"]))

        meta = {
            "index_name": index_name,
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "n_docs": self.n_docs,
            "n_terms": self.n_terms,
            "id_column": self.id_column,
            "text_column": self.text_column,
            "has_articles": self.articles is not None,
            "vectorizer_class": self.vectorizer.__class__.__name__,
            "term_doc_matrix_storage": matrix_storage,
            "term_doc_matrix_files": [path.name for path in matrix_artifact_paths],
            "term_doc_matrix_chunk_count": len(matrix_artifact_paths),
            "term_doc_matrix_max_chunk_bytes": MAX_TERM_DOC_MATRIX_CHUNK_BYTES,
            "term_doc_matrix_included": bool(include_matrix_artifacts),
            "vectorizer_files": [path.name for path in vectorizer_artifact["files"]],
            "terms_files": [path.name for path in terms_artifact["files"]],
            "doc_ids_files": [path.name for path in doc_ids_artifact["files"]],
            "articles_files": [] if articles_artifact is None else [path.name for path in articles_artifact["files"]],
            "search_backend": "postings",
            "postings_files": [
                path.name
                for info in posting_infos.values()
                for path in info["files"]
            ],
            "postings_data_files": [path.name for path in posting_infos["postings_data"]["files"]],
            "postings_doc_indices_files": [path.name for path in posting_infos["postings_doc_indices"]["files"]],
            "postings_indptr_files": [path.name for path in posting_infos["postings_indptr"]["files"]],
        }
        if extra_meta:
            meta.update(extra_meta)

        meta_artifact = _write_json_artifact(paths["meta"], meta)

        returned_paths = dict(paths)
        returned_paths["term_doc_matrix_files"] = matrix_artifact_paths
        returned_paths["vectorizer_files"] = vectorizer_artifact["files"]
        returned_paths["terms_files"] = terms_artifact["files"]
        returned_paths["doc_ids_files"] = doc_ids_artifact["files"]
        returned_paths["articles_files"] = [] if articles_artifact is None else articles_artifact["files"]
        returned_paths["meta_files"] = meta_artifact["files"]
        returned_paths["postings_files"] = [
            path
            for info in posting_infos.values()
            for path in info["files"]
        ]
        returned_paths["postings_data_files"] = posting_infos["postings_data"]["files"]
        returned_paths["postings_doc_indices_files"] = posting_infos["postings_doc_indices"]["files"]
        returned_paths["postings_indptr_files"] = posting_infos["postings_indptr"]["files"]
        return returned_paths

    @classmethod
    def load(cls, index_dir, index_name, load_articles=True):
        paths = cls.artifact_paths(index_dir, index_name)
        chunk_paths = cls.term_doc_matrix_chunk_paths(index_dir, index_name)
        log_runtime_event(
            "vector_index.load_start",
            index_name=index_name,
            chunk_count=len(chunk_paths),
        )

        required = [
            paths["vectorizer"],
            paths["terms"],
            paths["doc_ids"],
        ]
        if not _artifact_exists(paths["term_doc_matrix"]) and not chunk_paths:
            required.append(paths["term_doc_matrix"])
        missing = [str(path) for path in required if not _artifact_exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing required vector index artifacts: {', '.join(missing)}"
            )

        with _materialized_artifact_path(paths["vectorizer"]) as vectorizer_path:
            with open(vectorizer_path, "rb") as f:
                vectorizer = pickle.load(f)
        log_runtime_event("vector_index.vectorizer_loaded")
        term_doc_matrix = cls._load_term_doc_matrix(
            matrix_path=paths["term_doc_matrix"],
            chunk_paths=chunk_paths,
        )

        with _materialized_artifact_path(paths["terms"]) as terms_path:
            with open(terms_path, "r", encoding="utf-8") as f:
                terms = json.load(f)
        log_runtime_event("vector_index.terms_loaded", n_terms=len(terms))
        with _materialized_artifact_path(paths["doc_ids"]) as doc_ids_path:
            with open(doc_ids_path, "r", encoding="utf-8") as f:
                doc_ids = json.load(f)
        log_runtime_event("vector_index.doc_ids_loaded", n_doc_ids=len(doc_ids))

        meta = {}
        if _artifact_exists(paths["meta"]):
            with _materialized_artifact_path(paths["meta"]) as meta_path:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}

        articles = None
        if load_articles and _artifact_exists(paths["articles"]):
            log_runtime_event("vector_index.articles_load_start")
            with _materialized_artifact_path(paths["articles"]) as articles_path:
                articles = pd.read_pickle(articles_path)
            log_runtime_event(
                "vector_index.articles_load_done",
                n_article_rows=int(len(articles)),
            )
        elif not load_articles:
            log_runtime_event("vector_index.articles_load_skipped")

        id_column = meta.get("id_column", "id")
        text_column = meta.get("text_column", "body_text")

        instance = cls(
            term_doc_matrix=term_doc_matrix,
            terms=terms,
            doc_ids=doc_ids,
            vectorizer=vectorizer,
            articles=articles,
            id_column=id_column,
            text_column=text_column,
        )
        log_runtime_event(
            "vector_index.load_done",
            index_name=index_name,
            n_docs=instance.n_docs,
            n_terms=instance.n_terms,
        )
        return instance, meta


class TfidfPostingsIndex:
    def __init__(
        self,
        terms,
        doc_ids,
        vectorizer,
        postings_data,
        postings_doc_indices,
        postings_indptr,
        articles=None,
        id_column="id",
        text_column="body_text",
        temp_artifact_paths=None,
    ):
        if vectorizer is None:
            raise ValueError("vectorizer cannot be None.")

        normalized_terms = TfidfMatrixIndex._normalize_terms(terms)
        normalized_doc_ids = TfidfMatrixIndex._normalize_doc_ids(doc_ids)

        if len(postings_indptr) != len(normalized_terms) + 1:
            raise ValueError(
                "postings_indptr length must equal number of terms + 1."
            )
        if len(postings_data) != len(postings_doc_indices):
            raise ValueError(
                "postings_data length must match postings_doc_indices length."
            )
        if int(postings_indptr[-1]) != len(postings_doc_indices):
            raise ValueError(
                "Final postings_indptr entry must equal postings_doc_indices length."
            )

        self.terms = normalized_terms
        self.term_to_idx = {term: idx for idx, term in enumerate(self.terms)}
        self.n_terms = len(self.terms)

        self.doc_ids = normalized_doc_ids
        self.doc_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.doc_ids)}
        self.n_docs = len(self.doc_ids)

        self.vectorizer = vectorizer
        self.postings_data = postings_data
        self.postings_doc_indices = postings_doc_indices
        self.postings_indptr = postings_indptr
        self.id_column = id_column
        self.text_column = text_column
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
                id_column=self.id_column,
            )

    def search(self, query, top_n=100, return_articles=True):
        if top_n is None:
            top_n = self.n_docs
        top_n = int(top_n)
        if top_n <= 0:
            return []

        log_runtime_event(
            "postings_search.query_vector_start",
            query_chars=len(str(query or "")),
            top_n=top_n,
            n_docs=self.n_docs,
            n_terms=self.n_terms,
        )
        query_vec = self.vectorizer.transform([str(query)])
        query_term_indices = np.asarray(query_vec.indices, dtype=np.int32)
        query_term_weights = np.asarray(query_vec.data, dtype=np.float32)
        log_runtime_event(
            "postings_search.query_vector_done",
            query_term_count=int(len(query_term_indices)),
        )
        if query_term_indices.size == 0:
            log_runtime_event("postings_search.no_query_terms")
            return []

        scores = np.zeros(self.n_docs, dtype=np.float32)
        total_terms = int(query_term_indices.size)
        for term_pos, (term_idx, query_weight) in enumerate(
            zip(query_term_indices, query_term_weights),
            start=1,
        ):
            start = int(self.postings_indptr[int(term_idx)])
            end = int(self.postings_indptr[int(term_idx) + 1])
            if term_pos == total_terms or term_pos <= 5 or term_pos % 10 == 0:
                log_runtime_event(
                    "postings_search.term_progress",
                    term_index=term_pos,
                    term_total=total_terms,
                    posting_count=max(0, end - start),
                )
            if end <= start:
                continue

            doc_slice = np.asarray(
                self.postings_doc_indices[start:end],
                dtype=np.intp,
            )
            weight_slice = np.asarray(
                self.postings_data[start:end],
                dtype=np.float32,
            )
            scores[doc_slice] += np.float32(query_weight) * weight_slice

        candidate_doc_indices = np.flatnonzero(scores > 0)
        if candidate_doc_indices.size == 0:
            log_runtime_event("postings_search.no_candidates")
            return []

        candidate_scores = scores[candidate_doc_indices]
        top_n = min(top_n, int(candidate_doc_indices.size))
        log_runtime_event(
            "postings_search.topk_select_start",
            candidate_count=int(candidate_doc_indices.size),
        )
        if candidate_scores.size > top_n:
            top_positions = np.argpartition(candidate_scores, -top_n)[-top_n:]
            sorted_positions = top_positions[np.argsort(candidate_scores[top_positions])[::-1]]
        else:
            sorted_positions = np.argsort(candidate_scores)[::-1]
        log_runtime_event(
            "postings_search.topk_select_done",
            selected_count=int(len(sorted_positions)),
        )

        results = []
        for pos in sorted_positions:
            idx = int(candidate_doc_indices[pos])
            score = float(candidate_scores[pos])
            doc_id = self.doc_ids[idx]
            if return_articles and self.articles is not None:
                payload = self.articles.iloc[idx].to_dict()
            else:
                payload = doc_id
            results.append((payload, score))

        log_runtime_event("postings_search.results_done", result_count=len(results))
        return results

    @classmethod
    def load(cls, index_dir, index_name, load_articles=True):
        paths = TfidfMatrixIndex.artifact_paths(index_dir, index_name)
        log_runtime_event("postings_index.load_start", index_name=index_name)

        required = [
            paths["vectorizer"],
            paths["terms"],
            paths["doc_ids"],
            paths["postings_data"],
            paths["postings_doc_indices"],
            paths["postings_indptr"],
        ]
        missing = [str(path) for path in required if not _artifact_exists(path)]
        if missing:
            raise FileNotFoundError(
                f"Missing required postings index artifacts: {', '.join(missing)}"
            )

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
            postings_data_path, postings_data_temp = _materialized_artifact_path_for_mmap(paths["postings_data"])
            if postings_data_temp is not None:
                temp_artifact_paths.append(postings_data_temp)
            postings_data = np.load(postings_data_path, mmap_mode="r", allow_pickle=False)

            postings_doc_indices_path, postings_doc_indices_temp = _materialized_artifact_path_for_mmap(
                paths["postings_doc_indices"]
            )
            if postings_doc_indices_temp is not None:
                temp_artifact_paths.append(postings_doc_indices_temp)
            postings_doc_indices = np.load(
                postings_doc_indices_path,
                mmap_mode="r",
                allow_pickle=False,
            )

            postings_indptr_path, postings_indptr_temp = _materialized_artifact_path_for_mmap(paths["postings_indptr"])
            if postings_indptr_temp is not None:
                temp_artifact_paths.append(postings_indptr_temp)
            postings_indptr = np.load(postings_indptr_path, mmap_mode="r", allow_pickle=False)
        except Exception:
            _cleanup_temp_paths(temp_artifact_paths)
            raise
        log_runtime_event(
            "postings_index.arrays_loaded",
            posting_count=int(len(postings_data)),
            posting_term_count=int(len(postings_indptr) - 1),
        )

        meta = {}
        if _artifact_exists(paths["meta"]):
            with _materialized_artifact_path(paths["meta"]) as meta_path:
                with open(meta_path, "r", encoding="utf-8") as f:
                    meta = json.load(f) or {}

        articles = None
        if load_articles and _artifact_exists(paths["articles"]):
            log_runtime_event("postings_index.articles_load_start")
            with _materialized_artifact_path(paths["articles"]) as articles_path:
                articles = pd.read_pickle(articles_path)
            log_runtime_event(
                "postings_index.articles_load_done",
                n_article_rows=int(len(articles)),
            )
        elif not load_articles:
            log_runtime_event("postings_index.articles_load_skipped")

        instance = cls(
            terms=terms,
            doc_ids=doc_ids,
            vectorizer=vectorizer,
            postings_data=postings_data,
            postings_doc_indices=postings_doc_indices,
            postings_indptr=postings_indptr,
            articles=articles,
            id_column=meta.get("id_column", "id"),
            text_column=meta.get("text_column", "body_text"),
            temp_artifact_paths=temp_artifact_paths,
        )
        log_runtime_event(
            "postings_index.load_done",
            index_name=index_name,
            n_docs=instance.n_docs,
            n_terms=instance.n_terms,
        )
        return instance, meta

    @classmethod
    def has_artifacts(cls, index_dir, index_name):
        return TfidfMatrixIndex.has_postings_artifacts(index_dir, index_name)

    @classmethod
    def artifacts_within_size_limit(cls, index_dir, index_name):
        return TfidfMatrixIndex.postings_artifacts_within_size_limit(index_dir, index_name)

    @classmethod
    def artifact_infos(cls, index_dir, index_name):
        return TfidfMatrixIndex.posting_artifact_infos(index_dir, index_name)

    @classmethod
    def repartition_artifacts(cls, index_dir, index_name):
        paths = TfidfMatrixIndex.artifact_paths(index_dir, index_name)
        infos = {}
        for key in ("postings_data", "postings_doc_indices", "postings_indptr"):
            artifact_path = paths[key]
            chunk_paths = _artifact_chunk_paths(artifact_path)
            if chunk_paths:
                infos[key] = {
                    "path": artifact_path,
                    "files": chunk_paths,
                    "storage": "chunked",
                }
                continue

            if not artifact_path.exists():
                raise FileNotFoundError(f"Missing postings artifact: {artifact_path}")

            if artifact_path.stat().st_size > MAX_VECTOR_INDEX_ARTIFACT_BYTES:
                infos[key] = _repartition_existing_artifact(artifact_path)
            else:
                infos[key] = {
                    "path": artifact_path,
                    "files": [artifact_path],
                    "storage": "single_file",
                }
        return infos


def load_search_index(
    index_dir,
    index_name,
    load_articles=False,
    allow_matrix_fallback=False,
):
    if TfidfPostingsIndex.has_artifacts(index_dir=index_dir, index_name=index_name):
        log_runtime_event("search_index.load_postings", index_name=index_name)
        return TfidfPostingsIndex.load(
            index_dir=index_dir,
            index_name=index_name,
            load_articles=load_articles,
        )

    log_runtime_event(
        "search_index.missing_postings",
        index_name=index_name,
        allow_matrix_fallback=bool(allow_matrix_fallback),
    )
    if allow_matrix_fallback:
        return TfidfMatrixIndex.load(
            index_dir=index_dir,
            index_name=index_name,
            load_articles=load_articles,
        )

    raise RuntimeError(
        "Postings artifacts are missing for the search index. Rebuild the search index "
        "offline to generate postings files before deploying this version."
    )
