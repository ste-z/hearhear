import json
import pickle
import io
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import load_npz, save_npz
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from backend.runtime_debug import log_runtime_event


DEFAULT_TFIDF_PARAMS = {
    "analyzer": "word",
    "token_pattern": r"(?u)\b[^\W\d_]+(?:[-'][^\W\d_]+)*\b",
    "lowercase": True,
    "strip_accents": "unicode",
    "stop_words": "english",
    "min_df": 2,
}

# Keep matrix chunks comfortably below the common 100 MB git hosting limit.
MAX_TERM_DOC_MATRIX_CHUNK_BYTES = 95 * 1024 * 1024


class VectorizedText:
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
        if terms is None:
            raise ValueError("terms cannot be None.")
        normalized = []
        seen = set()
        for term in list(terms):
            value = str(term).strip()
            if not value:
                raise ValueError("terms contains empty values.")
            if value in seen:
                raise ValueError(f"Duplicate term found: '{value}'.")
            normalized.append(value)
            seen.add(value)
        return normalized

    @staticmethod
    def _normalize_doc_ids(doc_ids):
        if doc_ids is None:
            raise ValueError("doc_ids cannot be None.")
        normalized_ids = VectorizedText._normalize_id_series(doc_ids, "doc_ids")
        return normalized_ids.tolist()

    @staticmethod
    def _normalize_id_series(id_values, field_name):
        if id_values is None:
            raise ValueError(f"{field_name} cannot be None.")

        normalized = pd.Series(id_values, dtype="string").fillna("").str.strip()
        if (normalized == "").any():
            raise ValueError(f"{field_name} contains blank ids.")

        duplicates = normalized[normalized.duplicated()].unique().tolist()
        if duplicates:
            preview = ", ".join(duplicates[:5])
            raise ValueError(f"{field_name} has duplicate ids (sample: {preview}).")

        return normalized

    def _normalize_articles(self, articles):
        if not isinstance(articles, pd.DataFrame):
            raise TypeError("articles must be a pandas DataFrame.")
        if self.id_column not in articles.columns:
            raise ValueError(f"articles must contain id column '{self.id_column}'.")

        normalized = articles.reset_index(drop=True).copy()
        normalized[self.id_column] = self._normalize_id_series(
            normalized[self.id_column], self.id_column
        )

        indexed = normalized.set_index(self.id_column, drop=False)
        missing_ids = [doc_id for doc_id in self.doc_ids if doc_id not in indexed.index]
        if missing_ids:
            preview = ", ".join(missing_ids[:5])
            raise ValueError(
                f"articles is missing ids from doc_ids (sample: {preview})."
            )

        return indexed.loc[self.doc_ids].reset_index(drop=True)

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
        log_runtime_event("vector_search.cosine_similarity_start")
        cosine_similarities = cosine_similarity(query_vec, self.term_doc_matrix).ravel()
        log_runtime_event(
            "vector_search.cosine_similarity_done",
            similarity_count=int(cosine_similarities.shape[0]),
        )
        log_runtime_event("vector_search.argsort_start")
        top_indices = np.argsort(cosine_similarities)[::-1][:top_n]
        log_runtime_event("vector_search.argsort_done")

        results = []
        for idx in top_indices:
            score = float(cosine_similarities[idx])
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
        }

    @classmethod
    def term_doc_matrix_chunk_paths(cls, index_dir, index_name):
        matrix_path = cls.artifact_paths(index_dir, index_name)["term_doc_matrix"]
        chunk_prefix = f"{matrix_path.name}.part-"
        return sorted(
            path
            for path in matrix_path.parent.glob(f"{chunk_prefix}*")
            if path.is_file() and not path.name.endswith(".tmp")
        )

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
            serialized = io.BytesIO()
            for idx, chunk_path in enumerate(chunk_paths, start=1):
                log_runtime_event(
                    "vector_index.matrix_chunk_read",
                    chunk_index=idx,
                    chunk_total=len(chunk_paths),
                    chunk_name=chunk_path.name,
                )
                with open(chunk_path, "rb") as f:
                    serialized.write(f.read())
            serialized.seek(0)
            log_runtime_event("vector_index.matrix_chunked_load_npz_start")
            matrix = load_npz(serialized)
            log_runtime_event(
                "vector_index.matrix_chunked_load_done",
                n_rows=int(matrix.shape[0]),
                n_cols=int(matrix.shape[1]),
            )
            return matrix

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

    def save(self, index_dir, index_name, extra_meta=None):
        paths = self.artifact_paths(index_dir, index_name)
        Path(index_dir).mkdir(parents=True, exist_ok=True)
        existing_chunk_paths = self.term_doc_matrix_chunk_paths(index_dir, index_name)

        with open(paths["vectorizer"], "wb") as f:
            pickle.dump(self.vectorizer, f)

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

        with open(paths["terms"], "w", encoding="utf-8") as f:
            json.dump(self.terms, f)
        with open(paths["doc_ids"], "w", encoding="utf-8") as f:
            json.dump(self.doc_ids, f)

        if self.articles is not None:
            self.articles.to_pickle(paths["articles"])

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
        }
        if extra_meta:
            meta.update(extra_meta)

        with open(paths["meta"], "w", encoding="utf-8") as f:
            json.dump(meta, f)

        returned_paths = dict(paths)
        returned_paths["term_doc_matrix_files"] = matrix_artifact_paths
        return returned_paths

    @classmethod
    def load(cls, index_dir, index_name):
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
        if not paths["term_doc_matrix"].exists() and not chunk_paths:
            required.append(paths["term_doc_matrix"])
        missing = [str(path) for path in required if not path.exists()]
        if missing:
            raise FileNotFoundError(
                f"Missing required vector index artifacts: {', '.join(missing)}"
            )

        with open(paths["vectorizer"], "rb") as f:
            vectorizer = pickle.load(f)
        log_runtime_event("vector_index.vectorizer_loaded")
        term_doc_matrix = cls._load_term_doc_matrix(
            matrix_path=paths["term_doc_matrix"],
            chunk_paths=chunk_paths,
        )

        with open(paths["terms"], "r", encoding="utf-8") as f:
            terms = json.load(f)
        log_runtime_event("vector_index.terms_loaded", n_terms=len(terms))
        with open(paths["doc_ids"], "r", encoding="utf-8") as f:
            doc_ids = json.load(f)
        log_runtime_event("vector_index.doc_ids_loaded", n_doc_ids=len(doc_ids))

        meta = {}
        if paths["meta"].exists():
            with open(paths["meta"], "r", encoding="utf-8") as f:
                meta = json.load(f) or {}

        articles = None
        if paths["articles"].exists():
            log_runtime_event("vector_index.articles_load_start")
            articles = pd.read_pickle(paths["articles"])
            log_runtime_event(
                "vector_index.articles_load_done",
                n_article_rows=int(len(articles)),
            )

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
