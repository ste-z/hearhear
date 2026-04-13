import argparse
import json
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.indexing.artifacts import (
    _artifact_exists,
    _artifact_within_size_limit,
    _materialized_artifact_path,
    _write_json_artifact,
)
from backend.text_processing.indexing.settings import (
    DEFAULT_TFIDF_PARAMS,
    MAX_VECTOR_INDEX_ARTIFACT_BYTES,
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
from backend.text_processing.text_processor import (
    TfidfMatrixIndex,
    TfidfPostingsIndex,
)


DEFAULT_INDEX_NAME = "guardian_tfidf"


def _load_index_meta(paths):
    meta_path = paths["meta"]
    if not _artifact_exists(meta_path):
        return {}

    try:
        with _materialized_artifact_path(meta_path) as materialized_meta_path:
            with open(materialized_meta_path, "r", encoding="utf-8") as f:
                return json.load(f) or {}
    except Exception:
        return {}


def _is_existing_index_fresh(index_dir, index_name, db_row_count, expected_years=None):
    paths = TfidfMatrixIndex.artifact_paths(index_dir, index_name)
    if not _artifact_exists(paths["meta"]):
        return False

    meta = _load_index_meta(paths)
    if not meta:
        return False

    try:
        stored_count = int(meta.get("db_row_count"))
    except (TypeError, ValueError):
        return False

    stored_vectorizer_params = meta.get("vectorizer_params")
    if stored_vectorizer_params != DEFAULT_TFIDF_PARAMS:
        return False

    required_paths = [
        paths["vectorizer"],
        paths["terms"],
        paths["doc_ids"],
        paths["postings_data"],
        paths["postings_doc_indices"],
        paths["postings_indptr"],
    ]
    if meta.get("has_articles"):
        required_paths.append(paths["articles"])

    if not all(_artifact_exists(path) for path in required_paths):
        return False

    if not all(_artifact_within_size_limit(path) for path in required_paths + [paths["meta"]]):
        return False

    if meta.get("search_backend") != "postings":
        return False

    normalized_expected_years = _normalized_years(expected_years)
    if normalized_expected_years is not None:
        try:
            stored_source_years = _normalized_years(meta.get("source_years"))
        except (TypeError, ValueError):
            return False
        if stored_source_years != normalized_expected_years:
            return False

    if not TfidfPostingsIndex.artifacts_within_size_limit(
        index_dir=index_dir,
        index_name=index_name,
    ):
        return False

    return stored_count == int(db_row_count)
def preprocess_tfidf_index(
    db_path=DEFAULT_DB_PATH,
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_INDEX_NAME,
    force_rebuild=False,
    years=None,
):
    db_path = Path(db_path)
    index_dir = Path(index_dir)
    normalized_years = _normalized_years(years)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    db_row_count = _current_db_row_count(db_path)
    log_runtime_event(
        "tfidf_preprocess.start",
        db_path=str(db_path),
        index_name=index_name,
        db_row_count=db_row_count,
        force_rebuild=bool(force_rebuild),
        requested_years=normalized_years,
    )
    if not force_rebuild and _is_existing_index_fresh(
        index_dir,
        index_name,
        db_row_count,
        expected_years=normalized_years,
    ):
        log_runtime_event(
            "tfidf_preprocess.up_to_date",
            index_name=index_name,
            db_row_count=db_row_count,
        )
        return {
            "built": False,
            "reason": "up_to_date",
            "db_row_count": db_row_count,
            "index_dir": str(index_dir),
            "index_name": index_name,
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
        "tfidf_preprocess.source_ready",
        source_kind=source_kind,
        article_count=int(len(articles)),
        source_years=source_years,
    )

    if articles.empty:
        raise ValueError("No guardian_articles rows found; cannot build TF-IDF index.")

    vectorizer = TfidfVectorizer(**DEFAULT_TFIDF_PARAMS)
    vectorized_text = TfidfMatrixIndex.from_articles(
        articles=articles,
        vectorizer=vectorizer,
        text_column="body_text",
        id_column="id",
        include_text_in_articles=False,
    )
    paths = vectorized_text.save(
        index_dir=index_dir,
        index_name=index_name,
        extra_meta={
            "db_row_count": int(db_row_count),
            "source_db_path": _relative_db_path_for_meta(db_path),
            "text_source": source_kind,
            "source_years": source_years,
            "vectorizer_params": dict(DEFAULT_TFIDF_PARAMS),
        },
        include_matrix_artifacts=False,
    )

    return {
        "built": True,
        "db_row_count": db_row_count,
        "index_dir": str(index_dir),
        "index_name": index_name,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def ensure_postings_artifacts(
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_INDEX_NAME,
):
    index_dir = Path(index_dir)
    paths = TfidfMatrixIndex.artifact_paths(index_dir, index_name)
    matrix_paths = TfidfMatrixIndex.term_doc_matrix_chunk_paths(index_dir, index_name)
    if paths["term_doc_matrix"].exists():
        matrix_paths = matrix_paths + [paths["term_doc_matrix"]]

    log_runtime_event(
        "postings_ensure.start",
        index_name=index_name,
    )
    posting_paths = {
        "postings_data": paths["postings_data"],
        "postings_doc_indices": paths["postings_doc_indices"],
        "postings_indptr": paths["postings_indptr"],
    }
    built_postings = False

    if not TfidfPostingsIndex.has_artifacts(index_dir=index_dir, index_name=index_name):
        try:
            vectorized_text, _meta = TfidfMatrixIndex.load(
                index_dir=index_dir,
                index_name=index_name,
                load_articles=False,
            )
        except FileNotFoundError as exc:
            raise FileNotFoundError(
                "Cannot derive postings artifacts because the legacy TF-IDF matrix artifacts "
                "are missing. Run a full TF-IDF rebuild instead."
            ) from exc
        posting_paths = vectorized_text.save_postings_artifacts(
            index_dir=index_dir,
            index_name=index_name,
        )
        built_postings = True
    elif not TfidfPostingsIndex.artifacts_within_size_limit(
        index_dir=index_dir,
        index_name=index_name,
    ):
        log_runtime_event(
            "postings_ensure.repartition_start",
            index_name=index_name,
            max_artifact_bytes=int(MAX_VECTOR_INDEX_ARTIFACT_BYTES),
        )
        posting_paths = TfidfPostingsIndex.repartition_artifacts(
            index_dir=index_dir,
            index_name=index_name,
        )
        built_postings = True
        log_runtime_event(
            "postings_ensure.repartition_done",
            index_name=index_name,
            postings_files=sum(len(info["files"]) for info in posting_paths.values()),
        )
    else:
        posting_paths = TfidfPostingsIndex.artifact_infos(
            index_dir=index_dir,
            index_name=index_name,
        )

    if matrix_paths:
        TfidfMatrixIndex._unlink_paths(matrix_paths)

    meta = _load_index_meta(paths)
    meta.update(
        {
            "search_backend": "postings",
            "postings_files": [
                path.name
                for info in posting_paths.values()
                for path in info["files"]
            ],
            "postings_data_files": [path.name for path in posting_paths["postings_data"]["files"]],
            "postings_doc_indices_files": [
                path.name for path in posting_paths["postings_doc_indices"]["files"]
            ],
            "postings_indptr_files": [
                path.name for path in posting_paths["postings_indptr"]["files"]
            ],
            "term_doc_matrix_storage": "omitted",
            "term_doc_matrix_files": [],
            "term_doc_matrix_chunk_count": 0,
            "term_doc_matrix_max_chunk_bytes": int(MAX_VECTOR_INDEX_ARTIFACT_BYTES),
            "term_doc_matrix_included": False,
        }
    )
    _write_json_artifact(paths["meta"], meta)

    if not built_postings and not matrix_paths:
        log_runtime_event(
            "postings_ensure.up_to_date",
            index_name=index_name,
        )
        return {
            "built": False,
            "reason": "up_to_date",
            "index_dir": str(index_dir),
            "index_name": index_name,
        }

    return {
        "built": True,
        "index_dir": str(index_dir),
        "index_name": index_name,
        "paths": {
            key: {
                "path": str(info["path"]),
                "files": [str(path) for path in info["files"]],
                "storage": info["storage"],
            }
            for key, info in posting_paths.items()
        },
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Precompute TF-IDF vector index artifacts.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--force-rebuild", action="store_true")
    parser.add_argument("--ensure-postings", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    if args.ensure_postings:
        result = ensure_postings_artifacts(
            index_dir=args.index_dir,
            index_name=args.index_name,
        )
    else:
        result = preprocess_tfidf_index(
            db_path=args.db_path,
            index_dir=args.index_dir,
            index_name=args.index_name,
            force_rebuild=args.force_rebuild,
        )
    print(json.dumps(result, indent=2))
