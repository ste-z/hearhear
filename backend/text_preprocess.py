import argparse
import json
import sqlite3
from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

from backend.data_import import normalize_guardian_article_columns
from backend.text_processor import DEFAULT_TFIDF_PARAMS, VectorizedText


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "instance" / "data.db"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "backend" / "data" / "processed" / "vector_index"
DEFAULT_INDEX_NAME = "guardian_tfidf"


def _load_guardian_articles_from_sqlite(db_path):
    query = """
        SELECT
            id,
            title,
            summary,
            date,
            url,
            author_raw,
            contributors,
            n_contributors,
            keywords,
            year,
            body_text
        FROM guardian_articles
        ORDER BY id
    """

    with sqlite3.connect(db_path) as conn:
        articles = pd.read_sql_query(query, conn)

    if articles.empty:
        return articles

    articles = normalize_guardian_article_columns(
        articles,
        list_columns=("contributors", "keywords"),
        int_columns=("n_contributors", "year"),
    )
    return articles


def _current_db_row_count(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM guardian_articles")
        row = cursor.fetchone()
        return int(row[0]) if row else 0


def _is_existing_index_fresh(index_dir, index_name, db_row_count):
    meta_path = VectorizedText.artifact_paths(index_dir, index_name)["meta"]
    if not meta_path.exists():
        return False

    try:
        with open(meta_path, "r", encoding="utf-8") as f:
            meta = json.load(f) or {}
    except Exception:
        return False

    try:
        stored_count = int(meta.get("db_row_count"))
    except (TypeError, ValueError):
        return False

    return stored_count == int(db_row_count)


def _relative_db_path_for_meta(db_path):
    try:
        return str(Path(db_path).resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(Path(db_path))


def preprocess_tfidf_index(
    db_path=DEFAULT_DB_PATH,
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_INDEX_NAME,
    force_rebuild=False,
):
    db_path = Path(db_path)
    index_dir = Path(index_dir)

    if not db_path.exists():
        raise FileNotFoundError(f"Database not found: {db_path}")

    db_row_count = _current_db_row_count(db_path)
    if not force_rebuild and _is_existing_index_fresh(index_dir, index_name, db_row_count):
        return {
            "built": False,
            "reason": "up_to_date",
            "db_row_count": db_row_count,
            "index_dir": str(index_dir),
            "index_name": index_name,
        }

    articles = _load_guardian_articles_from_sqlite(db_path)
    if articles.empty:
        raise ValueError("No guardian_articles rows found; cannot build TF-IDF index.")

    vectorizer = TfidfVectorizer(**DEFAULT_TFIDF_PARAMS)
    vectorized_text = VectorizedText.from_articles(
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
        },
    )

    return {
        "built": True,
        "db_row_count": db_row_count,
        "index_dir": str(index_dir),
        "index_name": index_name,
        "paths": {key: str(value) for key, value in paths.items()},
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Precompute TF-IDF vector index artifacts.")
    parser.add_argument("--db-path", default=str(DEFAULT_DB_PATH))
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--index-name", default=DEFAULT_INDEX_NAME)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    result = preprocess_tfidf_index(
        db_path=args.db_path,
        index_dir=args.index_dir,
        index_name=args.index_name,
        force_rebuild=args.force_rebuild,
    )
    print(json.dumps(result, indent=2))
