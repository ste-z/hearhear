import sqlite3
from pathlib import Path

import pandas as pd

from backend.imports.data_import import (
    load_and_clean_guardian_years,
    normalize_guardian_article_columns,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
DEFAULT_DB_PATH = PROJECT_ROOT / "instance" / "data.db"
DEFAULT_INDEX_DIR = PROJECT_ROOT / "data" / "processed" / "vector_index"
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "guardian_by_year"
DEFAULT_MIN_BODY_TEXT_CHARS = 1000


def _normalized_years(years):
    if years is None:
        return None
    return sorted({int(year) for year in years if year is not None})


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

    return normalize_guardian_article_columns(
        articles,
        list_columns=("contributors", "keywords"),
        int_columns=("n_contributors", "year"),
    )


def _available_guardian_years(raw_data_dir):
    years = []
    for path in Path(raw_data_dir).glob("guardian_opinion_*.csv"):
        suffix = path.stem.rsplit("_", 1)[-1]
        if suffix.isdigit():
            years.append(int(suffix))
    return sorted(set(years))


def _load_guardian_articles_from_raw(
    raw_data_dir=DEFAULT_RAW_DATA_DIR,
    years=None,
    min_body_text_chars=DEFAULT_MIN_BODY_TEXT_CHARS,
):
    resolved_years = list(years or _available_guardian_years(raw_data_dir))
    if not resolved_years:
        raise FileNotFoundError(f"No Guardian raw CSVs found in {raw_data_dir}")

    return load_and_clean_guardian_years(
        years=resolved_years,
        folder=raw_data_dir,
        drop_duplicates=True,
        min_body_text_chars=min_body_text_chars,
    )


def _current_db_row_count(db_path):
    with sqlite3.connect(db_path) as conn:
        cursor = conn.execute("SELECT COUNT(*) FROM guardian_articles")
        row = cursor.fetchone()
        return int(row[0]) if row else 0


def _db_years(db_path):
    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT DISTINCT year FROM guardian_articles WHERE year IS NOT NULL ORDER BY year"
        ).fetchall()
    return [int(year) for (year,) in rows]


def _db_has_complete_body_text(db_path):
    with sqlite3.connect(db_path) as conn:
        total_row = conn.execute("SELECT COUNT(*) FROM guardian_articles").fetchone()
        nonempty_row = conn.execute(
            """
            SELECT COUNT(*)
            FROM guardian_articles
            WHERE TRIM(COALESCE(body_text, '')) != ''
            """
        ).fetchone()

    total_count = int(total_row[0]) if total_row else 0
    nonempty_count = int(nonempty_row[0]) if nonempty_row else 0
    return total_count > 0 and nonempty_count == total_count


def _filter_articles_to_years(articles, years=None):
    if articles is None:
        return pd.DataFrame()
    if articles.empty:
        return articles

    normalized_years = _normalized_years(years)
    if not normalized_years or "year" not in articles.columns:
        return articles

    filtered = articles.copy()
    article_years = pd.to_numeric(filtered["year"], errors="coerce").astype("Int64")
    return filtered.loc[article_years.isin(normalized_years)].reset_index(drop=True)


def _relative_db_path_for_meta(db_path):
    try:
        return str(Path(db_path).resolve().relative_to(PROJECT_ROOT.resolve()))
    except Exception:
        return str(Path(db_path))
