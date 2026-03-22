import os
from pathlib import Path

import pandas as pd
from sqlalchemy import and_, func

from backend.data_import import load_and_clean_guardian_years
from models import GuardianArticle, db


DEFAULT_YEARS = set(range(2010, 2026))
DEFAULT_MIN_BODY_TEXT_CHARS = 1000
DEFAULT_BATCH_SIZE = 500
DEFAULT_BUNDLED_INDEX_DIR = Path(__file__).resolve().parent.parent / "backend" / "data" / "processed" / "vector_index"
DEFAULT_BUNDLED_INDEX_NAME = "guardian_tfidf"
STORE_GUARDIAN_BODY_TEXT_ENV = "STORE_GUARDIAN_BODY_TEXT_IN_DB"


def _is_missing(value):
    if value is None:
        return True
    if isinstance(value, str) and value.strip().lower() in {"", "nan", "none", "null", "<na>"}:
        return True
    try:
        return value != value
    except Exception:
        return False


def _clean_str(value):
    return "" if _is_missing(value) else str(value)


def _clean_list(value):
    if isinstance(value, list):
        return value
    return []


def _clean_datetime(value):
    if _is_missing(value):
        return None

    if isinstance(value, str):
        parsed = pd.to_datetime(value, utc=True, errors="coerce")
        if _is_missing(parsed):
            return None
        return parsed.to_pydatetime()

    if hasattr(value, "to_pydatetime"):
        py_dt = value.to_pydatetime()
        return None if _is_missing(py_dt) else py_dt
    return value


def _normalized_years(years):
    return {int(year) for year in set(years or [])}


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_store_body_text():
    return _env_flag(STORE_GUARDIAN_BODY_TEXT_ENV, default=False)


def _existing_data_needs_refresh(expected_years=None, allow_missing_body_text=False):
    expected_year_set = _normalized_years(expected_years)

    missing_author_exists = db.session.query(GuardianArticle.id).filter(
        and_(
            GuardianArticle.n_contributors == 0,
            func.trim(func.coalesce(GuardianArticle.author_raw, "")) == "",
        )
    ).limit(1).first() is not None

    if allow_missing_body_text:
        missing_body_exists = False
        short_body_exists = False
    else:
        missing_body_exists = db.session.query(GuardianArticle.id).filter(
            func.trim(func.coalesce(GuardianArticle.body_text, "")) == "",
        ).limit(1).first() is not None

        short_body_exists = db.session.query(GuardianArticle.id).filter(
            func.length(func.coalesce(GuardianArticle.body_text, "")) < DEFAULT_MIN_BODY_TEXT_CHARS,
        ).limit(1).first() is not None

    missing_summary_exists = db.session.query(GuardianArticle.id).filter(
        func.trim(func.coalesce(GuardianArticle.summary, "")) == "",
    ).limit(1).first() is not None

    existing_years = {
        int(year)
        for (year,) in db.session.query(GuardianArticle.year).distinct().all()
        if year is not None
    }
    year_range_mismatch = bool(expected_year_set) and existing_years != expected_year_set

    return any([
        missing_author_exists,
        missing_body_exists,
        missing_summary_exists,
        short_body_exists,
        year_range_mismatch,
    ])


def _clear_stored_body_text():
    updated = (
        db.session.query(GuardianArticle)
        .filter(func.trim(func.coalesce(GuardianArticle.body_text, "")) != "")
        .update({GuardianArticle.body_text: ""}, synchronize_session=False)
    )
    db.session.commit()
    return int(updated or 0)


def _persist_guardian_articles(df, batch_size=DEFAULT_BATCH_SIZE, store_body_text=False):
    if df is None or df.empty:
        print("Guardian dataset is empty; skipping initialization.")
        return

    batch = []
    columns = list(df.columns)
    for row in df.itertuples(index=False, name=None):
        row_data = dict(zip(columns, row))
        authors = _clean_list(row_data.get("authors") or row_data.get("contributors"))
        author_display = ", ".join(authors)
        article = GuardianArticle(
            id=_clean_str(row_data.get("id")),
            title=_clean_str(row_data.get("title")),
            summary=_clean_str(row_data.get("summary")),
            date=_clean_datetime(row_data.get("date")),
            url=_clean_str(row_data.get("url")),
            author_raw=author_display or _clean_str(row_data.get("author_raw")),
            contributors=authors,
            n_contributors=int(row_data.get("n_contributors") or len(authors)),
            keywords=_clean_list(row_data.get("keywords")),
            body_text=_clean_str(row_data.get("body_text")) if store_body_text else "",
            section_id="",
            section_name="",
            year=int(row_data.get("year") or 0),
        )
        batch.append(article)

        if len(batch) >= batch_size:
            db.session.bulk_save_objects(batch)
            db.session.commit()
            batch.clear()

    if batch:
        db.session.bulk_save_objects(batch)
        db.session.commit()

    print(f"Database initialized with {GuardianArticle.query.count()} Guardian articles.")


def _load_bundled_guardian_articles():
    articles_path = DEFAULT_BUNDLED_INDEX_DIR / f"{DEFAULT_BUNDLED_INDEX_NAME}_articles.pkl"
    if not articles_path.exists():
        return pd.DataFrame()

    try:
        articles = pd.read_pickle(articles_path)
    except Exception as exc:
        print(f"Warning: failed to load bundled article snapshot from {articles_path}: {exc}")
        return pd.DataFrame()

    if not isinstance(articles, pd.DataFrame):
        print(f"Warning: bundled article snapshot at {articles_path} is not a DataFrame.")
        return pd.DataFrame()

    return articles.reset_index(drop=True).copy()


def _seed_guardian_articles(
    project_root,
    years=DEFAULT_YEARS,
    min_body_text_chars=DEFAULT_MIN_BODY_TEXT_CHARS,
    batch_size=DEFAULT_BATCH_SIZE,
    bundled_articles=None,
    store_body_text=False,
):
    if bundled_articles is None:
        bundled_articles = _load_bundled_guardian_articles()
    if not bundled_articles.empty:
        print("Seeding Guardian articles from bundled vector index metadata.")
        _persist_guardian_articles(
            bundled_articles,
            batch_size=batch_size,
            store_body_text=store_body_text,
        )
        return "bundled_vector_index"

    data_folder = project_root / "backend" / "data" / "raw" / "guardian_by_year"
    df = load_and_clean_guardian_years(
        years=years,
        folder=data_folder,
        drop_duplicates=True,
        min_body_text_chars=min_body_text_chars,
    )
    _persist_guardian_articles(
        df,
        batch_size=batch_size,
        store_body_text=store_body_text,
    )
    return "raw_source"


def initialize_offline_data_pipeline(
    app,
    project_root,
    years=DEFAULT_YEARS,
    min_body_text_chars=DEFAULT_MIN_BODY_TEXT_CHARS,
):
    """
    Ensure both offline assets are ready:
      1) SQLite guardian_articles table
      2) TF-IDF vector index artifacts
    """
    with app.app_context():
        db.create_all()

        store_body_text = _should_store_body_text()
        bundled_articles = _load_bundled_guardian_articles()
        bundled_articles_available = not bundled_articles.empty
        existing_count = GuardianArticle.query.count()
        should_seed = existing_count == 0

        if existing_count > 0 and _existing_data_needs_refresh(
            expected_years=years,
            allow_missing_body_text=(not store_body_text) or bundled_articles_available,
        ):
            print("Existing Guardian rows do not match the configured source data. Rebuilding dataset.")
            GuardianArticle.query.delete()
            db.session.commit()
            should_seed = True

        if should_seed:
            _seed_guardian_articles(
                project_root=project_root,
                years=years,
                min_body_text_chars=min_body_text_chars,
                bundled_articles=bundled_articles,
                store_body_text=store_body_text,
            )
        elif not store_body_text:
            cleared_rows = _clear_stored_body_text()
            if cleared_rows:
                print(f"Cleared stored article body text from {cleared_rows} Guardian rows.")

        try:
            from backend.text_preprocess import (
                DEFAULT_INDEX_DIR,
                DEFAULT_INDEX_NAME,
                preprocess_tfidf_index,
            )

            preprocess_tfidf_index(
                db_path=Path(app.instance_path) / "data.db",
                index_dir=DEFAULT_INDEX_DIR,
                index_name=DEFAULT_INDEX_NAME,
                force_rebuild=False,
            )
            print("TF-IDF artifacts are ready.")
        except Exception as exc:
            print(f"Warning: TF-IDF precompute failed; search may fail until rebuilt. Details: {exc}")
