from pathlib import Path

from sqlalchemy import and_, func

from backend.data_import import load_and_clean_guardian_years
from models import GuardianArticle, db


DEFAULT_YEARS = set(range(2010, 2026))
DEFAULT_MIN_BODY_TEXT_CHARS = 1000
DEFAULT_BATCH_SIZE = 500


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
    if hasattr(value, "to_pydatetime"):
        py_dt = value.to_pydatetime()
        return None if _is_missing(py_dt) else py_dt
    return value


def _normalized_years(years):
    return {int(year) for year in set(years or [])}


def _existing_data_needs_refresh(expected_years=None):
    expected_year_set = _normalized_years(expected_years)

    missing_author_exists = db.session.query(GuardianArticle.id).filter(
        and_(
            GuardianArticle.n_contributors == 0,
            func.trim(func.coalesce(GuardianArticle.author_raw, "")) == "",
        )
    ).limit(1).first() is not None

    missing_body_exists = db.session.query(GuardianArticle.id).filter(
        func.trim(func.coalesce(GuardianArticle.body_text, "")) == "",
    ).limit(1).first() is not None

    missing_summary_exists = db.session.query(GuardianArticle.id).filter(
        func.trim(func.coalesce(GuardianArticle.summary, "")) == "",
    ).limit(1).first() is not None

    short_body_exists = db.session.query(GuardianArticle.id).filter(
        func.length(func.coalesce(GuardianArticle.body_text, "")) < DEFAULT_MIN_BODY_TEXT_CHARS,
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


def _seed_guardian_articles(
    project_root,
    years=DEFAULT_YEARS,
    min_body_text_chars=DEFAULT_MIN_BODY_TEXT_CHARS,
    batch_size=DEFAULT_BATCH_SIZE,
):
    data_folder = project_root / "backend" / "data" / "raw" / "guardian_by_year"
    df = load_and_clean_guardian_years(
        years=years,
        folder=data_folder,
        drop_duplicates=True,
        min_body_text_chars=min_body_text_chars,
    )
    if df.empty:
        print("Guardian dataset is empty; skipping initialization.")
        return

    batch = []
    for row in df.to_dict(orient="records"):
        authors = _clean_list(row.get("authors") or row.get("contributors"))
        author_display = ", ".join(authors)
        article = GuardianArticle(
            id=_clean_str(row.get("id")),
            title=_clean_str(row.get("title")),
            summary=_clean_str(row.get("summary")),
            date=_clean_datetime(row.get("date")),
            url=_clean_str(row.get("url")),
            author_raw=author_display or _clean_str(row.get("author_raw")),
            contributors=authors,
            n_contributors=int(row.get("n_contributors") or len(authors)),
            keywords=_clean_list(row.get("keywords")),
            body_text=_clean_str(row.get("body_text")),
            section_id="",
            section_name="",
            year=int(row.get("year") or 0),
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

        existing_count = GuardianArticle.query.count()
        should_seed = existing_count == 0

        if existing_count > 0 and _existing_data_needs_refresh(expected_years=years):
            print("Existing Guardian rows do not match the configured source data. Rebuilding dataset.")
            GuardianArticle.query.delete()
            db.session.commit()
            should_seed = True

        if should_seed:
            _seed_guardian_articles(
                project_root=project_root,
                years=years,
                min_body_text_chars=min_body_text_chars,
            )

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
                force_rebuild=should_seed,
            )
            print("TF-IDF artifacts are ready.")
        except Exception as exc:
            print(f"Warning: TF-IDF precompute failed; search may fail until rebuilt. Details: {exc}")
