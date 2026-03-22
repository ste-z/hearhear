import os
from pathlib import Path

import pandas as pd
from sqlalchemy import and_, func

from backend.claim_store import (
    PACKAGED_CLAIM_RESULTS_DIR,
    expected_claim_record_count,
    iter_claim_records,
)
from backend.data_import import load_and_clean_guardian_years
from backend.runtime_debug import log_runtime_event
from models import GuardianArticle, GuardianArticleClaim, db


DEFAULT_YEARS = set(range(2010, 2026))
DEFAULT_MIN_BODY_TEXT_CHARS = 1000
DEFAULT_BATCH_SIZE = 500
DEFAULT_CLAIM_BATCH_SIZE = 500
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


def _filter_articles_to_years(articles, years=None):
    if articles is None:
        return pd.DataFrame()
    if not isinstance(articles, pd.DataFrame):
        raise TypeError("articles must be a pandas DataFrame.")
    if articles.empty:
        return articles.reset_index(drop=True).copy()

    expected_years = _normalized_years(years)
    if not expected_years or "year" not in articles.columns:
        return articles.reset_index(drop=True).copy()

    normalized = articles.reset_index(drop=True).copy()
    article_years = pd.to_numeric(normalized["year"], errors="coerce").astype("Int64")
    return normalized.loc[article_years.isin(expected_years)].reset_index(drop=True).copy()


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


def _claim_row_count():
    return int(db.session.query(func.count(GuardianArticleClaim.article_id)).scalar() or 0)


def _existing_claims_need_refresh(claim_root_dir=PACKAGED_CLAIM_RESULTS_DIR):
    expected_count = expected_claim_record_count(root_dir=claim_root_dir)
    existing_count = _claim_row_count()

    if expected_count is not None and existing_count != expected_count:
        return True

    missing_summary_exists = db.session.query(GuardianArticleClaim.article_id).filter(
        func.trim(func.coalesce(GuardianArticleClaim.central_claim_summary, "")) == "",
    ).limit(1).first() is not None

    return missing_summary_exists


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


def _persist_guardian_claims(
    claim_root_dir=PACKAGED_CLAIM_RESULTS_DIR,
    batch_size=DEFAULT_CLAIM_BATCH_SIZE,
):
    claim_root_dir = Path(claim_root_dir)
    if not claim_root_dir.exists():
        print(f"Claim results directory not found at {claim_root_dir}; skipping claim initialization.")
        return

    batch = []
    for record in iter_claim_records(root_dir=claim_root_dir):
        claim = GuardianArticleClaim(
            article_id=_clean_str(record.get("article_id")),
            title=_clean_str(record.get("title")),
            year=record.get("year"),
            central_claim_summary=_clean_str(record.get("central_claim_summary")),
            has_clear_central_thesis=record.get("has_clear_central_thesis"),
            thesis_sentence_id=_clean_str(record.get("thesis_sentence_id")),
            thesis_sentence=_clean_str(record.get("thesis_sentence")),
            support_sentence_ids=_clean_list(record.get("support_sentence_ids")),
            support_sentences=_clean_list(record.get("support_sentences")),
            secondary_claim_ids=_clean_list(record.get("secondary_claim_ids")),
            secondary_claim_sentences=_clean_list(record.get("secondary_claim_sentences")),
        )
        batch.append(claim)

        if len(batch) >= batch_size:
            db.session.bulk_save_objects(batch)
            db.session.commit()
            batch.clear()

    if batch:
        db.session.bulk_save_objects(batch)
        db.session.commit()

    print(
        f"Claims database initialized with {_claim_row_count()} rows from {claim_root_dir}."
    )


def _seed_guardian_claims(
    claim_root_dir=PACKAGED_CLAIM_RESULTS_DIR,
    batch_size=DEFAULT_CLAIM_BATCH_SIZE,
):
    _persist_guardian_claims(
        claim_root_dir=claim_root_dir,
        batch_size=batch_size,
    )
    return "packaged_claim_results"


def _load_bundled_guardian_articles(years=None):
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

    return _filter_articles_to_years(articles, years=years)


def _seed_guardian_articles(
    project_root,
    years=DEFAULT_YEARS,
    min_body_text_chars=DEFAULT_MIN_BODY_TEXT_CHARS,
    batch_size=DEFAULT_BATCH_SIZE,
    bundled_articles=None,
    store_body_text=False,
):
    if bundled_articles is None:
        bundled_articles = _load_bundled_guardian_articles(years=years)
    else:
        bundled_articles = _filter_articles_to_years(bundled_articles, years=years)
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


def _warm_runtime_assets():
    try:
        from search_helpers import build_vector_processor

        log_runtime_event("startup_warm.vector_index_start")
        vector_index = build_vector_processor(force_rebuild=False)
        log_runtime_event(
            "startup_warm.vector_index_done",
            n_docs=getattr(vector_index, "n_docs", None),
            n_terms=getattr(vector_index, "n_terms", None),
        )
        print("TF-IDF search index loaded into memory.")
    except Exception as exc:
        print(
            "Warning: TF-IDF warm-up failed; the first search may still cold-start. "
            f"Details: {exc}"
        )

    try:
        from backend.nli_processor import load_nli_bundle

        log_runtime_event("startup_warm.nli_start")
        bundle = load_nli_bundle()
        log_runtime_event(
            "startup_warm.nli_done",
            model_name=bundle.get("model_name"),
            device=str(bundle.get("device")),
        )
        print("NLI model loaded into memory.")
    except Exception as exc:
        print(
            "Warning: NLI warm-up failed; the first stance rerank may still cold-start. "
            f"Details: {exc}"
        )


def initialize_offline_data_pipeline(
    app,
    project_root,
    years=DEFAULT_YEARS,
    min_body_text_chars=DEFAULT_MIN_BODY_TEXT_CHARS,
):
    """
    Ensure all offline assets are ready:
      1) SQLite guardian_articles table
      2) SQLite guardian_article_claims table
      3) TF-IDF vector index artifacts
    """
    with app.app_context():
        db.create_all()

        store_body_text = _should_store_body_text()
        bundled_articles = _load_bundled_guardian_articles(years=years)
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

        existing_claim_count = _claim_row_count()
        should_seed_claims = existing_claim_count == 0
        if existing_claim_count > 0 and _existing_claims_need_refresh():
            print("Existing Guardian claim rows do not match the packaged claim data. Rebuilding claim table.")
            GuardianArticleClaim.query.delete()
            db.session.commit()
            should_seed_claims = True

        if should_seed_claims:
            _seed_guardian_claims()

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
                years=years,
            )
            print("TF-IDF artifacts are ready.")
        except Exception as exc:
            print(f"Warning: TF-IDF precompute failed; search may fail until rebuilt. Details: {exc}")

        _warm_runtime_assets()
