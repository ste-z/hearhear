import os
import re
from pathlib import Path

import pandas as pd
from sqlalchemy import and_, func, or_, text

from backend.claims.claim_store import (
    PACKAGED_CLAIM_RESULTS_DIR,
    expected_claim_record_count,
    iter_claim_records,
)
from backend.imports.data_import import load_and_clean_guardian_years
from backend.db.models import GuardianArticle, GuardianArticleClaim, db
from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.indexing.corpus import (
    _filter_articles_to_years,
    _normalized_years,
)


DEFAULT_YEARS = set(range(2015, 2026))
DEFAULT_MIN_BODY_TEXT_CHARS = 1000
DEFAULT_BATCH_SIZE = 500
DEFAULT_CLAIM_BATCH_SIZE = 500
DEFAULT_BUNDLED_INDEX_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "vector_index"
DEFAULT_BUNDLED_INDEX_NAME = "guardian_tfidf_svd"
DEFAULT_ANALYSIS_EXPORT_DIR = Path(__file__).resolve().parent.parent / "data" / "processed" / "analysis_exports"
DEFAULT_SVD_DIMENSION_SUMMARY_EXPORT_TOP_TERMS = 10
STORE_GUARDIAN_BODY_TEXT_ENV = "STORE_GUARDIAN_BODY_TEXT_IN_DB"
WORD_COUNT_PATTERN = re.compile(r"\b[\w'-]+\b")


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


def _article_character_count(value):
    return len(_clean_str(value))


def _article_word_count(value):
    return len(WORD_COUNT_PATTERN.findall(_clean_str(value)))


def _clean_count(value, fallback=0):
    try:
        if _is_missing(value):
            return int(fallback)
        return max(0, int(value))
    except (TypeError, ValueError):
        return int(fallback)


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


def _env_flag(name, default=False):
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _should_store_body_text():
    return _env_flag(STORE_GUARDIAN_BODY_TEXT_ENV, default=False)


def _ensure_guardian_article_length_columns():
    rows = db.session.execute(text("PRAGMA table_info(guardian_articles)")).fetchall()
    existing_columns = {str(row[1]) for row in rows}

    column_specs = {
        "body_character_count": "INTEGER NOT NULL DEFAULT 0",
        "body_word_count": "INTEGER NOT NULL DEFAULT 0",
    }
    for column_name, column_spec in column_specs.items():
        if column_name not in existing_columns:
            db.session.execute(
                text(f"ALTER TABLE guardian_articles ADD COLUMN {column_name} {column_spec}")
            )

    db.session.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_guardian_articles_body_character_count "
        "ON guardian_articles (body_character_count)"
    ))
    db.session.execute(text(
        "CREATE INDEX IF NOT EXISTS ix_guardian_articles_body_word_count "
        "ON guardian_articles (body_word_count)"
    ))
    db.session.commit()


def _article_length_lookup_from_dataframe(articles):
    if articles is None or articles.empty or "id" not in articles.columns:
        return {}

    lookup = {}
    for row in articles.itertuples(index=False):
        row_data = row._asdict()
        article_id = _clean_str(row_data.get("id"))
        if not article_id:
            continue

        body_text = _clean_str(row_data.get("body_text"))
        character_count = _clean_count(
            row_data.get("body_character_count"),
            fallback=_clean_count(
                row_data.get("body_text_length"),
                fallback=_article_character_count(body_text),
            ),
        )
        word_count = _clean_count(
            row_data.get("body_word_count"),
            fallback=_article_word_count(body_text),
        )
        lookup[article_id] = (character_count, word_count)

    return lookup


def _populate_missing_article_length_metadata(bundled_articles=None, batch_size=DEFAULT_BATCH_SIZE):
    bundled_lookup = _article_length_lookup_from_dataframe(bundled_articles)
    missing_query = GuardianArticle.query.filter(
        or_(
            GuardianArticle.body_character_count <= 0,
            GuardianArticle.body_word_count <= 0,
        )
    )

    updated = 0
    pending = 0
    for article in missing_query.yield_per(batch_size):
        body_text = _clean_str(article.body_text)
        fallback_counts = bundled_lookup.get(article.id)
        fallback_character_count = fallback_counts[0] if fallback_counts else 0
        fallback_word_count = fallback_counts[1] if fallback_counts else 0
        character_count = _clean_count(
            article.body_character_count,
            fallback=_article_character_count(body_text) or fallback_character_count,
        )
        word_count = _clean_count(
            article.body_word_count,
            fallback=_article_word_count(body_text) or fallback_word_count,
        )

        if character_count <= 0 and fallback_character_count > 0:
            character_count = fallback_character_count
        if word_count <= 0 and fallback_word_count > 0:
            word_count = fallback_word_count

        if (
            article.body_character_count != character_count
            or article.body_word_count != word_count
        ):
            article.body_character_count = character_count
            article.body_word_count = word_count
            updated += 1
            pending += 1

        if pending >= batch_size:
            db.session.commit()
            pending = 0

    if pending:
        db.session.commit()

    return updated


def _missing_article_length_metadata_exists():
    return db.session.query(GuardianArticle.id).filter(
        or_(
            GuardianArticle.body_character_count <= 0,
            GuardianArticle.body_word_count <= 0,
        )
    ).limit(1).first() is not None


def _export_startup_svd_dimension_summaries(processor):
    if processor is None or not hasattr(processor, "export_dimension_summaries"):
        return None

    from backend.text_processing.svd_processor import DEFAULT_SVD_INDEX_NAME

    output_path = (
        DEFAULT_ANALYSIS_EXPORT_DIR
        / f"{DEFAULT_SVD_INDEX_NAME}_all_dimensions_summary.csv"
    )

    log_runtime_event(
        "startup_warm.svd_dimension_export_start",
        output_path=str(output_path),
        top_terms=DEFAULT_SVD_DIMENSION_SUMMARY_EXPORT_TOP_TERMS,
    )
    export_path, df = processor.export_dimension_summaries(
        output_path=output_path,
        dimensions=None,
        top_n=DEFAULT_SVD_DIMENSION_SUMMARY_EXPORT_TOP_TERMS,
    )
    log_runtime_event(
        "startup_warm.svd_dimension_export_done",
        output_path=str(export_path),
        row_count=len(df),
        top_terms=DEFAULT_SVD_DIMENSION_SUMMARY_EXPORT_TOP_TERMS,
    )
    return export_path


def _existing_data_needs_refresh(expected_years=None, allow_missing_body_text=False):
    expected_year_set = set(_normalized_years(expected_years) or [])

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

    missing_length_metadata_exists = _missing_article_length_metadata_exists()

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
        missing_length_metadata_exists,
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
        body_text = _clean_str(row_data.get("body_text"))
        body_character_count = _clean_count(
            row_data.get("body_character_count"),
            fallback=_clean_count(
                row_data.get("body_text_length"),
                fallback=_article_character_count(body_text),
            ),
        )
        body_word_count = _clean_count(
            row_data.get("body_word_count"),
            fallback=_article_word_count(body_text),
        )
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
            body_text=body_text if store_body_text else "",
            body_character_count=body_character_count,
            body_word_count=body_word_count,
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
    if not store_body_text and not bundled_articles.empty:
        print("Seeding Guardian articles from bundled vector index metadata.")
        _persist_guardian_articles(
            bundled_articles,
            batch_size=batch_size,
            store_body_text=store_body_text,
        )
        return "bundled_vector_index"

    data_folder = project_root / "data" / "raw" / "guardian_by_year"
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
    retrieval_labels = {
        "tfidf": "TF-IDF",
        "svd": "SVD",
        "minilm": "Enhanced Semantic (MiniLM)",
    }
    try:
        from backend.text_processing.search_helpers import (
            DEFAULT_RETRIEVAL_MODEL,
            build_retrieval_processor,
            normalize_retrieval_model,
        )

        retrieval_model = normalize_retrieval_model(DEFAULT_RETRIEVAL_MODEL)
        label = retrieval_labels.get(
            retrieval_model,
            retrieval_model.replace("_", " ").upper(),
        )
        log_runtime_event(
            "startup_warm.vector_index_start",
            retrieval_model=retrieval_model,
        )
        vector_index = build_retrieval_processor(
            retrieval_model=retrieval_model,
            force_rebuild=False,
            ensure_preprocessed=True,
        )
        log_runtime_event(
            "startup_warm.vector_index_done",
            retrieval_model=retrieval_model,
            n_docs=getattr(vector_index, "n_docs", None),
            n_terms=getattr(vector_index, "n_terms", None),
        )
        if retrieval_model == "svd":
            try:
                export_path = _export_startup_svd_dimension_summaries(
                    vector_index
                )
                if export_path is not None:
                    print(
                        "SVD dimension summary exported to "
                        f"{export_path}."
                    )
            except Exception as exc:
                print(
                    "Warning: SVD dimension summary export failed; "
                    f"startup will continue. Details: {exc}"
                )
        print(f"{label} retrieval artifacts ensured and loaded into memory.")
    except Exception as exc:
        print(
            "Warning: retrieval warm-up initialization failed; startup will continue. "
            f"Details: {exc}"
        )


def initialize_offline_data_pipeline(
    app,
    project_root,
    years=DEFAULT_YEARS,
    min_body_text_chars=DEFAULT_MIN_BODY_TEXT_CHARS,
    warm_runtime_assets=True,
):
    """
    Ensure all offline assets are ready:
      1) SQLite guardian_articles table
      2) SQLite guardian_article_claims table
      3) Default retrieval index artifacts
    """
    with app.app_context():
        db.create_all()
        _ensure_guardian_article_length_columns()

        store_body_text = _should_store_body_text()
        bundled_articles = None
        existing_count = GuardianArticle.query.count()
        should_seed = existing_count == 0
        if existing_count > 0 and _missing_article_length_metadata_exists():
            bundled_articles = _load_bundled_guardian_articles(years=years)
            updated_length_rows = _populate_missing_article_length_metadata(
                bundled_articles=bundled_articles,
            )
            if updated_length_rows:
                print(
                    "Backfilled article length metadata for "
                    f"{updated_length_rows} Guardian rows."
                )

        if existing_count > 0 and _existing_data_needs_refresh(
            expected_years=years,
            allow_missing_body_text=not store_body_text,
        ):
            print("Existing Guardian rows do not match the configured source data. Rebuilding dataset.")
            GuardianArticle.query.delete()
            db.session.commit()
            should_seed = True

        if should_seed:
            if bundled_articles is None:
                bundled_articles = _load_bundled_guardian_articles(years=years)
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

        if warm_runtime_assets:
            _warm_runtime_assets()
