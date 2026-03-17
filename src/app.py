import os
import re
import sys
from pathlib import Path
from dotenv import load_dotenv
from flask import Flask
from sqlalchemy import and_, func

from flask_cors import CORS

# src/ directory and project root (one level up)
project_root = Path(__file__).resolve().parent.parent

# Allow importing backend modules before importing src modules that depend on them.
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

load_dotenv()

from models import db, GuardianArticle
from routes import register_routes
from backend.data_import import load_and_clean_guardian_years

# Serve React build files from <project_root>/frontend/dist
app = Flask(__name__,
    static_folder=str(project_root / "frontend" / "dist"),
    static_url_path="")
CORS(app)

# Configure SQLite database - using 3 slashes for relative path
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///data.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

# Initialize database with app
db.init_app(app)

# Register routes
register_routes(app)

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


def _existing_data_needs_refresh():
    """
    Check whether existing rows violate notebook-aligned cleaning rules.
    """
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
        func.length(func.coalesce(GuardianArticle.body_text, "")) < 1000,
    ).limit(1).first() is not None

    return any([
        missing_author_exists,
        missing_body_exists,
        missing_summary_exists,
        short_body_exists,
    ])


def init_db():
    with app.app_context():
        # Create all tables
        db.create_all()

        existing_count = GuardianArticle.query.count()
        should_seed = existing_count == 0

        if existing_count > 0 and _existing_data_needs_refresh():
            print("Existing Guardian rows do not match cleaning rules. Rebuilding dataset.")
            GuardianArticle.query.delete()
            db.session.commit()
            should_seed = True

        # Initialize database with Guardian article data if empty / refreshed.
        if should_seed:
            data_folder = project_root / "backend" / "data" / "raw" / "guardian_by_year"
            years = set(range(2015, 2025)) 

            df = load_and_clean_guardian_years(
                years=years,
                folder=data_folder,
                drop_duplicates=True,
                min_body_text_chars=1000,
            )
            if df.empty:
                print("Guardian dataset is empty; skipping initialization.")
                return

            batch = []
            batch_size = 500
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

init_db()

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0", port=5001)
