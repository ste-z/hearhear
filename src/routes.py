"""
Routes: React app serving and Guardian article search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os
from flask import send_from_directory, request, jsonify
from sqlalchemy import or_
from models import GuardianArticle
from search_helpers import build_matches, build_vector_processor, serialize_article
from io import BytesIO
from pypdf import PdfReader

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────

def keyword_search(query):
    if not query or not query.strip():
        return []

    query = query.strip()
    if len(query) < 3:
        return []
    results = (
        GuardianArticle.query.filter(
            or_(
                GuardianArticle.title.ilike(f"%{query}%"),
                GuardianArticle.summary.ilike(f"%{query}%"),
            )
        )
        .order_by(GuardianArticle.date.desc())
        .limit(100)
        .all()
    )

    return [serialize_article(article) for article in results]

def tfidf_cos_search(query):
    if not query or not query.strip():
        return []
    query = query.strip()
    if len(query) < 3:
        return []

    processor = build_vector_processor()
    if processor is None:
        return []

    ranked = processor.search(query, top_n=100)
    return build_matches(ranked)

def json_search(query):
    """
    Primary search used by /api/articles and optional LLM retrieval.
    Defaults to TF-IDF cosine search, falls back to keyword SQL search.
    """
    try:
        return tfidf_cos_search(query)
    except Exception:
        return keyword_search(query)

def _extract_pdf_text(uploaded_file, max_pages=20, max_chars=20000):
    """
    Extract text from an uploaded PDF file.
    Limits pages/chars so very large PDFs do not overwhelm search.
    """
    if uploaded_file is None:
        return ""

    filename = (uploaded_file.filename or "").lower()
    content_type = (uploaded_file.mimetype or "").lower()

    if not filename.endswith(".pdf") and content_type != "application/pdf":
        return ""

    try:
        file_bytes = uploaded_file.read()
        if not file_bytes:
            return ""

        reader = PdfReader(BytesIO(file_bytes))
        chunks = []

        for page in reader.pages[:max_pages]:
            page_text = page.extract_text() or ""
            page_text = page_text.strip()
            if page_text:
                chunks.append(page_text)
            if sum(len(c) for c in chunks) >= max_chars:
                break

        text = "\n\n".join(chunks).strip()
        return text[:max_chars]
    except Exception:
        return ""
    finally:
        try:
            uploaded_file.stream.seek(0)
        except Exception:
            pass


def _extract_search_text():
    """
    Support:
      - GET query params
      - POST JSON payloads
      - POST form payloads
      - optional uploaded PDF under field name 'pdf'

    Returns combined search text from typed text and extracted PDF text.
    """
    typed_text = ""

    if request.method == "POST":
        if request.is_json:
            payload = request.get_json(silent=True) or {}
            typed_text = (
                payload.get("q")
                or payload.get("query")
                or payload.get("text")
                or payload.get("title")
                or ""
            )
            return str(typed_text).strip()

        typed_text = (
            request.form.get("q")
            or request.form.get("query")
            or request.form.get("text")
            or request.form.get("title")
            or ""
        )

        uploaded_pdf = request.files.get("pdf")
        pdf_text = _extract_pdf_text(uploaded_pdf)

        parts = [str(typed_text).strip(), pdf_text.strip()]
        combined = "\n\n".join(part for part in parts if part)
        return combined.strip()

    typed_text = request.args.get("q", "") or request.args.get("title", "")
    return str(typed_text).strip()


def register_routes(app):
    @app.route('/', defaults={'path': ''})
    @app.route('/<path:path>')
    def serve(path):
        if path != "" and os.path.exists(os.path.join(app.static_folder, path)):
            return send_from_directory(app.static_folder, path)
        else:
            return send_from_directory(app.static_folder, 'index.html')

    @app.route("/api/config")
    def config():
        return jsonify({"use_llm": USE_LLM})

    @app.route("/api/articles", methods=["GET", "POST"])
    @app.route("/api/articles/search", methods=["POST"])
    def articles_search():
        text = _extract_search_text()
        return jsonify(json_search(text))

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
