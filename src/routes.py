"""
Routes: React app serving and Guardian article search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os
from flask import send_from_directory, request, jsonify
from sqlalchemy import or_
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge
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

def tfidf_cos_search(query, top_n=100):
    if not query or not query.strip():
        return []
    query = query.strip()
    if len(query) < 3:
        return []

    processor = build_vector_processor()
    if processor is None:
        return []

    ranked = processor.search(query, top_n=top_n)
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


def stance_search(topic, opinion, topic_weight=0.5, stance_weight=0.5, top_n=20):
    from backend.stance_rerank import rerank_article_matches

    topic_text = str(topic or "").strip()
    opinion_text = str(opinion or "").strip()
    if len(topic_text) < 2 or len(opinion_text) < 2:
        return []

    topic_matches = tfidf_cos_search(topic_text, top_n=top_n)
    if not topic_matches:
        return []

    return rerank_article_matches(
        article_matches=topic_matches,
        topic=topic_text,
        opinion=opinion_text,
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        top_n=top_n,
    )


def essay_claim_candidates(essay_text, top_n=5):
    from backend.nli_processor import score_claim_sentences
    from backend.sentence_splitter import sentence_rows_from_text

    resolved_text = str(essay_text or "").strip()
    if len(resolved_text) < 3:
        return {
            "essay_text": resolved_text,
            "sentence_count": 0,
            "candidates": [],
        }

    sentence_rows = sentence_rows_from_text(resolved_text, prefix="essay_s")
    candidates = score_claim_sentences(sentence_rows, top_n=top_n)
    return {
        "essay_text": resolved_text,
        "sentence_count": len(sentence_rows),
        "candidates": candidates,
    }


def essay_search(
    essay_text,
    selected_thesis_sentence,
    selected_thesis_id=None,
    topic_weight=0.5,
    stance_weight=0.5,
    top_n=20,
):
    from backend.stance_rerank import rerank_article_matches_by_statement

    resolved_essay = str(essay_text or "").strip()
    resolved_thesis = str(selected_thesis_sentence or "").strip()
    if len(resolved_essay) < 3:
        return []

    topic_matches = tfidf_cos_search(resolved_essay, top_n=top_n)
    if not topic_matches:
        return []
    if not resolved_thesis:
        return topic_matches

    reranked = rerank_article_matches_by_statement(
        article_matches=topic_matches,
        statement=resolved_thesis,
        topic_weight=topic_weight,
        stance_weight=stance_weight,
        top_n=top_n,
    )
    for match in reranked:
        match["selected_thesis_sentence"] = resolved_thesis
        match["selected_thesis_id"] = selected_thesis_id
        match["essay_query_text"] = resolved_essay
    return reranked

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


def _request_payload():
    if request.method == "GET":
        return request.args.to_dict(flat=True)

    if request.is_json:
        return request.get_json(silent=True) or {}

    return request.form.to_dict(flat=True)


def _coerce_float(value, default):
    try:
        return float(value)
    except (TypeError, ValueError):
        return float(default)


def _coerce_int(value, default, minimum=1, maximum=100):
    try:
        resolved = int(value)
    except (TypeError, ValueError):
        resolved = int(default)
    return max(int(minimum), min(int(maximum), resolved))


def _extract_request_context():
    payload = _request_payload()
    mode = str(payload.get("mode") or "essay").strip().lower()
    topic = str(payload.get("topic") or "").strip()
    opinion = str(payload.get("opinion") or "").strip()
    topic_weight = _coerce_float(payload.get("topic_weight"), 0.5)
    stance_weight = _coerce_float(payload.get("stance_weight"), 0.5)
    rerank_top_k = _coerce_int(payload.get("top_k"), 20, minimum=1, maximum=100)
    candidate_top_n = _coerce_int(payload.get("candidate_top_n"), 5, minimum=1, maximum=10)
    selected_thesis_sentence = str(payload.get("selected_thesis_sentence") or "").strip()
    selected_thesis_id = str(payload.get("selected_thesis_id") or "").strip() or None

    typed_text = (
        payload.get("q")
        or payload.get("query")
        or payload.get("text")
        or payload.get("title")
        or ""
    )
    typed_text = str(typed_text).strip()

    pdf_text = ""
    if request.method == "POST" and not request.is_json:
        pdf_text = _extract_pdf_text(request.files.get("pdf"))

    parts = [typed_text, pdf_text.strip()]
    essay_text = "\n\n".join(part for part in parts if part).strip()

    return {
        "mode": mode,
        "topic": topic,
        "opinion": opinion,
        "topic_weight": topic_weight,
        "stance_weight": stance_weight,
        "rerank_top_k": rerank_top_k,
        "candidate_top_n": candidate_top_n,
        "selected_thesis_sentence": selected_thesis_sentence,
        "selected_thesis_id": selected_thesis_id,
        "essay_text": essay_text,
    }


def _api_error_response(exc):
    if isinstance(exc, RequestEntityTooLarge):
        return jsonify({"error": "Uploaded file is too large. Try a smaller PDF."}), 413

    if isinstance(exc, ValueError):
        return jsonify({"error": str(exc)}), 400

    if isinstance(exc, RuntimeError):
        return jsonify({"error": str(exc)}), 500

    if isinstance(exc, HTTPException):
        message = str(exc.description or exc.name or "Request failed.")
        return jsonify({"error": message}), int(exc.code or 500)

    return jsonify({"error": "Unexpected server error while processing the request."}), 500


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
        try:
            context = _extract_request_context()
            if context["mode"] == "stance":
                results = stance_search(
                    topic=context["topic"],
                    opinion=context["opinion"],
                    topic_weight=context["topic_weight"],
                    stance_weight=context["stance_weight"],
                    top_n=context["rerank_top_k"],
                )
            elif context["mode"] == "essay":
                results = essay_search(
                    essay_text=context["essay_text"],
                    selected_thesis_sentence=context["selected_thesis_sentence"],
                    selected_thesis_id=context["selected_thesis_id"],
                    topic_weight=context["topic_weight"],
                    stance_weight=context["stance_weight"],
                    top_n=context["rerank_top_k"],
                )
            else:
                results = json_search(context["essay_text"])
            return jsonify(results)
        except Exception as exc:
            app.logger.exception("API request to /api/articles failed")
            return _api_error_response(exc)

    @app.route("/api/essay/claim-candidates", methods=["POST"])
    def essay_claim_candidates_route():
        try:
            context = _extract_request_context()
            return jsonify(
                essay_claim_candidates(
                    essay_text=context["essay_text"],
                    top_n=context["candidate_top_n"],
                )
            )
        except Exception as exc:
            app.logger.exception("API request to /api/essay/claim-candidates failed")
            return _api_error_response(exc)

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
