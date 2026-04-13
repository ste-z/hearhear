"""
Routes: React app serving and Guardian article search API.

To enable AI chat, set USE_LLM = True below. See backend/api/llm_routes.py for AI code.
"""
import os

from flask import send_from_directory, request, jsonify
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from backend.runtime.runtime_debug import log_runtime_event
from backend.services.essay_service import essay_claim_candidates, essay_search
from backend.services.pdf_service import extract_pdf_text
from backend.services.retrieval_service import (
    available_article_year_range,
    attach_query_svd_chart_dimensions,
    DEFAULT_AUTO_RERANK_THRESHOLDS,
    DEFAULT_RERANK_SELECTION_MODE,
    DEFAULT_RETRIEVAL_MODEL,
    json_search,
    MAX_AUTO_RERANK_CANDIDATES,
    normalize_retrieval_model,
    normalize_rerank_selection_mode,
    retrieval_query_svd_corpus_chart_dimensions,
    retrieval_query_svd_dimensions,
    SUPPORTED_RERANK_SELECTION_MODES,
    SUPPORTED_RETRIEVAL_MODELS,
    stance_search,
)
from backend.stance_processing.stance_rerank import DEFAULT_NORMALIZE_TOPIC_SCORES

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────


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


def _coerce_optional_int(value, label):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return int(text)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a whole number.")


def _coerce_optional_float(value, label):
    if value is None:
        return None
    text = str(value).strip()
    if not text:
        return None
    try:
        return float(text)
    except (TypeError, ValueError):
        raise ValueError(f"{label} must be a number.")


def _coerce_bool(value, default=False):
    if value is None:
        return bool(default)
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)

    text = str(value).strip().lower()
    if text in {"1", "true", "yes", "on"}:
        return True
    if text in {"0", "false", "no", "off"}:
        return False
    return bool(default)


def _extract_request_context():
    payload = _request_payload()
    mode = str(payload.get("mode") or "essay").strip().lower()
    topic = str(payload.get("topic") or "").strip()
    opinion = str(payload.get("opinion") or "").strip()
    topic_weight = _coerce_float(payload.get("topic_weight"), 0.4)
    stance_weight = _coerce_float(payload.get("stance_weight"), 0.4)
    recency_weight = _coerce_float(payload.get("recency_weight"), 0.2)
    rerank_top_k = _coerce_int(payload.get("top_k"), 20, minimum=1, maximum=100)
    candidate_top_n = _coerce_int(payload.get("candidate_top_n"), 5, minimum=1, maximum=10)
    normalize_topic_scores = _coerce_bool(
        payload.get("normalize_topic_scores"),
        DEFAULT_NORMALIZE_TOPIC_SCORES,
    )
    retrieval_model = normalize_retrieval_model(
        payload.get("retrieval_model")
        or payload.get("search_backend")
        or DEFAULT_RETRIEVAL_MODEL
    )
    rerank_selection_mode = normalize_rerank_selection_mode(
        payload.get("rerank_selection_mode"),
        DEFAULT_RERANK_SELECTION_MODE,
    )
    rerank_threshold = _coerce_optional_float(
        payload.get("rerank_threshold"),
        "Automatic rerank threshold",
    )
    year_start = _coerce_optional_int(payload.get("year_start"), "Start year")
    year_end = _coerce_optional_int(payload.get("year_end"), "End year")
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
        pdf_text = extract_pdf_text(request.files.get("pdf"))

    parts = [typed_text, pdf_text.strip()]
    essay_text = "\n\n".join(part for part in parts if part).strip()

    return {
        "mode": mode,
        "topic": topic,
        "opinion": opinion,
        "topic_weight": topic_weight,
        "stance_weight": stance_weight,
        "recency_weight": recency_weight,
        "rerank_top_k": rerank_top_k,
        "candidate_top_n": candidate_top_n,
        "normalize_topic_scores": normalize_topic_scores,
        "retrieval_model": retrieval_model,
        "rerank_selection_mode": rerank_selection_mode,
        "rerank_threshold": rerank_threshold,
        "year_start": year_start,
        "year_end": year_end,
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
        min_article_year, max_article_year = available_article_year_range()
        return jsonify({
            "use_llm": USE_LLM,
            "default_retrieval_model": DEFAULT_RETRIEVAL_MODEL,
            "default_normalize_topic_scores": DEFAULT_NORMALIZE_TOPIC_SCORES,
            "supported_retrieval_models": list(SUPPORTED_RETRIEVAL_MODELS),
            "default_rerank_selection_mode": DEFAULT_RERANK_SELECTION_MODE,
            "supported_rerank_selection_modes": list(SUPPORTED_RERANK_SELECTION_MODES),
            "default_auto_rerank_thresholds": dict(DEFAULT_AUTO_RERANK_THRESHOLDS),
            "max_auto_rerank_candidates": MAX_AUTO_RERANK_CANDIDATES,
            "min_article_year": min_article_year,
            "max_article_year": max_article_year,
        })

    @app.route("/api/articles", methods=["GET", "POST"])
    @app.route("/api/articles/search", methods=["POST"])
    def articles_search():
        try:
            context = _extract_request_context()
            log_runtime_event(
                "articles_search.start",
                mode=context["mode"],
                retrieval_model=context["retrieval_model"],
                essay_chars=len(context["essay_text"]),
                topic_chars=len(context["topic"]),
                opinion_chars=len(context["opinion"]),
                rerank_top_k=context["rerank_top_k"],
                normalize_topic_scores=context["normalize_topic_scores"],
                rerank_selection_mode=context["rerank_selection_mode"],
                rerank_threshold=context["rerank_threshold"],
            )
            empty_results_message = None
            if context["mode"] == "stance":
                search_payload = stance_search(
                    topic=context["topic"],
                    opinion=context["opinion"],
                    topic_weight=context["topic_weight"],
                    stance_weight=context["stance_weight"],
                    recency_weight=context["recency_weight"],
                    top_n=context["rerank_top_k"],
                    retrieval_model=context["retrieval_model"],
                    year_start=context["year_start"],
                    year_end=context["year_end"],
                    normalize_topic_scores=context["normalize_topic_scores"],
                    rerank_selection_mode=context["rerank_selection_mode"],
                    rerank_threshold=context["rerank_threshold"],
                )
            elif context["mode"] == "essay":
                search_payload = essay_search(
                    essay_text=context["essay_text"],
                    selected_thesis_sentence=context["selected_thesis_sentence"],
                    selected_thesis_id=context["selected_thesis_id"],
                    topic_weight=context["topic_weight"],
                    stance_weight=context["stance_weight"],
                    recency_weight=context["recency_weight"],
                    top_n=context["rerank_top_k"],
                    retrieval_model=context["retrieval_model"],
                    year_start=context["year_start"],
                    year_end=context["year_end"],
                    normalize_topic_scores=context["normalize_topic_scores"],
                    rerank_selection_mode=context["rerank_selection_mode"],
                    rerank_threshold=context["rerank_threshold"],
                )
            else:
                search_payload = {
                    "results": json_search(
                        context["essay_text"],
                        retrieval_model=context["retrieval_model"],
                        year_start=context["year_start"],
                        year_end=context["year_end"],
                    ),
                    "empty_results_message": None,
                }
            results = list(search_payload.get("results") or [])
            empty_results_message = search_payload.get("empty_results_message")
            if context["mode"] == "stance":
                query_text = context["topic"]
            else:
                query_text = context["essay_text"]
            query_svd_dimensions = retrieval_query_svd_dimensions(
                query=query_text,
                retrieval_model=context["retrieval_model"],
            )
            query_svd_corpus_chart_dimensions = retrieval_query_svd_corpus_chart_dimensions(
                query=query_text,
                retrieval_model=context["retrieval_model"],
            )
            results = attach_query_svd_chart_dimensions(
                results,
                query_dimensions=query_svd_dimensions,
                retrieval_model=context["retrieval_model"],
            )
            log_runtime_event(
                "articles_search.done",
                mode=context["mode"],
                retrieval_model=context["retrieval_model"],
                result_count=len(results),
                empty_results_message=empty_results_message,
            )
            return jsonify({
                "results": results,
                "query_svd_dimensions": query_svd_dimensions,
                "query_svd_corpus_chart_dimensions": query_svd_corpus_chart_dimensions,
                "empty_results_message": empty_results_message,
            })
        except Exception as exc:
            app.logger.exception("API request to /api/articles failed")
            return _api_error_response(exc)

    @app.route("/api/essay/claim-candidates", methods=["POST"])
    def essay_claim_candidates_route():
        try:
            context = _extract_request_context()
            log_runtime_event(
                "essay_claim_candidates.start",
                essay_chars=len(context["essay_text"]),
                candidate_top_n=context["candidate_top_n"],
            )
            return jsonify(
                essay_claim_candidates(
                    essay_text=context["essay_text"],
                    top_n=context["candidate_top_n"],
                )
            )
        except Exception as exc:
            app.logger.exception("API request to /api/essay/claim-candidates failed")
            return _api_error_response(exc)

    @app.route("/api/essay/extract-text", methods=["POST"])
    def essay_extract_text_route():
        try:
            essay_text = extract_pdf_text(request.files.get("pdf"))
            if not essay_text:
                raise ValueError(
                    "We couldn't read text from that PDF. Try another file or paste the essay manually."
                )
            return jsonify({"essay_text": essay_text})
        except Exception as exc:
            app.logger.exception("API request to /api/essay/extract-text failed")
            return _api_error_response(exc)

    if USE_LLM:
        from backend.api.llm_routes import register_chat_route
        register_chat_route(app, json_search)
