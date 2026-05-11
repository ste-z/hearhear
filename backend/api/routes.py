"""
Routes: React app serving and Guardian article search API.

To enable AI chat, set USE_LLM = True below. See backend/api/llm_routes.py for AI code.
"""
import os
import gc
import json

from flask import Response, send_from_directory, request, jsonify, stream_with_context
from werkzeug.exceptions import HTTPException, RequestEntityTooLarge

from backend.runtime.runtime_debug import log_runtime_event
from backend.runtime.search_progress import (
    normalize_progress_id,
    publish_search_progress,
    remove_progress_channel,
    stream_search_progress,
)
from backend.services.filters.article_filters import (
    available_article_character_range,
    available_article_reading_time_range,
    available_article_word_range,
    available_article_year_range,
    normalize_article_character_range,
    normalize_article_reading_time_range,
    normalize_article_word_range,
    normalize_article_year_range,
)
from backend.services.filters.text_filters import normalize_avoid_words
from backend.services.essay_service import essay_claim_candidates, essay_search
from backend.services.pdf_service import extract_pdf_text
from backend.services.retrieval_service import (
    attach_query_svd_chart_dimensions,
    DEFAULT_AUTO_RERANK_THRESHOLDS,
    DEFAULT_CHUNK_ARTICLE_TOP_K,
    DEFAULT_CHUNK_AUTO_THRESHOLDS,
    DEFAULT_CHUNK_CANDIDATE_TOP_K,
    DEFAULT_RERANK_SELECTION_MODE,
    DEFAULT_RETRIEVAL_MODEL,
    json_search,
    MAX_CHUNK_CANDIDATE_TOP_K,
    MAX_AUTO_RERANK_CANDIDATES,
    normalize_retrieval_model,
    normalize_rerank_selection_mode,
    retrieval_query_typo_suggestion,
    retrieval_query_svd_corpus_chart_dimensions,
    retrieval_query_svd_dimensions,
    similar_articles,
    SUPPORTED_RERANK_SELECTION_MODES,
    SUPPORTED_RETRIEVAL_MODELS,
    stance_search,
)
from backend.stance_processing.stance_rerank import (
    DEFAULT_CHUNKING_MODE,
    DEFAULT_NORMALIZE_TOPIC_SCORES,
    DEFAULT_STANCE_METHOD,
    SUPPORTED_CHUNKING_MODES,
    SUPPORTED_STANCE_METHODS,
    normalize_chunking_mode,
    normalize_stance_method,
)

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = True
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


def _first_payload_value(payload, *keys):
    for key in keys:
        if key in payload:
            return payload.get(key)
    return None


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


def _coerce_string_list(value):
    if value is None:
        return []
    if isinstance(value, str):
        raw_values = value.split(",")
    else:
        try:
            raw_values = list(value)
        except TypeError:
            raw_values = [value]

    normalized = []
    seen = set()
    for raw_value in raw_values:
        text = str(raw_value or "").strip()
        if not text or text in seen:
            continue
        normalized.append(text)
        seen.add(text)
    return normalized


def _progress_callback(progress_id):
    normalized_id = normalize_progress_id(progress_id)
    if not normalized_id:
        return None

    def publish(stage, label, progress, **fields):
        publish_search_progress(
            normalized_id,
            stage=stage,
            label=label,
            progress=progress,
            **fields,
        )

    return publish


def _preload_runtime_artifact(payload):
    artifact = str(payload.get("artifact") or "").strip().lower()

    if artifact == "retrieval_index":
        from backend.text_processing.search_helpers import build_retrieval_processor

        retrieval_model = normalize_retrieval_model(
            payload.get("retrieval_model") or DEFAULT_RETRIEVAL_MODEL
        )
        processor = build_retrieval_processor(
            retrieval_model=retrieval_model,
            force_rebuild=False,
            ensure_preprocessed=True,
        )
        if retrieval_model == "minilm" and _coerce_bool(payload.get("load_model"), False):
            from backend.text_processing.minilm_processor import load_minilm_bundle

            load_minilm_bundle()
        return {
            "artifact": artifact,
            "retrieval_model": retrieval_model,
            "n_docs": getattr(processor, "n_docs", None),
            "n_terms": getattr(processor, "n_terms", None),
        }

    if artifact == "chunk_index":
        from backend.services.chunk_retrieval_service import build_chunk_retrieval_index

        retrieval_model = normalize_retrieval_model(
            payload.get("retrieval_model") or DEFAULT_RETRIEVAL_MODEL
        )
        if retrieval_model == "tfidf":
            retrieval_model = "svd"
        chunking_mode = normalize_chunking_mode(
            payload.get("chunking_mode") or DEFAULT_CHUNKING_MODE
        )
        if chunking_mode == "none":
            chunking_mode = "semantic"

        if _coerce_bool(payload.get("release_retrieval_indexes"), False):
            from backend.text_processing.search_helpers import unload_retrieval_processors

            unloaded = unload_retrieval_processors(keep_models=[])
            if unloaded:
                gc.collect()

        index = build_chunk_retrieval_index(
            retrieval_model=retrieval_model,
            chunking_mode=chunking_mode,
        )
        if retrieval_model == "minilm" and _coerce_bool(payload.get("load_model"), False):
            from backend.text_processing.minilm_processor import load_minilm_bundle

            load_minilm_bundle()
        return {
            "artifact": artifact,
            "retrieval_model": retrieval_model,
            "chunking_mode": chunking_mode,
            "n_chunks": getattr(index, "n_chunks", None),
        }

    if artifact == "nli_model":
        from backend.stance_processing.nli_processor import load_nli_bundle

        bundle = load_nli_bundle()
        return {
            "artifact": artifact,
            "model_name": bundle.get("model_name"),
            "device": str(bundle.get("device")),
        }

    supported = ", ".join(("retrieval_index", "chunk_index", "nli_model"))
    raise ValueError(f"Unsupported artifact {artifact!r}. Supported artifacts: {supported}.")


def _unload_unused_runtime_artifacts(payload):
    from backend.text_processing.search_helpers import unload_retrieval_processors
    from backend.services.chunk_retrieval_service import unload_chunk_retrieval_indexes

    keep_retrieval_models = _coerce_string_list(payload.get("keep_retrieval_models"))
    keep_chunk_indexes = payload.get("keep_chunk_indexes")
    if not isinstance(keep_chunk_indexes, list):
        keep_chunk_indexes = []

    unloaded_retrieval_models = unload_retrieval_processors(
        keep_models=keep_retrieval_models,
    )
    unloaded_chunk_indexes = unload_chunk_retrieval_indexes(
        keep_indexes=keep_chunk_indexes,
    )

    unloaded_minilm_model = False
    if _coerce_bool(payload.get("unload_minilm_model"), False):
        from backend.text_processing.minilm_processor import unload_minilm_bundle

        unloaded_minilm_model = bool(unload_minilm_bundle())

    unloaded_nli_model = False
    if _coerce_bool(payload.get("unload_nli_model"), False):
        from backend.stance_processing.nli_processor import unload_nli_bundle

        unloaded_nli_model = bool(unload_nli_bundle())

    unloaded_any = any([
        unloaded_retrieval_models,
        unloaded_chunk_indexes,
        unloaded_minilm_model,
        unloaded_nli_model,
    ])
    if unloaded_any:
        gc.collect()

    return {
        "unloaded_retrieval_models": unloaded_retrieval_models,
        "unloaded_chunk_indexes": [
            {"retrieval_model": model, "chunking_mode": chunking_mode}
            for model, chunking_mode in unloaded_chunk_indexes
        ],
        "unloaded_minilm_model": unloaded_minilm_model,
        "unloaded_nli_model": unloaded_nli_model,
    }


def _extract_request_context():
    payload = _request_payload()
    mode = str(payload.get("mode") or "essay").strip().lower()
    topic = str(payload.get("topic") or "").strip()
    opinion = str(payload.get("opinion") or "").strip()
    topic_weight = _coerce_float(payload.get("topic_weight"), 0.4)
    stance_weight = _coerce_float(payload.get("stance_weight"), 0.4)
    recency_weight = _coerce_float(payload.get("recency_weight"), 0.2)
    rerank_top_k = _coerce_int(payload.get("top_k"), 20, minimum=1, maximum=100)
    chunk_candidate_top_k = _coerce_int(
        payload.get("chunk_candidate_top_k")
        or payload.get("chunk_top_k")
        or payload.get("chunk_candidate_limit"),
        DEFAULT_CHUNK_CANDIDATE_TOP_K,
        minimum=25,
        maximum=MAX_CHUNK_CANDIDATE_TOP_K,
    )
    chunk_article_top_k = _coerce_int(
        payload.get("chunk_article_top_k")
        or payload.get("chunks_per_article"),
        DEFAULT_CHUNK_ARTICLE_TOP_K,
        minimum=1,
        maximum=10,
    )
    candidate_top_n = _coerce_int(payload.get("candidate_top_n"), 5, minimum=1, maximum=10)
    normalize_topic_scores = _coerce_bool(
        payload.get("normalize_topic_scores"),
        DEFAULT_NORMALIZE_TOPIC_SCORES,
    )
    raw_chunking_mode = payload.get("chunking_mode") or payload.get("chunking")
    legacy_use_chunking = _coerce_bool(
        payload.get("use_chunking")
        if "use_chunking" in payload
        else payload.get("paragraph_chunking"),
        False,
    )
    chunking_mode = normalize_chunking_mode(raw_chunking_mode, DEFAULT_CHUNKING_MODE)
    if raw_chunking_mode is None and legacy_use_chunking and chunking_mode == "none":
        chunking_mode = "paragraph"
    use_chunking = chunking_mode != "none"
    stance_method = normalize_stance_method(
        payload.get("stance_method")
        or payload.get("agreement_method")
        or DEFAULT_STANCE_METHOD
    )
    if use_chunking:
        stance_method = "llm"
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
    character_start = _coerce_optional_int(
        _first_payload_value(
            payload,
            "character_start",
            "article_character_start",
            "article_char_start",
        ),
        "Minimum article length",
    )
    character_end = _coerce_optional_int(
        _first_payload_value(
            payload,
            "character_end",
            "article_character_end",
            "article_char_end",
        ),
        "Maximum article length",
    )
    word_start = _coerce_optional_int(
        _first_payload_value(
            payload,
            "word_start",
            "article_word_start",
        ),
        "Minimum article word count",
    )
    word_end = _coerce_optional_int(
        _first_payload_value(
            payload,
            "word_end",
            "article_word_end",
        ),
        "Maximum article word count",
    )
    reading_time_start = _coerce_optional_int(
        _first_payload_value(
            payload,
            "reading_time_start",
            "article_reading_time_start",
            "reading_minutes_start",
        ),
        "Minimum article reading time",
    )
    reading_time_end = _coerce_optional_int(
        _first_payload_value(
            payload,
            "reading_time_end",
            "article_reading_time_end",
            "reading_minutes_end",
        ),
        "Maximum article reading time",
    )
    words_to_avoid = normalize_avoid_words(
        _first_payload_value(
            payload,
            "words_to_avoid",
            "avoid_words",
            "excluded_words",
            "avoid_terms",
        )
    )
    selected_thesis_sentence = str(payload.get("selected_thesis_sentence") or "").strip()
    selected_thesis_id = str(payload.get("selected_thesis_id") or "").strip() or None
    topic_feedback_irrelevant_article_ids = _coerce_string_list(
        payload.get("topic_feedback_irrelevant_article_ids")
        or payload.get("irrelevant_article_ids")
        or payload.get("not_relevant_article_ids")
    )
    skip_typo_correction = _coerce_bool(
        payload.get("skip_typo_correction")
        or payload.get("ignore_typo_correction"),
        False,
    )
    llm_label_irrelevant = _coerce_bool(
        _first_payload_value(
            payload,
            "llm_label_irrelevant",
            "llm_label_irrelevant_articles",
            "label_irrelevant_articles",
        ),
        True,
    )
    search_progress_id = normalize_progress_id(
        payload.get("search_progress_id")
        or payload.get("progress_id")
    )

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
        "chunk_candidate_top_k": chunk_candidate_top_k,
        "chunk_article_top_k": chunk_article_top_k,
        "candidate_top_n": candidate_top_n,
        "normalize_topic_scores": normalize_topic_scores,
        "use_chunking": use_chunking,
        "chunking_mode": chunking_mode,
        "stance_method": stance_method,
        "retrieval_model": retrieval_model,
        "rerank_selection_mode": rerank_selection_mode,
        "rerank_threshold": rerank_threshold,
        "year_start": year_start,
        "year_end": year_end,
        "character_start": character_start,
        "character_end": character_end,
        "word_start": word_start,
        "word_end": word_end,
        "reading_time_start": reading_time_start,
        "reading_time_end": reading_time_end,
        "words_to_avoid": words_to_avoid,
        "selected_thesis_sentence": selected_thesis_sentence,
        "selected_thesis_id": selected_thesis_id,
        "topic_feedback_irrelevant_article_ids": topic_feedback_irrelevant_article_ids,
        "skip_typo_correction": skip_typo_correction,
        "llm_label_irrelevant": llm_label_irrelevant,
        "search_progress_id": search_progress_id,
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
        min_article_characters, max_article_characters = available_article_character_range()
        min_article_words, max_article_words = available_article_word_range()
        min_article_reading_minutes, max_article_reading_minutes = available_article_reading_time_range()
        llm_agreement_available = bool(
            (os.getenv("SPARK_API_KEY") or os.getenv("API_KEY") or "").strip()
        )
        default_stance_method = (
            "llm"
            if llm_agreement_available and "llm" in SUPPORTED_STANCE_METHODS
            else DEFAULT_STANCE_METHOD
        )
        return jsonify({
            "use_llm": USE_LLM,
            "default_retrieval_model": DEFAULT_RETRIEVAL_MODEL,
            "default_normalize_topic_scores": DEFAULT_NORMALIZE_TOPIC_SCORES,
            "default_stance_method": default_stance_method,
            "supported_stance_methods": list(SUPPORTED_STANCE_METHODS),
            "default_use_chunking": False,
            "default_chunking_mode": DEFAULT_CHUNKING_MODE,
            "supported_chunking_modes": list(SUPPORTED_CHUNKING_MODES),
            "llm_agreement_available": llm_agreement_available,
            "supported_retrieval_models": list(SUPPORTED_RETRIEVAL_MODELS),
            "default_rerank_selection_mode": DEFAULT_RERANK_SELECTION_MODE,
            "supported_rerank_selection_modes": list(SUPPORTED_RERANK_SELECTION_MODES),
            "default_auto_rerank_thresholds": dict(DEFAULT_AUTO_RERANK_THRESHOLDS),
            "default_chunk_auto_rerank_thresholds": dict(DEFAULT_CHUNK_AUTO_THRESHOLDS),
            "default_chunk_candidate_top_k": DEFAULT_CHUNK_CANDIDATE_TOP_K,
            "default_chunk_article_top_k": DEFAULT_CHUNK_ARTICLE_TOP_K,
            "max_auto_rerank_candidates": MAX_AUTO_RERANK_CANDIDATES,
            "max_chunk_candidate_top_k": MAX_CHUNK_CANDIDATE_TOP_K,
            "min_article_year": min_article_year,
            "max_article_year": max_article_year,
            "min_article_characters": min_article_characters,
            "max_article_characters": max_article_characters,
            "min_article_words": min_article_words,
            "max_article_words": max_article_words,
            "min_article_reading_minutes": min_article_reading_minutes,
            "max_article_reading_minutes": max_article_reading_minutes,
        })

    @app.route("/api/runtime/preload", methods=["POST"])
    def preload_runtime_artifact_route():
        try:
            payload = _request_payload()
            result = _preload_runtime_artifact(payload)
            log_runtime_event(
                "runtime_preload.done",
                artifact=result.get("artifact"),
                retrieval_model=result.get("retrieval_model"),
                chunking_mode=result.get("chunking_mode"),
                model_name=result.get("model_name"),
            )
            return jsonify({
                "ok": True,
                **result,
            })
        except Exception as exc:
            app.logger.exception("API request to /api/runtime/preload failed")
            return _api_error_response(exc)

    @app.route("/api/runtime/unload", methods=["POST"])
    def unload_runtime_artifact_route():
        try:
            payload = _request_payload()
            result = _unload_unused_runtime_artifacts(payload)
            log_runtime_event(
                "runtime_unload.done",
                unloaded_retrieval_models=result.get("unloaded_retrieval_models"),
                unloaded_chunk_indexes=result.get("unloaded_chunk_indexes"),
                unloaded_minilm_model=result.get("unloaded_minilm_model"),
                unloaded_nli_model=result.get("unloaded_nli_model"),
            )
            return jsonify({
                "ok": True,
                **result,
            })
        except Exception as exc:
            app.logger.exception("API request to /api/runtime/unload failed")
            return _api_error_response(exc)

    @app.route("/api/articles/progress/<progress_id>", methods=["GET"])
    def articles_progress(progress_id):
        normalized_id = normalize_progress_id(progress_id)
        if not normalized_id:
            return jsonify({"error": "Invalid search progress id."}), 400

        def generate():
            try:
                for event in stream_search_progress(normalized_id):
                    if event.get("type") == "heartbeat":
                        yield ": heartbeat\n\n"
                        continue
                    yield f"event: progress\ndata: {json.dumps(event)}\n\n"
            finally:
                remove_progress_channel(normalized_id)

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={
                "Cache-Control": "no-cache",
                "X-Accel-Buffering": "no",
            },
        )

    @app.route("/api/typo-check", methods=["POST"])
    def typo_check():
        """Fast spell-check endpoint. Returns just the typo suggestion for a
        query without doing any retrieval — lets the frontend show "did you
        mean ..." in the proofreading slip immediately, in parallel with the
        slower full search request to /api/articles.
        """
        try:
            payload = request.get_json(silent=True) or {}
            query = str(payload.get("query") or "").strip()
            retrieval_model = normalize_retrieval_model(
                payload.get("retrieval_model") or DEFAULT_RETRIEVAL_MODEL
            )
            if not query:
                return jsonify({"typo_suggestion": None})
            suggestion = retrieval_query_typo_suggestion(
                query,
                retrieval_model=retrieval_model,
            )
            return jsonify({"typo_suggestion": suggestion})
        except Exception as exc:
            app.logger.exception("API request to /api/typo-check failed")
            return _api_error_response(exc)

    @app.route("/api/articles", methods=["GET", "POST"])
    @app.route("/api/articles/search", methods=["POST"])
    def articles_search():
        progress = None
        try:
            context = _extract_request_context()
            progress = _progress_callback(context["search_progress_id"])
            if progress:
                progress(
                    "received",
                    "Preparing search request",
                    0.03,
                    mode=context["mode"],
                )
            log_runtime_event(
                "articles_search.start",
                mode=context["mode"],
                retrieval_model=context["retrieval_model"],
                essay_chars=len(context["essay_text"]),
                topic_chars=len(context["topic"]),
                opinion_chars=len(context["opinion"]),
                rerank_top_k=context["rerank_top_k"],
                normalize_topic_scores=context["normalize_topic_scores"],
                use_chunking=context["use_chunking"],
                chunking_mode=context["chunking_mode"],
                stance_method=context["stance_method"],
                rerank_selection_mode=context["rerank_selection_mode"],
                rerank_threshold=context["rerank_threshold"],
                avoid_word_count=(
                    len(context["words_to_avoid"])
                    if context["retrieval_model"] == "tfidf"
                    else 0
                ),
                rocchio_irrelevant_count=len(context["topic_feedback_irrelevant_article_ids"]),
            )
            empty_results_message = None
            # Publish topic-relevance results to the SSE channel as soon as
            # that step finishes. Stage 2 uses this to scatter cards while the
            # slower agreement rerank is still running.
            def _emit_topic_done(topic_matches):
                if not progress:
                    return
                progress(
                    "topic_results",
                    "Topic candidates ready",
                    0.4,
                    topic_articles=list(topic_matches),
                )

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
                    character_start=context["character_start"],
                    character_end=context["character_end"],
                    word_start=context["word_start"],
                    word_end=context["word_end"],
                    reading_time_start=context["reading_time_start"],
                    reading_time_end=context["reading_time_end"],
                    words_to_avoid=context["words_to_avoid"],
                    normalize_topic_scores=context["normalize_topic_scores"],
                    stance_method=context["stance_method"],
                    use_chunking=context["use_chunking"],
                    chunking_mode=context["chunking_mode"],
                    rerank_selection_mode=context["rerank_selection_mode"],
                    rerank_threshold=context["rerank_threshold"],
                    chunk_candidate_top_k=context["chunk_candidate_top_k"],
                    chunk_article_top_k=context["chunk_article_top_k"],
                    topic_feedback_irrelevant_article_ids=context["topic_feedback_irrelevant_article_ids"],
                    llm_label_irrelevant=context["llm_label_irrelevant"],
                    progress_callback=progress,
                    on_topic_done=_emit_topic_done,
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
                    character_start=context["character_start"],
                    character_end=context["character_end"],
                    word_start=context["word_start"],
                    word_end=context["word_end"],
                    reading_time_start=context["reading_time_start"],
                    reading_time_end=context["reading_time_end"],
                    words_to_avoid=context["words_to_avoid"],
                    normalize_topic_scores=context["normalize_topic_scores"],
                    stance_method=context["stance_method"],
                    use_chunking=context["use_chunking"],
                    chunking_mode=context["chunking_mode"],
                    rerank_selection_mode=context["rerank_selection_mode"],
                    rerank_threshold=context["rerank_threshold"],
                    chunk_candidate_top_k=context["chunk_candidate_top_k"],
                    chunk_article_top_k=context["chunk_article_top_k"],
                    topic_feedback_irrelevant_article_ids=context["topic_feedback_irrelevant_article_ids"],
                    llm_label_irrelevant=context["llm_label_irrelevant"],
                    progress_callback=progress,
                    on_topic_done=_emit_topic_done,
                )
            else:
                if progress:
                    progress("topic", "Scoring topic relevance", 0.1)
                search_payload = {
                    "results": json_search(
                        context["essay_text"],
                        top_n=context["rerank_top_k"],
                        retrieval_model=context["retrieval_model"],
                        year_start=context["year_start"],
                        year_end=context["year_end"],
                        character_start=context["character_start"],
                        character_end=context["character_end"],
                        word_start=context["word_start"],
                        word_end=context["word_end"],
                        reading_time_start=context["reading_time_start"],
                        reading_time_end=context["reading_time_end"],
                        words_to_avoid=context["words_to_avoid"],
                        topic_feedback_irrelevant_article_ids=context["topic_feedback_irrelevant_article_ids"],
                    ),
                    "empty_results_message": None,
                }
                if progress:
                    progress("metadata", "Preparing result details", 0.88)
            results = list(search_payload.get("results") or [])
            empty_results_message = search_payload.get("empty_results_message")
            if context["mode"] == "stance":
                query_text = context["topic"]
            else:
                query_text = context["essay_text"]
            if progress:
                progress("metadata", "Preparing result details", 0.9)
            typo_suggestion = (
                retrieval_query_typo_suggestion(
                    query_text,
                    retrieval_model=context["retrieval_model"],
                )
                if context["mode"] == "stance" and not context["skip_typo_correction"]
                else None
            )
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
                query=query_text,
            )
            log_runtime_event(
                "articles_search.done",
                mode=context["mode"],
                retrieval_model=context["retrieval_model"],
                result_count=len(results),
                empty_results_message=empty_results_message,
            )
            if progress:
                progress(
                    "complete",
                    "Search complete",
                    1.0,
                    result_count=len(results),
                )
            return jsonify({
                "results": results,
                "query_svd_dimensions": query_svd_dimensions,
                "query_svd_corpus_chart_dimensions": query_svd_corpus_chart_dimensions,
                "empty_results_message": empty_results_message,
                "typo_suggestion": typo_suggestion,
            })
        except Exception as exc:
            app.logger.exception("API request to /api/articles failed")
            if progress:
                progress(
                    "error",
                    "Search failed",
                    1.0,
                    message=str(exc),
                )
            return _api_error_response(exc)

    @app.route("/api/articles/similar", methods=["POST"])
    def similar_articles_route():
        try:
            payload = _request_payload()
            article_id = payload.get("article_id") or payload.get("id")
            retrieval_model = payload.get("retrieval_model") or payload.get("model")
            limit = _coerce_int(payload.get("limit"), 10, minimum=1, maximum=25)
            offset = _coerce_int(payload.get("offset"), 0, minimum=0, maximum=5000)
            year_start, year_end = normalize_article_year_range(
                payload.get("year_start"),
                payload.get("year_end"),
            )
            character_start, character_end = normalize_article_character_range(
                _first_payload_value(
                    payload,
                    "character_start",
                    "article_character_start",
                    "article_char_start",
                ),
                _first_payload_value(
                    payload,
                    "character_end",
                    "article_character_end",
                    "article_char_end",
                ),
            )
            word_start, word_end = normalize_article_word_range(
                _first_payload_value(
                    payload,
                    "word_start",
                    "article_word_start",
                ),
                _first_payload_value(
                    payload,
                    "word_end",
                    "article_word_end",
                ),
            )
            reading_time_start, reading_time_end = normalize_article_reading_time_range(
                _first_payload_value(
                    payload,
                    "reading_time_start",
                    "article_reading_time_start",
                    "reading_minutes_start",
                ),
                _first_payload_value(
                    payload,
                    "reading_time_end",
                    "article_reading_time_end",
                    "reading_minutes_end",
                ),
            )
            results = similar_articles(
                article_id=article_id,
                retrieval_model=retrieval_model,
                limit=limit,
                offset=offset,
                year_start=year_start,
                year_end=year_end,
                character_start=character_start,
                character_end=character_end,
                word_start=word_start,
                word_end=word_end,
                reading_time_start=reading_time_start,
                reading_time_end=reading_time_end,
            )
            log_runtime_event(
                "similar_articles.done",
                article_id=str(article_id or ""),
                retrieval_model=str(retrieval_model or "default"),
                result_count=len(results.get("results") or []),
                has_more=bool(results.get("has_more")),
            )
            return jsonify(results)
        except Exception as exc:
            app.logger.exception("API request to /api/articles/similar failed")
            return _api_error_response(exc)

    @app.route("/api/visualization/project_query", methods=["POST"])
    def visualization_project_query_route():
        """Embed the query with the relevant retrieval processor, run it
        through the matching UMAP model, and return the projected 2D
        coordinate in the same int16 space the frontend renders.
        """
        try:
            payload = _request_payload()
            query = str(payload.get("query") or payload.get("topic") or "").strip()
            source = str(payload.get("source") or payload.get("retrieval_model") or "").strip().lower()
            if source in ("tfidf", "tf-idf", "tf_idf"):
                # The atlas only has UMAP projections for MiniLM and SVD;
                # SVD is the natural fallback for the TF-IDF retrieval mode
                # (SVD is built on top of the same TF-IDF vectors).
                source = "svd"
            if source not in ("minilm", "svd"):
                source = "svd"
            from backend.services.visualization_service import project_query

            x, y = project_query(query, source=source)
            log_runtime_event(
                "visualization.project_query.done",
                source=source,
                query_chars=len(query),
            )
            return jsonify({"x": x, "y": y, "source": source})
        except Exception as exc:
            app.logger.exception("API request to /api/visualization/project_query failed")
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
