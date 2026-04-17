"""
LLM routes — only loaded when USE_LLM = True in routes.py.
Adds POST /api/chat for LLM-driven Guardian article RAG and
POST /api/llm/agreement-scores for direct agreement scoring.

Setup:
  1. Add API_KEY=your_key to .env
  2. Set USE_LLM = True in routes.py
"""
import json
import os
import re
import logging
from flask import request, jsonify, Response, stream_with_context

from backend.stance_processing.llm_processor import (
    create_spark_client,
    score_llm_article_agreement,
)

logger = logging.getLogger(__name__)


def llm_search_decision(client, user_message):
    """Ask the LLM whether to search the DB and which word to use."""
    messages = [
        {
            "role": "system",
            "content": (
                "You have access to a database of Guardian opinion article titles, "
                "summaries, and metadata. Decide whether the user's message needs "
                "article retrieval. Reply with exactly: YES followed by one concise "
                "search query, or NO if article data is not needed."
            ),
        },
        {"role": "user", "content": user_message},
    ]
    response = client.chat(messages)
    content = (response.get("content") or "").strip().upper()
    logger.info(f"LLM search decision: {content}")
    if re.search(r"\bNO\b", content) and not re.search(r"\bYES\b", content):
        return False, None
    yes_match = re.search(r"\bYES\s+(.+)", content)
    if yes_match:
        return True, yes_match.group(1).strip().lower()
    if re.search(r"\bYES\b", content):
        return True, user_message
    return False, None


def register_chat_route(app, json_search):
    """Register the /api/chat SSE endpoint. Called from backend/api/routes.py."""

    @app.route("/api/chat", methods=["POST"])
    def chat():
        data = request.get_json() or {}
        user_message = (data.get("message") or "").strip()
        if not user_message:
            return jsonify({"error": "Message is required"}), 400

        api_key = os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
        if not api_key:
            return jsonify({"error": "SPARK_API_KEY or API_KEY not set — add it to your .env file"}), 500

        try:
            client = create_spark_client(api_key=api_key)
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        use_search, search_term = llm_search_decision(client, user_message)

        if use_search:
            articles = json_search(search_term or user_message, top_n=8)
            context_text = "\n\n---\n\n".join(
                (
                    f"Title: {article.get('title')}\n"
                    f"Summary: {article.get('summary')}\n"
                    f"Date: {article.get('date')}\n"
                    f"Author: {article.get('author_display') or article.get('author_raw')}\n"
                    f"URL: {article.get('url')}"
                )
                for article in articles
            ) or "No matching Guardian articles found."
            messages = [
                {"role": "system", "content": "Answer questions about Guardian opinion articles using only the article information provided."},
                {"role": "user", "content": f"Article information:\n\n{context_text}\n\nUser question: {user_message}"},
            ]
        else:
            messages = [
                {"role": "system", "content": "You are a helpful assistant for Guardian opinion article research."},
                {"role": "user", "content": user_message},
            ]

        def generate():
            if use_search and search_term:
                yield f"data: {json.dumps({'search_term': search_term})}\n\n"
            try:
                for chunk in client.chat(messages, stream=True):
                    if chunk.get("content"):
                        yield f"data: {json.dumps({'content': chunk['content']})}\n\n"
            except Exception as e:
                logger.error(f"Streaming error: {e}")
                yield f"data: {json.dumps({'error': 'Streaming error occurred'})}\n\n"

        return Response(
            stream_with_context(generate()),
            mimetype="text/event-stream",
            headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
        )

    @app.route("/api/llm/agreement-scores", methods=["POST"])
    def agreement_scores():
        data = request.get_json() or {}
        thesis = (
            data.get("thesis")
            or data.get("statement")
            or data.get("selected_thesis_sentence")
            or ""
        )
        articles = data.get("articles") or data.get("results") or []
        if not str(thesis or "").strip():
            return jsonify({"error": "Thesis is required"}), 400
        if not isinstance(articles, list) or not articles:
            return jsonify({"error": "A non-empty articles list is required"}), 400

        try:
            scores = score_llm_article_agreement(
                articles=articles,
                thesis=thesis,
            )
        except RuntimeError as exc:
            return jsonify({"error": str(exc)}), 500
        except Exception:
            logger.exception("Agreement scoring request failed")
            return jsonify({"error": "LLM agreement scoring failed"}), 500

        return jsonify({"scores": scores})
