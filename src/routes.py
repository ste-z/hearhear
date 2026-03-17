"""
Routes: React app serving and Guardian article search API.

To enable AI chat, set USE_LLM = True below. See llm_routes.py for AI code.
"""
import os
from flask import send_from_directory, request, jsonify
from sqlalchemy import or_
from models import GuardianArticle

# ── AI toggle ────────────────────────────────────────────────────────────────
USE_LLM = False
# USE_LLM = True
# ─────────────────────────────────────────────────────────────────────────────


def json_search(query):
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

    matches = []
    for article in results:
        authors = article.contributors if isinstance(article.contributors, list) else []
        author_display = ", ".join(authors) if authors else article.author_raw
        matches.append({
            "id": article.id,
            "title": article.title,
            "summary": article.summary,
            "date": article.date.isoformat() if article.date else None,
            "url": article.url,
            "authors": authors,
            "author_display": author_display,
            "author_raw": article.author_raw,
            "n_contributors": article.n_contributors,
            "keywords": article.keywords or [],
            "year": article.year,
        })
    return matches


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

    @app.route("/api/articles")
    def articles_search():
        text = request.args.get("q", "") or request.args.get("title", "")
        return jsonify(json_search(text))

    if USE_LLM:
        from llm_routes import register_chat_route
        register_chat_route(app, json_search)
