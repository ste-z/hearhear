import json
import os
import re

from backend.runtime.runtime_debug import log_runtime_event


DEFAULT_LLM_BATCH_SIZE = 20
DEFAULT_MAX_ARTICLE_CHARS = 10000
SPARK_API_KEY_ENV_NAMES = ("SPARK_API_KEY", "API_KEY")

LLM_AGREEMENT_SYSTEM_PROMPT = """
You evaluate how much retrieved news opinion articles agree with a user's thesis.

Use only the article context supplied by the application. Do not rely on outside
knowledge of the publication, author, topic, or article. Treat topical relevance
and agreement as different signals: an article can be very relevant while
strongly disagreeing with the thesis.

For each article, assign an agreement score from 0 to 1:
- 1.00 means the article's central claim strongly supports the user's thesis.
- 0.75 means the article mostly supports the thesis, with qualifications.
- 0.50 means the article is neutral, mixed, unclear, or does not provide enough
  evidence about agreement.
- 0.25 means the article mostly disagrees with the thesis, with qualifications.
- 0.00 means the article's central claim strongly contradicts the thesis.

Also assign an irrelevant flag:
- 1 means the article is completely unrelated to the user's topic/thesis.
- 0 means the article is related or even broadly related.
Be conservative. When in doubt, use 0. Energy policy is related to climate
change. An article about free buses is not related to free speech.

Return valid JSON only: a single array in the exact same order as the
article_ids list. Each item must be [agreement_score, irrelevant_flag]. Do not
return article IDs, labels, rationale, markdown, comments, or prose. Example:
[[0.9, 0], [0.5, 0], [0.5, 1]]
""".strip()


def spark_api_key(api_key=None):
    if api_key:
        return str(api_key).strip()

    for env_name in SPARK_API_KEY_ENV_NAMES:
        value = os.getenv(env_name)
        if value and value.strip():
            return value.strip()

    env_list = " or ".join(SPARK_API_KEY_ENV_NAMES)
    raise RuntimeError(
        f"Spark API key not set. Add {env_list} to your .env file before using LLM agreement scoring."
    )


def create_spark_client(api_key=None):
    try:
        from infosci_spark_client import LLMClient
    except ImportError as exc:
        raise RuntimeError(
            "infosci_spark_client is not installed. Run `pip install -r requirements.txt` "
            "before using LLM agreement scoring."
        ) from exc

    return LLMClient(api_key=spark_api_key(api_key=api_key))


def _clean_text(value):
    if value is None:
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def _clip_text(value, max_chars):
    text = _clean_text(value)
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return f"{text[:max(0, max_chars - 3)].rstrip()}..."


def _clean_list(value, max_items=4, max_chars=320):
    if not isinstance(value, list):
        return []
    cleaned = []
    for item in value[:max_items]:
        text = _clip_text(item, max_chars=max_chars)
        if text:
            cleaned.append(text)
    return cleaned


def _article_id(article, index):
    raw_id = None
    if isinstance(article, dict):
        raw_id = article.get("id") or article.get("article_id")
    else:
        raw_id = getattr(article, "id", None) or getattr(article, "article_id", None)

    resolved = _clean_text(raw_id)
    return resolved or f"article_{index + 1}"


def _article_value(article, key, default=None):
    if isinstance(article, dict):
        return article.get(key, default)
    return getattr(article, key, default)


def _article_prompt_payload(article, index, max_article_chars=DEFAULT_MAX_ARTICLE_CHARS):
    half_budget = max(320, int(max_article_chars / 2))
    return {
        "article_id": _article_id(article, index),
        "title": _clip_text(_article_value(article, "title"), max_chars=320),
        "summary": _clip_text(_article_value(article, "summary"), max_chars=half_budget),
        "central_claim": _clip_text(
            _article_value(article, "central_claim_summary"),
            max_chars=half_budget,
        ),
        "thesis_sentence": _clip_text(
            _article_value(article, "thesis_sentence"),
            max_chars=420,
        ),
        "support_sentences": _clean_list(_article_value(article, "support_sentences")),
        "secondary_claims": _clean_list(
            _article_value(article, "secondary_claim_sentences"),
            max_items=3,
            max_chars=280,
        ),
        "body_excerpt": _clip_text(
            _article_value(article, "body_text"),
            max_chars=max_article_chars,
        ),
    }


def build_llm_agreement_messages(
    thesis,
    articles,
    max_article_chars=DEFAULT_MAX_ARTICLE_CHARS,
    start_index=0,
):
    article_payload = [
        _article_prompt_payload(
            article,
            index=start_index + offset,
            max_article_chars=max_article_chars,
        )
        for offset, article in enumerate(articles)
    ]
    article_ids = [article["article_id"] for article in article_payload]
    user_prompt = (
        "User thesis:\n"
        f"{_clean_text(thesis)}\n\n"
        "article_ids in scoring order:\n"
        f"{json.dumps(article_ids, ensure_ascii=False)}\n\n"
        "Retrieved articles in the same order:\n"
        f"{json.dumps(article_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Return exactly {len(article_ids)} [score, irrelevant] pairs as a JSON array in the article_ids order."
    )
    return [
        {"role": "system", "content": LLM_AGREEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _response_content(response):
    if isinstance(response, dict):
        for key in ("content", "message", "text", "response"):
            value = response.get(key)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return json.dumps(response)
    return str(response or "").strip()


def _try_load_json(text):
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return None


def _extract_json_payload(content):
    text = _clean_text(content)
    if not text:
        raise RuntimeError("The LLM returned an empty agreement-scoring response.")

    parsed = _try_load_json(text)
    if parsed is not None:
        return parsed

    fence_match = re.search(r"```(?:json)?\s*(.*?)```", text, flags=re.IGNORECASE | re.DOTALL)
    if fence_match:
        parsed = _try_load_json(fence_match.group(1).strip())
        if parsed is not None:
            return parsed

    for open_char, close_char in (("{", "}"), ("[", "]")):
        start = text.find(open_char)
        end = text.rfind(close_char)
        if start >= 0 and end > start:
            parsed = _try_load_json(text[start:end + 1])
            if parsed is not None:
                return parsed

    raise RuntimeError("The LLM agreement response was not valid JSON.")


def _score_values_from_payload(payload):
    if isinstance(payload, list):
        return payload
    if isinstance(payload, dict):
        scores = payload.get("scores")
        if isinstance(scores, list):
            return scores
    raise RuntimeError("The LLM agreement JSON must be a list of scores.")


def _coerce_unit_score(value, default=0.5):
    try:
        score = float(value)
    except (TypeError, ValueError):
        score = float(default)
    return max(0.0, min(1.0, score))


def _coerce_irrelevant_flag(value):
    if isinstance(value, bool):
        return bool(value)
    if isinstance(value, (int, float)):
        return int(value) == 1

    text = _clean_text(value).lower()
    if text in {"1", "true", "yes", "y", "irrelevant", "unrelated"}:
        return True
    return False


def _score_pair(value):
    if isinstance(value, (list, tuple)):
        score_value = value[0] if value else 0.5
        irrelevant_value = value[1] if len(value) > 1 else 0
        return score_value, irrelevant_value

    if isinstance(value, dict):
        score_value = (
            value.get("agreement_score")
            if "agreement_score" in value
            else value.get("score", 0.5)
        )
        irrelevant_value = (
            value.get("irrelevant")
            if "irrelevant" in value
            else value.get("is_irrelevant", 0)
        )
        return score_value, irrelevant_value

    return value, 0


def _score_row(article_id, score_value):
    score_value, irrelevant_value = _score_pair(score_value)
    score = _coerce_unit_score(score_value, default=0.5)
    return {
        "article_id": article_id,
        "agreement_score": score,
        "stance_score": (score * 2.0) - 1.0,
        "llm_irrelevant": _coerce_irrelevant_flag(irrelevant_value),
    }


def _normalize_batch_scores(payload, article_ids):
    score_values = _score_values_from_payload(payload)
    expected_count = len(article_ids)
    returned_count = len(score_values)
    if returned_count != expected_count:
        log_runtime_event(
            "llm_agreement.score_count_mismatch",
            expected_count=expected_count,
            returned_count=returned_count,
        )

    rows = []
    for index, article_id in enumerate(article_ids):
        score_value = score_values[index] if index < returned_count else 0.5
        rows.append(_score_row(article_id, score_value))
    return rows


def score_llm_article_agreement(
    articles,
    thesis,
    client=None,
    api_key=None,
    batch_size=DEFAULT_LLM_BATCH_SIZE,
    max_article_chars=DEFAULT_MAX_ARTICLE_CHARS,
):
    article_rows = list(articles or [])
    cleaned_thesis = _clean_text(thesis)
    if not article_rows or not cleaned_thesis:
        return []

    resolved_batch_size = max(1, int(batch_size or DEFAULT_LLM_BATCH_SIZE))
    resolved_client = client or create_spark_client(api_key=api_key)
    rows = []

    log_runtime_event(
        "llm_agreement.start",
        article_count=len(article_rows),
        thesis_chars=len(cleaned_thesis),
        batch_size=resolved_batch_size,
        max_article_chars=int(max_article_chars),
    )

    total_batches = (
        len(article_rows) + resolved_batch_size - 1
    ) // resolved_batch_size
    for batch_index, start_index in enumerate(
        range(0, len(article_rows), resolved_batch_size),
        start=1,
    ):
        batch_articles = article_rows[start_index:start_index + resolved_batch_size]
        batch_article_ids = [
            _article_id(article, index=start_index + offset)
            for offset, article in enumerate(batch_articles)
        ]
        messages = build_llm_agreement_messages(
            thesis=cleaned_thesis,
            articles=batch_articles,
            max_article_chars=max_article_chars,
            start_index=start_index,
        )
        log_runtime_event(
            "llm_agreement.batch_start",
            batch_index=batch_index,
            batch_total=total_batches,
            batch_size=len(batch_articles),
        )
        response = resolved_client.chat(messages)
        payload = _extract_json_payload(_response_content(response))
        rows.extend(_normalize_batch_scores(payload, batch_article_ids))
        log_runtime_event(
            "llm_agreement.batch_done",
            batch_index=batch_index,
            batch_total=total_batches,
        )

    log_runtime_event("llm_agreement.done", row_count=len(rows))
    return rows
