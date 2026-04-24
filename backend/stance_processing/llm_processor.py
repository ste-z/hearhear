import csv
import json
import os
import re
import sys
from pathlib import Path

from backend.runtime.runtime_debug import log_runtime_event
from backend.text_processing.paragraph_splitter import paragraph_rows_from_text


DEFAULT_LLM_BATCH_SIZE = 50
DEFAULT_MAX_ARTICLE_CHARS = 50000
DEFAULT_LLM_PARAGRAPH_BATCH_SIZE = 100
DEFAULT_MAX_PARAGRAPHS_PER_ARTICLE = 40
DEFAULT_MAX_PARAGRAPH_CHARS = 2500
DEFAULT_RELEVANT_PARAGRAPH_COUNT = 5
DEFAULT_CHUNKING_MODE = "paragraph"
DEFAULT_SEMANTIC_BREAK_SIMILARITY_THRESHOLD = 0.75
SPARK_API_KEY_ENV_NAMES = ("SPARK_API_KEY", "API_KEY")
PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_GUARDIAN_RAW_DATA_DIR = PROJECT_ROOT / "data" / "raw" / "guardian_by_year"

LLM_ARTICLE_AGREEMENT_SYSTEM_PROMPT = """
You evaluate how much retrieved news opinion articles agree with a user's thesis or essay.

Use only the article context supplied by the application. Do not rely on outside
knowledge of the publication, author, topic, or article. Treat topical relevance
and agreement as different signals: an article can be very relevant while
strongly disagreeing with the user's position.

For each article, assign an agreement score from 0 to 1:
- 1.00 means the article's central claim strongly supports the user's position.
- 0.75 means the article mostly supports the user's position, with qualifications.
- 0.50 means the article is neutral, mixed, unclear, or does not provide enough
  evidence about agreement.
- 0.25 means the article mostly disagrees with the user's position, with qualifications.
- 0.00 means the article's central claim strongly contradicts the user's position.

Also assign an irrelevant flag:
- 1 means the article is completely unrelated to the user's topic or position.
- 0 means the article is related or even broadly related.
Be conservative. When in doubt, use 0. Energy policy is related to climate
change. An article about free buses is not related to free speech.

Return valid JSON only: a single array with exactly one object for each
article_id. Each object must include article_id, agreement_score, and
irrelevant. Do not return labels, rationale, markdown, comments, or prose.
Example:
[{"article_id": "article_1", "agreement_score": 0.9, "irrelevant": 0}]
""".strip()

LLM_PARAGRAPH_AGREEMENT_SYSTEM_PROMPT = """
You evaluate retrieved article chunks against a user's thesis or essay. A chunk may be a
paragraph or a semantic group of adjacent sentences.

Use only the chunk context supplied by the application. Do not rely on
outside knowledge of the publication, author, topic, or article. Judge each
chunk independently, not whether the whole article agrees.

For each chunk, assign an agreement score from 0 to 1:
- 1.00 means the chunk directly supports the user's position.
- 0.75 means the chunk mostly supports the user's position, with qualifications.
- 0.50 means the chunk is related but neutral, descriptive, background,
  mixed, unclear, or not stance-bearing.
- 0.25 means the chunk mostly pushes against the user's position, with qualifications.
- 0.00 means the chunk directly contradicts the user's position.

Also assign an irrelevant flag:
- 1 means the chunk is completely unrelated to the broad topic, issue,
  actors, policy area, cause, consequence, or debate in the user's position.
- 0 means the chunk is related or even broadly related.
Be conservative. When in doubt, use 0. Background, evidence, context,
counterarguments, nearby subtopics, causes, consequences, and policy details are
relevant if they connect to the broad issue. Energy policy is related to climate
change. A chunk about free buses is not related to free speech.

Return valid JSON only: a single array with exactly one object for each
chunk_id. Each object must include chunk_id, agreement_score, and irrelevant.
Do not return labels, rationale, markdown, comments, or prose. Example:
[{"chunk_id": "chunk_1", "agreement_score": 0.9, "irrelevant": 0}]
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


def _raw_text(value):
    if value is None:
        return ""
    return str(value).strip()


def _article_year(article):
    raw_year = _article_value(article, "year")
    try:
        return int(raw_year)
    except (TypeError, ValueError):
        pass

    date_value = _article_value(article, "date")
    date_text = _clean_text(date_value)
    match = re.search(r"\b(19|20)\d{2}\b", date_text)
    if not match:
        return None
    try:
        return int(match.group(0))
    except (TypeError, ValueError):
        return None


def _normalize_chunking_mode(value):
    text = str(value or DEFAULT_CHUNKING_MODE).strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"semantic", "semantic_chunking", "semantic_chunks"}:
        return "semantic"
    return "paragraph"


def _set_large_csv_field_limit():
    limit = sys.maxsize
    while True:
        try:
            csv.field_size_limit(limit)
            return
        except OverflowError:
            limit = int(limit / 10)
        except Exception:
            return


def _raw_article_paths_for_years(years, raw_data_dir=DEFAULT_GUARDIAN_RAW_DATA_DIR):
    raw_dir = Path(raw_data_dir)
    if years:
        return [
            raw_dir / f"guardian_opinion_{year}.csv"
            for year in sorted({int(year) for year in years if year is not None})
        ]
    return sorted(raw_dir.glob("guardian_opinion_*.csv"))


def _article_body_text_lookup_from_raw(article_ids, year_by_article_id=None):
    pending_ids = {
        _clean_text(article_id)
        for article_id in article_ids
        if _clean_text(article_id)
    }
    if not pending_ids:
        return {}

    year_lookup = dict(year_by_article_id or {})
    candidate_years = {
        year_lookup.get(article_id)
        for article_id in pending_ids
        if year_lookup.get(article_id) is not None
    }
    candidate_paths = _raw_article_paths_for_years(candidate_years)
    if not candidate_paths:
        candidate_paths = _raw_article_paths_for_years(None)

    found = {}
    scanned_paths = set()

    def scan_paths(paths):
        _set_large_csv_field_limit()
        for path in paths:
            if not pending_ids:
                break
            if not path.exists() or path in scanned_paths:
                continue
            scanned_paths.add(path)
            try:
                with path.open("r", encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    for row in reader:
                        article_id = _clean_text(row.get("id"))
                        if article_id not in pending_ids:
                            continue
                        body_text = _raw_text(row.get("body_text"))
                        if body_text:
                            found[article_id] = body_text
                        pending_ids.discard(article_id)
                        if not pending_ids:
                            break
            except Exception:
                continue

    scan_paths(candidate_paths)
    if pending_ids and candidate_years:
        scan_paths(_raw_article_paths_for_years(None))

    if found:
        log_runtime_event(
            "llm_chunking.raw_body_lookup_done",
            requested_count=len(article_ids),
            found_count=len(found),
            scanned_file_count=len(scanned_paths),
        )
    return found


def _article_body_text_lookup(articles):
    article_ids = []
    year_by_article_id = {}
    for index, article in enumerate(articles):
        article_id = _article_id(article, index)
        if article_id and article_id not in article_ids:
            article_ids.append(article_id)
        article_year = _article_year(article)
        if article_id and article_year is not None:
            year_by_article_id[article_id] = article_year

    if not article_ids:
        return {}

    body_lookup = {}
    try:
        from backend.db.models import GuardianArticle
    except Exception:
        GuardianArticle = None

    if GuardianArticle is not None:
        try:
            rows = (
                GuardianArticle.query.with_entities(
                    GuardianArticle.id,
                    GuardianArticle.body_text,
                )
                .filter(GuardianArticle.id.in_(article_ids))
                .all()
            )
            for row in rows:
                body_text = _raw_text(row.body_text)
                if body_text:
                    body_lookup[row.id] = body_text
        except Exception:
            body_lookup = {}

    missing_ids = [
        article_id
        for article_id in article_ids
        if not _raw_text(body_lookup.get(article_id))
    ]
    if missing_ids:
        body_lookup.update(
            _article_body_text_lookup_from_raw(
                missing_ids,
                year_by_article_id=year_by_article_id,
            )
        )

    return body_lookup


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
        "User thesis or essay:\n"
        f"{_clean_text(thesis)}\n\n"
        "article_ids in scoring order:\n"
        f"{json.dumps(article_ids, ensure_ascii=False)}\n\n"
        "Retrieved articles in the same order:\n"
        f"{json.dumps(article_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Return exactly {len(article_ids)} JSON objects, one per article_id."
    )
    return [
        {"role": "system", "content": LLM_ARTICLE_AGREEMENT_SYSTEM_PROMPT},
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
        if isinstance(scores, (list, dict)):
            return scores
        if any(key in payload for key in ("article_id", "chunk_id", "paragraph_id", "id")):
            return [payload]
        if all(isinstance(key, str) for key in payload.keys()):
            return payload
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


def _score_item_id(value, id_keys):
    if not isinstance(value, dict):
        return ""
    for key in id_keys:
        text = _clean_text(value.get(key))
        if text:
            return text
    return ""


def _ordered_score_values(
    score_values,
    ordered_ids,
    event_prefix,
    id_keys,
    default_score,
):
    expected_count = len(ordered_ids)
    returned_count = len(score_values) if hasattr(score_values, "__len__") else 0
    if returned_count != expected_count:
        log_runtime_event(
            f"{event_prefix}.score_count_mismatch",
            expected_count=expected_count,
            returned_count=returned_count,
        )

    score_by_id = {}
    if isinstance(score_values, dict):
        score_by_id = {
            _clean_text(item_id): score_value
            for item_id, score_value in score_values.items()
            if _clean_text(item_id)
        }
    elif isinstance(score_values, list):
        score_by_id = {
            item_id: score_value
            for score_value in score_values
            for item_id in [_score_item_id(score_value, id_keys)]
            if item_id
        }

    ordered_id_set = set(ordered_ids)
    if score_by_id:
        missing_ids = [
            item_id for item_id in ordered_ids
            if item_id not in score_by_id
        ]
        extra_count = len(set(score_by_id) - ordered_id_set)
        if missing_ids or extra_count:
            log_runtime_event(
                f"{event_prefix}.score_id_mismatch",
                missing_count=len(missing_ids),
                extra_count=extra_count,
            )
        return [
            score_by_id.get(item_id, default_score)
            for item_id in ordered_ids
        ]

    if isinstance(score_values, list):
        return [
            score_values[index] if index < returned_count else default_score
            for index, _item_id in enumerate(ordered_ids)
        ]

    return [default_score for _item_id in ordered_ids]


def _score_row(article_id, score_value):
    score_value, irrelevant_value = _score_pair(score_value)
    score = _coerce_unit_score(score_value, default=0.5)
    return {
        "article_id": article_id,
        "agreement_score": score,
        "stance_score": (score * 2.0) - 1.0,
        "llm_irrelevant": _coerce_irrelevant_flag(irrelevant_value),
    }


def _paragraph_prompt_payload(paragraph_row):
    payload = {
        "chunk_id": paragraph_row["paragraph_id"],
        "article_id": paragraph_row["article_id"],
        "chunk_index": paragraph_row["paragraph_index"],
        "text": paragraph_row["paragraph"],
    }
    topic_score = paragraph_row.get("topic_score")
    if topic_score is not None:
        payload["topic_score"] = topic_score
    return payload


def build_llm_paragraph_agreement_messages(thesis, paragraph_rows):
    paragraph_payload = [
        _paragraph_prompt_payload(row)
        for row in paragraph_rows
    ]
    paragraph_ids = [row["chunk_id"] for row in paragraph_payload]
    user_prompt = (
        "User thesis or essay:\n"
        f"{_clean_text(thesis)}\n\n"
        "chunk_ids in scoring order:\n"
        f"{json.dumps(paragraph_ids, ensure_ascii=False)}\n\n"
        "Retrieved chunks in the same order:\n"
        f"{json.dumps(paragraph_payload, ensure_ascii=False, indent=2)}\n\n"
        f"Return exactly {len(paragraph_ids)} JSON objects, one per chunk_id."
    )
    return [
        {"role": "system", "content": LLM_PARAGRAPH_AGREEMENT_SYSTEM_PROMPT},
        {"role": "user", "content": user_prompt},
    ]


def _normalize_paragraph_batch_scores(payload, paragraph_rows):
    score_values = _score_values_from_payload(payload)
    paragraph_ids = [row["paragraph_id"] for row in paragraph_rows]
    ordered_scores = _ordered_score_values(
        score_values=score_values,
        ordered_ids=paragraph_ids,
        event_prefix="llm_paragraph_agreement",
        id_keys=("chunk_id", "paragraph_id", "id"),
        default_score=[0.5, 1],
    )

    rows = []
    for index, paragraph_row in enumerate(paragraph_rows):
        score_value = ordered_scores[index]
        score_value, irrelevant_value = _score_pair(score_value)
        score = _coerce_unit_score(score_value, default=0.5)
        rows.append(
            {
                "paragraph_id": paragraph_row["paragraph_id"],
                "article_id": paragraph_row["article_id"],
                "paragraph_index": paragraph_row["paragraph_index"],
                "paragraph": paragraph_row["paragraph"],
                "sentence_start_index": paragraph_row.get("sentence_start_index"),
                "sentence_end_index": paragraph_row.get("sentence_end_index"),
                "topic_score": paragraph_row.get("topic_score"),
                "topic_chunk_rank": paragraph_row.get("topic_chunk_rank"),
                "agreement_score": score,
                "stance_score": (score * 2.0) - 1.0,
                "llm_irrelevant": _coerce_irrelevant_flag(irrelevant_value),
            }
        )
    return rows


def _article_source_text(article, article_id, body_lookup):
    body_text = _article_value(article, "body_text")
    if not body_text:
        body_text = body_lookup.get(article_id)
    if body_text:
        return body_text

    parts = [
        _article_value(article, "central_claim_summary"),
        _article_value(article, "summary"),
        _article_value(article, "thesis_sentence"),
    ]
    return "\n\n".join(_clean_text(part) for part in parts if _clean_text(part))


def _provided_chunk_rows_for_article(article, article_id):
    if not isinstance(article, dict):
        return []

    raw_chunks = article.get("topic_relevant_chunks") or article.get("retrieval_chunks")
    if not isinstance(raw_chunks, list):
        return []

    rows = []
    for fallback_index, chunk in enumerate(raw_chunks):
        if not isinstance(chunk, dict):
            continue
        text = _clean_text(chunk.get("text") or chunk.get("paragraph"))
        if not text:
            continue
        try:
            chunk_index = int(chunk.get("chunk_index"))
        except (TypeError, ValueError):
            chunk_index = fallback_index
        rows.append({
            "paragraph_id": str(
                chunk.get("chunk_id")
                or chunk.get("paragraph_id")
                or f"{article_id}_chunk_{fallback_index}"
            ),
            "paragraph_index": chunk_index,
            "article_id": article_id,
            "paragraph": text,
            "sentence_start_index": chunk.get("sentence_start_index"),
            "sentence_end_index": chunk.get("sentence_end_index"),
            "topic_score": chunk.get("topic_score"),
            "topic_chunk_rank": chunk.get("chunk_rank"),
        })

    return rows


def _svd_processor_for_semantic_chunks():
    try:
        from backend.text_processing.search_helpers import build_retrieval_processor
    except Exception:
        return None

    try:
        return build_retrieval_processor(retrieval_model="svd")
    except Exception:
        return None


def _semantic_chunk_rows_from_text(*args, **kwargs):
    from backend.text_processing.semantic_chunker import semantic_chunk_rows_from_text

    return semantic_chunk_rows_from_text(*args, **kwargs)


def _chunk_rows_for_articles(
    articles,
    chunking_mode=DEFAULT_CHUNKING_MODE,
    max_paragraphs_per_article=DEFAULT_MAX_PARAGRAPHS_PER_ARTICLE,
    max_paragraph_chars=DEFAULT_MAX_PARAGRAPH_CHARS,
    semantic_break_threshold=DEFAULT_SEMANTIC_BREAK_SIMILARITY_THRESHOLD,
):
    resolved_mode = _normalize_chunking_mode(chunking_mode)
    body_lookup = _article_body_text_lookup(articles)
    svd_processor = (
        _svd_processor_for_semantic_chunks()
        if resolved_mode == "semantic"
        else None
    )
    paragraph_rows = []
    provided_article_count = 0
    provided_paragraph_count = 0
    generated_article_count = 0
    generated_paragraph_count = 0
    for article_index, article in enumerate(articles):
        article_id = _article_id(article, article_index)
        provided_rows = _provided_chunk_rows_for_article(article, article_id)
        if provided_rows:
            provided_article_count += 1
            provided_paragraph_count += len(provided_rows)
            paragraph_rows.extend(provided_rows)
            continue

        source_text = _article_source_text(article, article_id, body_lookup)
        if resolved_mode == "semantic":
            rows = _semantic_chunk_rows_from_text(
                source_text,
                article_id=article_id,
                prefix=f"a{article_index}_sc",
                svd_processor=svd_processor,
                similarity_threshold=semantic_break_threshold,
                max_chars=max_paragraph_chars,
            )
        else:
            rows = paragraph_rows_from_text(
                source_text,
                article_id=article_id,
                prefix=f"a{article_index}_p",
                min_chars=20,
                max_chars=max_paragraph_chars,
            )
        if max_paragraphs_per_article:
            rows = rows[:max(1, int(max_paragraphs_per_article))]
        if rows:
            generated_article_count += 1
            generated_paragraph_count += len(rows)
        paragraph_rows.extend(rows)
    if provided_paragraph_count:
        log_runtime_event(
            "llm_paragraph_agreement.retrieved_chunks_used",
            article_count=provided_article_count,
            paragraph_count=provided_paragraph_count,
            chunking_mode=resolved_mode,
        )
    if generated_paragraph_count:
        log_runtime_event(
            "llm_paragraph_agreement.generated_chunks_used",
            article_count=generated_article_count,
            paragraph_count=generated_paragraph_count,
            chunking_mode=resolved_mode,
        )
    return paragraph_rows


def _top_relevant_paragraphs(paragraph_scores, limit=DEFAULT_RELEVANT_PARAGRAPH_COUNT):
    ranked = sorted(
        paragraph_scores,
        key=lambda row: (
            abs(float(row.get("agreement_score", 0.5)) - 0.5),
            float(row.get("agreement_score", 0.5)),
        ),
        reverse=True,
    )
    return [
        {
            "paragraph_id": row["paragraph_id"],
            "paragraph_index": row["paragraph_index"],
            "text": row["paragraph"],
            "agreement_score": row["agreement_score"],
            "topic_score": row.get("topic_score"),
            "topic_chunk_rank": row.get("topic_chunk_rank"),
            "sentence_start_index": row.get("sentence_start_index"),
            "sentence_end_index": row.get("sentence_end_index"),
        }
        for row in ranked[:max(1, int(limit))]
    ]


def _aggregate_paragraph_scores(article_rows, paragraph_scores):
    grouped = {}
    for paragraph_score in paragraph_scores:
        grouped.setdefault(paragraph_score["article_id"], []).append(paragraph_score)

    rows = []
    for index, article in enumerate(article_rows):
        article_id = _article_id(article, index)
        scores = grouped.get(article_id, [])
        related_scores = [
            score for score in scores
            if not bool(score.get("llm_irrelevant"))
        ]
        if not related_scores:
            rows.append(
                {
                    "article_id": article_id,
                    "agreement_score": 0.5,
                    "stance_score": 0.0,
                    "llm_irrelevant": True,
                    "llm_relevant_paragraphs": [],
                    "llm_chunk_count": len(scores),
                    "llm_related_chunk_count": 0,
                }
            )
            continue

        weights = []
        for score in related_scores:
            try:
                weight = float(score.get("topic_score"))
            except (TypeError, ValueError):
                weight = 0.0
            weights.append(max(0.0, weight))

        total_weight = sum(weights)
        if total_weight > 0:
            agreement_score = sum(
                float(score["agreement_score"]) * weight
                for score, weight in zip(related_scores, weights)
            ) / total_weight
        else:
            agreement_score = sum(
                float(score["agreement_score"]) for score in related_scores
            ) / len(related_scores)
        rows.append(
            {
                "article_id": article_id,
                "agreement_score": agreement_score,
                "stance_score": (agreement_score * 2.0) - 1.0,
                "llm_irrelevant": False,
                "llm_relevant_paragraphs": _top_relevant_paragraphs(related_scores),
                "llm_chunk_count": len(scores),
                "llm_related_chunk_count": len(related_scores),
            }
        )
    return rows


def _normalize_batch_scores(payload, article_ids):
    score_values = _score_values_from_payload(payload)
    ordered_scores = _ordered_score_values(
        score_values=score_values,
        ordered_ids=article_ids,
        event_prefix="llm_agreement",
        id_keys=("article_id", "id"),
        default_score=0.5,
    )

    rows = []
    for index, article_id in enumerate(article_ids):
        score_value = ordered_scores[index]
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


def score_llm_article_agreement_by_paragraphs(
    articles,
    thesis,
    client=None,
    api_key=None,
    batch_size=DEFAULT_LLM_PARAGRAPH_BATCH_SIZE,
    chunking_mode=DEFAULT_CHUNKING_MODE,
    max_paragraphs_per_article=DEFAULT_MAX_PARAGRAPHS_PER_ARTICLE,
    max_paragraph_chars=DEFAULT_MAX_PARAGRAPH_CHARS,
    semantic_break_threshold=DEFAULT_SEMANTIC_BREAK_SIMILARITY_THRESHOLD,
):
    article_rows = list(articles or [])
    cleaned_thesis = _clean_text(thesis)
    if not article_rows or not cleaned_thesis:
        return []

    resolved_chunking_mode = _normalize_chunking_mode(chunking_mode)
    paragraph_rows = _chunk_rows_for_articles(
        article_rows,
        chunking_mode=resolved_chunking_mode,
        max_paragraphs_per_article=max_paragraphs_per_article,
        max_paragraph_chars=max_paragraph_chars,
        semantic_break_threshold=semantic_break_threshold,
    )
    if not paragraph_rows:
        return [
            {
                "article_id": _article_id(article, index),
                "agreement_score": 0.5,
                "stance_score": 0.0,
                "llm_irrelevant": True,
                "llm_relevant_paragraphs": [],
                "llm_chunk_count": 0,
                "llm_related_chunk_count": 0,
                "llm_chunking_mode": resolved_chunking_mode,
            }
            for index, article in enumerate(article_rows)
        ]

    resolved_batch_size = max(1, int(batch_size or DEFAULT_LLM_PARAGRAPH_BATCH_SIZE))
    resolved_client = client or create_spark_client(api_key=api_key)
    paragraph_scores = []

    log_runtime_event(
        "llm_paragraph_agreement.start",
        article_count=len(article_rows),
        paragraph_count=len(paragraph_rows),
        thesis_chars=len(cleaned_thesis),
        batch_size=resolved_batch_size,
        chunking_mode=resolved_chunking_mode,
    )

    total_batches = (
        len(paragraph_rows) + resolved_batch_size - 1
    ) // resolved_batch_size
    for batch_index, start_index in enumerate(
        range(0, len(paragraph_rows), resolved_batch_size),
        start=1,
    ):
        batch_rows = paragraph_rows[start_index:start_index + resolved_batch_size]
        messages = build_llm_paragraph_agreement_messages(
            thesis=cleaned_thesis,
            paragraph_rows=batch_rows,
        )
        log_runtime_event(
            "llm_paragraph_agreement.batch_start",
            batch_index=batch_index,
            batch_total=total_batches,
            batch_size=len(batch_rows),
        )
        response = resolved_client.chat(messages)
        payload = _extract_json_payload(_response_content(response))
        paragraph_scores.extend(_normalize_paragraph_batch_scores(payload, batch_rows))
        log_runtime_event(
            "llm_paragraph_agreement.batch_done",
            batch_index=batch_index,
            batch_total=total_batches,
        )

    rows = _aggregate_paragraph_scores(article_rows, paragraph_scores)
    for row in rows:
        row["llm_chunking_mode"] = resolved_chunking_mode
    log_runtime_event(
        "llm_paragraph_agreement.done",
        row_count=len(rows),
        paragraph_score_count=len(paragraph_scores),
        chunking_mode=resolved_chunking_mode,
    )
    return rows
