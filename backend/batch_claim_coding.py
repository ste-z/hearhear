import argparse
import json
import os
import re
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from backend.data_import import load_and_clean_guardian_years
from backend.sentence_splitter import (
    normalize_whitespace,
    sentence_lookup,
    sentence_table_from_text,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_RAW_DATA_DIR = PROJECT_ROOT / "backend" / "data" / "raw" / "guardian_by_year"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "backend" / "data" / "processed" / "openai_claim_batches"
DEFAULT_MODEL = "gpt-5-nano"
DEFAULT_MODEL_SNAPSHOT = "gpt-5-nano-2025-08-07"
DEFAULT_REASONING_EFFORT = "minimal"
DEFAULT_MAX_OUTPUT_TOKENS = 250
DEFAULT_COMPLETION_WINDOW = "24h"
DEFAULT_ARTICLES_PER_BATCH = 300
DEFAULT_MAX_ESTIMATED_ENQUEUED_TOKENS = 1_500_000
PROMPT_VERSION = "openai_claim_coding_v2"
TERMINAL_BATCH_STATUSES = {"completed", "failed", "cancelled", "expired"}
ACTIVE_BATCH_STATUSES = {"submitted", "validating", "in_progress", "finalizing", "cancelling"}

CLAIM_CODING_INSTRUCTIONS = """
You are annotating argument structure in opinion and analysis articles.

Your task is NOT generic summarization, NOT general salience detection, and NOT evidence extraction for an external query.
Your task is to identify the sentences that best capture the AUTHOR'S OWN ENDORSED ARGUMENT.

What to extract:
- the dominant central thesis of the article
- up to 2 major support sentences for that same thesis
- up to 3 secondary claim-bearing sentences if the article also advances other important endorsed claims

Key definitions:
- A thesis sentence is the single sentence that best states the author's main endorsed position.
- A support sentence states a major general reason, justification, principle, benefit, harm, or causal consequence supporting that thesis.
- A claim-bearing sentence expresses a debatable proposition, judgment, recommendation, normative position, or important causal argument relevant to the author's position.
- A secondary claim is an additional endorsed claim-bearing proposition that is important in the article but is not simply the same thesis sentence and is not merely an example or implementation detail.

What does NOT count as a good support or secondary claim unless it clearly states a broader proposition:
- concrete examples
- imagined scenarios
- case instances
- anecdotes
- operational or implementation details
- quoted speech
- opponent views
- background or scene-setting
- purely descriptive or factual context

Requirements for `central_claim_summary`:
- It must be exactly one sentence.
- It must state the article's central claim as a standalone proposition.
- Do NOT use attributional framing such as:
  - "the author argues"
  - "the article says"
  - "the writer believes"
  - "this piece claims"
- State the claim directly.
- Preserve the original polarity and modality where possible, such as "should", "must", "should not", "cannot", "is", or "is not".

Selection principles:
1. Prioritize the author's endorsed position, not merely a position mentioned in the article.
2. Prefer claim-bearing and reason-bearing sentences over generally important or topic-relevant sentences.
3. Prefer generalizable argumentative propositions over specific illustrations or applications.
4. Support sentences must support the same dominant thesis.
5. Support sentences should add distinct reasons rather than repeating the same point.
6. Secondary claims must be genuine endorsed claim-bearing propositions, not examples, restatements, or minor elaborations.
7. Choose only from the provided sentence IDs.
8. Return exactly one `thesis_sentence_id`.
9. If the article has no sharply stated thesis, still return the best thesis approximation and set `has_clear_central_thesis` to false.
10. Return 0 to 2 `support_sentence_ids`. Do not force supports if no good support sentences exist.
11. Return 0 to 3 `secondary_claim_ids`. Do not force secondary claims if none are strong enough.
12. Do not repeat the thesis ID in the support or secondary lists.
13. Do not repeat any sentence ID across fields.
14. If a sentence is mainly an example, scenario, anecdote, or implementation detail, do not select it unless it also clearly states a broader claim.

Tie-breaking guidance:
- Prefer sentences that directly state a position over sentences that merely imply it.
- Prefer sentences that are reusable as standalone propositions.
- Prefer sentences that express the article's organizing argument over sentences that only elaborate one branch of it.
- Prefer broader supporting reasons over narrow logistical details.
"""


CLAIM_CODING_SCHEMA = {
    'type': 'object',
    'additionalProperties': False,
    'properties': {
        'central_claim_summary': {
            'type': 'string',
            'description': 'Exactly one sentence stating the article’s central endorsed claim as a standalone proposition, without attributional framing.',
        },
        'has_clear_central_thesis': {
            'type': 'boolean',
            'description': 'Whether the article has a clear single dominant thesis rather than a diffuse or multi-claim structure.',
        },
        'thesis_sentence_id': {
            'type': 'string',
            'pattern': '^s\\d+$',
            'description': 'The single sentence ID that best states the author’s main endorsed position.',
        },
        'support_sentence_ids': {
            'type': 'array',
            'items': {'type': 'string', 'pattern': '^s\\d+$'},
            'maxItems': 2,
            'description': 'Up to 2 sentence IDs that provide distinct major support for the same dominant thesis. These should be general argumentative reasons, not merely examples or implementation details.',
        },
        'secondary_claim_ids': {
            'type': 'array',
            'items': {'type': 'string', 'pattern': '^s\\d+$'},
            'maxItems': 3,
            'description': 'Up to 3 additional endorsed claim-bearing sentence IDs that express important secondary or parallel claims, excluding the thesis, supports, examples, and mere restatements.',
        },
    },
    'required': [
        'central_claim_summary',
        'has_clear_central_thesis',
        'thesis_sentence_id',
        'support_sentence_ids',
        'secondary_claim_ids',
    ],
}

def build_claim_coding_input(sentence_df, article_title=None):
    lines = []
    if article_title:
        lines.append(f"ARTICLE TITLE: {article_title}")
        lines.append("")

    lines.append("SENTENCES:")
    for row in sentence_df.itertuples(index=False):
        lines.append(f"{row.sentence_id}: {row.sentence}")

    lines.append("")
    lines.append("Return only the JSON object that matches the schema.")
    return "\n".join(lines)


def build_responses_body(
    sentence_df,
    article_title=None,
    model=DEFAULT_MODEL,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
):
    return {
        "model": model,
        "instructions": CLAIM_CODING_INSTRUCTIONS,
        "input": build_claim_coding_input(sentence_df, article_title=article_title),
        "reasoning": {"effort": reasoning_effort},
        "max_output_tokens": int(max_output_tokens),
        "store": False,
        "text": {
            "verbosity": "low",
            "format": {
                "type": "json_schema",
                "name": "article_claim_coding",
                "description": "Structured coding of an article's dominant thesis sentence and supporting claim-bearing sentences.",
                "strict": True,
                "schema": CLAIM_CODING_SCHEMA,
            },
        },
    }


def _serialize_value(value):
    if value is None:
        return None
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    if hasattr(value, "isoformat"):
        try:
            return value.isoformat()
        except Exception:
            pass
    return str(value)


def _safe_custom_id(article_id, idx):
    slug = re.sub(r"[^A-Za-z0-9_-]+", "_", str(article_id)).strip("_")
    if not slug:
        slug = f"article_{idx}"
    return f"claim-{idx:06d}-{slug[:80]}"


def _timestamp_slug():
    return datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")


def _sdk_to_dict(obj):
    if obj is None:
        return None
    if hasattr(obj, "model_dump"):
        return obj.model_dump()
    if isinstance(obj, dict):
        return obj
    return json.loads(json.dumps(obj, default=str))


def _write_jsonl(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")


def _write_json(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _path_exists(path):
    return bool(path) and Path(path).exists()


def _load_client():
    load_dotenv(PROJECT_ROOT / ".env")
    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("Set OPENAI_API_KEY in your .env file or shell before using the Batch API script.")
    return OpenAI(api_key=api_key)


def _filter_articles(articles, article_ids=None, title_query=None, limit=None):
    filtered = articles.copy()

    if article_ids:
        normalized_ids = {str(article_id).strip() for article_id in article_ids}
        filtered = filtered.loc[filtered["id"].astype("string").str.strip().isin(normalized_ids)]

    if title_query:
        filtered = filtered.loc[
            filtered["title"].astype("string").str.contains(title_query, case=False, na=False)
        ]

    filtered = filtered.reset_index(drop=True)
    if limit is not None:
        filtered = filtered.head(int(limit)).reset_index(drop=True)

    return filtered


def _chunk_articles(articles, articles_per_batch):
    articles_per_batch = int(articles_per_batch)
    if articles_per_batch <= 0:
        raise ValueError("articles_per_batch must be a positive integer.")

    chunks = []
    for start in range(0, len(articles), articles_per_batch):
        chunk = articles.iloc[start : start + articles_per_batch].reset_index(drop=True)
        chunks.append(chunk)
    return chunks


def _estimate_text_tokens(text):
    return max(1, (len(text) + 3) // 4)


def estimate_request_enqueued_tokens(body):
    body_text = json.dumps(body, ensure_ascii=False)
    return _estimate_text_tokens(body_text) + int(body.get("max_output_tokens", 0))


def build_batch_files(
    articles,
    output_dir=DEFAULT_OUTPUT_DIR,
    model=DEFAULT_MODEL,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    source_years=None,
    run_dir=None,
):
    output_dir = Path(output_dir)
    run_dir = Path(run_dir) if run_dir is not None else output_dir / f"claim_coding_batch_{_timestamp_slug()}"
    run_dir.mkdir(parents=True, exist_ok=True)

    request_rows = []
    manifest_rows = []
    estimated_enqueued_tokens = 0

    for idx, (_, article_row) in enumerate(articles.iterrows()):
        sentence_df = sentence_table_from_text(article_row["body_text"])
        if sentence_df.empty:
            continue

        custom_id = _safe_custom_id(article_row.get("id", idx), idx)
        body = build_responses_body(
            sentence_df=sentence_df,
            article_title=article_row.get("title"),
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
        )
        estimated_enqueued_tokens += estimate_request_enqueued_tokens(body)

        request_rows.append(
            {
                "custom_id": custom_id,
                "method": "POST",
                "url": "/v1/responses",
                "body": body,
            }
        )
        manifest_rows.append(
            {
                "custom_id": custom_id,
                "article_id": _serialize_value(article_row.get("id")),
                "title": _serialize_value(article_row.get("title")),
                "date": _serialize_value(article_row.get("date")),
                "author_display": _serialize_value(article_row.get("author_display")),
                "year": _serialize_value(article_row.get("year")),
                "prompt_version": PROMPT_VERSION,
                "model": model,
                "reasoning_effort": reasoning_effort,
                "max_output_tokens": int(max_output_tokens),
                "sentence_table": sentence_df.to_dict(orient="records"),
            }
        )

    if not request_rows:
        raise ValueError("No batch requests were generated.")

    requests_path = run_dir / "batch_input.jsonl"
    manifest_path = run_dir / "batch_manifest.jsonl"
    summary_path = run_dir / "batch_prepare_summary.json"

    _write_jsonl(requests_path, request_rows)
    _write_jsonl(manifest_path, manifest_rows)

    summary = {
        "run_dir": str(run_dir),
        "requests_path": str(requests_path),
        "manifest_path": str(manifest_path),
        "request_count": len(request_rows),
        "article_count": int(len(articles)),
        "estimated_enqueued_tokens": int(estimated_enqueued_tokens),
        "source_years": list(source_years) if source_years is not None else None,
        "prompt_version": PROMPT_VERSION,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": int(max_output_tokens),
        "prepared_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_json(summary_path, summary)
    return summary


def build_chunked_batch_plan(
    articles,
    output_dir=DEFAULT_OUTPUT_DIR,
    articles_per_batch=DEFAULT_ARTICLES_PER_BATCH,
    model=DEFAULT_MODEL,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    source_years=None,
):
    output_dir = Path(output_dir)
    parent_run_dir = output_dir / f"claim_coding_batch_series_{_timestamp_slug()}"
    parent_run_dir.mkdir(parents=True, exist_ok=True)

    article_chunks = _chunk_articles(articles, articles_per_batch)
    total_chunks = len(article_chunks)
    chunk_rows = []

    for chunk_index, chunk_articles in enumerate(article_chunks, start=1):
        chunk_run_dir = parent_run_dir / f"chunk_{chunk_index:03d}_of_{total_chunks:03d}"
        prepared = build_batch_files(
            articles=chunk_articles,
            output_dir=output_dir,
            model=model,
            reasoning_effort=reasoning_effort,
            max_output_tokens=max_output_tokens,
            source_years=source_years,
            run_dir=chunk_run_dir,
        )
        chunk_rows.append(
            {
                "chunk_index": chunk_index,
                "total_chunks": total_chunks,
                "status": "prepared",
                "article_count": int(len(chunk_articles)),
                "request_count": prepared["request_count"],
                "estimated_enqueued_tokens": prepared["estimated_enqueued_tokens"],
                "run_dir": prepared["run_dir"],
                "requests_path": prepared["requests_path"],
                "manifest_path": prepared["manifest_path"],
                "prepared_summary_path": str(Path(prepared["run_dir"]) / "batch_prepare_summary.json"),
                "batch_id": None,
                "submission_path": None,
                "submit_error": None,
            }
        )

    plan = {
        "plan_path": str(parent_run_dir / "chunk_plan.json"),
        "parent_run_dir": str(parent_run_dir),
        "prompt_version": PROMPT_VERSION,
        "source_years": list(source_years) if source_years is not None else None,
        "article_count": int(len(articles)),
        "articles_per_batch": int(articles_per_batch),
        "total_chunks": total_chunks,
        "model": model,
        "reasoning_effort": reasoning_effort,
        "max_output_tokens": int(max_output_tokens),
        "created_at": datetime.now(timezone.utc).isoformat(),
        "chunks": chunk_rows,
    }
    _write_json(plan["plan_path"], plan)
    return plan


def summarize_chunk_plan(plan):
    chunks = plan.get("chunks", [])
    prepared_chunks = 0
    active_chunks = 0
    completed_chunks = 0
    failed_chunks = 0
    next_pending_chunk_index = None

    for chunk in chunks:
        status = chunk.get("status")
        if status == "prepared":
            prepared_chunks += 1
            if next_pending_chunk_index is None:
                next_pending_chunk_index = chunk.get("chunk_index")
        elif status in ACTIVE_BATCH_STATUSES:
            active_chunks += 1
        elif status == "completed":
            completed_chunks += 1
        elif status in {"failed", "submit_failed", "cancelled", "expired"}:
            failed_chunks += 1

    return {
        "total_chunks": len(chunks),
        "prepared_chunks": prepared_chunks,
        "active_chunks": active_chunks,
        "completed_chunks": completed_chunks,
        "failed_chunks": failed_chunks,
        "next_pending_chunk_index": next_pending_chunk_index,
        "done": prepared_chunks == 0 and active_chunks == 0,
    }


def sync_chunk_plan(
    plan_path,
    auto_download=True,
    auto_parse=True,
):
    plan_path = Path(plan_path)
    plan = _read_json(plan_path)
    updated_chunks = []

    for chunk in plan["chunks"]:
        batch_id = chunk.get("batch_id")
        if not batch_id:
            continue

        batch = retrieve_batch(batch_id)
        batch_status = batch.get("status")
        chunk["status"] = batch_status
        chunk["status_checked_at"] = datetime.now(timezone.utc).isoformat()
        chunk["batch_path"] = str(Path(chunk["run_dir"]) / f"{batch_id}_batch.json")
        _write_json(chunk["batch_path"], batch)

        request_counts = batch.get("request_counts") or {}
        chunk["request_counts"] = request_counts
        chunk["usage"] = batch.get("usage")
        chunk["batch_errors"] = batch.get("errors")
        updated_chunks.append(
            {
                "chunk_index": chunk["chunk_index"],
                "batch_id": batch_id,
                "status": batch_status,
                "request_counts": request_counts,
            }
        )

        if auto_download and batch_status in TERMINAL_BATCH_STATUSES:
            try:
                artifacts = download_batch_artifacts(batch_id=batch_id, output_dir=chunk["run_dir"])
                chunk["batch_path"] = artifacts.get("batch_path")
                chunk["output_file_path"] = artifacts.get("output_file_path")
                chunk["error_file_path"] = artifacts.get("error_file_path")
            except Exception as exc:
                chunk["download_error"] = f"{type(exc).__name__}: {exc}"

        if auto_parse and batch_status == "completed" and _path_exists(chunk.get("output_file_path")):
            try:
                parsed = parse_batch_results(
                    output_file_path=chunk["output_file_path"],
                    manifest_path=chunk["manifest_path"],
                    error_file_path=chunk.get("error_file_path"),
                    output_dir=chunk["run_dir"],
                )
                chunk["parsed_jsonl_path"] = parsed.get("parsed_jsonl_path")
                chunk["parsed_csv_path"] = parsed.get("parsed_csv_path")
                chunk["parsed_count"] = parsed.get("count")
                chunk["parse_error"] = None
            except Exception as exc:
                chunk["parse_error"] = f"{type(exc).__name__}: {exc}"

    plan["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(plan_path, plan)
    return {
        "plan_path": str(plan_path),
        "parent_run_dir": plan["parent_run_dir"],
        "summary": summarize_chunk_plan(plan),
        "updated_chunks": updated_chunks,
    }


def submit_chunk_plan(
    plan_path,
    max_chunks_to_submit=None,
    max_estimated_enqueued_tokens=DEFAULT_MAX_ESTIMATED_ENQUEUED_TOKENS,
    completion_window=DEFAULT_COMPLETION_WINDOW,
):
    plan_path = Path(plan_path)
    plan = _read_json(plan_path)

    submitted_chunks = []
    submitted_count = 0
    submitted_estimated_tokens = 0
    active_estimated_tokens = sum(
        int(chunk.get("estimated_enqueued_tokens") or 0)
        for chunk in plan["chunks"]
        if chunk.get("batch_id") and chunk.get("status") in ACTIVE_BATCH_STATUSES
    )

    for chunk in plan["chunks"]:
        if chunk.get("batch_id") or chunk.get("status") in ACTIVE_BATCH_STATUSES:
            continue

        if max_chunks_to_submit is not None and submitted_count >= int(max_chunks_to_submit):
            break

        chunk_estimated_tokens = int(chunk.get("estimated_enqueued_tokens") or 0)
        would_exceed_budget = (
            active_estimated_tokens + submitted_estimated_tokens + chunk_estimated_tokens
            > int(max_estimated_enqueued_tokens)
        )
        if would_exceed_budget:
            break

        try:
            submission = submit_batch_from_prepared_files(
                requests_path=chunk["requests_path"],
                manifest_path=chunk["manifest_path"],
                output_dir=chunk["run_dir"],
                completion_window=completion_window,
            )
            batch = submission["batch"]
            submission_path = Path(chunk["run_dir"]) / "batch_submission.json"

            chunk["status"] = "submitted"
            chunk["batch_id"] = batch["id"]
            chunk["submission_path"] = str(submission_path)
            chunk["submitted_at"] = submission["submitted_at"]
            chunk["submit_error"] = None

            submitted_count += 1
            submitted_estimated_tokens += chunk_estimated_tokens
            submitted_chunks.append(
                {
                    "chunk_index": chunk["chunk_index"],
                    "batch_id": batch["id"],
                    "status": batch["status"],
                    "estimated_enqueued_tokens": chunk_estimated_tokens,
                    "run_dir": chunk["run_dir"],
                }
            )
        except Exception as exc:
            chunk["status"] = "submit_failed"
            chunk["submit_error"] = f"{type(exc).__name__}: {exc}"
            break

    plan["updated_at"] = datetime.now(timezone.utc).isoformat()
    _write_json(plan_path, plan)

    remaining_chunk_indices = [
        chunk["chunk_index"]
        for chunk in plan["chunks"]
        if not chunk.get("batch_id") and chunk.get("status") != "submit_failed"
    ]

    return {
        "plan_path": str(plan_path),
        "parent_run_dir": plan["parent_run_dir"],
        "active_estimated_enqueued_tokens": active_estimated_tokens,
        "submitted_chunk_count": submitted_count,
        "submitted_estimated_enqueued_tokens": submitted_estimated_tokens,
        "remaining_chunk_indices": remaining_chunk_indices,
        "submitted_chunks": submitted_chunks,
    }


def run_chunk_plan(
    plan_path,
    max_chunks_to_submit=None,
    max_estimated_enqueued_tokens=DEFAULT_MAX_ESTIMATED_ENQUEUED_TOKENS,
    completion_window=DEFAULT_COMPLETION_WINDOW,
    poll_seconds=120,
    until_done=False,
    auto_download=True,
    auto_parse=True,
):
    cycles = []

    while True:
        sync_result = sync_chunk_plan(
            plan_path=plan_path,
            auto_download=auto_download,
            auto_parse=auto_parse,
        )
        submit_result = submit_chunk_plan(
            plan_path=plan_path,
            max_chunks_to_submit=max_chunks_to_submit,
            max_estimated_enqueued_tokens=max_estimated_enqueued_tokens,
            completion_window=completion_window,
        )
        plan = _read_json(plan_path)
        summary = summarize_chunk_plan(plan)

        cycles.append(
            {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "sync": sync_result,
                "submit": submit_result,
                "summary": summary,
            }
        )

        if not until_done or summary["done"]:
            break

        time.sleep(float(poll_seconds))

    return {
        "plan_path": str(plan_path),
        "parent_run_dir": _read_json(plan_path)["parent_run_dir"],
        "cycles": cycles,
        "final_summary": summarize_chunk_plan(_read_json(plan_path)),
    }


def submit_batch_from_prepared_files(
    requests_path,
    manifest_path,
    output_dir=None,
    completion_window=DEFAULT_COMPLETION_WINDOW,
):
    requests_path = Path(requests_path)
    manifest_path = Path(manifest_path)
    run_dir = Path(output_dir) if output_dir else requests_path.parent
    run_dir.mkdir(parents=True, exist_ok=True)

    client = _load_client()
    with requests_path.open("rb") as f:
        input_file = client.files.create(file=f, purpose="batch")

    batch = client.batches.create(
        input_file_id=input_file.id,
        endpoint="/v1/responses",
        completion_window=completion_window,
        metadata={
            "task": "article_claim_coding",
            "prompt_version": PROMPT_VERSION,
            "manifest_file": manifest_path.name,
        },
    )

    submission = {
        "run_dir": str(run_dir),
        "requests_path": str(requests_path),
        "manifest_path": str(manifest_path),
        "input_file": _sdk_to_dict(input_file),
        "batch": _sdk_to_dict(batch),
        "submitted_at": datetime.now(timezone.utc).isoformat(),
    }
    submission_path = run_dir / "batch_submission.json"
    submission_path.write_text(json.dumps(submission, indent=2), encoding="utf-8")
    return submission


def retrieve_batch(batch_id):
    client = _load_client()
    batch = client.batches.retrieve(batch_id)
    return _sdk_to_dict(batch)


def _download_file_content(client, file_id, target_path):
    file_response = client.files.content(file_id)
    text = getattr(file_response, "text", None)
    if text is None:
        text = str(file_response)
    Path(target_path).write_text(text, encoding="utf-8")
    return str(target_path)


def download_batch_artifacts(batch_id, output_dir=DEFAULT_OUTPUT_DIR):
    client = _load_client()
    batch = client.batches.retrieve(batch_id)
    batch_dict = _sdk_to_dict(batch)

    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    batch_path = output_dir / f"{batch_id}_batch.json"
    batch_path.write_text(json.dumps(batch_dict, indent=2), encoding="utf-8")

    result = {
        "batch_id": batch_id,
        "status": batch_dict.get("status"),
        "batch_path": str(batch_path),
        "output_file_path": None,
        "error_file_path": None,
    }

    output_file_id = batch_dict.get("output_file_id")
    error_file_id = batch_dict.get("error_file_id")
    if output_file_id:
        result["output_file_path"] = _download_file_content(
            client=client,
            file_id=output_file_id,
            target_path=output_dir / f"{batch_id}_output.jsonl",
        )
    if error_file_id:
        result["error_file_path"] = _download_file_content(
            client=client,
            file_id=error_file_id,
            target_path=output_dir / f"{batch_id}_errors.jsonl",
        )

    return result


def dedupe_preserve_order(values):
    deduped = []
    seen = set()
    for value in values:
        if value in seen:
            continue
        deduped.append(value)
        seen.add(value)
    return deduped


def validate_claim_coding_output(payload, valid_ids):
    thesis_id = payload["thesis_sentence_id"]
    support_ids = dedupe_preserve_order(payload.get("support_sentence_ids", []))[:2]
    secondary_ids = dedupe_preserve_order(payload.get("secondary_claim_ids", []))[:3]

    if thesis_id not in valid_ids:
        raise ValueError(f"thesis_sentence_id {thesis_id!r} is not one of the provided sentence IDs.")

    invalid_support = [sid for sid in support_ids if sid not in valid_ids]
    invalid_secondary = [sid for sid in secondary_ids if sid not in valid_ids]
    if invalid_support:
        raise ValueError(f"Invalid support sentence IDs returned: {invalid_support}")
    if invalid_secondary:
        raise ValueError(f"Invalid secondary claim IDs returned: {invalid_secondary}")
    if thesis_id in support_ids or thesis_id in secondary_ids:
        raise ValueError("The thesis sentence ID must not be repeated in support or additional lists.")

    overlap = set(support_ids) & set(secondary_ids)
    if overlap:
        raise ValueError(f"Support and additional candidate IDs overlap: {sorted(overlap)}")

    return {
        "central_claim_summary": normalize_whitespace(payload["central_claim_summary"]),
        "has_clear_central_thesis": bool(payload["has_clear_central_thesis"]),
        "thesis_sentence_id": thesis_id,
        "support_sentence_ids": support_ids,
        "secondary_claim_ids": secondary_ids,
    }


def extract_output_text_from_response_body(body):
    output_text = body.get("output_text")
    if output_text:
        return output_text

    chunks = []
    for item in body.get("output", []):
        if item.get("type") != "message":
            continue
        for content in item.get("content", []):
            if content.get("type") == "output_text":
                chunks.append(content.get("text", ""))
    return "".join(chunks)


def parse_batch_results(output_file_path, manifest_path, error_file_path=None, output_dir=None):
    output_rows = _read_jsonl(output_file_path)
    manifest_rows = _read_jsonl(manifest_path)
    manifest_by_custom_id = {row["custom_id"]: row for row in manifest_rows}

    parsed_rows = []
    for row in output_rows:
        custom_id = row.get("custom_id")
        manifest = manifest_by_custom_id.get(custom_id)
        if manifest is None:
            continue

        sentence_table = manifest.get("sentence_table", [])
        lookup = sentence_lookup(sentence_table)
        valid_ids = set(lookup)
        response = row.get("response") or {}
        response_body = response.get("body") or {}
        output_text = extract_output_text_from_response_body(response_body)

        if not output_text:
            parsed_rows.append(
                {
                    "custom_id": custom_id,
                    "article_id": manifest.get("article_id"),
                    "title": manifest.get("title"),
                    "error": "No output_text found in batch response body.",
                }
            )
            continue

        try:
            payload = json.loads(output_text)
            validated = validate_claim_coding_output(payload, valid_ids=valid_ids)
            parsed_rows.append(
                {
                    "custom_id": custom_id,
                    "article_id": manifest.get("article_id"),
                    "title": manifest.get("title"),
                    "central_claim_summary": validated["central_claim_summary"],
                    "has_clear_central_thesis": validated["has_clear_central_thesis"],
                    "thesis_sentence_id": validated["thesis_sentence_id"],
                    "thesis_sentence": lookup.get(validated["thesis_sentence_id"]),
                    "support_sentence_ids": validated["support_sentence_ids"],
                    "support_sentences": [lookup[sid] for sid in validated["support_sentence_ids"]],
                    "secondary_claim_ids": validated["secondary_claim_ids"],
                    "secondary_claim_sentences": [
                        lookup[sid] for sid in validated["secondary_claim_ids"]
                    ],
                    "response_status_code": response.get("status_code"),
                    "response_request_id": response.get("request_id"),
                    "model": response_body.get("model"),
                    "usage": response_body.get("usage"),
                    "error": None,
                }
            )
        except Exception as exc:
            parsed_rows.append(
                {
                    "custom_id": custom_id,
                    "article_id": manifest.get("article_id"),
                    "title": manifest.get("title"),
                    "error": f"{type(exc).__name__}: {exc}",
                    "raw_output_text": output_text,
                }
            )

    error_rows = _read_jsonl(error_file_path) if error_file_path and Path(error_file_path).exists() else []
    for row in error_rows:
        custom_id = row.get("custom_id")
        manifest = manifest_by_custom_id.get(custom_id, {})
        parsed_rows.append(
            {
                "custom_id": custom_id,
                "article_id": manifest.get("article_id"),
                "title": manifest.get("title"),
                "error": row.get("error"),
            }
        )

    result = {
        "parsed_rows": parsed_rows,
        "count": len(parsed_rows),
    }

    if output_dir:
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        parsed_jsonl_path = output_dir / "parsed_claim_coding_results.jsonl"
        parsed_csv_path = output_dir / "parsed_claim_coding_results.csv"
        _write_jsonl(parsed_jsonl_path, parsed_rows)
        pd.DataFrame(parsed_rows).to_csv(parsed_csv_path, index=False)
        result["parsed_jsonl_path"] = str(parsed_jsonl_path)
        result["parsed_csv_path"] = str(parsed_csv_path)

    return result


def submit_guardian_claim_batch(
    years,
    raw_data_dir=DEFAULT_RAW_DATA_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    min_body_text_chars=1000,
    article_ids=None,
    title_query=None,
    limit=None,
    model=DEFAULT_MODEL,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    completion_window=DEFAULT_COMPLETION_WINDOW,
):
    articles = load_and_clean_guardian_years(
        years=years,
        folder=raw_data_dir,
        min_body_text_chars=min_body_text_chars,
    )
    filtered_articles = _filter_articles(
        articles=articles,
        article_ids=article_ids,
        title_query=title_query,
        limit=limit,
    )
    if filtered_articles.empty:
        raise ValueError("No articles matched the requested years and filters.")

    prepared = build_batch_files(
        articles=filtered_articles,
        output_dir=output_dir,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        source_years=years,
    )
    submission = submit_batch_from_prepared_files(
        requests_path=prepared["requests_path"],
        manifest_path=prepared["manifest_path"],
        output_dir=prepared["run_dir"],
        completion_window=completion_window,
    )
    return {
        "prepared": prepared,
        "submission": submission,
    }


def submit_guardian_claim_batches_chunked(
    years,
    raw_data_dir=DEFAULT_RAW_DATA_DIR,
    output_dir=DEFAULT_OUTPUT_DIR,
    min_body_text_chars=1000,
    article_ids=None,
    title_query=None,
    limit=None,
    model=DEFAULT_MODEL,
    reasoning_effort=DEFAULT_REASONING_EFFORT,
    max_output_tokens=DEFAULT_MAX_OUTPUT_TOKENS,
    completion_window=DEFAULT_COMPLETION_WINDOW,
    articles_per_batch=DEFAULT_ARTICLES_PER_BATCH,
    max_chunks_to_submit=None,
    max_estimated_enqueued_tokens=DEFAULT_MAX_ESTIMATED_ENQUEUED_TOKENS,
):
    articles = load_and_clean_guardian_years(
        years=years,
        folder=raw_data_dir,
        min_body_text_chars=min_body_text_chars,
    )
    filtered_articles = _filter_articles(
        articles=articles,
        article_ids=article_ids,
        title_query=title_query,
        limit=limit,
    )
    if filtered_articles.empty:
        raise ValueError("No articles matched the requested years and filters.")

    plan = build_chunked_batch_plan(
        articles=filtered_articles,
        output_dir=output_dir,
        articles_per_batch=articles_per_batch,
        model=model,
        reasoning_effort=reasoning_effort,
        max_output_tokens=max_output_tokens,
        source_years=years,
    )
    submission = submit_chunk_plan(
        plan_path=plan["plan_path"],
        max_chunks_to_submit=max_chunks_to_submit,
        max_estimated_enqueued_tokens=max_estimated_enqueued_tokens,
        completion_window=completion_window,
    )
    return {
        "plan": plan,
        "submission": submission,
    }


def _parse_args():
    parser = argparse.ArgumentParser(description="Submit and manage OpenAI Batch API claim-coding jobs.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    submit_parser = subparsers.add_parser("submit", help="Prepare batch input from Guardian articles and submit it.")
    submit_parser.add_argument("--years", nargs="+", type=int, required=True)
    submit_parser.add_argument("--raw-data-dir", default=str(DEFAULT_RAW_DATA_DIR))
    submit_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))
    submit_parser.add_argument("--min-body-text-chars", type=int, default=1000)
    submit_parser.add_argument("--article-ids", nargs="*")
    submit_parser.add_argument("--title-query")
    submit_parser.add_argument("--limit", type=int)
    submit_parser.add_argument("--model", default=DEFAULT_MODEL)
    submit_parser.add_argument("--use-snapshot", action="store_true")
    submit_parser.add_argument("--reasoning-effort", default=DEFAULT_REASONING_EFFORT)
    submit_parser.add_argument("--max-output-tokens", type=int, default=DEFAULT_MAX_OUTPUT_TOKENS)
    submit_parser.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW)
    submit_parser.add_argument("--articles-per-batch", type=int, default=DEFAULT_ARTICLES_PER_BATCH)
    submit_parser.add_argument("--max-chunks-to-submit", type=int)
    submit_parser.add_argument(
        "--max-estimated-enqueued-tokens",
        type=int,
        default=DEFAULT_MAX_ESTIMATED_ENQUEUED_TOKENS,
    )
    submit_parser.add_argument("--no-auto-chunk", action="store_true")

    status_parser = subparsers.add_parser("status", help="Retrieve the status for an existing batch.")
    status_parser.add_argument("--batch-id", required=True)

    download_parser = subparsers.add_parser("download", help="Download output and error files for a completed batch.")
    download_parser.add_argument("--batch-id", required=True)
    download_parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    parse_parser = subparsers.add_parser("parse", help="Parse downloaded batch output and join it with the manifest.")
    parse_parser.add_argument("--output-file-path", required=True)
    parse_parser.add_argument("--manifest-path", required=True)
    parse_parser.add_argument("--error-file-path")
    parse_parser.add_argument("--output-dir")

    submit_plan_parser = subparsers.add_parser(
        "submit-plan",
        help="Submit pending chunks from an existing chunk plan.",
    )
    submit_plan_parser.add_argument("--plan-path", required=True)
    submit_plan_parser.add_argument("--max-chunks-to-submit", type=int)
    submit_plan_parser.add_argument(
        "--max-estimated-enqueued-tokens",
        type=int,
        default=DEFAULT_MAX_ESTIMATED_ENQUEUED_TOKENS,
    )
    submit_plan_parser.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW)

    status_plan_parser = subparsers.add_parser(
        "status-plan",
        help="Refresh and summarize all chunk statuses for an existing chunk plan.",
    )
    status_plan_parser.add_argument("--plan-path", required=True)
    status_plan_parser.add_argument("--no-auto-download", action="store_true")
    status_plan_parser.add_argument("--no-auto-parse", action="store_true")

    run_plan_parser = subparsers.add_parser(
        "run-plan",
        help="Refresh chunk statuses, download/parse completed outputs, and submit the next safe chunk(s).",
    )
    run_plan_parser.add_argument("--plan-path", required=True)
    run_plan_parser.add_argument("--max-chunks-to-submit", type=int)
    run_plan_parser.add_argument(
        "--max-estimated-enqueued-tokens",
        type=int,
        default=DEFAULT_MAX_ESTIMATED_ENQUEUED_TOKENS,
    )
    run_plan_parser.add_argument("--completion-window", default=DEFAULT_COMPLETION_WINDOW)
    run_plan_parser.add_argument("--poll-seconds", type=float, default=120)
    run_plan_parser.add_argument("--until-done", action="store_true")
    run_plan_parser.add_argument("--no-auto-download", action="store_true")
    run_plan_parser.add_argument("--no-auto-parse", action="store_true")

    return parser.parse_args()


def main():
    args = _parse_args()

    if args.command == "submit":
        model = DEFAULT_MODEL_SNAPSHOT if args.use_snapshot else args.model
        if args.no_auto_chunk or not args.articles_per_batch:
            result = submit_guardian_claim_batch(
                years=args.years,
                raw_data_dir=args.raw_data_dir,
                output_dir=args.output_dir,
                min_body_text_chars=args.min_body_text_chars,
                article_ids=args.article_ids,
                title_query=args.title_query,
                limit=args.limit,
                model=model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                completion_window=args.completion_window,
            )
        else:
            result = submit_guardian_claim_batches_chunked(
                years=args.years,
                raw_data_dir=args.raw_data_dir,
                output_dir=args.output_dir,
                min_body_text_chars=args.min_body_text_chars,
                article_ids=args.article_ids,
                title_query=args.title_query,
                limit=args.limit,
                model=model,
                reasoning_effort=args.reasoning_effort,
                max_output_tokens=args.max_output_tokens,
                completion_window=args.completion_window,
                articles_per_batch=args.articles_per_batch,
                max_chunks_to_submit=args.max_chunks_to_submit,
                max_estimated_enqueued_tokens=args.max_estimated_enqueued_tokens,
            )
        print(json.dumps(result, indent=2))
        return

    if args.command == "status":
        result = retrieve_batch(args.batch_id)
        print(json.dumps(result, indent=2))
        return

    if args.command == "download":
        result = download_batch_artifacts(
            batch_id=args.batch_id,
            output_dir=args.output_dir,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "parse":
        result = parse_batch_results(
            output_file_path=args.output_file_path,
            manifest_path=args.manifest_path,
            error_file_path=args.error_file_path,
            output_dir=args.output_dir,
        )
        printable = {key: value for key, value in result.items() if key != "parsed_rows"}
        printable["count"] = result["count"]
        print(json.dumps(printable, indent=2))
        return

    if args.command == "submit-plan":
        result = submit_chunk_plan(
            plan_path=args.plan_path,
            max_chunks_to_submit=args.max_chunks_to_submit,
            max_estimated_enqueued_tokens=args.max_estimated_enqueued_tokens,
            completion_window=args.completion_window,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "status-plan":
        result = sync_chunk_plan(
            plan_path=args.plan_path,
            auto_download=not args.no_auto_download,
            auto_parse=not args.no_auto_parse,
        )
        print(json.dumps(result, indent=2))
        return

    if args.command == "run-plan":
        result = run_chunk_plan(
            plan_path=args.plan_path,
            max_chunks_to_submit=args.max_chunks_to_submit,
            max_estimated_enqueued_tokens=args.max_estimated_enqueued_tokens,
            completion_window=args.completion_window,
            poll_seconds=args.poll_seconds,
            until_done=args.until_done,
            auto_download=not args.no_auto_download,
            auto_parse=not args.no_auto_parse,
        )
        print(json.dumps(result, indent=2))
        return


if __name__ == "__main__":
    main()
