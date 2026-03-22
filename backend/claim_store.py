import gzip
import json
import re
from pathlib import Path

from backend.runtime_debug import log_runtime_event


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BATCH_CLAIM_RESULTS_DIR = (
    PROJECT_ROOT / "backend" / "data" / "processed" / "openai_claim_batches"
)
PACKAGED_CLAIM_RESULTS_DIR = (
    PROJECT_ROOT / "backend" / "data" / "processed" / "claim_coding_clean"
)


_ARTICLE_YEAR_RE = re.compile(r"/(20\d{2})/")


def default_claim_results_dir():
    if PACKAGED_CLAIM_RESULTS_DIR.exists():
        return PACKAGED_CLAIM_RESULTS_DIR
    return RAW_BATCH_CLAIM_RESULTS_DIR


def _claim_result_files(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return []
    paths = list(root.rglob("claims_*.jsonl"))
    paths.extend(root.rglob("claims_*.jsonl.gz"))
    return sorted(set(paths))


def _iter_jsonl(path):
    path = Path(path)
    open_fn = gzip.open if path.suffix == ".gz" else path.open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def _normalize_article_id(article_id):
    return str(article_id or "").strip()


def _normalize_year(value):
    if value is None:
        return None
    text = str(value).strip()
    if len(text) == 4 and text.isdigit():
        return int(text)
    return None


def _normalize_bool(value):
    if value is None:
        return None
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"1", "true", "yes", "y", "on"}:
            return True
        if normalized in {"0", "false", "no", "n", "off"}:
            return False
    return None


def _normalize_list(value):
    if isinstance(value, list):
        return value
    return []


def _infer_year_from_article_id(article_id):
    match = _ARTICLE_YEAR_RE.search(str(article_id or ""))
    if not match:
        return None
    return int(match.group(1))


def normalized_claim_record(row):
    article_id = _normalize_article_id(row.get("article_id"))
    if not article_id:
        return None
    if row.get("error"):
        return None
    if not row.get("central_claim_summary"):
        return None

    return {
        "article_id": article_id,
        "title": row.get("title"),
        "year": _normalize_year(row.get("year")) or _infer_year_from_article_id(article_id),
        "central_claim_summary": str(row.get("central_claim_summary") or "").strip(),
        "has_clear_central_thesis": _normalize_bool(row.get("has_clear_central_thesis")),
        "thesis_sentence_id": row.get("thesis_sentence_id"),
        "thesis_sentence": row.get("thesis_sentence"),
        "support_sentence_ids": _normalize_list(row.get("support_sentence_ids")),
        "support_sentences": _normalize_list(row.get("support_sentences")),
        "secondary_claim_ids": _normalize_list(row.get("secondary_claim_ids")),
        "secondary_claim_sentences": _normalize_list(row.get("secondary_claim_sentences")),
    }


def claim_manifest(root_dir=None):
    resolved_root_dir = default_claim_results_dir() if root_dir is None else Path(root_dir)
    manifest_path = Path(resolved_root_dir) / "manifest.json"
    if not manifest_path.exists():
        return {}
    try:
        with open(manifest_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}
    except Exception:
        return {}


def expected_claim_record_count(root_dir=None):
    manifest = claim_manifest(root_dir=root_dir)
    try:
        return int(manifest.get("record_count"))
    except (TypeError, ValueError):
        return None


def iter_claim_records(root_dir=None):
    resolved_root_dir = default_claim_results_dir() if root_dir is None else Path(root_dir)
    for path in _claim_result_files(resolved_root_dir):
        for row in _iter_jsonl(path):
            record = normalized_claim_record(row)
            if record is None:
                continue
            yield record


def _claim_payload(record):
    return {
        "central_claim_summary": record["central_claim_summary"],
        "has_clear_central_thesis": record["has_clear_central_thesis"],
        "thesis_sentence_id": record["thesis_sentence_id"],
        "thesis_sentence": record["thesis_sentence"],
        "support_sentence_ids": record["support_sentence_ids"] or [],
        "support_sentences": record["support_sentences"] or [],
        "secondary_claim_ids": record["secondary_claim_ids"] or [],
        "secondary_claim_sentences": record["secondary_claim_sentences"] or [],
        "claim_source_path": "sqlite:guardian_article_claims",
        "claim_available": True,
    }


def get_claim_records(article_refs):
    requested_ids = []
    for ref in article_refs or []:
        if isinstance(ref, dict):
            article_id = _normalize_article_id(ref.get("id") or ref.get("article_id"))
        else:
            article_id = _normalize_article_id(ref)
        if not article_id or article_id in requested_ids:
            continue
        requested_ids.append(article_id)

    if not requested_ids:
        return {}

    log_runtime_event(
        "claim_store.db_lookup_start",
        requested_count=len(requested_ids),
    )

    from models import GuardianArticleClaim, db

    rows = (
        db.session.query(
            GuardianArticleClaim.article_id,
            GuardianArticleClaim.central_claim_summary,
            GuardianArticleClaim.has_clear_central_thesis,
            GuardianArticleClaim.thesis_sentence_id,
            GuardianArticleClaim.thesis_sentence,
            GuardianArticleClaim.support_sentence_ids,
            GuardianArticleClaim.support_sentences,
            GuardianArticleClaim.secondary_claim_ids,
            GuardianArticleClaim.secondary_claim_sentences,
        )
        .filter(GuardianArticleClaim.article_id.in_(requested_ids))
        .all()
    )
    records = {
        row.article_id: _claim_payload({
            "central_claim_summary": row.central_claim_summary,
            "has_clear_central_thesis": row.has_clear_central_thesis,
            "thesis_sentence_id": row.thesis_sentence_id,
            "thesis_sentence": row.thesis_sentence,
            "support_sentence_ids": row.support_sentence_ids,
            "support_sentences": row.support_sentences,
            "secondary_claim_ids": row.secondary_claim_ids,
            "secondary_claim_sentences": row.secondary_claim_sentences,
        })
        for row in rows
    }

    log_runtime_event(
        "claim_store.db_lookup_done",
        requested_count=len(requested_ids),
        fetched_count=len(records),
    )
    return records


def get_claim_record(article_id):
    key = _normalize_article_id(article_id)
    if not key:
        return None
    return get_claim_records([key]).get(key)
