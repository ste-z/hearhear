import json
import gzip
from pathlib import Path
from threading import Lock


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_BATCH_CLAIM_RESULTS_DIR = (
    PROJECT_ROOT / "backend" / "data" / "processed" / "openai_claim_batches"
)
PACKAGED_CLAIM_RESULTS_DIR = (
    PROJECT_ROOT / "backend" / "data" / "processed" / "claim_coding_clean"
)


_claim_store_cache = None
_claim_store_fingerprint = None
_claim_store_lock = Lock()


def default_claim_results_dir():
    packaged_files = _claim_result_files(PACKAGED_CLAIM_RESULTS_DIR)
    if packaged_files:
        return PACKAGED_CLAIM_RESULTS_DIR
    return RAW_BATCH_CLAIM_RESULTS_DIR


def _claim_result_files(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return []
    paths = list(root.rglob("parsed_claim_coding_results.jsonl"))
    paths.extend(root.rglob("claims_*.jsonl"))
    paths.extend(root.rglob("claims_*.jsonl.gz"))
    return sorted(set(paths))


def _fingerprint(paths):
    return tuple(
        (str(path), path.stat().st_mtime_ns, path.stat().st_size)
        for path in paths
    )


def _read_jsonl(path):
    rows = []
    path = Path(path)
    open_fn = gzip.open if path.suffix == ".gz" else path.open
    with open_fn(path, "rt", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def load_claim_store(root_dir=None, force_reload=False):
    global _claim_store_cache, _claim_store_fingerprint

    resolved_root_dir = default_claim_results_dir() if root_dir is None else Path(root_dir)

    claim_files = _claim_result_files(resolved_root_dir)
    fingerprint = _fingerprint(claim_files)
    cache_ok = (
        not force_reload
        and _claim_store_cache is not None
        and _claim_store_fingerprint == fingerprint
    )
    if cache_ok:
        return _claim_store_cache

    with _claim_store_lock:
        claim_files = _claim_result_files(resolved_root_dir)
        fingerprint = _fingerprint(claim_files)
        cache_ok = (
            not force_reload
            and _claim_store_cache is not None
            and _claim_store_fingerprint == fingerprint
        )
        if cache_ok:
            return _claim_store_cache

        by_article_id = {}
        source_count = 0

        for path in claim_files:
            source_count += 1
            source_mtime_ns = path.stat().st_mtime_ns
            rows = _read_jsonl(path)
            for row in rows:
                article_id = str(row.get("article_id") or "").strip()
                if not article_id:
                    continue
                if row.get("error"):
                    continue
                if not row.get("central_claim_summary"):
                    continue

                existing = by_article_id.get(article_id)
                if existing and existing["_source_mtime_ns"] > source_mtime_ns:
                    continue

                record = dict(row)
                record["_source_path"] = str(path)
                record["_source_mtime_ns"] = source_mtime_ns
                by_article_id[article_id] = record

        claim_store = {
            "by_article_id": by_article_id,
            "article_count": len(by_article_id),
            "source_count": source_count,
            "files": [str(path) for path in claim_files],
            "root_dir": str(resolved_root_dir),
        }
        _claim_store_cache = claim_store
        _claim_store_fingerprint = fingerprint
        return claim_store


def get_claim_record(article_id, root_dir=None, force_reload=False):
    store = load_claim_store(root_dir=root_dir, force_reload=force_reload)
    key = str(article_id or "").strip()
    if not key:
        return None
    return store["by_article_id"].get(key)
