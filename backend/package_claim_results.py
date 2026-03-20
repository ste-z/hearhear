import argparse
import gzip
import json
import re
from datetime import datetime, timezone
from pathlib import Path

from backend.claim_store import (
    PACKAGED_CLAIM_RESULTS_DIR,
    RAW_BATCH_CLAIM_RESULTS_DIR,
)


DEFAULT_MAX_SHARD_BYTES = 90_000_000


def _claim_result_files(root_dir):
    root = Path(root_dir)
    if not root.exists():
        return []
    return sorted(root.rglob("parsed_claim_coding_results.jsonl"))


def _read_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rows.append(json.loads(line))
    return rows


def _infer_year(article_id):
    match = re.search(r"/(20\d{2})/", str(article_id or ""))
    if not match:
        return "unknown"
    return match.group(1)


def _minimal_record(row):
    article_id = str(row.get("article_id") or "").strip()
    if not article_id:
        return None
    if row.get("error"):
        return None
    if not row.get("central_claim_summary"):
        return None

    return {
        "article_id": article_id,
        "title": row.get("title"),
        "year": _infer_year(article_id),
        "central_claim_summary": row.get("central_claim_summary"),
        "has_clear_central_thesis": row.get("has_clear_central_thesis"),
        "thesis_sentence_id": row.get("thesis_sentence_id"),
        "thesis_sentence": row.get("thesis_sentence"),
        "support_sentence_ids": row.get("support_sentence_ids") or [],
        "support_sentences": row.get("support_sentences") or [],
        "secondary_claim_ids": row.get("secondary_claim_ids") or [],
        "secondary_claim_sentences": row.get("secondary_claim_sentences") or [],
    }


def collect_claim_records(source_dir):
    by_article_id = {}
    for path in _claim_result_files(source_dir):
        source_mtime_ns = path.stat().st_mtime_ns
        for row in _read_jsonl(path):
            record = _minimal_record(row)
            if record is None:
                continue

            article_id = record["article_id"]
            existing = by_article_id.get(article_id)
            if existing and existing["_source_mtime_ns"] > source_mtime_ns:
                continue

            record["_source_mtime_ns"] = source_mtime_ns
            by_article_id[article_id] = record

    cleaned = []
    for article_id in sorted(by_article_id):
        record = dict(by_article_id[article_id])
        record.pop("_source_mtime_ns", None)
        cleaned.append(record)
    return sorted(
        cleaned,
        key=lambda record: (
            str(record.get("year") or "unknown"),
            str(record.get("article_id") or ""),
        ),
    )


def _cleanup_output_dir(output_dir):
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    for pattern in ("claims_*.jsonl", "claims_*.jsonl.gz"):
        for path in output_dir.glob(pattern):
            path.unlink()
    manifest_path = output_dir / "manifest.json"
    if manifest_path.exists():
        manifest_path.unlink()


def _write_shard(records, shard_path):
    with gzip.open(shard_path, "wt", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False))
            f.write("\n")
    return shard_path.stat().st_size


def write_shards(
    records,
    output_dir,
    source_dir,
    max_shard_bytes=DEFAULT_MAX_SHARD_BYTES,
):
    output_dir = Path(output_dir)
    _cleanup_output_dir(output_dir)

    manifest_files = []
    current_records = []
    current_bytes = 0
    current_year = None
    part_idx = 1

    def flush():
        nonlocal current_records, current_bytes, current_year, part_idx
        if not current_records:
            return

        shard_name = f"claims_{current_year}_part_{part_idx:03d}.jsonl.gz"
        shard_path = output_dir / shard_name
        file_size = _write_shard(current_records, shard_path)
        manifest_files.append(
            {
                "filename": shard_name,
                "year": current_year,
                "record_count": len(current_records),
                "size_bytes": file_size,
            }
        )
        part_idx += 1
        current_records = []
        current_bytes = 0

    for record in records:
        line = json.dumps(record, ensure_ascii=False) + "\n"
        encoded_length = len(line.encode("utf-8"))
        record_year = str(record.get("year") or "unknown")

        if current_year is None:
            current_year = record_year
            part_idx = 1

        should_roll_year = record_year != current_year
        should_roll_size = current_records and (current_bytes + encoded_length) > int(max_shard_bytes)
        if should_roll_year or should_roll_size:
            flush()
            if should_roll_year:
                current_year = record_year
                part_idx = 1

        current_records.append(record)
        current_bytes += encoded_length

    flush()

    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_dir": str(Path(source_dir)),
        "output_dir": str(output_dir),
        "record_count": len(records),
        "file_count": len(manifest_files),
        "max_shard_bytes": int(max_shard_bytes),
        "files": manifest_files,
    }
    manifest_path = output_dir / "manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest


def build_clean_claim_folder(
    source_dir=RAW_BATCH_CLAIM_RESULTS_DIR,
    output_dir=PACKAGED_CLAIM_RESULTS_DIR,
    max_shard_bytes=DEFAULT_MAX_SHARD_BYTES,
):
    records = collect_claim_records(source_dir)
    return write_shards(
        records=records,
        output_dir=output_dir,
        source_dir=source_dir,
        max_shard_bytes=max_shard_bytes,
    )


def parse_args():
    parser = argparse.ArgumentParser(
        description="Package parsed claim-coding results into clean sharded files for app/server use."
    )
    parser.add_argument(
        "--source-dir",
        default=str(RAW_BATCH_CLAIM_RESULTS_DIR),
        help="Directory containing raw batch run folders with parsed_claim_coding_results.jsonl files.",
    )
    parser.add_argument(
        "--output-dir",
        default=str(PACKAGED_CLAIM_RESULTS_DIR),
        help="Directory where clean sharded claim files should be written.",
    )
    parser.add_argument(
        "--max-shard-bytes",
        type=int,
        default=DEFAULT_MAX_SHARD_BYTES,
        help="Maximum uncompressed payload size to pack into each shard before starting a new one.",
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = build_clean_claim_folder(
        source_dir=args.source_dir,
        output_dir=args.output_dir,
        max_shard_bytes=args.max_shard_bytes,
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
