import argparse
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path

try:
    from dotenv import load_dotenv
except ModuleNotFoundError:
    load_dotenv = None

from backend.text_processing.indexing.artifacts import (
    _artifact_exists,
    _materialized_artifact_path,
    _write_json_artifact,
)
from backend.text_processing.svd_processor import (
    DEFAULT_INDEX_DIR,
    DEFAULT_SVD_INDEX_NAME,
    TruncatedSvdIndex,
    load_svd_index,
)


DEFAULT_SVD_DIMENSION_LABEL_TOP_TERMS = 10
DEFAULT_SVD_DIMENSION_LABEL_BATCH_SIZE = 20
SVD_DIMENSION_LABEL_PROMPT_VERSION = "svd_dimension_labels_v1"


def svd_dimension_label_artifact_path(
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_SVD_INDEX_NAME,
):
    return Path(index_dir) / f"{index_name}_dimension_labels.json"


def _clean_label_text(value, max_chars=80):
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    text = re.sub(
        r"^(concept|dimension)\s+\d+\s*[:\-]\s*",
        "",
        text,
        flags=re.IGNORECASE,
    ).strip()
    if max_chars > 0 and len(text) > max_chars:
        text = f"{text[:max_chars - 3].rstrip()}..."
    return text


def _parse_svd_dimension_labels(raw_content, requested_indices):
    text = str(raw_content or "").strip()
    if not text:
        raise ValueError("Empty SVD label response.")

    fence_match = re.search(
        r"```(?:json)?\s*(.*?)```",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    candidate = fence_match.group(1).strip() if fence_match else text
    parsed = json.loads(candidate)

    raw_labels = parsed.get("labels") if isinstance(parsed, dict) else parsed
    if not isinstance(raw_labels, list):
        raise ValueError("SVD label response must contain a labels array.")

    requested = {int(index) for index in requested_indices}
    labels = []
    seen = set()
    for item in raw_labels:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("dimension_index"))
        except (TypeError, ValueError):
            continue
        if index not in requested or index in seen:
            continue
        label = _clean_label_text(item.get("label"))
        if not label:
            continue
        labels.append({"dimension_index": index, "label": label})
        seen.add(index)

    if not labels:
        raise ValueError("SVD label response did not include usable labels.")
    return labels


def _term_list(term_weights, top_n):
    terms = []
    for term, _weight in list(term_weights or [])[:top_n]:
        text = str(term or "").strip()
        if text:
            terms.append(text)
    return terms


def _dimension_label_input(processor, dimension, top_terms):
    summary = processor.dimension_summary_record(
        dimension=dimension,
        top_n=top_terms,
        format_terms=False,
    )
    return {
        "dimension_index": int(dimension),
        "positive_terms": _term_list(summary.get("positive_terms"), top_terms),
        "negative_terms": _term_list(summary.get("negative_terms"), top_terms),
        "absolute_terms": _term_list(summary.get("absolute_terms"), top_terms),
    }


def build_svd_dimension_label_inputs(processor, top_terms=DEFAULT_SVD_DIMENSION_LABEL_TOP_TERMS):
    return [
        _dimension_label_input(processor, dimension, int(top_terms))
        for dimension in range(int(processor.n_components))
    ]


def _label_prompt_messages(label_inputs):
    system_prompt = (
        "You label latent SVD dimensions for a news opinion search interface. "
        "Each dimension is represented by its top positive, negative, and absolute terms. "
        "Write a short human-readable topic label for each dimension, 2 to 6 words long. "
        "Use broad topic language such as 'Immigration and asylum policy' or "
        "'Climate and energy politics'. Do not include the words Concept or Dimension, "
        "do not include numbers, and do not explain. Return valid JSON only with this "
        "shape: {\"labels\":[{\"dimension_index\":0,\"label\":\"Example topic\"}]}"
    )
    user_prompt = json.dumps({"dimensions": label_inputs}, ensure_ascii=False)
    return [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": user_prompt},
    ]


def _batched(items, batch_size):
    size = max(1, int(batch_size))
    for start in range(0, len(items), size):
        yield items[start:start + size]


def _load_svd_meta(index_dir, index_name):
    meta_path = TruncatedSvdIndex.artifact_paths(index_dir, index_name)["meta"]
    if not _artifact_exists(meta_path):
        return {}
    with _materialized_artifact_path(meta_path) as materialized_path:
        with open(materialized_path, "r", encoding="utf-8") as f:
            return json.load(f) or {}


def load_svd_dimension_label_artifact(
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_SVD_INDEX_NAME,
    require_fresh=True,
):
    path = svd_dimension_label_artifact_path(index_dir=index_dir, index_name=index_name)
    if not _artifact_exists(path):
        return None

    try:
        with _materialized_artifact_path(path) as materialized_path:
            with open(materialized_path, "r", encoding="utf-8") as f:
                artifact = json.load(f) or {}
    except Exception:
        return None

    if not require_fresh:
        return artifact

    try:
        current_meta = _load_svd_meta(index_dir=index_dir, index_name=index_name)
    except Exception:
        return None

    if artifact.get("index_name") != index_name:
        return None
    if artifact.get("source_svd_saved_at_utc") != current_meta.get("saved_at_utc"):
        return None
    try:
        if int(artifact.get("n_components")) != int(current_meta.get("n_components")):
            return None
    except (TypeError, ValueError):
        return None
    if artifact.get("prompt_version") != SVD_DIMENSION_LABEL_PROMPT_VERSION:
        return None

    return artifact


def cached_svd_dimension_labels(
    dimension_indices,
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_SVD_INDEX_NAME,
    require_fresh=True,
):
    requested = []
    seen = set()
    for raw_index in dimension_indices:
        try:
            index = int(raw_index)
        except (TypeError, ValueError):
            continue
        if index in seen:
            continue
        requested.append(index)
        seen.add(index)

    labels_by_index = cached_svd_dimension_label_map(
        requested,
        index_dir=index_dir,
        index_name=index_name,
        require_fresh=require_fresh,
    )

    return [
        {"dimension_index": index, "label": labels_by_index[index]}
        for index in requested
        if index in labels_by_index
    ]


def cached_svd_dimension_label_map(
    dimension_indices=None,
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_SVD_INDEX_NAME,
    require_fresh=True,
):
    artifact = load_svd_dimension_label_artifact(
        index_dir=index_dir,
        index_name=index_name,
        require_fresh=require_fresh,
    )
    if not artifact:
        return {}

    requested = None
    if dimension_indices is not None:
        requested = set()
        for raw_index in dimension_indices:
            try:
                requested.add(int(raw_index))
            except (TypeError, ValueError):
                continue

    labels_by_index = {}
    for item in artifact.get("labels") or []:
        if not isinstance(item, dict):
            continue
        try:
            index = int(item.get("dimension_index"))
        except (TypeError, ValueError):
            continue
        if requested is not None and index not in requested:
            continue
        label = _clean_label_text(item.get("label"))
        if label:
            labels_by_index[index] = label

    return labels_by_index


def build_svd_dimension_label_artifact(
    api_key=None,
    index_dir=DEFAULT_INDEX_DIR,
    index_name=DEFAULT_SVD_INDEX_NAME,
    top_terms=DEFAULT_SVD_DIMENSION_LABEL_TOP_TERMS,
    batch_size=DEFAULT_SVD_DIMENSION_LABEL_BATCH_SIZE,
):
    if load_dotenv is not None:
        load_dotenv()

    resolved_api_key = api_key or os.getenv("SPARK_API_KEY") or os.getenv("API_KEY")
    if not resolved_api_key:
        raise RuntimeError("SPARK_API_KEY or API_KEY is required to precompute SVD labels.")

    from backend.stance_processing.llm_processor import create_spark_client

    processor, meta = load_svd_index(
        index_dir=index_dir,
        index_name=index_name,
        load_articles=False,
    )
    label_inputs = build_svd_dimension_label_inputs(
        processor=processor,
        top_terms=top_terms,
    )
    client = create_spark_client(api_key=resolved_api_key)

    labels_by_index = {}
    for batch in _batched(label_inputs, batch_size):
        response = client.chat(_label_prompt_messages(batch))
        parsed_labels = _parse_svd_dimension_labels(
            response.get("content"),
            requested_indices=[item["dimension_index"] for item in batch],
        )
        for item in parsed_labels:
            labels_by_index[int(item["dimension_index"])] = item["label"]

    missing_indices = [
        int(item["dimension_index"])
        for item in label_inputs
        if int(item["dimension_index"]) not in labels_by_index
    ]
    if missing_indices:
        raise RuntimeError(
            "LLM label generation did not return labels for dimensions: "
            + ", ".join(str(index) for index in missing_indices)
        )

    labels = []
    for item in label_inputs:
        index = int(item["dimension_index"])
        labels.append(
            {
                "dimension_index": index,
                "dimension_label": index + 1,
                "label": labels_by_index.get(index, ""),
                "positive_terms": item["positive_terms"],
                "negative_terms": item["negative_terms"],
                "absolute_terms": item["absolute_terms"],
            }
        )

    artifact = {
        "index_name": index_name,
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_svd_saved_at_utc": meta.get("saved_at_utc"),
        "n_components": int(processor.n_components),
        "top_terms": int(top_terms),
        "label_model": "spark",
        "prompt_version": SVD_DIMENSION_LABEL_PROMPT_VERSION,
        "labels": labels,
    }
    path = svd_dimension_label_artifact_path(
        index_dir=index_dir,
        index_name=index_name,
    )
    _write_json_artifact(path, artifact)
    return path, artifact


def _parse_args():
    parser = argparse.ArgumentParser(
        description="Precompute human-readable labels for SVD dimensions."
    )
    parser.add_argument("--index-dir", default=str(DEFAULT_INDEX_DIR))
    parser.add_argument("--index-name", default=DEFAULT_SVD_INDEX_NAME)
    parser.add_argument("--top-terms", type=int, default=DEFAULT_SVD_DIMENSION_LABEL_TOP_TERMS)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_SVD_DIMENSION_LABEL_BATCH_SIZE)
    return parser.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    output_path, output_artifact = build_svd_dimension_label_artifact(
        index_dir=args.index_dir,
        index_name=args.index_name,
        top_terms=args.top_terms,
        batch_size=args.batch_size,
    )
    print(
        json.dumps(
            {
                "path": str(output_path),
                "label_count": len(output_artifact.get("labels") or []),
                "n_components": output_artifact.get("n_components"),
                "prompt_version": output_artifact.get("prompt_version"),
            },
            indent=2,
        )
    )
