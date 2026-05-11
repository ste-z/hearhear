"""Preprocessing CLI that projects article embeddings to 2D via UMAP for the
"explore" atlas visualization.

Produces (per source = minilm | svd):
- data/processed/vector_index/guardian_article_umap2d_{source}_coords.npy
    Master float32 (N, 2) coordinates aligned to the canonical MiniLM doc_id order.
- data/processed/vector_index/guardian_article_umap2d_{source}_model.pkl
    The fitted UMAP estimator (so the backend can later call .transform(query)).
- frontend/public/embedding_atlas_{source}.bin
    int16 little-endian quantized flat buffer for the frontend (~265 KB).

Plus shared:
- frontend/public/embedding_atlas_meta.json
    Columnar metadata aligned to the canonical doc_id order (titles, years,
    sections, urls).
- data/processed/vector_index/guardian_article_umap2d_meta.json
    UMAP hyperparameters + provenance.

Run:
    python -m backend.text_processing.embedding_projection --all
    python -m backend.text_processing.embedding_projection --source minilm
    python -m backend.text_processing.embedding_projection --metadata
    python -m backend.text_processing.embedding_projection --source minilm --limit 2000  (smoke test)
"""

import argparse
import json
import pickle
import sqlite3
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

from backend.text_processing.indexing.corpus import (
    DEFAULT_DB_PATH,
    DEFAULT_INDEX_DIR,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FRONTEND_PUBLIC = PROJECT_ROOT / "frontend" / "public"

SOURCES = ("minilm", "svd")

SOURCE_PATHS = {
    "minilm": {
        "embeddings": DEFAULT_INDEX_DIR / "guardian_article_minilm_semantic_embeddings.npy",
        "doc_ids": DEFAULT_INDEX_DIR / "guardian_article_minilm_semantic_doc_ids.json",
    },
    "svd": {
        "embeddings": DEFAULT_INDEX_DIR / "guardian_tfidf_svd_doc_embeddings.npy",
        "doc_ids": DEFAULT_INDEX_DIR / "guardian_tfidf_svd_doc_ids.json",
    },
}

CANONICAL_SOURCE = "minilm"  # the canonical doc_id ordering used for shared metadata
INT16_RANGE = 30000  # leaves headroom inside int16's [-32768, 32767]


def _output_coords_npy(source: str) -> Path:
    return DEFAULT_INDEX_DIR / f"guardian_article_umap2d_{source}_coords.npy"


def _output_model_pkl(source: str) -> Path:
    return DEFAULT_INDEX_DIR / f"guardian_article_umap2d_{source}_model.pkl"


def _output_umap_meta() -> Path:
    return DEFAULT_INDEX_DIR / "guardian_article_umap2d_meta.json"


def _output_public_bin(source: str) -> Path:
    return FRONTEND_PUBLIC / f"embedding_atlas_{source}.bin"


def _output_public_meta() -> Path:
    return FRONTEND_PUBLIC / "embedding_atlas_meta.json"


def _load_doc_ids(path: Path) -> list[str]:
    with open(path) as fh:
        ids = json.load(fh)
    return [str(d) for d in ids]


def _quantize_to_int16(coords: np.ndarray) -> np.ndarray:
    """Center to origin then scale so |max| == INT16_RANGE."""
    centered = coords - coords.mean(axis=0, keepdims=True)
    max_abs = float(np.abs(centered).max())
    if max_abs <= 0.0:
        scaled = centered
    else:
        scaled = centered * (INT16_RANGE / max_abs)
    return np.clip(scaled, -32767, 32767).round().astype(np.int16)


def _save_int16_bin(int16_coords: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    # Force little-endian regardless of host order, so the browser DataView reads it
    # consistently on every machine.
    np.ascontiguousarray(int16_coords, dtype="<i2").tofile(path)


def project_source(
    *,
    source: str,
    canonical_doc_ids: list[str],
    n_neighbors: int = 30,
    min_dist: float = 0.1,
    metric: str = "cosine",
    random_state: int = 42,
    sample_limit: int | None = None,
) -> dict:
    """Run UMAP on one source's embeddings and persist all artifacts."""
    if source not in SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {SOURCES}")

    import umap  # lazy import — only paid when actually running

    paths = SOURCE_PATHS[source]
    print(f"[{source}] Loading {paths['embeddings'].name}", flush=True)
    raw_embeddings = np.load(paths["embeddings"], mmap_mode="r")
    embeddings = np.asarray(raw_embeddings, dtype=np.float32)  # materializes if float16
    print(
        f"[{source}] embeddings shape={embeddings.shape} dtype={embeddings.dtype}",
        flush=True,
    )

    source_doc_ids = _load_doc_ids(paths["doc_ids"])
    if len(source_doc_ids) != embeddings.shape[0]:
        raise RuntimeError(
            f"[{source}] doc_ids length {len(source_doc_ids)} != embeddings rows {embeddings.shape[0]}"
        )

    # Reorder embeddings to match the canonical doc_id ordering.
    source_id_to_row = {doc_id: i for i, doc_id in enumerate(source_doc_ids)}
    missing = [d for d in canonical_doc_ids if d not in source_id_to_row]
    if missing:
        raise RuntimeError(
            f"[{source}] {len(missing)} canonical doc_ids missing from this source; "
            "rebuild upstream indices before continuing."
        )
    canonical_rows = np.fromiter(
        (source_id_to_row[d] for d in canonical_doc_ids),
        dtype=np.int64,
        count=len(canonical_doc_ids),
    )
    aligned = embeddings[canonical_rows]
    print(f"[{source}] aligned shape={aligned.shape}", flush=True)

    if sample_limit is not None:
        aligned = aligned[:sample_limit]
        print(f"[{source}] sampled to {aligned.shape[0]} rows (smoke test)", flush=True)

    print(
        f"[{source}] Fitting UMAP "
        f"(n_neighbors={n_neighbors}, min_dist={min_dist}, metric={metric}) ...",
        flush=True,
    )
    started = time.time()
    reducer = umap.UMAP(
        n_components=2,
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        metric=metric,
        random_state=random_state,
        low_memory=True,
        verbose=True,
    )
    coords_2d = reducer.fit_transform(aligned).astype(np.float32, copy=False)
    elapsed = time.time() - started
    print(f"[{source}] UMAP done in {elapsed:.1f}s — coords shape={coords_2d.shape}", flush=True)

    # 1. Save master float32 coords
    coords_npy = _output_coords_npy(source)
    coords_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(coords_npy, coords_2d)
    print(f"[{source}] wrote {coords_npy} ({coords_npy.stat().st_size} bytes)", flush=True)

    # 2. Save UMAP model pickle (Phase 2 backend transform)
    model_pkl = _output_model_pkl(source)
    with open(model_pkl, "wb") as fh:
        pickle.dump(reducer, fh, protocol=pickle.HIGHEST_PROTOCOL)
    print(
        f"[{source}] wrote {model_pkl} ({model_pkl.stat().st_size / 1024 / 1024:.1f} MB)",
        flush=True,
    )

    # 3. Quantize to int16 and write public/.bin
    int16_coords = _quantize_to_int16(coords_2d)
    public_bin = _output_public_bin(source)
    _save_int16_bin(int16_coords, public_bin)
    print(f"[{source}] wrote {public_bin} ({public_bin.stat().st_size} bytes)", flush=True)

    return {
        "source": source,
        "n_docs": int(coords_2d.shape[0]),
        "n_neighbors": n_neighbors,
        "min_dist": min_dist,
        "metric": metric,
        "random_state": random_state,
        "elapsed_seconds": round(elapsed, 1),
        "coords_npy": str(coords_npy),
        "model_pkl": str(model_pkl),
        "public_bin": str(public_bin),
    }


def build_shared_metadata(
    *,
    canonical_doc_ids: list[str],
    db_path: Path = DEFAULT_DB_PATH,
) -> dict:
    """Read titles/sections/urls/years from SQLite, write columnar JSON aligned to canonical order."""
    print(f"[meta] Reading metadata from {db_path}", flush=True)
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    try:
        df = pd.read_sql_query(
            "SELECT id, title, section_name, url, year FROM guardian_articles",
            conn,
        )
    finally:
        conn.close()
    print(f"[meta] loaded {len(df)} rows from sqlite", flush=True)

    df = df.set_index("id")
    df = df.reindex(canonical_doc_ids)

    # The Guardian `section_name` column is empty in this corpus; derive the
    # section from the article ID prefix (e.g. "commentisfree/2015/..." →
    # "commentisfree"). Falls back to the column when it does happen to be
    # populated, and "uncategorized" otherwise.
    def _resolve_section(doc_id: str, raw_name) -> str:
        if isinstance(raw_name, str) and raw_name.strip():
            return raw_name.strip()
        if isinstance(doc_id, str) and "/" in doc_id:
            return doc_id.split("/", 1)[0]
        return "uncategorized"

    section_values = [
        _resolve_section(doc_id, raw)
        for doc_id, raw in zip(canonical_doc_ids, df["section_name"].tolist())
    ]
    unique_sections = sorted(set(section_values))
    section_to_idx = {name: idx for idx, name in enumerate(unique_sections)}
    section_indices = [section_to_idx[name] for name in section_values]

    titles = df["title"].fillna("").astype(str).tolist()
    urls = df["url"].fillna("").astype(str).tolist()
    years_raw = df["year"].fillna(0)
    years = years_raw.astype(int).tolist()

    payload = {
        "n_docs": len(canonical_doc_ids),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "canonical_source": CANONICAL_SOURCE,
        "sections": unique_sections,
        "ids": canonical_doc_ids,
        "titles": titles,
        "years": years,
        "section_indices": section_indices,
        "urls": urls,
    }

    out_path = _output_public_meta()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(payload, fh, ensure_ascii=False, separators=(",", ":"))
    print(
        f"[meta] wrote {out_path} "
        f"({out_path.stat().st_size / 1024 / 1024:.2f} MB, {len(unique_sections)} sections)",
        flush=True,
    )
    return {
        "n_docs": payload["n_docs"],
        "n_sections": len(unique_sections),
        "path": str(out_path),
        "size_bytes": out_path.stat().st_size,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", choices=SOURCES, help="Project a single source.")
    parser.add_argument("--all", action="store_true", help="Project both sources + metadata.")
    parser.add_argument("--metadata", action="store_true", help="Build only the shared metadata JSON.")
    parser.add_argument("--n-neighbors", type=int, default=30)
    parser.add_argument("--min-dist", type=float, default=0.1)
    parser.add_argument("--metric", type=str, default="cosine")
    parser.add_argument("--random-state", type=int, default=42)
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Subsample N rows for fast smoke testing.",
    )
    args = parser.parse_args()

    if not (args.source or args.all or args.metadata):
        parser.error("Pass at least one of --source, --all, --metadata.")

    canonical_doc_ids = _load_doc_ids(SOURCE_PATHS[CANONICAL_SOURCE]["doc_ids"])
    if args.limit is not None:
        canonical_doc_ids = canonical_doc_ids[: args.limit]
        print(f"[main] Using subsample of {len(canonical_doc_ids)} canonical doc_ids", flush=True)

    sources_to_run: list[str] = []
    if args.all:
        sources_to_run = list(SOURCES)
    elif args.source:
        sources_to_run = [args.source]

    per_source_results: dict[str, dict] = {}
    for src in sources_to_run:
        per_source_results[src] = project_source(
            source=src,
            canonical_doc_ids=canonical_doc_ids,
            n_neighbors=args.n_neighbors,
            min_dist=args.min_dist,
            metric=args.metric,
            random_state=args.random_state,
            sample_limit=None,  # already subsampled in canonical_doc_ids
        )

    if args.metadata or args.all:
        meta_summary = build_shared_metadata(canonical_doc_ids=canonical_doc_ids)
    else:
        meta_summary = None

    # Write/update the umap meta JSON
    umap_meta_path = _output_umap_meta()
    existing_meta = {}
    if umap_meta_path.exists():
        try:
            with open(umap_meta_path) as fh:
                existing_meta = json.load(fh)
        except Exception:
            existing_meta = {}
    existing_sources = dict(existing_meta.get("sources") or {})
    existing_sources.update(per_source_results)
    meta_payload = {
        "saved_at_utc": datetime.now(timezone.utc).isoformat(),
        "reducer": "umap",
        "n_components": 2,
        "n_neighbors": args.n_neighbors,
        "min_dist": args.min_dist,
        "metric": args.metric,
        "random_state": args.random_state,
        "canonical_source": CANONICAL_SOURCE,
        "n_docs": len(canonical_doc_ids),
        "sources": existing_sources,
        "shared_metadata": meta_summary,
    }
    with open(umap_meta_path, "w", encoding="utf-8") as fh:
        json.dump(meta_payload, fh, indent=2, ensure_ascii=False)
    print(f"[main] wrote {umap_meta_path}", flush=True)

    return 0


if __name__ == "__main__":
    sys.exit(main())
