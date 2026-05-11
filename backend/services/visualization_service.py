"""Phase 2 visualization service: per-source lazy loader for the UMAP models
that back the "explore" atlas, so future code can project ad-hoc query
embeddings into the same 2D space via ``umap_model.transform``.

The MVP atlas does not call into this module — the frontend computes the
query position as a weighted centroid of the search-result coordinates it
already has. This service exists for the optional next step: a backend
``/api/visualization/project_query`` route that embeds an arbitrary user
query and returns its 2D position so the standalone atlas search bar can
do true semantic projection.

The cache mirrors the ``_vector_processors`` pattern in
``backend/text_processing/search_helpers.py`` — per-source dict, single
shared lock, double-checked locking. If only the MiniLM atlas is hit, the
SVD pickle is never opened.
"""

from __future__ import annotations

import json
import pickle
from pathlib import Path
from threading import Lock
from typing import Literal, Tuple

import numpy as np

from backend.text_processing.indexing.artifacts import (
    _artifact_exists,
    _materialized_artifact_path,
)


_VALID_SOURCES = ("minilm", "svd")
Source = Literal["minilm", "svd"]


_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_INDEX_DIR = _PROJECT_ROOT / "data" / "processed" / "vector_index"

# Logical pickle paths. The actual on-disk file is either the single ``.pkl``
# or a sequence of ``.pkl.part-NNN`` chunks (split at 95 MB by
# ``_write_pickle_artifact`` so each chunk fits under GitHub's 100 MB limit).
UMAP_PICKLE_PATHS: dict[str, Path] = {
    "minilm": _INDEX_DIR / "guardian_article_umap2d_minilm_model.pkl",
    "svd": _INDEX_DIR / "guardian_article_umap2d_svd_model.pkl",
}

UMAP_META_PATH = _INDEX_DIR / "guardian_article_umap2d_meta.json"

_umap_models: dict[str, object] = {}
_umap_quantization: dict[str, dict] = {}
_umap_load_lock = Lock()


def get_umap_model(source: Source) -> object:
    """Return the fitted UMAP model for ``source``, loading lazily on first use.

    The pickle is either a single file or a sequence of ``.pkl.part-NNN``
    chunks; the chunked-artifact helper reassembles parts into a temp file
    before ``pickle.load`` (mirrors how the existing search indices load
    chunked ``.npy`` files). First call for a source pays the ~150 MB load
    cost (~2-4 s); subsequent calls return the cached instance. The other
    source's pickle is never opened if its endpoint is never hit.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {_VALID_SOURCES}")

    cached = _umap_models.get(source)
    if cached is not None:
        return cached

    with _umap_load_lock:
        cached = _umap_models.get(source)
        if cached is not None:
            return cached
        path = UMAP_PICKLE_PATHS[source]
        if not _artifact_exists(path):
            raise FileNotFoundError(
                f"UMAP model for source={source!r} missing at {path}. "
                f"Run: python -m backend.text_processing.embedding_projection --source {source}"
            )
        with _materialized_artifact_path(path) as resolved_pickle_path:
            with open(resolved_pickle_path, "rb") as fh:
                model = pickle.load(fh)
        _umap_models[source] = model
        return model


def _get_quantization(source: Source) -> dict:
    """Return ``{center: [cx, cy], scale: s}`` so the backend can map raw
    UMAP coords into the same int16 space the frontend renders.

    Cached after first read. The values are written into the umap meta
    JSON at preprocessing time (see ``embedding_projection.py``).
    """
    cached = _umap_quantization.get(source)
    if cached is not None:
        return cached
    if not UMAP_META_PATH.exists():
        raise FileNotFoundError(
            f"UMAP meta not found at {UMAP_META_PATH}. "
            "Run `python -m backend.text_processing.embedding_projection --all`."
        )
    with open(UMAP_META_PATH) as fh:
        meta = json.load(fh) or {}
    sources_meta = meta.get("sources") or {}
    source_meta = sources_meta.get(source) or {}
    quant = source_meta.get("quantization")
    if not quant:
        raise ValueError(
            f"Quantization params missing for source={source!r}. Re-run the "
            "preprocessing script so the updated meta is written."
        )
    _umap_quantization[source] = quant
    return quant


def _get_query_projector_processor(source: Source):
    """Build (or reuse) the retrieval processor whose embeddings UMAP was
    fit on. Both ``svd`` and ``minilm`` processors expose
    ``project_query(text, normalize=True) -> np.ndarray``.
    """
    from backend.text_processing.search_helpers import build_retrieval_processor

    return build_retrieval_processor(retrieval_model=source)


def project_query(query: str, source: Source) -> Tuple[float, float]:
    """Embed the query text with the corresponding retrieval processor,
    run it through the fitted UMAP model's ``.transform`` (this is the
    exact projection — no nearest-neighbor approximation), and map the
    raw 2D output into the int16 space the frontend uses to render.

    Returns ``(x, y)`` floats in roughly ``[-30000, 30000]``.
    """
    if source not in _VALID_SOURCES:
        raise ValueError(f"Unknown source {source!r}; expected one of {_VALID_SOURCES}")
    text = (query or "").strip()
    if not text:
        raise ValueError("Query must be non-empty.")

    processor = _get_query_projector_processor(source)
    embedding = processor.project_query(text, normalize=True)
    if embedding is None:
        raise ValueError(
            "Query had no projectable terms (empty after normalization)."
        )
    arr = np.asarray(embedding, dtype=np.float32).reshape(1, -1)

    model = get_umap_model(source)
    raw = model.transform(arr)  # type: ignore[attr-defined]
    raw_xy = np.asarray(raw, dtype=np.float32).reshape(-1)[:2]

    quant = _get_quantization(source)
    center = np.asarray(quant["center"], dtype=np.float32)
    scale = float(quant["scale"])
    xy = (raw_xy - center) * scale
    return float(xy[0]), float(xy[1])


def is_umap_artifact_present(source: Source) -> bool:
    """Cheap existence check that does NOT load the pickle.

    Returns True if either the single ``.pkl`` exists or one or more
    ``.pkl.part-NNN`` chunks exist.
    """
    return _artifact_exists(UMAP_PICKLE_PATHS[source])
