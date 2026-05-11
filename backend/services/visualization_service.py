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

import pickle
from pathlib import Path
from threading import Lock
from typing import Literal, Tuple

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

_umap_models: dict[str, object] = {}
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


def project_query_embedding(
    query_embedding,
    source: Source,
) -> Tuple[float, float]:
    """Project a single high-D query embedding into the 2D UMAP space.

    Parameters
    ----------
    query_embedding:
        Numpy 1-D array. Must match the source's input dimensionality:
        384 for ``minilm``, 100 for ``svd``.
    source:
        Which precomputed UMAP projection to use.

    Returns
    -------
    (x, y):
        2D coordinates in the same scale as the corresponding ``.npy``
        master coords (i.e. before int16 quantization).
    """
    import numpy as np

    model = get_umap_model(source)
    arr = np.asarray(query_embedding, dtype=np.float32).reshape(1, -1)
    coords = model.transform(arr)  # type: ignore[attr-defined]
    return float(coords[0, 0]), float(coords[0, 1])


def is_umap_artifact_present(source: Source) -> bool:
    """Cheap existence check that does NOT load the pickle.

    Returns True if either the single ``.pkl`` exists or one or more
    ``.pkl.part-NNN`` chunks exist.
    """
    return _artifact_exists(UMAP_PICKLE_PATHS[source])
