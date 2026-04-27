import numpy as np


DEFAULT_DENSE_DOT_BLOCK_SIZE = 8192


def top_positive_dot_candidates(
    matrix,
    query_vector,
    top_n,
    threshold=None,
    block_size=DEFAULT_DENSE_DOT_BLOCK_SIZE,
    candidate_filter=None,
):
    """Return top positive dot-product candidates without scoring all rows at once."""
    embeddings = np.asarray(matrix)
    query = np.asarray(query_vector, dtype=np.float32).reshape(-1)
    if embeddings.ndim != 2:
        raise ValueError("matrix must be a 2-dimensional array.")
    if query.size != int(embeddings.shape[1]):
        raise ValueError("query_vector dimension must match matrix column count.")

    row_count = int(embeddings.shape[0])
    if row_count <= 0 or query.size <= 0:
        return np.asarray([], dtype=np.intp), np.asarray([], dtype=np.float32)

    resolved_top_n = min(max(1, int(top_n)), row_count)
    resolved_block_size = min(max(1, int(block_size)), row_count)
    min_score = float(threshold) if threshold is not None else None

    selected_indices = np.asarray([], dtype=np.intp)
    selected_scores = np.asarray([], dtype=np.float32)

    for start in range(0, row_count, resolved_block_size):
        stop = min(start + resolved_block_size, row_count)
        block_scores = np.asarray(
            embeddings[start:stop] @ query,
            dtype=np.float32,
        ).reshape(-1)

        mask = block_scores > 0
        if min_score is not None:
            mask &= block_scores >= min_score
        if not np.any(mask):
            continue

        block_indices = np.flatnonzero(mask).astype(np.intp, copy=False) + start
        block_scores = block_scores[mask]

        if candidate_filter is not None:
            filter_mask = np.asarray(candidate_filter(block_indices), dtype=bool)
            if filter_mask.size != block_indices.size:
                raise ValueError("candidate_filter must return one boolean per index.")
            if not np.any(filter_mask):
                continue
            block_indices = block_indices[filter_mask]
            block_scores = block_scores[filter_mask]

        if block_scores.size > resolved_top_n:
            block_positions = np.argpartition(block_scores, -resolved_top_n)[
                -resolved_top_n:
            ]
            block_indices = block_indices[block_positions]
            block_scores = block_scores[block_positions]

        if selected_scores.size:
            merged_indices = np.concatenate([selected_indices, block_indices])
            merged_scores = np.concatenate([selected_scores, block_scores])
        else:
            merged_indices = block_indices
            merged_scores = block_scores

        if merged_scores.size > resolved_top_n:
            keep_positions = np.argpartition(merged_scores, -resolved_top_n)[
                -resolved_top_n:
            ]
            selected_indices = merged_indices[keep_positions]
            selected_scores = merged_scores[keep_positions]
        else:
            selected_indices = merged_indices
            selected_scores = merged_scores

    if selected_scores.size <= 0:
        return np.asarray([], dtype=np.intp), np.asarray([], dtype=np.float32)

    sorted_positions = np.argsort(selected_scores)[::-1]
    return (
        np.asarray(selected_indices[sorted_positions], dtype=np.intp),
        np.asarray(selected_scores[sorted_positions], dtype=np.float32),
    )
