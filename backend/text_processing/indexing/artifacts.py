import json
import pickle
import tempfile
from contextlib import contextmanager
from pathlib import Path

import numpy as np

from backend.text_processing.indexing.settings import MAX_VECTOR_INDEX_ARTIFACT_BYTES


def _artifact_chunk_paths(path):
    final_path = Path(path)
    chunk_prefix = f"{final_path.name}.part-"
    return sorted(
        chunk_path
        for chunk_path in final_path.parent.glob(f"{chunk_prefix}*")
        if chunk_path.is_file() and not chunk_path.name.endswith(".tmp")
    )


def _artifact_files(path):
    final_path = Path(path)
    chunk_paths = _artifact_chunk_paths(final_path)
    if chunk_paths:
        return chunk_paths
    if final_path.exists():
        return [final_path]
    return []


def _artifact_exists(path):
    return bool(_artifact_files(path))


def _artifact_within_size_limit(path, max_bytes=MAX_VECTOR_INDEX_ARTIFACT_BYTES):
    files = _artifact_files(path)
    if not files:
        return False
    return all(file_path.stat().st_size <= int(max_bytes) for file_path in files)


def _cleanup_temp_paths(paths):
    for path in list(paths or []):
        resolved = Path(path)
        if resolved.exists():
            resolved.unlink()


def _new_temp_artifact_path(final_path):
    resolved_final_path = Path(final_path)
    suffix = "".join(resolved_final_path.suffixes) or ".tmp"
    with tempfile.NamedTemporaryFile(
        suffix=suffix,
        dir=resolved_final_path.parent,
        delete=False,
    ) as temp_file:
        return Path(temp_file.name)


def _write_artifact_chunks(
    source_path,
    final_path,
    max_chunk_bytes=MAX_VECTOR_INDEX_ARTIFACT_BYTES,
):
    resolved_source_path = Path(source_path)
    resolved_final_path = Path(final_path)
    directory = resolved_final_path.parent
    temp_chunk_paths = []
    final_chunk_paths = []
    chunk_prefix = f"{resolved_final_path.name}.part-"

    with open(resolved_source_path, "rb") as source:
        chunk_idx = 0
        while True:
            chunk = source.read(int(max_chunk_bytes))
            if not chunk:
                break

            final_chunk_path = directory / f"{chunk_prefix}{chunk_idx:03d}"
            temp_chunk_path = directory / f"{chunk_prefix}{chunk_idx:03d}.tmp"
            with open(temp_chunk_path, "wb") as f:
                f.write(chunk)

            temp_chunk_paths.append(temp_chunk_path)
            final_chunk_paths.append(final_chunk_path)
            chunk_idx += 1

    if not final_chunk_paths:
        raise ValueError(f"Serialized artifact was empty: {resolved_final_path.name}")

    for temp_chunk_path, final_chunk_path in zip(temp_chunk_paths, final_chunk_paths):
        temp_chunk_path.replace(final_chunk_path)

    return final_chunk_paths


def _write_artifact_from_temp(
    source_path,
    final_path,
    max_chunk_bytes=MAX_VECTOR_INDEX_ARTIFACT_BYTES,
):
    resolved_source_path = Path(source_path)
    resolved_final_path = Path(final_path)
    existing_chunk_paths = _artifact_chunk_paths(resolved_final_path)

    try:
        if resolved_source_path.stat().st_size > int(max_chunk_bytes):
            artifact_files = _write_artifact_chunks(
                source_path=resolved_source_path,
                final_path=resolved_final_path,
                max_chunk_bytes=max_chunk_bytes,
            )
            stale_chunk_paths = [
                path for path in existing_chunk_paths if path not in artifact_files
            ]
            _cleanup_temp_paths(stale_chunk_paths)
            if resolved_final_path.exists():
                resolved_final_path.unlink()
            storage = "chunked"
        else:
            _cleanup_temp_paths(existing_chunk_paths)
            resolved_source_path.replace(resolved_final_path)
            resolved_source_path = None
            artifact_files = [resolved_final_path]
            storage = "single_file"
    finally:
        if resolved_source_path is not None and resolved_source_path.exists():
            resolved_source_path.unlink()

    return {
        "path": resolved_final_path,
        "files": artifact_files,
        "storage": storage,
    }


def _write_pickle_artifact(path, value):
    temp_path = _new_temp_artifact_path(path)
    with open(temp_path, "wb") as f:
        pickle.dump(value, f, protocol=pickle.HIGHEST_PROTOCOL)
    return _write_artifact_from_temp(temp_path, path)


def _write_dataframe_pickle_artifact(path, frame):
    temp_path = _new_temp_artifact_path(path)
    frame.to_pickle(temp_path)
    return _write_artifact_from_temp(temp_path, path)


def _write_json_artifact(path, value):
    temp_path = _new_temp_artifact_path(path)
    with open(temp_path, "w", encoding="utf-8") as f:
        json.dump(value, f)
    return _write_artifact_from_temp(temp_path, path)


def _write_npy_artifact(path, array):
    temp_path = _new_temp_artifact_path(path)
    with open(temp_path, "wb") as f:
        np.save(f, array, allow_pickle=False)
    return _write_artifact_from_temp(temp_path, path)


def _repartition_existing_artifact(
    path,
    max_chunk_bytes=MAX_VECTOR_INDEX_ARTIFACT_BYTES,
):
    resolved_path = Path(path)
    if not resolved_path.exists():
        raise FileNotFoundError(f"Artifact not found for repartition: {resolved_path}")
    if resolved_path.stat().st_size <= int(max_chunk_bytes):
        return {
            "path": resolved_path,
            "files": [resolved_path],
            "storage": "single_file",
        }

    temp_path = resolved_path.with_name(f"{resolved_path.name}.repartition.tmp")
    if temp_path.exists():
        temp_path.unlink()
    resolved_path.replace(temp_path)
    return _write_artifact_from_temp(
        temp_path,
        resolved_path,
        max_chunk_bytes=max_chunk_bytes,
    )


def _materialize_chunked_artifact(path, chunk_paths):
    resolved_path = Path(path)
    temp_path = _new_temp_artifact_path(resolved_path)
    with open(temp_path, "wb") as temp_file:
        for chunk_path in chunk_paths:
            with open(chunk_path, "rb") as source:
                while True:
                    block = source.read(1024 * 1024)
                    if not block:
                        break
                    temp_file.write(block)
    return temp_path


@contextmanager
def _materialized_artifact_path(path):
    resolved_path = Path(path)
    chunk_paths = _artifact_chunk_paths(resolved_path)
    temp_path = None
    try:
        if chunk_paths:
            temp_path = _materialize_chunked_artifact(resolved_path, chunk_paths)
            yield temp_path
        else:
            if not resolved_path.exists():
                raise FileNotFoundError(f"Artifact not found: {resolved_path}")
            yield resolved_path
    finally:
        if temp_path is not None and temp_path.exists():
            temp_path.unlink()


def _materialized_artifact_path_for_mmap(path):
    resolved_path = Path(path)
    chunk_paths = _artifact_chunk_paths(resolved_path)
    if chunk_paths:
        temp_path = _materialize_chunked_artifact(resolved_path, chunk_paths)
        return temp_path, temp_path

    if not resolved_path.exists():
        raise FileNotFoundError(f"Artifact not found: {resolved_path}")
    return resolved_path, None
