import queue as queue_module
import re
import threading
import time
from dataclasses import dataclass, field


_PROGRESS_ID_RE = re.compile(r"^[A-Za-z0-9_-]{8,96}$")
_CHANNEL_TTL_SECONDS = 300
_MAX_QUEUE_SIZE = 96
_channels = {}
_channels_lock = threading.Lock()


@dataclass
class _ProgressChannel:
    queue: queue_module.Queue = field(default_factory=lambda: queue_module.Queue(maxsize=_MAX_QUEUE_SIZE))
    created_at: float = field(default_factory=time.time)
    updated_at: float = field(default_factory=time.time)
    done: bool = False


def normalize_progress_id(value):
    text = str(value or "").strip()
    if not text or not _PROGRESS_ID_RE.match(text):
        return None
    return text


def _cleanup_old_channels(now=None):
    resolved_now = time.time() if now is None else float(now)
    stale_ids = [
        progress_id
        for progress_id, channel in _channels.items()
        if resolved_now - channel.updated_at > _CHANNEL_TTL_SECONDS
    ]
    for progress_id in stale_ids:
        _channels.pop(progress_id, None)


def _get_channel(progress_id):
    normalized_id = normalize_progress_id(progress_id)
    if not normalized_id:
        return None

    with _channels_lock:
        _cleanup_old_channels()
        channel = _channels.get(normalized_id)
        if channel is None:
            channel = _ProgressChannel()
            _channels[normalized_id] = channel
        channel.updated_at = time.time()
        return channel


def remove_progress_channel(progress_id):
    normalized_id = normalize_progress_id(progress_id)
    if not normalized_id:
        return
    with _channels_lock:
        _channels.pop(normalized_id, None)


def publish_search_progress(progress_id, stage, label, progress, **fields):
    channel = _get_channel(progress_id)
    if channel is None:
        return

    try:
        resolved_progress = float(progress)
    except (TypeError, ValueError):
        resolved_progress = 0.0
    resolved_progress = max(0.0, min(1.0, resolved_progress))
    normalized_stage = str(stage or "working").strip() or "working"
    event = {
        "stage": normalized_stage,
        "label": str(label or "").strip() or "Working",
        "progress": resolved_progress,
        "ts": time.time(),
    }
    for key, value in fields.items():
        if value is not None:
            event[str(key)] = value

    channel.updated_at = time.time()
    if normalized_stage in {"complete", "error"}:
        channel.done = True

    try:
        channel.queue.put_nowait(event)
    except queue_module.Full:
        try:
            channel.queue.get_nowait()
        except queue_module.Empty:
            pass
        channel.queue.put_nowait(event)


def stream_search_progress(progress_id, heartbeat_seconds=15):
    channel = _get_channel(progress_id)
    if channel is None:
        return

    while True:
        try:
            event = channel.queue.get(timeout=heartbeat_seconds)
        except queue_module.Empty:
            yield {"type": "heartbeat"}
            continue

        yield event
        if event.get("stage") in {"complete", "error"}:
            break
