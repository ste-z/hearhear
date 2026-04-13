import logging
import os
import resource
from datetime import datetime, timezone


def _logger():
    return logging.getLogger("hearhear.runtime")


def _rss_mb():
    try:
        rss_kb = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    except Exception:
        return None

    if os.uname().sysname.lower() == "darwin":
        return round(float(rss_kb) / (1024 * 1024), 1)
    return round(float(rss_kb) / 1024, 1)


def log_runtime_event(event, **fields):
    payload = {
        "ts_utc": datetime.now(timezone.utc).isoformat(),
        "event": str(event),
    }
    rss_mb = _rss_mb()
    if rss_mb is not None:
        payload["rss_mb"] = rss_mb

    for key, value in fields.items():
        if value is None:
            continue
        payload[str(key)] = value

    ordered = " ".join(f"{key}={payload[key]!r}" for key in payload)
    _logger().info(ordered)
