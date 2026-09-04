"""Driver-side patch: replace SentenceTransformer with a thin client of the embedding
daemon (_embed_daemon.py), so workers hold no resident embedder (~2.4 GB saved each).

Call ``install()`` BEFORE any pipeline import (same pattern as the memoization patch it
replaces). If the daemon is unreachable at install time, falls back to a memoized local
SentenceTransformer, so a worker never breaks because the daemon died.

Only the pipeline's actual call pattern is proxied: ``encode(text, convert_to_numpy=True)``
on a single string (ArangoDB/retrieval.py:127). Anything else raises loudly rather than
silently diverging.
"""
import json
import os
import urllib.request

import numpy as np

PORT = int(os.getenv("EMBED_DAEMON_PORT", "8876"))
URL = f"http://127.0.0.1:{PORT}"


def _daemon_alive() -> bool:
    try:
        with urllib.request.urlopen(f"{URL}/health", timeout=3) as r:
            return json.load(r).get("ok", False)
    except Exception:  # noqa: BLE001
        return False


class _STProxy:
    """Duck-types the one SentenceTransformer method the pipeline uses."""

    def __init__(self, model_name_or_path=None, *a, **k):
        self.model_name = str(model_name_or_path)

    def encode(self, text, convert_to_numpy=True, **kwargs):
        if not isinstance(text, str) or kwargs:
            raise NotImplementedError(
                f"embed proxy only supports encode(str, convert_to_numpy=True); "
                f"got type={type(text).__name__} kwargs={list(kwargs)}")
        req = urllib.request.Request(
            f"{URL}/encode", data=json.dumps({"text": text}).encode(),
            headers={"Content-Type": "application/json"})
        for attempt in range(3):
            try:
                with urllib.request.urlopen(req, timeout=120) as r:
                    vec = json.load(r)["vec"]
                return np.asarray(vec, dtype=np.float32)
            except Exception:  # noqa: BLE001
                if attempt == 2:
                    raise
        return None  # unreachable


def install() -> str:
    """Patch sentence_transformers.SentenceTransformer; returns the mode installed."""
    import sentence_transformers as _st
    if _daemon_alive():
        _st.SentenceTransformer = _STProxy
        return "daemon-proxy"
    # fallback: memoized local model (the previous behaviour)
    _cache: dict = {}
    _orig = _st.SentenceTransformer

    def _memo(model_name_or_path=None, *a, **k):
        key = str(model_name_or_path)
        if key not in _cache:
            _cache[key] = _orig(model_name_or_path, *a, **k)
        return _cache[key]

    _st.SentenceTransformer = _memo
    return "local-memoized"
