"""Shared embedding daemon: load Qwen3-Embedding-0.6B ONCE, serve encode() to all
benchmark workers over localhost, with a disk-backed cache and CPU-thread caps.

Rationale: each pipeline worker otherwise holds its own ~2.4 GB copy of the embedder and
its encodes burst across every core. One daemon = one resident copy; the cache makes
repeated agent queries (the common case across a 400-run benchmark) free; thread caps
keep the shared machine polite. Vectors are bitwise-identical to a local
``SentenceTransformer(...).encode(text, convert_to_numpy=True)`` call, which is the only
call pattern the pipeline uses (ArangoDB/retrieval.py).

Run:  .venv/bin/python benchmark/_embed_daemon.py   (listens on 127.0.0.1:8876)
  GET  /health          -> {"ok": true, "cache": N}
  POST /encode {"text"} -> {"vec": [...]}
"""
import hashlib
import json
import os
import pickle
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

os.environ.setdefault("OMP_NUM_THREADS", "2")
os.environ.setdefault("MKL_NUM_THREADS", "2")
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "")  # this box's GPU can't run our torch build

import torch  # noqa: E402
torch.set_num_threads(2)
from sentence_transformers import SentenceTransformer  # noqa: E402

PORT = int(os.getenv("EMBED_DAEMON_PORT", "8876"))
MODEL_NAME = os.getenv("EMBED_MODEL", "Qwen/Qwen3-Embedding-0.6B")
CACHE_PATH = Path(__file__).parent / "results" / "_embed_cache.pkl"

print(f"[daemon] loading {MODEL_NAME} ...", flush=True)
MODEL = SentenceTransformer(MODEL_NAME)
LOCK = threading.Lock()
CACHE: dict = {}
if CACHE_PATH.exists():
    try:
        CACHE = pickle.load(open(CACHE_PATH, "rb"))
        print(f"[daemon] cache loaded: {len(CACHE)} entries", flush=True)
    except Exception:  # noqa: BLE001
        CACHE = {}
_dirty = 0


def encode(text: str):
    global _dirty
    key = hashlib.sha256(text.encode()).hexdigest()
    with LOCK:
        if key in CACHE:
            return CACHE[key]
    vec = MODEL.encode(text, convert_to_numpy=True).tolist()
    with LOCK:
        CACHE[key] = vec
        _dirty += 1
        if _dirty >= 50:  # periodic persist
            pickle.dump(CACHE, open(CACHE_PATH, "wb"))
            _dirty = 0
    return vec


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):  # quiet
        pass

    def _send(self, obj, code=200):
        body = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self._send({"ok": True, "model": MODEL_NAME, "cache": len(CACHE)})
        else:
            self._send({"error": "unknown"}, 404)

    def do_POST(self):
        if self.path != "/encode":
            self._send({"error": "unknown"}, 404)
            return
        n = int(self.headers.get("Content-Length", 0))
        try:
            req = json.loads(self.rfile.read(n))
            self._send({"vec": encode(req["text"])})
        except Exception as e:  # noqa: BLE001
            self._send({"error": str(e)[:200]}, 500)


if __name__ == "__main__":
    print(f"[daemon] serving on 127.0.0.1:{PORT}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
