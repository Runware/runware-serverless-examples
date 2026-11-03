"""The server the platform talks to, and the readiness gate that hides the compile.

Runware hands this server the request's ``payload`` member alone, not the
envelope around it, so a local curl and a real invocation send the same body.
"""

import base64
import io
import json
import os
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image

import timm

PORT = int(os.environ.get("PORT", "8080"))
MODEL = "resnet50.a1_in1k"
SIZE = 224
DEFAULT_TOP = 5

state = {"ready": False, "compiled": False, "error": None}


def warm():
    """Load, compile and exercise the model. Runs in a thread so the server can
    answer its readiness probe with a 503 while this is still going."""
    try:
        model = timm.create_model(MODEL, pretrained=True).eval().cuda()
        config = timm.data.resolve_data_config({}, model=model)
        transform = timm.data.create_transform(**config)

        try:
            labels = timm.data.ImageNetInfo()
        except Exception:
            labels = None

        # No cudagraphs. Capturing a graph records the addresses its tensors live
        # at, and a worker that is checkpointed and restored does not get the same
        # addresses back, so a captured graph replays against memory that moved.
        compiled = torch.compile(model, mode="max-autotune-no-cudagraphs", dynamic=False)

        # This pass is the compile. It writes the cache the volume keeps, and it is
        # why readiness takes minutes the first time and seconds afterwards.
        with torch.inference_mode():
            compiled(torch.zeros(1, 3, SIZE, SIZE, device="cuda"))

        state.update(model=compiled, transform=transform, labels=labels, compiled=True, ready=True)
    except Exception as problem:  # noqa: BLE001
        state["error"] = f"{type(problem).__name__}: {problem}"


def classify(raw, top):
    image = Image.open(io.BytesIO(base64.b64decode(raw, validate=True))).convert("RGB")
    batch = state["transform"](image).unsqueeze(0).cuda()

    started = time.time()
    with torch.inference_mode():
        probabilities = state["model"](batch).softmax(dim=-1)[0]
    seconds = time.time() - started

    scores, indices = probabilities.topk(top)
    labels = state["labels"]

    return {
        "labels": [
            {
                "label": labels.index_to_description(int(i)) if labels else str(int(i)),
                "index": int(i),
                "probability": round(float(s), 5),
            }
            for s, i in zip(scores.tolist(), indices.tolist())
        ],
        "compiled": state["compiled"],
        "seconds": round(seconds, 4),
    }


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def _send(self, status, body):
        raw = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(raw)))
        self.end_headers()
        self.wfile.write(raw)

    def do_GET(self):
        if self.path == "/health":
            return self._send(200, {"status": "ok"})

        if self.path == "/ready":
            if state["error"]:
                return self._send(500, {"status": "failed", "error": state["error"]})
            if not state["ready"]:
                # The whole cold start mechanism: no traffic until the compile is
                # done, so no caller ever waits on it inside a request.
                return self._send(503, {"status": "compiling"})

            return self._send(200, {"status": "ok"})

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/classify", ""):
            return self._send(404, {"error": f"no endpoint at {self.path}"})
        if not state["ready"]:
            return self._send(503, {"error": "still compiling"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as problem:
            return self._send(400, {"error": f"bad body: {problem}"})

        top = payload.get("top")
        try:
            return self._send(200, classify(payload["image"], int(top) if top else DEFAULT_TOP))
        except KeyError:
            return self._send(400, {"error": "image is required"})
        except Exception as problem:  # noqa: BLE001
            return self._send(500, {"error": str(problem)})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


threading.Thread(target=warm, daemon=True).start()
ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
