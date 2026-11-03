"""The server the platform talks to. Stdlib only, so the image stays small.

Runware hands this server the request's ``payload`` member alone, not the
envelope around it, so a local curl and a real invocation send the same body.
"""

import base64
import json
import os
import subprocess
import tempfile
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PORT = int(os.environ.get("PORT", "8080"))
BLENDER = "/usr/local/bin/blender"
SCENE = "/app/scene.py"

DEFAULTS = {"samples": 48, "width": 960, "height": 540}


def render(frame, samples, width, height):
    with tempfile.TemporaryDirectory() as work:
        target = str(Path(work) / "frame")
        started = time.time()
        result = subprocess.run(
            [BLENDER, "--background", "--factory-startup", "--python", SCENE,
             "--", str(frame), str(samples), str(width), str(height), target],
            capture_output=True, text=True, timeout=570,
        )
        seconds = time.time() - started

        if result.returncode != 0:
            raise RuntimeError(result.stderr[-2000:] or result.stdout[-2000:])

        device = "CPU"
        for line in result.stdout.splitlines():
            if line.startswith("RUNWARE_DEVICE="):
                device = line.split("=", 1)[1].strip()

        # Blender appends the frame number to a path with no extension.
        written = sorted(Path(work).glob("frame*.png")) or sorted(Path(work).glob("*.png"))
        if not written:
            raise RuntimeError(f"blender wrote no image: {result.stdout[-1000:]}")

        return {
            "image": base64.b64encode(written[0].read_bytes()).decode("ascii"),
            "frame": frame,
            "device": device,
            "seconds": round(seconds, 2),
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
        # Readiness is immediate: Blender is in the image, so there are no
        # weights to fetch and nothing to load before the first frame.
        if self.path in ("/ready", "/health"):
            return self._send(200, {"status": "ok"})

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") not in ("/render", ""):
            return self._send(404, {"error": f"no endpoint at {self.path}"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as problem:
            return self._send(400, {"error": f"bad body: {problem}"})

        values = {**DEFAULTS}
        for key in DEFAULTS:
            if payload.get(key) is not None:
                values[key] = int(payload[key])

        try:
            return self._send(200, render(int(payload["frame"]), **values))
        except KeyError:
            return self._send(400, {"error": "frame is required"})
        except subprocess.TimeoutExpired:
            return self._send(504, {"error": "render did not finish in time"})
        except Exception as problem:  # noqa: BLE001
            return self._send(500, {"error": str(problem)})

    def log_message(self, fmt, *args):
        print(f"{self.address_string()} {fmt % args}", flush=True)


ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
