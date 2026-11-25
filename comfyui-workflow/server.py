"""The server the platform talks to, in front of a ComfyUI that only it can reach.

Runware hands this server the request's ``payload`` member alone, not the
envelope around it, so a local curl and a real invocation send the same body.
"""

import base64
import copy
import json
import os
import random
import re
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.request
import uuid
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

from huggingface_hub import hf_hub_download

PORT = int(os.environ.get("PORT", "8080"))
COMFYUI = "http://127.0.0.1:8188"
COMFYUI_DIR = Path("/opt/ComfyUI")
OUTPUT_DIR = Path("/tmp/comfyui/output")
WEIGHTS_DIR = Path("/weights")

# --gpu-only keeps every model on the card once it is loaded, which is what the
# checkpoint taken at readiness should hold. It also turns off ComfyUI's
# DynamicVRAM, which sizes itself against the card's free memory as it runs.
FLAGS = ["--gpu-only"]

REPO = "Comfy-Org/Z-Image-Turbo"
REVISION = "6fc90a3b1b653e935a0d175e260736de25b84df5"
FILES = (
    "split_files/diffusion_models/z_image_turbo_bf16.safetensors",
    "split_files/text_encoders/qwen_3_4b.safetensors",
    "split_files/vae/ae.safetensors",
)

WORKFLOW = json.loads((Path(__file__).parent / "workflow.json").read_text())
DEFAULTS = {"width": 1024, "height": 1024, "steps": 8}
LIMITS = {"width": (256, 2048), "height": (256, 2048), "steps": (1, 50)}

# Below requestSeconds in container.yaml, so a workflow that runs too long is
# answered with a 504 before the platform stops waiting.
WAIT_SECONDS = 570

# tqdm redraws a progress bar in place, and a log viewer shows every redraw.
PROGRESS_BAR = re.compile(r"\d+%\|")
ANSI = re.compile(r"\x1b\[[0-9;]*[A-Za-z]")

state = {"ready": False, "error": None, "process": None}
client_id = str(uuid.uuid4())


class Invalid(Exception):
    """The request cannot run as sent. Answered with a 400."""

    def __init__(self, message, details=None):
        super().__init__(message)
        self.details = details


class Failed(Exception):
    """The workflow ran and a node failed. Answered with a 500."""


def comfyui(method, path, body=None):
    request = urllib.request.Request(
        COMFYUI + path,
        method=method,
        data=json.dumps(body).encode() if body is not None else None,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(request, timeout=30) as response:
            return response.status, json.loads(response.read() or b"null")
    except urllib.error.HTTPError as problem:
        return problem.code, json.loads(problem.read() or b"null")


def download():
    # A file the volume already holds at this revision is not fetched again, so
    # only a worker on a node that has never run this app pays for it.
    for name in FILES:
        hf_hub_download(REPO, name, revision=REVISION, local_dir=WEIGHTS_DIR / REPO)


def relay(stream):
    for line in stream:
        line = ANSI.sub("", line).rstrip()
        if line and not PROGRESS_BAR.search(line):
            print(f"[comfyui] {line}", flush=True)


def start_comfyui():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    process = subprocess.Popen(
        [sys.executable, "main.py", "--listen", "127.0.0.1", "--port", "8188",
         "--disable-auto-launch", "--output-directory", str(OUTPUT_DIR), *FLAGS],
        cwd=COMFYUI_DIR,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    state["process"] = process
    threading.Thread(target=relay, args=(process.stdout,), daemon=True).start()

    while process.poll() is None:
        try:
            if comfyui("GET", "/system_stats")[0] == 200:
                return
        except OSError:
            pass
        time.sleep(1)

    raise RuntimeError(f"ComfyUI exited with code {process.returncode} before it listened")


def run(workflow):
    """Queue one workflow, wait for it, and return the images its output nodes saved."""
    prompt_id = str(uuid.uuid4())

    # ComfyUI reuses a node's last output while its inputs are unchanged, and a
    # save node it reuses writes no file: the history then names the file of an
    # earlier request, which this server has already deleted. A prefix of this
    # request's own makes every save node run again.
    workflow = {
        node_id: {**node, "inputs": {**node["inputs"], "filename_prefix": prompt_id}}
        if isinstance(node, dict) and "filename_prefix" in (node.get("inputs") or {})
        else node
        for node_id, node in workflow.items()
    }

    status, answer = comfyui("POST", "/prompt", {"prompt": workflow, "prompt_id": prompt_id, "client_id": client_id})
    if status == 400:
        error = answer.get("error") or {}
        raise Invalid(error.get("message") or "ComfyUI refused the workflow", answer.get("node_errors"))
    if status != 200:
        raise RuntimeError(f"ComfyUI answered {status} to the prompt: {answer}")

    # ComfyUI answers with the id at once and runs the prompt from its own queue.
    # The history entry for that id appears once it finishes, whether it
    # succeeded or failed.
    deadline = time.time() + WAIT_SECONDS
    while True:
        record = comfyui("GET", f"/history/{prompt_id}")[1].get(prompt_id)
        if record:
            break
        if time.time() > deadline:
            comfyui("POST", "/queue", {"delete": [prompt_id]})
            comfyui("POST", "/interrupt", {"prompt_id": prompt_id})
            raise TimeoutError
        time.sleep(0.1)

    comfyui("POST", "/history", {"delete": [prompt_id]})

    if record["status"]["status_str"] != "success":
        for event, data in record["status"]["messages"]:
            if event == "execution_error":
                raise Failed(f"{data['node_type']} failed: {data['exception_message'].strip()}")
        raise Failed("the workflow did not finish")

    images = []
    for output in record["outputs"].values():
        for image in output.get("images", []):
            # A preview node writes to ComfyUI's temp folder, not to the output.
            if image.get("type") != "output":
                continue
            path = OUTPUT_DIR / image["subfolder"] / image["filename"]
            images.append(base64.b64encode(path.read_bytes()).decode("ascii"))
            path.unlink()

    return images


def bounded(payload, key):
    value = payload.get(key)
    if value is None:
        return DEFAULTS[key]

    low, high = LIMITS[key]
    if type(value) is not int or not low <= value <= high:
        raise Invalid(f"{key} must be an integer from {low} to {high}")
    if key in ("width", "height") and value % 16:
        raise Invalid(f"{key} must be a multiple of 16")

    return value


def generate(payload):
    prompt = payload.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        raise Invalid("prompt is required")

    seed = payload.get("seed")
    if seed is None:
        seed = random.randrange(2**32)
    elif type(seed) is not int or not 0 <= seed < 2**64:
        raise Invalid("seed must be an integer from 0 to 2^64 - 1")

    workflow = fill(prompt, seed, bounded(payload, "steps"), bounded(payload, "width"), bounded(payload, "height"))

    started = time.time()
    images = run(workflow)
    return {"images": images, "seed": seed, "seconds": round(time.time() - started, 2)}


def fill(prompt, seed, steps, width, height):
    """The bundled workflow with the caller's values in it. The keys are its node ids."""
    workflow = copy.deepcopy(WORKFLOW)
    workflow["prompt"]["inputs"]["text"] = prompt
    workflow["sampler"]["inputs"]["seed"] = seed
    workflow["sampler"]["inputs"]["steps"] = steps
    workflow["latent"]["inputs"]["width"] = width
    workflow["latent"]["inputs"]["height"] = height
    return workflow


def run_workflow(payload):
    workflow = payload.get("workflow")
    if not isinstance(workflow, dict) or not workflow:
        raise Invalid("workflow is required, in ComfyUI's API format")

    started = time.time()
    images = run(workflow)
    return {"images": images, "seconds": round(time.time() - started, 2)}


ENDPOINTS = {"/generate": generate, "": generate, "/workflow": run_workflow}


def prepare():
    """Fetch the weights, start ComfyUI and run one generation. Runs in a thread so
    the server can answer its readiness probe with a 503 while it runs."""
    try:
        download()
        start_comfyui()
        # ComfyUI loads a model the first time a workflow needs it, so this
        # generation is what loads all three. The platform checkpoints the worker
        # when it turns ready, and the checkpoint holds them on the card.
        run(fill("warm up", seed=0, steps=1, width=DEFAULTS["width"], height=DEFAULTS["height"]))
        state["ready"] = True
    except Exception as problem:  # noqa: BLE001
        state["error"] = f"{type(problem).__name__}: {problem}"
        print(f"prepare failed: {state['error']}", flush=True)


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
        process = state["process"]
        if self.path == "/health":
            # Without ComfyUI every request fails, so the liveness probe fails too.
            if process is not None and process.poll() is not None:
                return self._send(500, {"status": "ComfyUI exited", "code": process.returncode})
            return self._send(200, {"status": "ok"})

        if self.path == "/ready":
            if state["error"]:
                return self._send(500, {"status": "failed", "error": state["error"]})
            if not state["ready"]:
                return self._send(503, {"status": "loading"})
            return self._send(200, {"status": "ok"})

        return self._send(404, {"error": "not found"})

    def do_POST(self):
        handle = ENDPOINTS.get(self.path.rstrip("/"))
        if handle is None:
            return self._send(404, {"error": f"no endpoint at {self.path}"})
        if not state["ready"]:
            return self._send(503, {"error": "still loading"})

        try:
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length) or b"{}")
        except (ValueError, json.JSONDecodeError) as problem:
            return self._send(400, {"error": f"bad body: {problem}"})

        try:
            return self._send(200, handle(payload))
        except Invalid as problem:
            body = {"error": str(problem)}
            if problem.details:
                body["nodeErrors"] = problem.details
            return self._send(400, body)
        except Failed as problem:
            return self._send(500, {"error": str(problem)})
        except TimeoutError:
            return self._send(504, {"error": "the workflow did not finish in time"})
        except Exception as problem:  # noqa: BLE001
            return self._send(500, {"error": f"{type(problem).__name__}: {problem}"})

    def log_message(self, fmt, *args):
        # The platform probes every second while a worker starts, and a line per
        # probe buries the rest of the log.
        if self.path in ("/ready", "/health"):
            return
        print(f"{self.address_string()} {fmt % args}", flush=True)


if __name__ == "__main__":
    threading.Thread(target=prepare, daemon=True).start()
    ThreadingHTTPServer(("0.0.0.0", PORT), Handler).serve_forever()
