# Running a ComfyUI workflow

ComfyUI running headless on a GPU, with Z-Image Turbo as the model. **One endpoint
fills in a bundled workflow** from a prompt, and **the other runs any workflow you
send** in ComfyUI's API format.

Z-Image Turbo is Apache 2.0 and ungated, so you deploy it with **no token**.

It is a container example because **ComfyUI is a server with its own queue**. The app
is `server.py`, a small server in front of it that the platform talks to.

## What the example teaches

**ComfyUI's interface does not run on Serverless.** Every invocation reaches your
server as a `POST` to an endpoint you declare, so there is no way to serve ComfyUI's
page or its websocket. Build the workflow in a ComfyUI on your own machine, then save
it with **Export (API)** from ComfyUI's menu. That file is what this app runs.

ComfyUI **returns a prompt id**, and the platform **waits for a result**. `POST /prompt`
returns at once and runs the workflow from ComfyUI's own queue. `server.py` waits for
that id to appear in `/history`, reads the saved images from ComfyUI's output folder
and returns them in its response, then deletes both the files and the history entry.

ComfyUI **reuses the output of a node** whose inputs have not changed. That is what
keeps the models loaded between requests. A save node it reuses writes no file, and
its history entry lists the file of the earlier request, which `server.py` has already
deleted. So `server.py` gives every node with a `filename_prefix` input **the
request's own id as its prefix**, and ComfyUI runs those nodes again. Without it, sending the same
request twice failed the second time with a missing file.

**ComfyUI listens on the loopback address**, `127.0.0.1:8188`, so only `server.py`
can reach it. The platform calls port 8080, which `container.yaml` declares. 8188 is
clear of **the four ports the platform reserves** inside a worker: 8000, 8888, 8889 and
9090.

**The weights go on the volume.** On its first start a worker downloads the three
files the workflow loads from `Comfy-Org/Z-Image-Turbo`, about 21 GB at a pinned
revision, into `/weights`. `extra_model_paths.yaml` adds that folder to the ones
ComfyUI searches. The download skips a file the volume already holds at that
revision, so only a worker on a node that has never run the app pays for it.
**Volumes belong to a node**, and in three starts in a row every worker landed on a
node that had never run the app, so each one downloaded the weights again, in 50 to
59 seconds.

**Readiness waits for a first generation.** ComfyUI loads a model the first time a
workflow needs it. A server that reported ready as soon as ComfyUI listened would
load all three models inside its first request. `server.py` runs one generation of
one step before `/ready` returns `200`, and that warm-up took 25 to 30 seconds. **The
platform checkpoints the worker when it becomes ready**, with the models on the card,
and the first real request then spent 2.2 seconds in ComfyUI.

`--gpu-only` **keeps the models on the card**. By default ComfyUI moves a model back
to host memory after it runs, and its DynamicVRAM sizes itself against the memory
free on the card as it goes. `--gpu-only` turns both off, so no request after the
warm-up loads a model again.

**The platform checks the types**, and `server.py` **checks the ranges**. A payload
with a field of the wrong type, or one `container.yaml` does not declare, is rejected
with a `422` before it reaches the server. The platform does not check `minimum` and
`maximum`, so a `width` of 99999 still reaches `server.py`, which returns `400` naming
the value out of range. The task then fails, and its error quotes that response.

`generate` is **the endpoint you ship**, and `workflow` is **the one you iterate
with**. `generate` takes a prompt and fills in `workflow.json`, the graph of ComfyUI's own
Z-Image Turbo template, so a caller never sends a graph. `workflow` takes a whole
graph and runs it as sent. Both can use only the nodes in the image and the models on
the volume.

The node ids in `workflow.json` are **names where an export has numbers**. ComfyUI
accepts names as ids, and `server.py` finds the three nodes it fills by name:
`prompt`, `sampler` and `latent`.

A custom node is **installed when the image builds**. Download it into
`/opt/ComfyUI/custom_nodes` at a pinned commit, the way the Dockerfile downloads
ComfyUI itself, and install its requirements. The Dockerfile then runs
`main.py --quick-test-for-ci`, which **imports every node and exits**, so a node that
fails to import fails the build instead of the first request. `--cpu` lets that check
run in a build that has no GPU.

`server.py` **relays ComfyUI's output to the app's logs**, with each line prefixed
`[comfyui]`. It removes the color codes ComfyUI writes and the progress bar a sampler
redraws on every step, which would otherwise reach the logs once per redraw. It also
leaves out the probe requests, which the platform sends every second while a worker
starts.

**Every PNG includes the workflow that made it**, with the prompt, in a text chunk
named `prompt`. That is ComfyUI's default, and `--disable-metadata` in `FLAGS` leaves
it out.

ComfyUI is **licensed GPL-3.0**. This repository does not include it: the Dockerfile
downloads release 0.46.3 when the image builds.

## Deploying the app

```bash
runware serverless deploy --container ./comfyui-workflow \
  --id comfyui-workflow \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

`--container` points at the directory, not at a file.

`startupSeconds` in `container.yaml` is 2400, **the maximum the platform allows**. A
first start downloads the weights and runs the warm-up before the worker is ready,
which took 87 to 98 seconds here, and the maximum keeps **a slower download from
failing the start**.

## Calling the app

```bash
curl https://api.serverless.runware.ai/v1/apps/comfyui-workflow/invoke-sync/generate \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"prompt": "A pair of white running shoes on a concrete step in early morning light, product photo, shallow depth of field", "seed": 42}}'
```

The images come back at `output.images`, each a base64 PNG. A container app's output
is **your server's response body as it was**, with no `predictions` wrapper. `seed` is
the one the image was made with, so you can **send it back for the same image** when
you left it out. The same seed and values gave the same pixels on two workers on
different nodes.

Measured on an RTX PRO 6000:

| | Task | ComfyUI alone |
| --- | --- | --- |
| Build | 3m19s to 3m30s | |
| First call on a node, weights downloaded | **2m56s** and 3m27s | 2.2s and 2.1s |
| A warm worker, 1024 × 1024 at 8 steps | 5.7s | **2.3s** |
| A warm worker, 832 × 1216 at 8 steps | 4.9s | 1.6s |
| The same request again | 2.9s | 0.2s |

A start spent 61 to 81 seconds pulling the image, 50 to 59 seconds downloading the
weights, and the warm-up's 25 to 30 seconds loading the models. **The same request
again comes from ComfyUI's cache**: only the save node runs.

To run your own workflow, send the exported file as `workflow`:

```bash
curl https://api.serverless.runware.ai/v1/apps/comfyui-workflow/invoke-sync/workflow \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"taskId\": \"$(uuidgen | tr 'A-Z' 'a-z')\", \"payload\": {\"workflow\": $(cat workflow_api.json)}}"
```

**Every saved image comes back**, from each `SaveImage` node in the graph. A
`PreviewImage` node writes to ComfyUI's temporary folder, so `server.py` **leaves its
images out**.

## Testing the app

A container example is tested by running the container, so this directory has no
`test_model.py`.

```bash
docker build -t comfyui-workflow .
docker run --rm --gpus all -p 8080:8080 -v "$PWD/weights:/weights" comfyui-workflow

curl http://localhost:8080/ready    # 503 while it downloads and loads, then 200
curl -X POST http://localhost:8080/generate -H 'Content-Type: application/json' \
  -d '{"prompt": "A pair of white running shoes on a concrete step in early morning light, product photo, shallow depth of field"}'
```

The local volume keeps the weights between runs, the way `/weights` does on a worker.
