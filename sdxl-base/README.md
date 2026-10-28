# `sdxl-base`

A model with real weights on a real GPU. `echo` and `image-tools` deliberately
have neither, so this is the one that shows where the heavy parts of an app go.

Stable Diffusion XL is ungated and its license allows commercial use, so this
deploys with no token of any kind.

## What it teaches

**Heavy imports belong inside `load`.** The build imports this file to read the
endpoints out of it, and it does that **before installing anything the model
needs**. `import torch` at the top of the file fails the build. Inside `load` it
runs in the worker, where torch exists.

**Dependencies are declared on the class.** The `requirements` list is what the
build installs.

Pin them, and pin them to something your GPU can run. An unpinned app that
resolved last month and floats today stops working without anyone changing it,
and a pin older than the card you deploy to fails at the first inference with
`no kernel image is available for execution on the device`, because the wheel
carries no kernels compiled for that architecture.

**Weights belong on a volume.** The worker's filesystem is part of what a
checkpoint captures, so weights left outside a volume are copied into every
checkpoint and fetched again on every cold start. `--volume /weights` is what
makes the second start fast.

**Nothing here names a GPU.** The hardware is set on the app, so the same file
serves whatever you deployed it to.

## Deploying it

```bash
runware serverless deploy model.py \
  --id sdxl-base \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

The first deploy downloads about 7 GB of weights into the volume. Later starts
read them from it.

Deploying again over a running app builds the new version first and keeps serving
the old one until it is ready, so a deploy is not an outage.

## Calling it

```bash
curl https://api.serverless.runware.ai/v1/apps/sdxl-base/invoke-async/predict \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"prompt": "a red bicycle leaning on a wall"}}'
```

The class declares no `@endpoint`, so `@serve` exposes `predict` at the default
path. The image comes back as a base64 PNG at `output.predictions[0].image`.

Measured on an RTX PRO 6000: **1m30s** for the first call, which starts a worker
and fills the volume, and **under 7 seconds** for the next one, which finds both
already there.

## Testing it

```bash
uv run --no-project --with pytest --with runware-serverless pytest -q
```

The tests run with neither torch nor diffusers installed, which is the same
position the build is in when it imports the file. A test that fails here is a
deploy that would have failed later.
