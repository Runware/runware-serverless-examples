# Running a model with weights

A text-to-image model on a GPU, with the weights on a volume so a restart does not
fetch them again.

Z-Image Turbo is Apache 2.0 and ungated, so you deploy it with **no token**.

## What it teaches

**Import your heavy dependencies inside `load`, not at the top of the file.** The
build imports your file to find your endpoints, and it does that without your
requirements installed, so a module-level `import torch` fails the build. Inside
`load` it runs in the worker, where torch is there.

**`load` is retried when the GPU is out of memory at startup**, waiting for a
larger allocation each time. The retry re-instantiates your class, so both
`__init__` and `load` have to be safe to run twice.

**List your dependencies in `requirements` on the class.** The build installs what
you list, and keeping it next to the code means the next person reading your file
finds it.

**Pin the versions, and pin them to a release your GPU can run.** Leave them open
and a version you never chose lands on your next build. Pin too far back and you
get `no kernel image is available for execution on the device` at the first
inference, because that wheel predates your card.

**Keep your weights on a volume.** A checkpoint captures the worker's filesystem,
so anything you download outside a volume is copied into every checkpoint and
fetched again on every cold start. Mount one with `--volume /weights` and your
second start reads from it.

**Nothing in `model.py` names hardware.** You choose the GPU when you deploy, so
the same file runs on whatever you give it.

## Deploying it

```bash
runware serverless deploy model.py \
  --id loading-weights \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

The first deploy downloads the weights into the volume. Later starts read them
from it.

Deploying again over a running app builds the new version first and keeps serving
the old one until it is ready, so a deploy is not an outage.

## Calling it

The parameter is a model, so its fields sit under the parameter's own name.

```bash
curl https://api.serverless.runware.ai/v1/apps/loading-weights/invoke-async/predict \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"request": {"prompt": "a red bicycle leaning on a wall"}}}'
```

The class declares no `@endpoint`, so `@serve` exposes `predict` at the default
path. The image comes back as a base64 PNG at `output.predictions[0].image`.

Eight steps is the default because this is a turbo model. Raising it buys little.

Measured on an RTX PRO 6000, three different things:

| | |
| --- | --- |
| First call, empty volume | **6m33s** |
| A restart, weights already on the volume | 35 seconds |
| A worker already serving | 9 seconds |

The volume does not keep a worker warm. It makes a **restart** cheap, which is
the gap between the first row and the second.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

The tests run with neither torch nor diffusers installed, the same position the
build is in when it imports the file. A test that fails here is a deploy that
would have failed later.
