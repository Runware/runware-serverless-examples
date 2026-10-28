# Generating an image and removing its background

A product shot generated and cut out of its background, in one call. Two different
models run in one worker, and the image between them never leaves the process.

A managed inference API cannot give you this, because **the only thing that crosses a
request boundary is a finished image**.

## What it teaches

**Your intermediate never leaves the process.** The image you hand to the matte is
the object the generator returned, so you do not encode it to PNG, upload it, queue
it again, or wait for a second worker to start. Two API calls pay all four of those
for a step that takes a fraction of a second on a card that is already warm and
already holding your image.

**Two models mean two dtypes.** This file loads the generator in bfloat16, and the
matte loads in whatever its checkpoint carries, and nothing lines the two up for
you. Hand a float32 batch to a half model and it fails at the first convolution,
with a message about bias types that tells you nothing about where it came from. So
read the dtype off the model you are about to call.

**Two models share one worker and one volume.** The generator is tens of gigabytes
and the matte is small, so the second one costs you almost nothing: one cold start,
one set of weights to fetch, one GPU to keep warm. Split them across two apps and
you pay all three twice.

**Size the GPU for the larger model.** Nothing in `model.py` names hardware, and
your app has to fit the generator, so the matte rides along on whatever you chose.

**`trust_remote_code=True` runs code from the model's own repository**, and that
code has imports of its own. BiRefNet needs `einops`, `kornia` and `timm`, which is
why they are in `requirements` for a model your file never imports.

Nothing warns you at build time. The build reads your endpoints without installing
anything, so a missing import only shows up when a worker starts, as a
`CrashLoopBackOff` and an `ImportError` in `apps logs`. Read the remote code for its
imports before you deploy it, the same way you read any dependency you are about to
run.

**Both models are permissively licensed and ungated.** Qwen-Image is Apache 2.0 and
BiRefNet is MIT, so you deploy with no token and nothing here depends on who you
are.

## Deploying it

```bash
runware serverless deploy model.py \
  --id chained-steps \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

The first deploy fetches both models into the volume. Later starts read them from
it, so a cold start you measure in minutes becomes one you measure in seconds.

## Calling it

```bash
curl https://api.serverless.runware.ai/v1/apps/chained-steps/invoke-async/predict \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"request": {"prompt": "a ceramic coffee cup on a plain surface, studio lighting"}}}'
```

The parameter is a model, so its fields sit under the parameter's own name. The
result is a base64 PNG **with an alpha channel** at
`output.predictions[0].image`.

Use `invoke-async` for the first call. A sync invoke has a deadline of roughly 105
seconds, and fetching the weights takes longer than that.

Measured on an RTX PRO 6000, three different things:

| | |
| --- | --- |
| First call, empty volume | **3m28s** |
| A restart, weights already on the volume | 44 seconds |
| A worker already serving | 14 seconds |

The volume does not keep a worker warm. It makes a **restart** cheap, which is
the gap between the first row and the second.

The cut-out comes back as a PNG in **RGBA**, with alpha running the full range from
fully transparent to fully opaque.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

Neither model is present here, the same position the build is in when it imports
the file. What the tests reach is the contract: that the published request
carries its bounds, and that the response says the size of what it returns.
