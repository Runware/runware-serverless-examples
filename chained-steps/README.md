# Two models, one request

A product shot generated and cut out of its background, in one call. Two
different models run in one worker, and the image between them never crosses a
request boundary.

This is the one shape a managed inference API cannot offer you, because **the only
thing that can cross a request boundary is a finished image**.

## What it teaches

**The intermediate stays in the process.** The image handed to the matte is the
object the generator produced. Not encoded to PNG, not uploaded, not queued a
second time, and not waiting on a second worker to start. Two calls to any API,
ours included, pay all four of those for a step that takes a fraction of a second
on a card that is already warm and already holding the image.

**Two models are two dtypes.** The generator is loaded in bfloat16 by this file;
the matte is loaded in whatever its checkpoint carries, and nothing lines the two
up for you. A float32 batch into a half model fails at the first convolution, with
a message about bias types that says nothing about where it came from. The handler
reads the dtype off the model it is about to call.

**Two models share one worker and one volume.** The generator is tens of
gigabytes and the matte is small, and the second one costs almost nothing on top
of the first: one cold start, one set of weights to fetch, one GPU to keep warm.
Splitting them across two apps doubles all three.

**The GPU choice is the larger model's.** Nothing in `model.py` names hardware,
and the app has to fit the generator, so the matte rides along for free.

**`trust_remote_code=True` executes code from the model's own repository**, and
that code has imports of its own. BiRefNet's needs `einops`, `kornia` and `timm`,
which appear in `requirements` for a model this file never imports.

Nothing warns you at build time. The build reads your endpoints without installing
anything, so a missing import only surfaces when a worker starts, as a
`CrashLoopBackOff` and an `ImportError` in `apps logs`. **Read the remote code for
its imports before you deploy it**, the same way you would read any dependency you
are about to run.

**Both models are permissively licensed and ungated.** Qwen-Image is Apache 2.0
and BiRefNet is MIT, so this deploys with no token and nothing here depends on who
is deploying it.

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
