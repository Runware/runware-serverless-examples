# Embeddings

A text embedding model that takes a list of texts and returns one vector per
text.

`bge-small-en-v1.5` is MIT and ungated, so this deploys with **no token**.

## What the example teaches

**Batch inside your handler, because the platform will not do it for you.** A
worker takes one task at a time. There is no setting that lets several requests
share a worker, so the only batching available to you is the batching you put
inside one request. That is why the request takes a list rather than a string.

What that buys, measured on an RTX PRO 6000 against a warm worker:

| Texts in one request | Time on the worker | Per text |
| --- | --- | --- |
| 1 | 0.62s | 620ms |
| 8 | 0.41s | 52ms |
| 32 | 0.54s | 17ms |
| 128 | 0.80s | 6.2ms |
| 256 | 0.65s | 2.5ms |

**The time barely moves between one text and 256**, and eight came back faster than one, because on a warm worker almost all of
it is fixed overhead rather than model work. Medians of repeated calls, and one
text ranged from 0.47s to 0.84s across five of them.

Sending those 256 texts one per request would pay 256 queue hops and 256 times
that fixed cost for the same work.

**The platform enforces the types it published, and nothing else.** Check this
before you rely on a schema to keep bad requests away from your worker. Measured
against this same app:

| Request | What happened |
| --- | --- |
| `texts` as a string | **422 in 0.4s**, no worker woken |
| `texts` holding integers | **422**, with a pointer per bad index |
| `normalize` as a string | **422 in 0.1s** |
| An empty list, against `min_length=1` | **reached the worker** and failed there |
| 257 texts, against `max_length=256` | **reached the worker** and failed there |

A type is checked before anything wakes. **A value constraint is enforced by the
app, not the platform**, and a request that breaks one wakes a worker before it
is refused.

**Forbid the fields you do not know, and the platform will enforce it for you.**
Pydantic ignores an unknown field by default, so a caller who writes `normalise`
instead of `normalize` gets a silent success and the wrong behavior.
`model_config = ConfigDict(extra="forbid")` turns that into an error.

It is published too. Measured before and after adding it, the same request
carrying a misspelled field went from **a completed task on a woken worker** to
**a 422 in half a second with nothing woken**, so the line saves the cost of the
mistake as well as the wrong answer.

**Return the vectors rather than storing them.** This app embeds and answers,
and the caller decides where the vectors go. That keeps it stateless, so it
scales out across workers.

## Deploying the app

```bash
runware serverless deploy model.py \
  --id embeddings \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

The model is 130MB, which is small enough that you might skip the volume. Mount it
anyway: a checkpoint captures the worker's filesystem, so an unmounted download is
copied into every one.

## Calling the app

```bash
curl https://api.serverless.runware.ai/v1/apps/embeddings/invoke-sync \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"request": {"texts": ["a red bicycle", "a blue car"]}}}'
```

The class declares no `@endpoint`, so `@serve` exposes `predict` at the default
path. The parameter is a model, so its fields sit under the parameter's own name.

The vectors come back at `output.predictions[0].vectors`, one per input and in the
same order. They are unit length unless you ask for `normalize: false`, so a dot
product between two of them is their cosine similarity.

Measured on an RTX PRO 6000:

| | |
| --- | --- |
| Build | 2m57s |
| First call, empty volume | **112s** |
| A restart, weights already on the volume | 14s |
| A warm worker | 0.6s |

The first call pays for the worker, the download and a warmup pass. A volume
does not keep a worker warm, it makes a **restart** cheap, which is the gap
between the second row and the third.

## Testing the app

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

The tests run with neither torch nor sentence-transformers installed, the same
position the build is in when it imports the file. They pin the request contract, which is what a caller depends on.
