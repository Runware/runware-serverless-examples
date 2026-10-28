# More than one card

A worker with more than one card. The platform runs **one serving process per
GPU**, so the interesting part is not the arithmetic, it is what each of those
processes is expected to do.

No weights, so a failure here is the world coming up, not a model.

## What it teaches

**Every rank runs your handler, with the same request.** Only rank 0 is reachable
from outside the sandbox. It copies each invocation to its siblings and holds all
of them at a barrier before any reaches your code. A collective inside a handler is safe for that reason, and it is also the constraint: **a rank that returns early
hangs every rank that arrived at the collective it skipped.**

So a handler with a shortcut in it is the one thing to avoid. The split in this
example gives the early ranks the remainder, and a rank whose share is empty still
reaches the all-reduce.

**The process group is already up.** `load` runs inside the platform's
initialization, which sets this rank's device and calls
`init_process_group("nccl")` before it, then waits on a barrier after. **A second
`init_process_group` raises**, so `load` here only reads the world it was handed.

**Only rank 0's return value reaches the caller.** The siblings answer rank 0,
which discards them. Nothing breaks if the ranks disagree, so a response that
should carry every rank's contribution has to gather it, as `reduce` does.

**Each rank works on its own card.** `torch.cuda.current_device()` is already this
rank's, because the platform set it before `load`. Allocating with
`device=self.device` is what keeps a rank off its siblings' memory, and it is why
no single card has to hold the whole range.

**Nothing in the file names a GPU count.** `--gpus-per-worker` does, and the same
file serves a world of one.

## Deploying it

```bash
runware serverless deploy model.py \
  --id multi-gpu \
  --gpu-type rtx-pro-6000 \
  --gpus-per-worker 2
```

`--gpus-per-worker` takes 1, 2, 4 or 8.

**A worker of N GPUs needs one node with N of them free, and not every type has
one.** This example was first deployed to a different type and the build went
through while the rollout did not: the build stayed `ready` and the app went to
`failed`, because a build knows nothing about where its worker will land.

`runware serverless apps events multi-gpu` is what says so, in one line. Reach for
it before reading any of your own code, because a worker that was never scheduled
ran none of it.

## Calling it

Call `world-view` first. It takes no payload and says whether the world came up at
all, and on which cards:

```bash
curl https://api.serverless.runware.ai/v1/apps/multi-gpu/invoke-sync/world-view \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {}}'
```

The endpoint path is the method name with underscores turned into hyphens, which
is why `world_view` answers at `world-view`.

Then `reduce`, which sums `range(size)` with each rank holding its own slice:

```bash
curl https://api.serverless.runware.ai/v1/apps/multi-gpu/invoke-sync/reduce \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"size": 4194304}}'
```

`partials` is what proves the work was distributed. A worker that ignored the
world would answer with one entry holding everything.

Measured on two RTX PRO 6000:

```json
{
  "total": 8796090925056,
  "partials": [2199022206976, 6597068718080],
  "worldSize": 2,
  "size": 4194304
}
```

Two different partials, each summed on its own card, and a total that matches the
same sum computed anywhere else.

The first call is slow in a way the other examples are not: a worker carrying
torch pulls a large image before it runs anything, which took about eight minutes
from the first invoke to the result. `runware serverless apps workers multi-gpu`
reports `pulling` while that happens, so you can see a long first call instead of
guessing at it. The next call answers in **5 seconds**.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

Torch is absent here, so what the tests reach is the split: that the shares are
contiguous, that they cover the range exactly, that no two differ by more than one,
and that eight ranks over two values still produce eight shares. That last one is
the case a handler would be tempted to shortcut.
