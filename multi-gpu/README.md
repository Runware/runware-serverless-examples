# Using more than one GPU

A worker across several cards. The platform runs **one serving process per GPU**,
so what changes is not the arithmetic, it is what each of those processes has to
do.

No weights, so a failure here is the world coming up, not a model.

## What it teaches

**Every rank runs your handler, with the same request.** Only rank 0 is reachable
from outside the sandbox, so it copies each invocation to its siblings and holds
all of them at a barrier before any of them reaches your code. That is what makes
a collective inside your handler safe.

It is also the constraint. **A rank that returns early hangs every rank that
arrived at the collective it skipped**, so never put a shortcut in a handler. The
split here gives the remainder to the early ranks, and a rank whose share is empty
still reaches the all-reduce.

**The process group is already up.** `load` runs inside the platform's
initialization, which sets your rank's device and calls `init_process_group("nccl")`
before it, then waits on a barrier after. A second `init_process_group` raises, so
your `load` only reads the world it was handed.

**Only rank 0's return value reaches your caller.** The siblings answer rank 0 and
it discards them, and nothing breaks when the ranks disagree. If your response
should carry every rank's work, gather it, the way `reduce` does.

**Each rank works on its own card.** `torch.cuda.current_device()` is already your
rank's, because the platform set it before `load`, so allocate with
`device=self.device` and you keep off your siblings' memory. That is also why no
single card has to hold the whole range.

**Nothing in the file names a GPU count.** You set that with `--gpus-per-worker`,
and the same file serves a world of one.

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

The first call is slow because a worker carrying torch pulls a large image before
it runs anything, which took about eight minutes from the first invoke to the
result. `runware serverless apps workers multi-gpu`
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
