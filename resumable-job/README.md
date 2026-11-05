# Resuming a long job

Work that takes longer than you want to hold a connection open, driven as **many
short calls that resume**. Each call runs a slice, writes a checkpoint, and tells you whether to call
again.

The training loop here is a small stand-in. What the example is about is the
checkpoint and where it goes.

## What the example teaches

**A code app is not cut off by a request timeout, but it still cannot run for
hours.** `requestTimeoutSecs` applies to container apps only. A code app
stores the field and does not apply it. What ends a long run is everything else: a worker is checkpointed and restored,
`load` is retried and re-instantiated when the GPU is short at startup, and an
idle worker is scaled away. **A long job on one worker loses all of its
work if that worker goes away.**

**The checkpoint cannot live on a volume.** A volume is **node-local**, so the
worker that serves your next call may be on a different node and see an empty
directory. A volume makes a *restart* cheap, it does not make a *resume*
possible.

So the checkpoint goes to object storage, which every worker can reach:

```python
self.store.put_object(Bucket=BUCKET, Key=f"checkpoints/{job}.pt", Body=buffer.getvalue())
```

**The job name selects the checkpoint.** The same name resumes the same
checkpoint, a new name starts fresh. There is no job registry to keep in step
with the storage.

**A job name becomes part of an object key, so validate it.** `pattern` on the
field keeps a slash or a `..` out of the key.

**The check still costs a worker.** Measured: a job named `../etc/passwd` is
refused by this app after a worker has woken, not by the platform before it.
The platform enforces the types it published and `additionalProperties`. The value
constraints, `pattern`, `minimum`, `maximum` and a list's length, **all reach the
worker**. Validate anyway, and expect to pay for a wake when someone sends a bad name.

**Return the progress in every answer.** Each call reports `steps_done`,
`total_steps` and `done`, so the caller's loop is three lines and you can see
where a job got to without a dashboard.

**Size the slice by what a failure should cost.** It is 200 steps here. A bigger
slice loses more work on each failure, a smaller one pays the fixed per-call
overhead more often.

## Deploying the app

It needs credentials for any S3-compatible store, so the secrets come first. One
secret holds one value, and `--value-file -` reads it from stdin so it never
reaches the process list.

```bash
printf '%s' "$S3_BUCKET"            | runware serverless secrets set S3_BUCKET --value-file -
printf '%s' "$S3_ACCESS_KEY_ID"     | runware serverless secrets set S3_ACCESS_KEY_ID --value-file -
printf '%s' "$S3_SECRET_ACCESS_KEY" | runware serverless secrets set S3_SECRET_ACCESS_KEY --value-file -

runware serverless deploy model.py \
  --id resumable-job \
  --gpu-type rtx-pro-6000 \
  --secret S3_BUCKET \
  --secret S3_ACCESS_KEY_ID \
  --secret S3_SECRET_ACCESS_KEY
```

The secret names are the environment variable names the code reads, so nothing
needs mapping. Attaching one under a different name is `--secret NAME=ENV_VAR`.

**`S3_ENDPOINT` and `S3_REGION` are the optional two.** Leave both unset and boto3
resolves them the way it does for AWS. Any other S3-compatible store needs the
endpoint, and some want the region set as well:

| Store | `S3_ENDPOINT` | `S3_REGION` |
| --- | --- | --- |
| AWS S3 | unset | unset |
| Cloudflare R2 | `https://<account>.r2.cloudflarestorage.com` | `auto` |
| MinIO and most others | the server's URL | unset, or whatever it expects |

Neither is a credential, so they go in as plain variables rather than secrets:

```bash
runware serverless deploy model.py --id resumable-job --gpu-type rtx-pro-6000 \
  --secret S3_BUCKET --secret S3_ACCESS_KEY_ID --secret S3_SECRET_ACCESS_KEY \
  --env-var S3_ENDPOINT=https://<account>.r2.cloudflarestorage.com \
  --env-var S3_REGION=auto
```

## Calling the app

Call it once to start:

```bash
curl https://api.serverless.runware.ai/v1/apps/resumable-job/invoke-sync \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"request": {"job": "nightly", "total_steps": 1000}}}'
```

Then call it again with the same `job` until `done` is true. Five calls at 200
steps a call:

```bash
while :; do
  out=$(curl -s https://api.serverless.runware.ai/v1/apps/resumable-job/invoke-sync \
    -H "Authorization: Bearer $RUNWARE_API_KEY" \
    -H "Content-Type: application/json" \
    -d "{\"taskId\": \"$(uuidgen | tr 'A-Z' 'a-z')\", \"payload\": {\"request\": {\"job\": \"nightly\", \"total_steps\": 1000}}}")
  echo "$out" | python3 -c 'import json,sys; r=json.load(sys.stdin)["output"]["predictions"][0]; print(r["steps_done"], "/", r["total_steps"]); exit(0 if not r["done"] else 1)' || break
done
```

`resumed_from` in each answer is the step the checkpoint it started from held, so
a call that resumed correctly says so.

Measured on an RTX PRO 6000, five calls to one job of 1000 steps:

| Call | `resumed_from` | `steps_done` | `loss` | Time |
| --- | --- | --- | --- | --- |
| 1, cold worker | 0 | 200 | 0.0444 | **106s** |
| 2 | 200 | 400 | 0.0331 | 1.8s |
| 3 | 400 | 600 | 0.0251 | 1.1s |
| 4 | 600 | 800 | 0.0189 | 1.3s |
| 5 | 800 | 1000, `done` | 0.0142 | 1.5s |

**The loss column is what proves the resume worked.** A step counter only shows
that a number was stored. A loss that keeps falling across five separate workers
shows that the **optimizer state** survived the round trip, which is easy to get
wrong and silent when you do.

A sixth call under a new job name came back at `resumed_from: 0` with a loss of
0.0444, exactly the first call's, which is what a fresh start should look like.

## Testing the app

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

The tests run with neither torch nor boto3 installed, and they pin the request
contract and the object key. **The S3 round trip is not covered here**, because it needs a bucket. Deploying
and calling the app covers that half, which is what the table above measures.
