# Compiling a model

`torch.compile` on an image classifier, with the compiler's own cache on the
volume so only the first worker pays to compile.

It is a container example because the code path has no compiler, which is the
first thing it teaches.

## What it teaches

**`torch.compile` cannot run on a code app, because there is no compiler.**
`torch.compile` hands its kernels to Triton, and Triton shells out to a C compiler
at runtime. A code worker has none. Measured from inside one with
`shutil.which`:

| Tool | Found |
| --- | --- |
| `cc`, `gcc`, `g++`, `clang` | none |
| `as`, `ld` | none |
| `make`, `ninja` | none |
| `nvcc`, `ptxas` | none |
| `python3-config` | `/usr/local/bin/python3-config` |

A code app that compiles anyway crash-loops: the container exits with code 3 on
every attempt, and **the task stays pending with no error** while `apps errors`
shows nothing. The reason only appears in `apps logs`, inside a Triton traceback:
`Failed to find C compiler`. That is [RUNSERV-1568](https://runware.atlassian.net/browse/RUNSERV-1568).

So the image is a `-devel` one, which ships the toolchain. **A dependency you
cannot install any other way is the clearest reason to choose the container
path**, more so than a server you already wrote.

**Readiness is where the compile goes.** The server starts, answers `/ready` with
`503`, and compiles in a background thread. The platform sends no traffic until
that turns `200`, so **no caller ever waits on a compile inside a request**.

```python
if not state["ready"]:
    return self._send(503, {"status": "compiling"})
```

A server that reports ready before its weights are loaded takes a request it
cannot serve.

**Raise `startupSeconds` to match.** It defaults to 300, and a first compile is
longer than that. This asks for 2400, the maximum, and the platform adds its own
30 minutes on top. A worker that has not answered by then is restarted, so a long compile looks
like a crash loop.

**The cache goes on the volume, and the paths are set in the Dockerfile.**

```dockerfile
ENV TORCHINDUCTOR_CACHE_DIR=/weights/inductor \
    TRITON_CACHE_DIR=/weights/triton
```

In `ENV` rather than in code because **torch reads all of them as it loads**, and
the first import anywhere in the process has already fixed them. Setting them
after that silently does nothing.

A volume holds anything your app **generates** once and should not generate
again, not only weights. The volumes page names a compiled kernel cache as one
of them.

**No cudagraphs.** `mode="max-autotune-no-cudagraphs"` is deliberate. Capturing a
CUDA graph records the addresses its tensors live at, and a worker that is
checkpointed and restored does not get the same addresses back, so a captured
graph replays against memory that moved.

**Import your dependencies at build time.** The Dockerfile imports torch, timm
and pillow in a `RUN` line. There is no GPU during a build, so it checks the
imports rather than the model, which is enough to catch a wheel that does not
load. Without it, a broken wheel costs a deploy and a request to find.

It also needs `--break-system-packages` to pip install at all, because this
image's python is system managed and pip refuses without it. A container has one
environment, so there is nothing else to install into.

## Deploying it

```bash
runware serverless deploy --container ./compiled-model \
  --id compiled-model \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

`--container` points at the directory, not at a file.

## Calling it

```bash
curl https://api.serverless.runware.ai/v1/apps/compiled-model/invoke-sync/classify \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d "{\"taskId\": \"7c9e6679-7425-40de-944b-e07fc1f90ae7\", \"payload\": {\"image\": \"$(base64 -i photo.jpg | tr -d '\n')\", \"top\": 5}}"
```

A container app takes the payload your server expects, with no parameter name
wrapping it, because there is no handler signature to bind to.

The labels come back at `output.labels`, best first. **A container app has no `predictions` wrapper**: `output` is your server's response body as it was, unlike a code app, whose handler result arrives at `output.predictions[0]`. `seconds` is
the forward pass alone, so you can see what the compile bought without the queue
in the way.

Measured on an RTX PRO 6000:

| | |
| --- | --- |
| Build | 10m41s |
| First call, empty volume: weights plus the compile | **14m24s** |
| A restart, compile cache already on the volume | **19s** |
| A warm worker | 1.1s |
| The forward pass itself, warm | **3.2ms** |

**The first compile takes 45 times as long as a restart that reads the cache.**
That is what makes every worker after the first one cheap.

The first forward pass after a restart took 2.59 seconds against 3.2ms warm, so
a worker that has read the cache still needs one pass before it is quick.

`max-autotune` is what makes the first start this long. `mode="default"` compiles in a fraction of the time and keeps most of the
benefit, which is the better choice if you do not want to pay a 14 minute cold
start even once.

## Testing it

There is nothing local to run, which is why this directory has no `test_model.py`:
a container example is tested by running the container.

```bash
docker build -t compiled-model .
docker run --rm --gpus all -p 8080:8080 \
  -e TORCHINDUCTOR_CACHE_DIR=/tmp/inductor -e TRITON_CACHE_DIR=/tmp/triton \
  compiled-model

curl http://localhost:8080/ready    # 503 while it compiles, then 200
```

`/ready` answers `503` until the compile finishes and `200` afterwards, so you can
watch the gate work before deploying anything.
