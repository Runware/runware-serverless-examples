# A render farm

Blender rendering one frame of a turntable on a GPU. **No model, no weights, no
machine learning anywhere in it.**

This is the container path's real pitch: the environment matters more than the
code. Blender needs system libraries, ships as a tarball with its own Python, and
has a command line rather than an API.

## What it teaches

**A container app is your server, so the whole thing runs under Docker first.**
That is the advantage the code path does not have. Build it, run it, curl it, and
only then deploy:

```bash
docker build -t render-farm .
docker run --rm -p 8080:8080 render-farm
curl http://localhost:8080/ready
curl -X POST http://localhost:8080/render -H 'Content-Type: application/json' -d '{"frame": 1, "samples": 16}'
```

The local call sends the same body a real invocation does, because **Runware hands
your server the `payload` member alone**, not the envelope around it.

**Declare what the builder cannot see.** A code app has its endpoints read out of
the source. Nothing can look inside an image and find them, so `container.yaml`
declares the port, the probes, the timeouts and a JSON Schema per endpoint.

**Do not bind 8000, 8888, 8889 or 9090.** The platform binds all four inside a
worker pod and every container in a pod shares one network namespace. Taking one
does not warn you, it just loses the bind or exposes you through a Service that was
never meant to carry your traffic. This one uses 8080.

**Readiness is a promise about your weights, and here there are none.** Blender is
in the image, so `/ready` answers `200` immediately and honestly. An app that
loaded something would return `503` until it was done, which is how a cold start is
held off the caller.

**Raise `requestSeconds` before you need it.** It defaults to 60 and a render is
not a 60 second job. `container.yaml` sets 600, and the platform waits that plus 30
seconds. **This is the one timeout a code app cannot set**: `requestTimeoutSecs` is
container-only, which is a real reason to pick this path for long work.

**Build the scene in code rather than shipping a `.blend`.** A binary asset in the
repository is one more thing to keep in step with the script that renders it. This
scene is six tori, a light and a camera, written in Blender's own Python.

**The fan-out is the caller's loop.** Modal's version of this renders every frame
in parallel from inside the app. There is no map primitive here: you call the
endpoint once per frame, and `--max-workers` is what decides how many of those run
at a time. One app, one queue, and the parallelism lives in your shell.

```bash
for frame in $(seq 1 48); do
  curl -s https://api.serverless.runware.ai/v1/apps/render-farm/invoke-async/render \
    -H "Authorization: Bearer $RUNWARE_API_KEY" -H "Content-Type: application/json" \
    -d "{\"taskId\": \"$(uuidgen | tr 'A-Z' 'a-z')\", \"payload\": {\"frame\": $frame}}" &
done; wait
```

## Deploying it

```bash
runware serverless deploy --container ./render-farm \
  --id render-farm \
  --gpu-type rtx-pro-6000 \
  --max-workers 4
```

`--container` points at the directory, not at a file, and the directory root holds
the `Dockerfile` and the `container.yaml` beside whatever the Dockerfile copies.

`--max-workers` defaults to 1, and for this app that defeats the point. Four
workers render four frames at once.

## Calling it

```bash
curl https://api.serverless.runware.ai/v1/apps/render-farm/invoke-sync/render \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"frame": 12, "samples": 64}}'
```

The frame comes back as a base64 PNG at `output.image`. **A container app has no `predictions` wrapper**: `output` is your server's response body as it was. `device`
says whether Cycles found a GPU or fell back to the CPU, which is the first thing
to read if a render was slower than you expected.

Measured on an RTX PRO 6000, at the default 960x540:

| | Task | The render alone |
| --- | --- | --- |
| Build | 1m7s | |
| First call, cold container | **29s** | 1.8s |
| A warm worker, 32 samples | 2.1s | 1.2s |
| A warm worker, 128 samples | 2.1s | 1.4s |
| A warm worker, 1280x720 at 96 samples | 2.3s | 1.9s |

`device` came back `GPU` on every one, so Cycles found the card without being
told which backend to use. Note how little the sample count moves the number: at
this resolution the render is dominated by scene setup rather than by sampling,
which is the kind of thing worth measuring before you tune it.


## Testing it

There is nothing local to run, which is why this directory has no `test_model.py`:
a container example is tested by running the container. The `docker run` above is
the test, and it exercises more of the real thing than any code app's suite can.
