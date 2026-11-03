# Rendering 3D frames

Blender rendering one frame of a turntable on a GPU, with **no model and no
weights in the app at all**.

It is the case the container path is for: the app needs an environment more than
it needs code. Blender links against system libraries, ships as a tarball with
its own Python, and has a command line rather than an API.

## What it teaches

**A container app is your server, so you can run the whole thing under Docker
first.** The code path cannot be tested that way. Build it, run it, curl it, and
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
worker pod and every container in a pod shares one network namespace. Taking one gives no warning: your server either loses the bind or becomes
reachable through a Service that was never meant to carry your traffic. This one uses 8080.

**Readiness is about your weights, and this app has none.** Blender is in the
image, so `/ready` answers `200` straight away. An app that loads something
returns `503` until it is done, which keeps a cold start off the caller.

**Raise `requestSeconds` before you need it.** It defaults to 60 and a render
takes longer than that. `container.yaml` sets 600, and the platform waits that
plus 30 seconds. **A code app cannot set this timeout at all**:
`requestTimeoutSecs` is container-only, which is one reason to choose this path
for long work.

**Build the scene in code rather than shipping a `.blend`.** A binary asset in the
repository is one more thing to keep in step with the script that renders it. This
scene is six tori, a light and a camera, written in Blender's own Python.

**The caller does the fan-out.** There is no map primitive: you call the
endpoint once per frame, and `--max-workers` decides how many of those run at a
time. One app, one queue, and the loop is yours.

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
told which backend to use. The sample count barely moves the number: at this resolution the render is
dominated by scene setup rather than by sampling.


## Testing it

There is nothing local to run, which is why this directory has no `test_model.py`:
a container example is tested by running the container. The `docker run` above is that test.
