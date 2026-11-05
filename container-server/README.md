# Running your own container

vLLM, deployed as it is. The image is the one its maintainers publish, the server is the
one they wrote, and the whole app is a `Dockerfile` with no `COPY` in it and a
`container.yaml` beside it.

Qwen3 4B Instruct is Apache 2.0 and ungated, so this deploys with **no token**.

## What the example teaches

**A container deploy ships an image, not code.** `--container` takes a directory whose root
holds a `Dockerfile` and a `container.yaml`. The directory is the whole build context, and
what runs is what your Dockerfile describes.

**`container.yaml` is the contract.** Nothing is discovered by inspection, because a builder
cannot look inside an image and find the routes. You name the port, the probes, the request
budget and every endpoint the platform may route to.

**The platform recognizes a vLLM image and serves it its own way.** vLLM's release pipeline
labels every `vllm/vllm-openai` image, and the builder reads that label off the image your
final stage builds on. When it is there, the platform installs NVIDIA Dynamo beside the
image's own vLLM and puts its launcher first on `PATH` as `vllm`, so the `vllm serve` in
your `CMD` starts a Dynamo worker and a Dynamo frontend on the port you named. Your image
is not modified and its packages are not touched.

**That path accepts one range of vLLM releases.** It is 0.28.x today. A vLLM outside it
fails the build with the reason in the log rather than starting and faulting later, and
an image of vLLM-Omni, which is a different server carrying the same label, is built as
you wrote it.

**Two flags are added for you.** `--enable-sleep-mode`, which is what lets the platform
move the model off the card between calls and put it back at the same addresses, and
`--max-num-seqs` when your app sets `MAX_CONCURRENCY`. A `--max-num-seqs` of your own is
left alone.

**An endpoint path can be several segments.** `path: v1/chat/completions` is both what
callers invoke and where the request lands, so vLLM's own route needs no translation. The
`target` key exists for the case where the two differ, and this example does not need it.

**Pick a port the platform does not already hold.** It binds 8000, 8888, 8889 and 9090
inside the pod, and 8000 is vLLM's default. A server left on one of those loses the bind to
whatever got there first.

**Readiness and liveness are separate routes you choose.** `/health` answers once a server
that can take requests is up, and `/live` reports the process. Liveness is only checked
after the startup probe has passed, which is what keeps a load of several minutes from
being mistaken for a dead process.

**What your server returns is the result.** A code app's answer arrives wrapped at
`output.predictions[0]`, and a container app's does not.

**A declared `input` is enforced before your container is called.** A request with no
`messages` comes back `422` with a pointer at the field and never reaches the worker. Leave
`input` out of an endpoint and the payload is forwarded as it stands.

## Deploying the app

```bash
runware serverless deploy --container . \
  --id container-server \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

`HF_HOME` is set to `/weights` in the Dockerfile, which is the path the volume backs.
Without it the weights land on the sandbox filesystem and are fetched again on every cold
start.

## Calling the app

```bash
curl https://api.serverless.runware.ai/v1/apps/container-server/invoke-sync/v1/chat/completions \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"model": "Qwen/Qwen3-4B-Instruct-2507", "messages": [{"role": "user", "content": "Name three uses for a serverless GPU."}]}}'
```

The body is an ordinary OpenAI chat-completions request and the reply is the server's own,
at `output.choices[0].message.content`.

The `input` block in `container.yaml` is closed, so a field it does not name is refused at
the queue rather than forwarded. Streaming is refused the same way. An invocation answers
once, so there is no response to stream into.

## Testing the app

There is nothing here to run. The app is an image and a config document, and both are
judged by the platform at deploy: an invalid `container.yaml` is refused on the spot, with
a pointer into the document.

A server that is not an HTTP server at all, or one that needs a second process beside it,
needs a small front of your own. Then the front is code, and it is testable like any other.
