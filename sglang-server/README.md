# Running an SGLang server

SGLang, deployed as it is. The image is the one its maintainers publish, the server is
the one they wrote, and the whole app is a `Dockerfile` with no `COPY` in it and a
`container.yaml` beside it.

Qwen3 4B Instruct is Apache 2.0 and ungated, so this deploys with **no token**.

## What the example teaches

**The endpoints you declare decide what the platform does with the image.** The same
`lmsysorg/sglang` image serves language models and, as SGLang Diffusion, image and video
models, with one command, and which one it runs is settled when the command runs. Nothing
the build can read says which. So the platform goes by your contract: every endpoint in
`container.yaml` being a language model's route is what makes this a language model's
image. Declare an image or a video route instead and the image runs exactly as you wrote
it, with no serving layer attached. This is the one thing an SGLang image has to get right
that a vLLM image does not.

**The platform recognizes an SGLang image and serves it its own way.** SGLang's release
pipeline labels every image it publishes, the builder reads that label off the image your
final stage builds on, and when the endpoints agree it installs NVIDIA Dynamo beside the
image's own SGLang and puts its launcher first on `PATH` as `sglang`. The `sglang serve`
in your entrypoint then starts a Dynamo worker and a Dynamo frontend on the port you
named. Your image is not modified and its packages are not touched.

**One SGLang release is served at a time.** The platform serves the exact release the
Dynamo it installs pins, because SGLang changes internals that Dynamo's worker reads from
one patch release to the next. A release it does not serve fails the build with the reason
in the log rather than starting and faulting later. That is stricter than the vLLM path,
which serves whole minor lines.

**`--enable-memory-saver` is added for you.** What it allocates is the only memory the
platform can move off the card between calls and bring back at the same addresses, which
is what lets the app scale to zero and come back warm. `--enable-metrics` arrives with it,
for the engine's own numbers.

**Your `--max-running-requests` is read, not replaced.** It is both SGLang's own limit on
how many requests the engine runs at once and how many the platform sends one worker, so
it is worth setting deliberately. A command without it is given 256. This is where the two
engines differ in practice: vLLM's `--max-num-seqs` is set for you from `MAX_CONCURRENCY`.

**30000 is SGLang's default and the platform does not hold it.** It binds 8000, 8888, 8889
and 9090 inside the pod, and none of them is SGLang's, so this Dockerfile has no port to
move. A vLLM image does.

**A declared `input` is enforced before your container is called.** A request with no
`messages` comes back `422` with a pointer at the field and never reaches the worker. Leave
`input` out of an endpoint and the payload is forwarded as it stands.

## Deploying the app

```bash
runware serverless deploy --container . \
  --id sglang-server \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

`HF_HOME` is set to `/weights` in the Dockerfile, which is the path the volume backs.
Without it the weights land on the sandbox filesystem and are fetched again on every cold
start.

## Calling the app

```bash
curl https://api.serverless.runware.ai/v1/apps/sglang-server/invoke-sync/v1/chat/completions \
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
