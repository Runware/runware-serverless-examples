# Running a language model

A language model behind an endpoint, and the shape of that endpoint is yours. No
weights of the image kind, so what the example is about is the request.

Qwen3 4B Instruct is Apache 2.0 and ungated, so this deploys with **no token of
any kind**.

## What it teaches

**A language model is weights and a handler.** Nothing about it is special to the
platform: you load it in `load`, you keep it on a volume, and you answer requests
from it. Everything `loading-weights` says applies unchanged.

**There is no standard chat endpoint here.** `messages`, `max_tokens` and
`temperature` look familiar because this file wrote them that way. The platform
publishes whatever your signature declares, so if you want an OpenAI-shaped body
you write one, and if you want three fields of your own you write those. Nothing
is translating between you and the caller.

The roles publish as an enum, so a fourth role is refused at the boundary and
never reaches your worker.

**The prompt template belongs to the model.** `apply_chat_template` reads the turn
markers the tokenizer ships with, and those are not yours to guess at. A prompt
assembled by hand is how an instruct model ends up answering nothing.

**`generate` serves one request at a time.** There is no continuous batching here
and no paged attention, so two callers queue behind each other for the whole
length of a generation. That is fine for a single consumer or an internal tool and
wrong for real traffic.

A server built for that work, vLLM or its equivalents, brings **its own HTTP
server**, which is a container deploy rather than this one. This example is the
honest floor: the smallest thing that answers, and the place where you can see why
you would move.

## Deploying it

```bash
runware serverless deploy model.py \
  --id chat-endpoint \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

## Calling it

```bash
curl https://api.serverless.runware.ai/v1/apps/chat-endpoint/invoke-sync/predict \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"request": {"messages": [{"role": "user", "content": "Name three uses for a serverless GPU."}]}}}'
```

The reply comes back at `output.predictions[0]`, with `content` and the number of
tokens generated.

Send the whole conversation each time. The worker keeps nothing between requests,
and a handler that did would be holding state a second worker does not have.

Measured on an RTX PRO 6000, three different things:

| | |
| --- | --- |
| First call, empty volume | **5m48s** |
| A restart, weights already on the volume | 56 seconds |
| A worker already serving | 13 seconds |

The volume does not keep a worker warm. It makes a **restart** cheap, which is
the gap between the first row and the second.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

Neither torch nor transformers is present here, the same position the build is in.
What the tests reach is the published request: the roles, the bounds, and that a
conversation validates into `Message` objects before your handler sees it.
