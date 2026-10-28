# Running a language model

Chat completions from your own weights, with a request shape you decide.

Qwen3 4B Instruct is Apache 2.0 and ungated, so you deploy it with **no token**.

## What it teaches

**A language model is weights and a handler.** Nothing about it is special to the
platform. You load it in `load`, you keep it on a volume, and you answer requests
from it.

**You decide the request shape.** `messages`, `max_tokens` and `temperature` look
familiar because this file wrote them that way, and the platform publishes whatever
your signature declares. Want an OpenAI-shaped body, write one. Want three fields of
your own, write those. Nothing translates between you and your caller.

The roles publish as an enum, so a fourth role is refused at the boundary and never
reaches your worker.

**Use the model's own prompt template.** `apply_chat_template` reads the turn
markers the tokenizer ships with, and you should not guess at those. Assemble the
prompt by hand and an instruct model will answer you with nothing.

**`generate` serves one request at a time.** You get no continuous batching and no
paged attention, so two callers queue behind each other for the whole length of a
generation. That is fine for a single consumer or an internal tool, and wrong for
real traffic.

A server built for that work brings **its own HTTP server**, which makes it a
container deploy instead of this one. Start here, and move when you can see the
queue.

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
