# `typed-endpoints`

What a caller of your app sees. The platform builds the per-app documentation out
of your handler's signature, so the signature is the contract, and a handler
written with plain parameters publishes almost nothing.

The work here is deliberately small, reading an image's metadata, so the shape of
the request and the response is the whole subject.

## What it teaches

**The response schema is the return annotation.** Nothing else is read, so what
you annotate is what callers are told. Measured by building the manifest for each
of these:

| `predict` returns | published response |
| --- | --- |
| `dict[str, str]` | `{"type": "object"}`, and nothing about what is in it |
| a framework object, such as a pipeline output | `{}`, meaning anything at all |
| a pydantic model | every field, with its type, its prose and its nesting |

**A pydantic request model carries what a parameter cannot.** An optional
parameter exports its type and nothing else. No default, and no description.

That is not only a documentation gap. The platform fills an omitted optional from
the published schema, so a parameter whose default it never saw arrives as an
explicit `null`, and the handler has to apply the default a second time:

```python
def generate(self, prompt: str, steps: int = 4) -> dict[str, object]:
    steps = 4 if steps is None else steps
```

A field on a model publishes its default, which is then the one the platform
sends. `palette` in this app arrives as `False`, and there is nothing to re-apply.

**Nested models and enums survive.** A model that holds another model publishes
both, and an enum publishes its values, so `orientation` documents itself as one
of three strings rather than as a string.

**`@serve(encode_response=...)` is the escape hatch, and it has a price.** It
takes the handler results and returns whatever the app should answer, which is the
seam for a worker whose wire format cannot be its return type. What to know
before reaching for it:

- The published response is still the return annotation, so an encoder that
  reshapes the answer leaves the documentation describing something else.
- It receives a **list**, one entry per row. A single request is a batch of one.
- It belongs to the app, not to an endpoint, so every endpoint answers through
  the same encoder.
- The function has to be defined **above** the class. A decorator's arguments are
  evaluated when the class statement runs.

Returning a model instead avoids all four, which is why this example does not use
it.

## Deploying it

```bash
runware serverless deploy model.py --id typed-endpoints --gpu-type l40s
```

No weights and no GPU work, so the hardware only has to exist. Every app is
allocated a GPU, which is why one is named here at all.

## Calling it

The parameter is a model, so its fields sit under the parameter's own name:

```bash
curl https://api.serverless.runware.ai/v1/apps/typed-endpoints/invoke-sync/describe \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"source": {"image": "'"$(base64 -i picture.png)"'", "palette": true}}}'
```

The report comes back at `output.predictions[0]`:

```json
{
  "format": "PNG",
  "size": { "width": 640, "height": 360 },
  "orientation": "landscape",
  "bytes": 1499,
  "palette": [{ "red": 200, "green": 30, "blue": 30 }]
}
```

That is the trade: typed fields in, documented fields out, one extra level in the
body.

Omitting `palette` entirely answers the same way, with an empty palette rather
than a validation error. **That is the published default arriving as `False`.** A
`null` would not validate against `bool` at all, which is how you can tell the two
cases apart from the outside.

Measured on an l40s: the first call takes about **1m45s** and comes back `202`
with the task pending, because a sync invoke has a deadline and the worker is
still starting. Poll `/v1/apps/typed-endpoints/tasks/{taskId}` for it. The next
call answers `200` in **8 seconds**, of which a little over a second is the
handler.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

Nothing here installs the app's own requirements, which is the position the build
is in when it imports the file to read the endpoints out of it. The tests assert
the published schemas directly, since those are what the app promises.
