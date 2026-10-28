# Typed requests and responses

What your callers see. The platform builds your app's documentation out of your
handler signature, so that signature is your contract, and plain parameters
publish almost none of it.

The work here is small on purpose, reading an image's metadata, so the request and
the response are the subject.

## What it teaches

**Your return annotation is the response schema.** The build reads nothing else, so
what you annotate is what your callers are told. Building the manifest for each of
these gives:

| `predict` returns | published response |
| --- | --- |
| `dict[str, str]` | `{"type": "object"}`, and nothing about what is in it |
| a framework object, such as a pipeline output | `{}`, meaning anything at all |
| a pydantic model | every field, with its type, its prose and its nesting |

**A pydantic request model carries what a parameter cannot.** An optional parameter
exports its type and nothing else, so no default and no description.

That costs you more than documentation. The platform fills an omitted optional from
the published schema, so a parameter whose default it never saw reaches you as an
explicit `null` and you apply the default a second time:

```python
def generate(self, prompt: str, steps: int = 4) -> dict[str, object]:
    steps = 4 if steps is None else steps
```

Put it on a model field and the default is published, so the platform sends it.
`palette` arrives here as `False` and you re-apply nothing.

**Nested models and enums survive.** A model holding another model publishes both,
and an enum publishes its values, so `orientation` documents itself as one of three
known strings instead of as a string.

**`@serve(encode_response=...)` is the escape hatch, and it costs you.** It takes
your handler results and returns whatever the app should answer, so a worker whose
wire format cannot be its return type has a seam to use. Know four things first:

- Your published response is still the return annotation, so an encoder that
  reshapes the answer leaves your documentation describing something else.
- It receives a **list**, one entry per row. A single request is a batch of one.
- It belongs to your app, not to an endpoint, so every endpoint answers through the
  same encoder.
- Define the function **above** the class. A decorator's arguments run when the
  class statement does.

Return a model instead and you avoid all four, which is why this example does.

## Deploying it

```bash
runware serverless deploy model.py --id typed-endpoints --gpu-type rtx-pro-6000
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

The trade: typed fields in, documented fields out, one extra level in the body.

Omitting `palette` entirely answers the same way, with an empty palette rather
than a validation error. **That is the published default arriving as `False`.** A
`null` would not validate against `bool` at all, which is how you can tell the two
cases apart from the outside.

Measured on an RTX PRO 6000: the first call takes about **1m45s** and comes
back `202` with the task pending, because a sync invoke has a deadline of roughly
105 seconds and the worker is still starting. Poll
`/v1/apps/typed-endpoints/tasks/{taskId}` for it. The next call answers `200` in
**2 seconds**.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

Nothing here installs the app's own requirements, which is the position the build
is in when it imports the file to read the endpoints out of it. The tests assert
the published schemas directly, since those are what the app promises.
