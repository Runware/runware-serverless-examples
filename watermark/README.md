# `watermark`

Lays a mark over an image, which is the smallest honest reason to want real
pixels in a handler. No weights, so the subject stays the way an image gets in
and out.

## What it teaches

**The annotation is what decodes the image.** `Base64RGBImage` and
`Base64RGBAImage` turn the base64 string a caller sent into a `PIL.Image` before
the handler runs, so `layers.base` is already pixels. The field publishes itself
as a described base64 string, so callers are told what to send:

```json
{ "type": "string", "format": "base64", "description": "Image in base64 format" }
```

**They only work on a field of a model.** Coercion looks for a Pydantic model to
validate the parameter against, and these are images rather than models, so a bare
`def predict(self, image: Base64RGBImage)` is never coerced and the handler
receives the string exactly as it arrived.

The published contract goes with it. Measured, the same annotation on a bare
parameter and on a field:

| | published | the handler gets |
| --- | --- | --- |
| `image: Base64RGBImage` | `{}` | `str` |
| the same type on a field | a described base64 string | a `PIL.Image` |

Nothing fails either way. One of them just answers a different question than the
one it was asked, and tells callers nothing while it does it.

**The names describe the conversion, not the result.** `Base64RGBImage` converts
to RGB, so anything becomes three channels and `format` is gone with the
conversion. `Base64RGBAImage` converts **nothing**: a grayscale PNG stays
grayscale, and an image without an alpha channel does not gain one. It is the
right choice for a mark whose transparency you want kept, and it is why this
handler converts the mark itself before compositing.

**They are input-only.** On a response field they refuse to serialize and a
placeholder goes out in their place, so `Stamped.image` is a plain string the
handler encodes. The annotation has nothing to say about an image on the way out.

**The app declares Pillow.** The annotations need it at serving time, and pinning
it keeps the app's behavior its own rather than whatever the runtime happens to
carry. Nothing in `model.py` imports Pillow, because the only thing it does with
the library is call methods on the images the annotations handed it.

## Deploying it

```bash
runware serverless deploy model.py --id watermark --gpu-type rtx-pro-6000
```

## Calling it

```bash
curl https://api.serverless.runware.ai/v1/apps/watermark/invoke-sync/predict \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"layers": {"base": "'"$(base64 -i photo.jpg)"'", "mark": "'"$(base64 -i logo.png)"'", "corner": "bottom-right", "opacity": 0.5}}}'
```

The class declares no `@endpoint`, so `@serve` exposes `predict` at the default
path. The result is a base64 PNG at `output.predictions[0].image`, the same size as
the base.

Sending only `base` and `mark` works, and the mark lands bottom right at the
declared opacity rather than at nothing. **Each of those defaults is published, so
the platform sends it**, and a `null` would not have validated against `Corner`,
`float` or `int` at all.

Measured on an RTX PRO 6000: the first call takes about **1m45s** and answers
`202` with the task pending, because a sync invoke has a deadline of roughly 105
seconds and the worker is still starting. Poll
`/v1/apps/watermark/tasks/{taskId}` for it. The next call answers `200` in
**2 seconds**.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

The extra is what carries Pillow and Pydantic, since the base install of the
package has no dependencies at all. The tests run the handler on real images,
because here the app's own dependency is one the annotations already required.
