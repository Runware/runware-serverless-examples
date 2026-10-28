# Taking an image as input

How an image reaches your code and how it leaves, which every app that takes one
has in common, whether it upscales, edits or runs a ControlNet pass.

**The work here is a stand-in.** Laying a mark over an image is pixel arithmetic on
the CPU, and nothing in `model.py` touches the GPU or loads a weight. You would not
deploy this app. It is the smallest thing that needs a real image on both sides, so
the image can be the whole subject. Every app gets a GPU because `--gpu-type` is
required to create one, and this example leaves it idle.

## What it teaches

**The annotation decodes the image for you.** `Base64RGBImage` and
`Base64RGBAImage` turn the base64 string your caller sent into a `PIL.Image` before
your handler runs, so `layers.base` is already pixels. The field publishes itself as
a described base64 string, so your callers know what to send:

```json
{ "type": "string", "format": "base64", "description": "Image in base64 format" }
```

**Put them on a field of a model, never on a parameter.** Coercion looks for a
Pydantic model to validate the parameter against, and these are images, not models,
so a bare `def predict(self, image: Base64RGBImage)` is never coerced and your
handler gets the string exactly as it arrived.

Your published contract goes with it. The same annotation, on a bare parameter and
on a field:

| | published | your handler gets |
| --- | --- | --- |
| `image: Base64RGBImage` | `{}` | `str` |
| the same type on a field | a described base64 string | a `PIL.Image` |

Nothing fails either way. One of them just answers a different question than the one
it was asked, and tells your callers nothing while it does it.

**The names describe the conversion, not the result.** `Base64RGBImage` converts to
RGB, so anything becomes three channels and the `format` goes with the conversion.
`Base64RGBAImage` converts **nothing**: a grayscale PNG stays grayscale, and an
image with no alpha channel does not gain one. Reach for it when you want a mark's
transparency kept, and convert the mark yourself, the way this handler does.

**They have nothing to say on the way out.** Both refuse to serialize, so a response
field annotated with one sends a placeholder instead of your image. That is why
`Stamped.image` is a plain string the handler encodes. **Coercion is a request-side
feature.**

**The app declares Pillow.** The annotations need it at serving time, and pinning it
means your app behaves the same wherever it runs. Nothing in `model.py` imports
Pillow, because all it does with the library is call methods on the images the
annotations handed it.

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
declared opacity, not at nothing. **Each of those defaults is published, so
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
