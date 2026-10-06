# Two endpoints, two shapes

`generate` takes a prompt and `upscale` takes an image, so **no single request schema covers
both**. Each handler's signature becomes its own, and the platform routes by path.

What they share is the app: one queue, one worker pool, one GPU type. That is set on the app
rather than in the file, which is why nothing in `model.py` names hardware or scaling. **Two
endpoints that genuinely need different GPUs are two apps.**

Neither loads weights, and the pixels are a fixed 1x1 PNG standing in for the real thing, so
this runs anywhere.

## Deploying it

```bash
runware serverless deploy model.py --id image-tools --gpu-type l40s
```

## Calling it

Both endpoints are on the same app, so **only the last segment of the path changes**.

```bash
curl https://api.serverless.runware.ai/v1/apps/image-tools/invoke-async/generate \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"prompt": "a red bicycle"}}'
```

`upscale` takes an image instead, and nothing in its body carries over from the call above.
`steps` and `scale` both have defaults in their signature, so you can leave them out. An
omitted optional field **arrives as an explicit null** rather than as a missing key, which is
why each handler applies its default a second time.

## Testing it

```bash
uv run --no-project --with pytest --with runware-serverless pytest
```
