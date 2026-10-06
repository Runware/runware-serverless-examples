# Echo

The smallest app that works. No weights and no GPU work, so **a failure in it is the platform
rather than your model**. Reach for it to prove the loop end to end, or when something else is
failing and you need to know which side the problem is on.

Two endpoints. `echo` hands the message back with its length, and `reverse_message` reverses
it and can repeat the result.

## Deploying it

```bash
runware serverless deploy model.py --id echo --gpu-type l40s
```

## Calling it

A handler answers on the app id and its own name, and the body is an envelope whose `payload`
holds the handler's arguments.

```bash
curl https://api.serverless.runware.ai/v1/apps/echo/invoke-sync/echo \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"message": "hello"}}'
```

Underscores in a handler's name **become hyphens in its path**, so `reverse_message` answers
on `/reverse-message`.

```bash
curl https://api.serverless.runware.ai/v1/apps/echo/invoke-sync/reverse-message \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "1b4e28ba-2fa1-11d2-883f-0016d3cca427", "payload": {"message": "hello", "repeat": 2}}'
```

## Testing it

```bash
uv run --no-project --with pytest --with runware-serverless pytest
```

`serve` and `endpoint` **hand the class and its methods back unchanged**, so this is tested
like any other Python: import the module, make an instance, call the method.
