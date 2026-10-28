# `config-and-secrets`

What a worker can see of its own configuration, and what it must never answer
with. No weights, so the subject is the configuration itself.

## What it teaches

**A secret and a plain variable arrive the same way.** Both land in `os.environ`,
and nothing in the code can tell them apart. `API_TOKEN` here comes from a secret
and `LOG_LEVEL` from `--env-var`, and `model.py` reads both with the same line.

That is why **they share one namespace**. A plain variable whose name collides
with an attached secret's injected name is rejected, and so is the reverse.
Neither silently wins, which is the right call: a credential quietly overridden by
a plain variable would slip past everyone.

**The value must not leave the worker.** A handler that returned it would publish a
credential to every caller and into the task record, where it outlives the request.
The response here carries presence and nothing else, and a test asserts the value
cannot appear in it.

So an app cannot tell you whether a secret holds the *right* value. That is
checked on the attachment, not from inside:

```
apps env-vars list     CACHE_DIR = /weights      the value is printed
attachments            DOCS_EXAMPLE_TOKEN -> API_TOKEN    only the names
```

**That asymmetry is the practical difference.** A plain variable's value comes back
from the API, a secret's never does, and an attachment is recorded as an audit
event on the app.

**The configuration is read once, in `load`.** Changing a variable or an
attachment records a new version and rolls the app, so the worker holding an old
value is replaced by one that read the new set. A handler re-reading `os.environ`
on every request is guarding against something that cannot happen.

**A secret's stored name need not be the name your code reads.** The attachment
decides which stored secret fills a variable, so the same code runs against a
production and a staging credential without an edit. This app always reads
`API_TOKEN`.

**Nothing is declared in `requirements`.** Pydantic is already on the worker,
because the platform's own libraries bring it.

## Deploying it

```bash
runware serverless deploy model.py \
  --id config-and-secrets \
  --gpu-type rtx-pro-6000 \
  --env-var LOG_LEVEL=info \
  --env-var CACHE_DIR=/weights
```

A value passed as `--env-var` is visible in the process list and recorded, so
anything that would hurt if it leaked belongs in a secret. `--env-file` reads
`KEY=VALUE` lines instead.

Then the secret, which is two steps because one secret usually serves several apps:

```bash
printf '%s' "$TOKEN" | runware serverless secrets set DOCS_EXAMPLE_TOKEN --value-file -
runware serverless secrets attach config-and-secrets DOCS_EXAMPLE_TOKEN --env-var-name API_TOKEN
```

`--value-file -` reads from stdin so the value never reaches the process list.
Attaching rolls the live app, so the next call sees it.

## Calling it

```bash
curl https://api.serverless.runware.ai/v1/apps/config-and-secrets/invoke-sync/predict \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {}}'
```

Before the attach, `API_TOKEN` comes back under `missing`. After it, `present`.
The roll between the two is the whole mechanism, visible from outside.

Measured on an RTX PRO 6000, with the secret attached:

```json
{
  "bindings": [
    { "name": "LOG_LEVEL", "present": true },
    { "name": "CACHE_DIR", "present": true },
    { "name": "API_TOKEN", "present": true }
  ],
  "missing": []
}
```

`API_TOKEN` is `present` while the stored secret is named `DOCS_EXAMPLE_TOKEN`,
which is the indirection doing its job.

The first call answered in **56 seconds**, under the sync deadline, so there is no
`202` to poll here. An app that declares no dependencies has a small image to pull,
and the difference against an example carrying torch is minutes.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

The tests set and unset the names themselves, so they cover the three cases a
deploy cannot: a name present, a name absent, and a value that must not reach the
response.
