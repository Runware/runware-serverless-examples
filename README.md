# Runware Serverless examples

Apps you can deploy as they are. Each one is the shortest way to learn something **the
others do not teach**, and **each explains itself**: open its directory and the README there
says what it teaches, how to deploy it, how to call it and how to test it.

These are also the code behind a page in
[the Serverless documentation](https://runware.ai/docs/serverless/examples). **The page
renders these files**, so what you read there is what runs here.

```bash
git clone https://github.com/Runware/runware-serverless-examples
cd runware-serverless-examples/echo
runware serverless deploy model.py --id echo --gpu-type l40s
```

You need the [Runware CLI](https://runware.ai/docs/tools/cli) and an API key. The
authoring package, [`runware-serverless`](https://pypi.org/project/runware-serverless/),
is **installed by the platform**, so your app does not list it.

## The examples

| Example | What it teaches |
| --- | --- |
| [`echo`](echo/) | The smallest app that works. No weights and no GPU work, so a failure in it is the platform rather than your model |
| [`image-tools`](image-tools/) | One app whose endpoints take unrelated request shapes, and where the GPU choice actually lives |

Each one's README carries its own deploy command and the calls for its own endpoints.

## Running the tests

`serve` and `endpoint` **hand the class and its methods back unchanged**, so an example
is tested like any other Python: import the module, make an instance, call the method.
Nothing to mock and no test client to stand up.

```bash
cd echo
uv run --no-project --with pytest --with runware-serverless pytest
```

Every example is tested in CI the same way, **from inside its own directory**, because
each one is an independent project and they all hold a module named `model`. One
pytest run cannot import two of them.

## Where the weights go

Neither example loads any, so neither shows the thing that matters most for a real model:
**what you download in `load` should land on a volume**, not on the sandbox filesystem.

```python
import os

_CACHE = "/root/.cache/huggingface"


@serve
class MyModel:
    def load(self) -> None:
        if hasattr(self, "pipeline"):
            return
        self.pipeline = SomePipeline.from_pretrained(REPO, cache_dir=_CACHE).to("cuda")
```

```bash
runware serverless deploy model.py --id my-model --gpu-type h100 \
  --volume /root/.cache/huggingface
```

Two things are doing work there. The **volume** keeps the weights across cold starts, so
you are not paying GPU time to fetch them again. The **`hasattr` guard** makes `load` safe
to run twice, which it has to be: it is retried if the GPU runs out of memory at startup.

See [Volumes](https://runware.ai/docs/serverless/volumes) and
[Writing a model](https://runware.ai/docs/serverless/writing-a-model).

## Adding one

Drop in a directory with `model.py`, `test_model.py` and a `README.md`. The README is what
makes the example usable without leaving the repository, so it carries the deploy command
and a call for every endpoint.

`python3 check_examples.py` refuses an example that is missing any of them, and CI runs it.

## What earns a place here

An example belongs here when it **teaches something none of the others do**: a shape of
work, or a constraint that is easy to get wrong. **Two examples that differ only in which
model they load are one example.**

Serverless is for **the work our [models API](https://runware.ai/docs/models-api/introduction)
does not cover**: your own weights, a model we do not host, a pipeline with your logic in
it, an unusual compute shape. An example built around a model we already serve is welcome
when **the point is what you wrapped around it**, rather than the model itself.
