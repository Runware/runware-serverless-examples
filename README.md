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
runware serverless deploy model.py --id echo --gpu-type rtx-pro-6000
```

You need the [Runware CLI](https://runware.ai/docs/tools/cli) and an API key. The
authoring package, [`runware-serverless`](https://pypi.org/project/runware-serverless/),
is **installed by the platform**, so your app does not list it.

## The examples

| Example | What it teaches |
| --- | --- |
| [`echo`](echo/) | The smallest app that runs: two endpoints, no weights, no GPU work. Deploy it to check the platform end to end |
| [`image-tools`](image-tools/) | Two endpoints with different request shapes in one app, and where the GPU choice actually lives |
| [`loading-weights`](loading-weights/) | Load a real model on a GPU: where the imports go, how to declare dependencies, and why weights belong on a volume |
| [`typed-endpoints`](typed-endpoints/) | Describe your API with Pydantic so callers see every field, and so defaults reach your handler instead of nulls |
| [`watermark`](watermark/) | Accept an image and return one, with the annotations that decode base64 into pixels before your handler runs |
| [`multi-gpu`](multi-gpu/) | Run a worker across several cards, where every rank executes your handler and only the first one answers |
| [`config-and-secrets`](config-and-secrets/) | Read configuration and credentials on a worker, and what the platform will and will not give you back |
| [`chained-steps`](chained-steps/) | Run two models in one request so the image between them never leaves the worker |
| [`chat-endpoint`](chat-endpoint/) | Serve chat completions from your own weights, with a request shape you define |

Each one's README carries its own deploy command and the calls for its own endpoints.

## Running the tests

`serve` and `endpoint` **hand the class and its methods back unchanged**, so an example
is tested like any other Python: import the module, make an instance, call the method.
Nothing to mock and no test client to stand up.

```bash
cd echo
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest
```

Every example is tested in CI the same way, **from inside its own directory**, because
each one is an independent project and they all hold a module named `model`. One
pytest run cannot import two of them.

## Adding one

Drop in a directory with `model.py`, `test_model.py` and a `README.md`. The README is **what
makes the example usable without leaving the repository**, so it carries the deploy command
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
