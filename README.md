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
runware serverless deploy model.py --id echo-app --gpu-type rtx-pro-6000
```

You need the [Runware CLI](https://runware.ai/docs/tools/cli) and an API key. The
authoring package, [`runware-serverless`](https://pypi.org/project/runware-serverless/),
is **installed by the platform**, so your app does not list it.

## The examples

| Example | What it teaches |
| --- | --- |
| [`echo`](echo/) | Deploy an app that loads nothing, to test the platform end to end or tell whether a failure comes from your model |
| [`image-tools`](image-tools/) | Serve two endpoints with different request schemas from one app, each routed by its path |
| [`loading-weights`](loading-weights/) | Run a text-to-image model on a GPU and keep its weights on a volume across restarts |
| [`typed-endpoints`](typed-endpoints/) | Type your handler parameters with Pydantic models and see the request schema the platform builds from them |
| [`watermark`](watermark/) | Accept an image and return one, letting the package's annotations decode the base64 before your handler runs |
| [`multi-gpu`](multi-gpu/) | Run one worker across several GPUs, with one serving process per GPU calling your handler |
| [`config-and-secrets`](config-and-secrets/) | Pass configuration and credentials to a worker as environment variables, and check which ones arrived |
| [`chained-steps`](chained-steps/) | Generate a product shot and cut it out of its background in one call, with both models on one worker |
| [`chat-endpoint`](chat-endpoint/) | Serve chat completions from Qwen3 4B Instruct and decide the request shape yourself |
| [`embeddings`](embeddings/) | Embed a list of texts in one request and get back one vector per text, in the order you sent them |
| [`compiled-model`](compiled-model/) | Compile an image classifier with torch.compile and keep the compiler's cache on a volume for later workers |
| [`resumable-job`](resumable-job/) | Split a long job into short calls that resume from the checkpoint the previous call saved |
| [`protein-folding`](protein-folding/) | Predict a protein's structure from its sequence with ESMFold and get it back as a PDB file |
| [`transcription`](transcription/) | Transcribe an audio file from a URL with Parakeet TDT 0.6B v3, and get the text with a start and end for each word |
| [`render-farm`](render-farm/) | Render a Blender frame on a GPU from a container that installs Blender and the system libraries it needs |
| [`container-server`](container-server/) | Deploy the vLLM image its maintainers publish, unchanged, and let the platform serve it with NVIDIA Dynamo |
| [`sglang-server`](sglang-server/) | Deploy the SGLang image its maintainers publish, and let the endpoints you declare settle that it is serving a language model |
| [`comfyui-workflow`](comfyui-workflow/) | Run ComfyUI headless and call it with a prompt or with a whole workflow exported from your own ComfyUI |

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

Drop in a directory with `model.py`, `test_model.py`, a `README.md` and a `runware.yaml`. The
README is **what makes the example usable without leaving the repository**, so it carries the
deploy command and a call for every endpoint. The `runware.yaml` holds the same deploy options
the README passes as flags, so a client can offer the deploy without a person reading them
first.

`python3 check_examples.py` refuses an example that is missing any of them, and CI runs it. It
also refuses a `runware.yaml` that disagrees with the flags its README passes.

## What earns a place here

An example belongs here when it **teaches something none of the others do**: a shape of
work, or a constraint that is easy to get wrong. **Two examples that differ only in which
model they load are one example.**

Serverless is for **the work our [models API](https://runware.ai/docs/models-api/introduction)
does not cover**: your own weights, a model we do not host, a pipeline with your logic in
it, an unusual compute shape. An example built around a model we already serve is welcome
when **the point is what you wrapped around it**, rather than the model itself.
