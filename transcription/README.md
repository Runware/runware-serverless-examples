# Transcribing speech

An audio file in, the text out with a span per word. Parakeet TDT 0.6B v3 on a
GPU, with the weights on a volume.

The model is CC-BY-4.0 and ungated, so this deploys with **no token**. The
attribution the license asks for falls on whoever deploys it.

## What the example teaches

**The audio arrives as a URL and the worker fetches it.** Every other example
that takes media takes base64. An hour of speech is tens of megabytes inside a
JSON body, and the file already lives somewhere reachable, so this app takes the
address instead. A worker can reach the network, which is the same thing the
resumable job relies on for its checkpoints.

**Chunk the audio.** Attention over a whole file is what runs a worker out of
memory, so this sends 300 seconds at a time. The cost is the boundary: a word
spoken across one comes back split in two.

**Set `max_new_tokens` yourself.** Generation otherwise stops at a
model-agnostic 1310 and a long chunk comes back **quietly cut off**, with no
error anywhere. Measured at about six tokens a second on ordinary speech, so a
five-minute chunk needs more than the default.

**The timestamps arrive per subword piece, not per word.** The transformers path
returns one span per piece:

```python
[{'token': 'He', 'start': 0.32, 'end': 0.48},
 {'token': ' h', 'start': 0.48, 'end': 0.64},
 {'token': 'op', 'start': 0.64, 'end': 0.8},
 {'token': 'ed', 'start': 0.8, 'end': 0.96}]
```

NeMo's own API hands you word and segment spans. This one hands you pieces, and
a piece that opens with a space opens a word. That grouping is nine lines and
the only real logic in the file, which is why the tests cover it and nothing
else.

**The feature extractor needs librosa, and nothing in this file imports it.**
`ParakeetFeatureExtractor` does, and the failure arrives on a worker as an
`ImportError` rather than at build time, because the build reads your endpoints
without installing anything. A worker that cannot start leaves the call
**pending with nothing in `apps errors`**, so the traceback is only in
`apps logs`. librosa brings scipy, scikit-learn, numba and
llvmlite with it, which is 56 packages for a 0.6B model. All of them ship
wheels, and that is what keeps this on the code path: a code worker has no C
compiler, so a dependency that builds at install time would force a container.

**Pin the weights to a revision.** `from_pretrained` takes `revision`, and
without it you are asking for whatever `main` holds today.

## Which model

Three models fit this job, and the choice is not the accuracy table.

| Model | License | Languages | Timestamps |
| --- | --- | --- | --- |
| `nvidia/parakeet-tdt-0.6b-v3` | CC-BY-4.0 | 25 European | per piece, and the throughput leader by a wide margin |
| `openai/whisper-large-v3` | Apache 2.0 | 99 | per segment, per word needs a separate alignment pass |
| `ibm-granite/granite-speech-4.1-2b` | Apache 2.0 | 6 | none, they are in the `-plus` variant |

Granite leads the English word error rate of the three and returns text only.
Whisper is the one to reach for when you need a language outside Parakeet's 25.
This app takes Parakeet because 0.6B keeps the cold start short and the spans
come back without a second model.

## Deploying the app

```bash
runware serverless deploy model.py \
  --id transcription \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

The weights are about 2.4GB. Mount the volume: a checkpoint captures the
worker's filesystem, so an unmounted download is copied into every one.

## Calling the app

Use `invoke-async` and poll. An hour of audio is not a synchronous request.

```bash
curl https://api.serverless.runware.ai/v1/apps/transcription/invoke-async \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"request": {"audio_url": "https://example.com/interview.flac"}}}'
```

The class declares no `@endpoint`, so `@serve` exposes `predict` at the default
path. The parameter is a model, so its fields sit under the parameter's own name.

The transcript comes back at `output.predictions[0]`: `text` for the whole
thing, `words` for the spans, `seconds` for the length of the audio rather than
of the request, and `chunks` for how many passes it took. Send
`"timestamps": false` and `words` comes back empty, which is the cheaper answer
when all you want is the text.

Measured on an RTX PRO 6000:

| Call | Audio | Task |
| --- | --- | --- |
| Build | | 2m55s |
| First call, empty volume | 10s | **2m6s** |
| A restart, weights on the volume | 10s | **15.3s** |
| A warm worker | 10s | 1.6s |
| A warm worker, `timestamps` off | 10s | 1.1s |
| 21 minutes of mp3, 5 chunks | 1278s | 24.6s |
| 36 minutes of ogg, 8 chunks | 2207s | 40.2s |

**The volume turns a two-minute first start into fifteen seconds.** That gap is the 2.4GB
download, paid once for the volume instead of once for every worker.

Thirty-six minutes of speech came back in forty seconds, and the fetch of a 20MB file is
inside that figure. It is inside the warm figures too, where it is most of a request on a
ten-second file.

## Testing the app

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

The tests run with neither torch nor transformers installed, the same position
the build is in when it imports the file. They pin the request contract and the
grouping of pieces into words.
