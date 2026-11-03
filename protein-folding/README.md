# Folding a protein

A GPU workload with nothing to do with images, video or text. A protein sequence
goes in and a predicted structure comes out, as PDB.

ESMFold is MIT licensed and its weights are ungated, so this deploys with **no
token**.

## What it teaches

**Serverless exists for the GPU work the models API does not cover.** That API
handles image, video, audio and text. This is the shape of app that
exists because your workload is not any of those: your own weights, your own
science, the same hardware.

**Pin the weights to a revision.** `from_pretrained` takes `revision`, and
without it you are asking for whatever `main` holds today. That is how you end up
with weights the installed version cannot load, and the failure arrives as a
worker that exits at startup rather than as a bad answer.

```python
EsmForProteinFolding.from_pretrained(MODEL, revision=REVISION, cache_dir=CACHE)
```

**Take the port, not the original, when the original needs a compiler.**
`fair-esm[esmfold]` pulls in OpenFold and wants `nvcc` to build kernels at
install time. **A code worker has no toolchain at all**, so that route cannot
run here. The transformers port dropped the OpenFold dependency, which is what
makes this a plain code app rather than a container.

**No alignment, so no third party in the request path.** ESMFold predicts from a
single sequence. Folding tools that want a multiple sequence alignment reach out
to an alignment server, and an app that calls a third party on every request
depends on that third party staying up.

**Publish the ceiling and enforce it yourself.** Attention over the sequence grows
with the square of its length. `max_length` is published, so it documents the
limit, but the platform checks the types it published and not a value constraint:
the rejection happens in this app, after a worker has woken.

**Load once, in `load`.** The weights are about 2.6GB and the first fold is what
pulls them. A volume keeps them off the checkpoint and out of the next cold start,
and the warmup fold means no request is the one paying for either.

**Print what you are doing.** `load` says what it is folding and when it is done.
A worker that dies quietly during startup leaves a task pending with no error, and
the logs are the only place the reason can appear.

## Deploying it

```bash
runware serverless deploy model.py \
  --id protein-folding \
  --gpu-type rtx-pro-6000 \
  --volume /weights
```

## Calling it

Use `invoke-async` and poll. A fold is tens of seconds, not milliseconds.

```bash
curl https://api.serverless.runware.ai/v1/apps/protein-folding/invoke-async \
  -H "Authorization: Bearer $RUNWARE_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"taskId": "7c9e6679-7425-40de-944b-e07fc1f90ae7", "payload": {"request": {"sequence": "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGCEKAVQVKVKALPDAQFEVVHSLAKWKRQTLGQHDFSAGEGLYTHMKALRPDEDRLSPLHSVYVDQWDWERVMGDGERQFSTLKSTVEAIWAGIKATEAAVSEE"}}}'
```

Then:

```bash
curl https://api.serverless.runware.ai/v1/apps/protein-folding/tasks/7c9e6679-7425-40de-944b-e07fc1f90ae7 \
  -H "Authorization: Bearer $RUNWARE_API_KEY"
```

The structure comes back as PDB text at `output.predictions[0].structure`. Save it
to a `.pdb` file and any molecular viewer will open it.

`confidence.mean_plddt` runs from 0 to 100 and is the model's own estimate of how
much to trust what it drew. **Above 70 is confident, below 50 is close to a
guess**, so a caller can tell the two apart without reading coordinates.

Measured on an RTX PRO 6000:

| | |
| --- | --- |
| First call, empty volume: weights plus a warmup fold | **4m36s** |
| A restart, weights already on the volume | **3m3s** |
| A warm worker, 154 residues | 1.2s |
| A warm worker, 22 residues | 1.3s |

**A restart is still minutes, and that is worth knowing before you size an app
around this.** The volume saves the download, not the load: 2.6GB still has to
reach the GPU and the warmup fold still runs. Keep `--min-workers 1` if a cold
caller cannot wait.

Length barely moves the warm number at these sizes, so the fold is mostly fixed
cost until the sequence gets long enough for the quadratic term to show.

## Testing it

```bash
uv run --no-project --with pytest --with 'runware-serverless[schema]' pytest -q
```

The tests run with neither torch nor transformers installed, the same position the
build is in when it imports the file. They pin the sequence contract, including
that a non-standard residue is refused here rather than inside the tokenizer, and
that the weights are pinned to a revision rather than to a moving reference.
