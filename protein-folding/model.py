"""Folding a protein: GPU work with nothing to do with media.

ESMFold predicts a structure from a single sequence, with no multiple sequence
alignment and so no call out to an alignment server. The whole app is one model
and one forward pass.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from runware_serverless import serve

MODEL = "facebook/esmfold_v1"

REVISION = "75a3841ee059df2bf4d56688166c8fb459ddd97a"
"""Pinned, deliberately.

A moving reference is how you end up with weights the installed version cannot
load: the checkpoint drifts, the code does not, and the worker dies at startup
with a constructor it cannot satisfy. A revision costs nothing and removes that."""

CACHE = "/weights"
"""Mount a volume here. The model is about 2.6GB, which is a download worth doing
once rather than on every cold start."""

WARMUP = "MKTAYIAKQRQISFVKSHFSRQ"
"""Folded once in ``load`` so the weights arrive before any request does."""

MAX_RESIDUES = 400
"""Attention over the sequence grows with the square of its length. The schema
publishes the ceiling, and this app enforces it: the platform checks the types it
published but not a value constraint."""

CHUNK = 64
"""Trades speed for peak memory inside the folding trunk. Lower it if a long
sequence runs out of room."""


class Request(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sequence: str = Field(
        min_length=10,
        max_length=MAX_RESIDUES,
        pattern=r"^[ACDEFGHIKLMNPQRSTVWY]+$",
        description="One protein chain, as standard single-letter amino acids.",
    )


class Folded(BaseModel):
    structure: str = Field(description="The predicted structure as PDB text.")
    format: str
    residues: int
    confidence: dict[str, float] = Field(
        description="Mean pLDDT for the structure, from 0 to 100. Above 70 is confident."
    )


@serve
class ProteinFolding:
    requirements = [
        "torch==2.14.1",
        "transformers==5.21.0",
        "accelerate==1.15.0",
    ]

    def load(self) -> None:
        if hasattr(self, "model"):
            return

        # Here and not at the top of the file. The build imports this file with the
        # platform's own dependencies, not yours, so nothing in requirements exists
        # until a worker runs.
        import torch
        from transformers import AutoTokenizer, EsmForProteinFolding

        self.torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(
            MODEL, revision=REVISION, cache_dir=CACHE
        )
        self.model = EsmForProteinFolding.from_pretrained(
            MODEL, revision=REVISION, cache_dir=CACHE
        )
        self.model = self.model.cuda().eval()
        self.model.trunk.set_chunk_size(CHUNK)

        print(f"warming up: folding {len(WARMUP)} residues", flush=True)
        self._fold(WARMUP)
        print("warmup complete", flush=True)

    def predict(self, request: Request) -> Folded:
        structure, confidence = self._fold(request.sequence)

        return Folded(
            structure=structure,
            format="pdb",
            residues=len(request.sequence),
            confidence=confidence,
        )

    def _fold(self, sequence: str) -> tuple[str, dict[str, float]]:
        tokens = self.tokenizer(
            [sequence], return_tensors="pt", add_special_tokens=False
        )["input_ids"].cuda()

        with self.torch.no_grad():
            output = self.model(tokens)

        # Measured: this model returns pLDDT on a 0 to 1 scale, while the
        # literature and every structure viewer use 0 to 100. Scaled once here,
        # and guarded so a version that already returns 0 to 100 is left alone.
        plddt = float(output["plddt"].mean())

        return self.model.output_to_pdb(output)[0], {
            "mean_plddt": round(plddt * 100 if plddt <= 1.0 else plddt, 2)
        }
