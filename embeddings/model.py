"""Many inputs in one request: a list of texts in, a list of vectors out."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field

from runware_serverless import serve

MODEL = "BAAI/bge-small-en-v1.5"
"""MIT and ungated, 384 dimensions, small enough that the GPU is never the limit."""

CACHE = "/weights"
"""Mount a volume here. Small as this model is, a checkpoint captures the worker's
filesystem, so weights left outside a volume are copied into every one."""

MAX_TEXTS = 256
"""The largest list this app accepts.

Measured: the platform checks the types it published and rejects a wrong one in
under half a second without waking anything, but it does not check a list's
length, so a request carrying more than this reaches the worker and fails there.
This app enforces the value constraint itself, and a request that breaks one
wakes a worker before it is refused."""

ENCODE_BATCH = 64
"""How many texts cross the GPU at once inside one request. Unrelated to how many
the request carries: the platform gives a worker one task at a time, so batching
is something you do inside your handler or not at all."""


class Request(BaseModel):
    # Without this pydantic ignores a field it does not know, so a caller who
    # misspells one gets a silent success. An unknown field is an error here.
    model_config = ConfigDict(extra="forbid")

    texts: list[str] = Field(
        min_length=1,
        max_length=MAX_TEXTS,
        description="The texts to embed. One request, one pass over the GPU.",
    )
    normalize: bool = Field(
        default=True,
        description="Return unit vectors, so a dot product is a cosine similarity.",
    )


class Embedded(BaseModel):
    vectors: list[list[float]] = Field(description="One vector per input, in order.")
    dimensions: int
    count: int


@serve
class Embeddings:
    requirements = [
        "torch==2.14.1",
        "transformers==5.21.0",
        "sentence-transformers==6.2.0",
    ]

    def load(self) -> None:
        if hasattr(self, "model"):
            return

        # Here and not at the top of the file. The build imports this file with the
        # platform's own dependencies, not yours, so nothing in requirements exists
        # until a worker runs.
        from sentence_transformers import SentenceTransformer

        self.model = SentenceTransformer(MODEL, cache_folder=CACHE, device="cuda")
        self.model.encode(["warmup"])

    def predict(self, request: Request) -> Embedded:
        vectors = self.model.encode(
            request.texts,
            batch_size=ENCODE_BATCH,
            normalize_embeddings=request.normalize,
        )

        return Embedded(
            vectors=vectors.tolist(),
            dimensions=int(vectors.shape[1]),
            count=len(request.texts),
        )
