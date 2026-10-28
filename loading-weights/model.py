"""Where the heavy parts of an app go, and why the weights belong on a volume."""

from __future__ import annotations

import base64
import io

from pydantic import BaseModel, Field

from runware_serverless import serve

MODEL = "Tongyi-MAI/Z-Image-Turbo"
"""Apache 2.0 and ungated, so this deploys with no token of any kind."""

CACHE = "/weights"
"""Mount a volume here. The worker's filesystem is part of what a checkpoint
captures, so weights left outside a volume are copied into every checkpoint and
fetched again on every cold start."""

DEFAULT_STEPS = 8
"""A turbo model, so the schedule is short. Raising this buys little."""


class Request(BaseModel):
    prompt: str = Field(description="What to generate.")
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=50)


class Generated(BaseModel):
    image: str = Field(description="The result as a base64 PNG.")
    steps: int


@serve
class LoadingWeights:
    requirements = [
        "torch==2.14.1",
        "diffusers==0.42.1",
        "transformers==5.21.0",
        "accelerate==1.15.0",
    ]

    def load(self) -> None:
        if hasattr(self, "pipeline"):
            return

        # Here and not at the top of the file. The build imports this file with the
        # platform's own dependencies, not yours, so nothing in requirements exists
        # until a worker runs.
        import torch
        from diffusers import DiffusionPipeline

        self.pipeline = DiffusionPipeline.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE
        ).to("cuda")

        # The first call compiles kernels and allocates, so no request should be
        # the one paying for it.
        self.pipeline(prompt="warmup", num_inference_steps=1)

    def predict(self, request: Request) -> Generated:
        image = self.pipeline(
            prompt=request.prompt, num_inference_steps=request.steps
        ).images[0]

        return Generated(image=_png(image), steps=request.steps)


def _png(image: object) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # pyright: ignore[reportAttributeAccessIssue]

    return base64.b64encode(buffer.getvalue()).decode("ascii")
