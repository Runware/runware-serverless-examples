"""A real model with weights: where the heavy parts go, and why they go there."""

from __future__ import annotations

import base64
import io

from runware_serverless import serve

MODEL = "stabilityai/stable-diffusion-xl-base-1.0"
"""Ungated and usable commercially, so this deploys with no token of any kind."""

CACHE = "/weights"
"""Mount a volume here. The worker's filesystem is part of what a checkpoint
captures, so weights left outside a volume are copied into every checkpoint and
fetched again on every cold start."""

DEFAULT_STEPS = 25


@serve
class StableDiffusionXL:
    requirements = [
        "torch==2.14.1",
        "diffusers==0.42.1",
        "transformers==5.21.0",
        "accelerate==1.15.0",
    ]

    def load(self) -> None:
        if hasattr(self, "pipeline"):
            return

        # Here rather than at the top of the file. The build imports this file to
        # read the endpoints out of it, and it does that before installing any of
        # the requirements above, so a torch import at module scope fails it.
        import torch
        from diffusers import StableDiffusionXLPipeline

        self.pipeline = StableDiffusionXLPipeline.from_pretrained(
            MODEL,
            torch_dtype=torch.float16,
            variant="fp16",
            use_safetensors=True,
            cache_dir=CACHE,
        ).to("cuda")

        # The first call compiles kernels and allocates, so a request should not
        # be the one paying for it.
        self.pipeline(prompt="warmup", num_inference_steps=1)

    def predict(self, prompt: str, steps: int = DEFAULT_STEPS) -> dict[str, str]:
        steps = DEFAULT_STEPS if steps is None else steps
        image = self.pipeline(prompt=prompt, num_inference_steps=steps).images[0]

        return {"image": _png(image), "prompt": prompt}


def _png(image: object) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # pyright: ignore[reportAttributeAccessIssue]

    return base64.b64encode(buffer.getvalue()).decode("ascii")
