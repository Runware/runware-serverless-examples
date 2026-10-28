"""Two models in one request, so the image between them never crosses a boundary."""

from __future__ import annotations

import base64
import io

from pydantic import BaseModel, Field

from runware_serverless import serve

GENERATOR = "Qwen/Qwen-Image"
"""Apache 2.0 and ungated, so this deploys with no token."""

MATTE = "ZhengPeng7/BiRefNet"
"""MIT and ungated. Loading it runs code from its own repository, which is what
trust_remote_code means, and that code has imports of its own: einops, kornia and
timm are in requirements below because the matte needs them, not this file."""

CACHE = "/weights"
"""Mount a volume here. The generator alone is tens of gigabytes, so without one
it is fetched again on every cold start and copied into every checkpoint."""

DEFAULT_STEPS = 30
MATTE_SIZE = 1024


class Request(BaseModel):
    prompt: str = Field(description="What to generate before the background is removed.")
    steps: int = Field(default=DEFAULT_STEPS, ge=1, le=100)


class Generated(BaseModel):
    image: str = Field(description="The cut-out as a base64 PNG, with an alpha channel.")
    width: int
    height: int


@serve
class ChainedSteps:
    requirements = [
        "torch==2.14.1",
        "torchvision==0.29.1",
        "diffusers==0.42.1",
        "transformers==5.21.0",
        "accelerate==1.15.0",
        "einops==0.8.2",
        "kornia==0.9.0",
        "timm==1.0.30",
    ]

    def load(self) -> None:
        if hasattr(self, "generator"):
            return

        import torch
        from diffusers import DiffusionPipeline
        from transformers import AutoModelForImageSegmentation

        self.generator = DiffusionPipeline.from_pretrained(
            GENERATOR, torch_dtype=torch.bfloat16, cache_dir=CACHE
        ).to("cuda")

        self.matte = AutoModelForImageSegmentation.from_pretrained(
            MATTE, trust_remote_code=True, cache_dir=CACHE
        ).to("cuda")
        self.matte.eval()

        self.generator(prompt="warmup", num_inference_steps=1)

    def predict(self, request: Request) -> Generated:
        import torch

        generated = self.generator(
            prompt=request.prompt, num_inference_steps=request.steps
        ).images[0]

        # The one thing a second API call cannot do. The image handed to the matte
        # is the object the generator produced: not encoded, not uploaded, not
        # queued again, and not waiting on a second worker to start.
        with torch.inference_mode():
            alpha = self._alpha(generated)

        generated.putalpha(alpha)

        return Generated(
            image=_png(generated), width=generated.width, height=generated.height
        )

    def _alpha(self, image: object) -> object:
        import torch
        from torchvision import transforms

        prepare = transforms.Compose(
            [
                transforms.Resize((MATTE_SIZE, MATTE_SIZE)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        # Matched to the matte's own dtype, not assumed. Two models in one worker
        # are two dtypes, and the checkpoint decides each one: a float32 batch into
        # a half model fails at the first convolution, with a message about bias
        # types that says nothing about where it came from.
        weights = next(self.matte.parameters())
        batch = prepare(image).unsqueeze(0).to(weights.device, weights.dtype)  # pyright: ignore[reportArgumentType]
        mask = self.matte(batch)[-1].sigmoid().float().cpu()[0].squeeze()

        return transforms.ToPILImage()(mask).resize(image.size)  # pyright: ignore[reportAttributeAccessIssue]


def _png(image: object) -> str:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")  # pyright: ignore[reportAttributeAccessIssue]

    return base64.b64encode(buffer.getvalue()).decode("ascii")
