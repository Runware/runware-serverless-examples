"""An image arrives as base64, and the annotation is what turns it into pixels."""

from __future__ import annotations

import base64
import io
from enum import Enum

from pydantic import BaseModel, Field

from runware_serverless import serve
from runware_serverless.schema import Base64RGBAImage, Base64RGBImage

DEFAULT_OPACITY = 0.65
DEFAULT_MARGIN = 24
OPAQUE = 255


class Corner(str, Enum):
    top_left = "top-left"
    top_right = "top-right"
    bottom_left = "bottom-left"
    bottom_right = "bottom-right"


class Layers(BaseModel):
    base: Base64RGBImage = Field(description="The image to stamp, base64 encoded.")
    mark: Base64RGBAImage = Field(description="The mark to lay over it, base64 encoded.")
    corner: Corner = Field(default=Corner.bottom_right, description="Where the mark goes.")
    opacity: float = Field(default=DEFAULT_OPACITY, ge=0, le=1)
    margin: int = Field(default=DEFAULT_MARGIN, ge=0, description="Gap from the two edges, in pixels.")


class Stamped(BaseModel):
    image: str = Field(description="The result as a base64 PNG.")
    width: int
    height: int


@serve
class Watermark:
    requirements = ["pillow==13.0.0"]

    def load(self) -> None:
        pass

    def predict(self, layers: Layers) -> Stamped:
        canvas = layers.base.convert("RGBA")
        mark = _faded(layers.mark, layers.opacity)
        canvas.alpha_composite(mark, _position(canvas.size, mark.size, layers))

        flattened = canvas.convert("RGB")
        buffer = io.BytesIO()
        flattened.save(buffer, format="PNG")

        return Stamped(
            image=base64.b64encode(buffer.getvalue()).decode("ascii"),
            width=flattened.width,
            height=flattened.height,
        )


def _faded(mark: object, opacity: float) -> object:
    # Converted here rather than trusted: the annotation on `mark` preserves the
    # mode it arrived in, so a mark with no alpha channel reaches this unchanged.
    faded = mark.convert("RGBA")  # pyright: ignore[reportAttributeAccessIssue]
    faded.putalpha(faded.getchannel("A").point(lambda value: round(value * opacity)))

    return faded


def _position(
    canvas: tuple[int, int],
    mark: tuple[int, int],
    layers: Layers,
) -> tuple[int, int]:
    right = canvas[0] - mark[0] - layers.margin
    bottom = canvas[1] - mark[1] - layers.margin

    return {
        Corner.top_left: (layers.margin, layers.margin),
        Corner.top_right: (right, layers.margin),
        Corner.bottom_left: (layers.margin, bottom),
        Corner.bottom_right: (right, bottom),
    }[layers.corner]
