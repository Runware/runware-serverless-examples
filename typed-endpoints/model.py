"""A request and a response the platform can describe, which is what pydantic buys."""

from __future__ import annotations

import base64
import io
from enum import Enum

from pydantic import BaseModel, Field

from runware_serverless import endpoint, serve

MAX_PALETTE = 8


class Orientation(str, Enum):
    landscape = "landscape"
    portrait = "portrait"
    square = "square"


class Source(BaseModel):
    image: str = Field(description="The image to read, base64 encoded, with no data URI prefix.")
    palette: bool = Field(default=False, description="Also return the most common colors.")


class Size(BaseModel):
    width: int
    height: int


class Color(BaseModel):
    red: int
    green: int
    blue: int


class Report(BaseModel):
    format: str = Field(description="The container the bytes are in, such as PNG or JPEG.")
    size: Size
    orientation: Orientation
    bytes: int = Field(description="How many bytes the image decoded to.")
    palette: list[Color] = Field(default_factory=list)


@serve
class TypedEndpoints:
    requirements = ["pillow==13.0.0"]

    def load(self) -> None:
        pass

    @endpoint
    def describe(self, source: Source) -> Report:
        # Here rather than at the top of the file. The build imports this file
        # with the platform's own dependencies, not the app's, so nothing in
        # requirements exists until a worker runs.
        from PIL import Image

        raw = base64.b64decode(source.image, validate=True)
        with Image.open(io.BytesIO(raw)) as image:
            width, height = image.size

            return Report(
                format=image.format or "unknown",
                size=Size(width=width, height=height),
                orientation=_orientation(width, height),
                bytes=len(raw),
                palette=_palette(image) if source.palette else [],
            )


def _orientation(width: int, height: int) -> Orientation:
    if width > height:
        return Orientation.landscape
    if height > width:
        return Orientation.portrait

    return Orientation.square


def _palette(image: object) -> list[Color]:
    # Quantizing rather than counting pixels: getcolors returns nothing at all
    # once an image holds more distinct colors than the limit it was given, which
    # every photograph does.
    quantized = image.convert("RGB").quantize(colors=MAX_PALETTE)  # pyright: ignore[reportAttributeAccessIssue]
    channels = quantized.getpalette() or []
    by_frequency = sorted(quantized.getcolors() or [], reverse=True)

    return [
        Color(
            red=channels[index * 3],
            green=channels[index * 3 + 1],
            blue=channels[index * 3 + 2],
        )
        for _, index in by_frequency
    ]
