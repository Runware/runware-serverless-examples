"""The annotations are the part worth testing: they decide what the handler receives."""

from __future__ import annotations

import base64
import io

from model import Corner, Layers, Stamped, Watermark, _position
from PIL import Image
from pydantic import BaseModel

from runware_serverless.schema import Base64RGBImage


def _b64(image: Image.Image, fmt: str = "PNG") -> str:
    buffer = io.BytesIO()
    image.save(buffer, format=fmt)

    return base64.b64encode(buffer.getvalue()).decode("ascii")


def _layers(**overrides: object) -> Layers:
    return Layers(
        base=_b64(Image.new("RGB", (200, 100), (10, 20, 30))),
        mark=_b64(Image.new("RGBA", (40, 20), (255, 255, 255, 255))),
        **overrides,
    )


def test_the_annotation_hands_the_handler_pixels_not_a_string() -> None:
    layers = _layers()

    assert isinstance(layers.base, Image.Image)
    assert layers.base.size == (200, 100)


def test_rgb_flattens_and_rgba_preserves_whatever_arrived() -> None:
    """The names describe the conversion, not the result.

    ``Base64RGBImage`` converts, so anything becomes RGB and the container the
    bytes arrived in is lost with it. ``Base64RGBAImage`` converts nothing, so a
    grayscale image stays grayscale rather than gaining an alpha channel.
    """
    gray = _b64(Image.new("L", (8, 8), 128))
    layers = Layers(base=gray, mark=gray)

    assert layers.base.mode == "RGB"
    assert layers.base.format is None
    assert layers.mark.mode == "L"


def test_the_field_publishes_a_described_base64_string() -> None:
    field = Layers.model_json_schema()["properties"]["base"]

    assert field["type"] == "string"
    assert field["format"] == "base64"


def test_these_types_do_not_survive_a_response() -> None:
    """They coerce on the way in and refuse to serialize on the way out.

    Which is why ``Stamped.image`` is a plain string the handler encodes itself.
    """
    image = _b64(Image.new("RGB", (8, 8)))

    class Wrong(BaseModel):
        image: Base64RGBImage

    assert image not in Wrong(image=image).model_dump_json()


def test_every_corner_lands_inside_the_canvas() -> None:
    for corner in Corner:
        x, y = _position((200, 100), (40, 20), _layers(corner=corner, margin=10))

        assert (x, y) in {(10, 10), (150, 10), (10, 70), (150, 70)}


def test_the_result_is_a_png_the_size_of_the_base() -> None:
    stamped = Watermark().predict(_layers())

    assert isinstance(stamped, Stamped)
    assert (stamped.width, stamped.height) == (200, 100)
    with Image.open(io.BytesIO(base64.b64decode(stamped.image))) as out:
        assert out.format == "PNG"
        assert out.size == (200, 100)


def test_a_fully_transparent_mark_changes_nothing() -> None:
    stamped = Watermark().predict(_layers(opacity=0))

    with Image.open(io.BytesIO(base64.b64decode(stamped.image))) as out:
        assert out.getpixel((190, 90)) == (10, 20, 30)
