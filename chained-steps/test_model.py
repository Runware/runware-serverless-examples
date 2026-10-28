"""The published bounds are what a test can reach without a GPU."""

from __future__ import annotations

import base64

from model import DEFAULT_STEPS, ChainedSteps, Generated, Request, _png


def test_the_file_imports_without_the_model_s_own_dependencies() -> None:
    """The build imports this file with none of the app's requirements installed.

    Importing the test module is the same position, since torch and diffusers are
    absent here too. The assertion names what that proves.
    """
    assert ChainedSteps.predict.__name__ == "predict"


def test_requirements_are_declared_and_pinned() -> None:
    assert all("==" in line for line in ChainedSteps.requirements)
    assert any(line.startswith("diffusers") for line in ChainedSteps.requirements)


def test_steps_is_published_with_its_bounds() -> None:
    field = Request.model_json_schema()["properties"]["steps"]

    assert (field["default"], field["minimum"], field["maximum"]) == (DEFAULT_STEPS, 1, 100)


def test_the_response_declares_the_size_of_what_it_returns() -> None:
    assert set(Generated.model_json_schema()["required"]) == {"image", "width", "height"}


def test_an_image_leaves_as_base64() -> None:
    class Fake:
        def save(self, buffer: object, format: str) -> None:  # noqa: A002
            buffer.write(b"pixels")  # pyright: ignore[reportAttributeAccessIssue]

    assert base64.b64decode(_png(Fake())) == b"pixels"
