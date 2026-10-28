"""What is testable without a GPU is the part that decides whether a deploy works."""

from __future__ import annotations

import base64

from model import DEFAULT_STEPS, Generated, LoadingWeights, Request, _png


def test_the_file_imports_without_the_model_s_own_dependencies() -> None:
    """The build imports this file with none of the app's requirements installed.

    Importing the test module is the same position, since torch and diffusers are
    absent here too. The assertion names what that proves.
    """
    assert LoadingWeights.predict.__name__ == "predict"


def test_requirements_are_declared_and_pinned() -> None:
    assert all("==" in line for line in LoadingWeights.requirements)
    assert any(line.startswith("torch") for line in LoadingWeights.requirements)


def test_steps_is_published_with_its_default_and_bounds() -> None:
    """An omitted step count arrives as the published default, not as null."""
    field = Request.model_json_schema()["properties"]["steps"]

    assert (field["default"], field["minimum"], field["maximum"]) == (DEFAULT_STEPS, 1, 50)


def test_the_schedule_is_short_because_the_model_is_a_turbo() -> None:
    assert DEFAULT_STEPS <= 10


def test_the_response_declares_what_it_returns() -> None:
    assert set(Generated.model_json_schema()["required"]) == {"image", "steps"}


def test_an_image_leaves_as_base64() -> None:
    class Fake:
        def save(self, buffer: object, format: str) -> None:  # noqa: A002
            buffer.write(b"pixels")  # pyright: ignore[reportAttributeAccessIssue]

    assert base64.b64decode(_png(Fake())) == b"pixels"
