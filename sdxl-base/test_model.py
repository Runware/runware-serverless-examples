"""What is testable without a GPU is the part that decides whether a deploy works."""

from __future__ import annotations

import base64

from model import DEFAULT_STEPS, StableDiffusionXL, _png


def test_the_file_imports_without_the_model_s_own_dependencies() -> None:
    """The build imports this file before installing anything in requirements.

    Importing the test module already proved it, since torch and diffusers are
    absent here too. The assertion names what that proves.
    """
    assert StableDiffusionXL.predict.__name__ == "predict"


def test_requirements_are_declared_on_the_class() -> None:
    assert isinstance(StableDiffusionXL.requirements, list)
    assert all(isinstance(line, str) for line in StableDiffusionXL.requirements)
    assert any(line.startswith("torch") for line in StableDiffusionXL.requirements)


def test_steps_has_a_default_an_omitted_field_can_fall_back_to() -> None:
    assert DEFAULT_STEPS == 25


def test_an_image_leaves_as_base64() -> None:
    class Fake:
        def save(self, buffer: object, format: str) -> None:  # noqa: A002
            buffer.write(b"pixels")  # pyright: ignore[reportAttributeAccessIssue]

    assert base64.b64decode(_png(Fake())) == b"pixels"
