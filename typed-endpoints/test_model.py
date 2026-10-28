"""The contract this app publishes is the thing worth pinning."""

from __future__ import annotations

from model import Color, Orientation, Report, Source, TypedEndpoints, _orientation, _palette


def test_the_file_imports_without_the_model_s_own_dependencies() -> None:
    """The build imports this file with none of the app's requirements installed.

    Importing the test module is the same position, since nothing here installs
    them either. The assertion names what that proves.
    """
    assert TypedEndpoints.describe.__name__ == "describe"


def test_requirements_are_declared_and_pinned() -> None:
    assert all("==" in line for line in TypedEndpoints.requirements)


def test_the_request_publishes_its_defaults_and_its_prose() -> None:
    """What a scalar parameter cannot carry, and the reason for the model.

    An optional scalar exports its type and nothing else, so the platform has no
    default to send and the handler receives an explicit null. A pydantic field
    exports both, which is what makes ``palette=False`` arrive as False.
    """
    field = Source.model_json_schema()["properties"]["palette"]

    assert field["default"] is False
    assert field["description"]


def test_the_response_describes_every_field_including_the_nested_ones() -> None:
    schema = Report.model_json_schema(mode="serialization")

    assert set(schema["required"]) == {"format", "size", "orientation", "bytes"}
    assert schema["properties"]["size"]["$ref"] == "#/$defs/Size"
    assert schema["$defs"]["Orientation"]["enum"] == ["landscape", "portrait", "square"]


def test_orientation_covers_the_three_cases() -> None:
    assert _orientation(2, 1) is Orientation.landscape
    assert _orientation(1, 2) is Orientation.portrait
    assert _orientation(1, 1) is Orientation.square


def test_the_palette_is_ordered_by_frequency() -> None:
    class Fake:
        def convert(self, mode: str) -> Fake:
            return self

        def quantize(self, colors: int) -> Fake:
            return self

        def getpalette(self) -> list[int]:
            return [1, 2, 3, 4, 5, 6]

        def getcolors(self) -> list[tuple[int, int]]:
            return [(10, 1), (90, 0)]

    assert _palette(Fake()) == [Color(red=1, green=2, blue=3), Color(red=4, green=5, blue=6)]
