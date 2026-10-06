"""Each endpoint is a method, so each test is a call."""

from model import BASE_SIZE, PIXEL_PNG, ImageTools


def test_generate_reports_its_own_handler() -> None:
    model = ImageTools()
    model.load()

    result = model.generate("a red bicycle")

    assert result["handler"] == "generate"
    assert result["steps"] == 4
    assert result["image"] == PIXEL_PNG


def test_upscale_takes_a_different_shape_entirely() -> None:
    model = ImageTools()
    model.load()

    result = model.upscale(PIXEL_PNG, scale=4)

    assert result["handler"] == "upscale"
    assert result["width"] == BASE_SIZE * 4


def test_each_handler_applies_its_own_default() -> None:
    model = ImageTools()
    model.load()

    assert model.generate("a red bicycle", steps=None)["steps"] == 4
    assert model.upscale(PIXEL_PNG, scale=None)["scale"] == 2
