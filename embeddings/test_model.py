"""Tests that run with neither torch nor sentence-transformers installed."""

import pytest
from pydantic import ValidationError

from model import MAX_TEXTS, Request


def test_a_single_text_is_a_valid_request() -> None:
    assert Request(texts=["one"]).texts == ["one"]


def test_normalizing_is_the_published_default() -> None:
    """The default is on the model field, so the platform sends it.

    A plain parameter default would not be published, and the field would
    arrive as an explicit ``null`` instead.
    """
    assert Request(texts=["one"]).normalize is True


def test_an_empty_list_is_refused() -> None:
    with pytest.raises(ValidationError):
        Request(texts=[])


def test_more_texts_than_the_ceiling_are_refused() -> None:
    with pytest.raises(ValidationError):
        Request(texts=["x"] * (MAX_TEXTS + 1))


def test_the_ceiling_itself_is_allowed() -> None:
    assert len(Request(texts=["x"] * MAX_TEXTS).texts) == MAX_TEXTS


def test_an_unknown_field_is_refused() -> None:
    """A misspelled field is an error, not something to ignore quietly."""
    with pytest.raises(ValidationError):
        Request(texts=["one"], normalise=True)
