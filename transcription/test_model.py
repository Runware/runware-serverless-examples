"""Tests that run with neither torch nor transformers installed."""

import pytest
from pydantic import ValidationError

from model import REVISION, Request, Transcript, Word, _words


def test_an_https_url_is_a_valid_request() -> None:
    assert Request(audio_url="https://example.com/talk.flac").audio_url.endswith(".flac")


def test_timestamps_are_the_published_default() -> None:
    """The default is on the model field, so the platform sends it.

    A plain parameter default would not be published, and the field would
    arrive as an explicit ``null`` instead.
    """
    assert Request(audio_url="https://example.com/talk.flac").timestamps is True


def test_a_plain_http_url_is_refused() -> None:
    with pytest.raises(ValidationError):
        Request(audio_url="http://example.com/talk.flac")


def test_a_local_path_is_refused() -> None:
    """A worker fetches what it is given, so the scheme is part of the contract."""
    with pytest.raises(ValidationError):
        Request(audio_url="file:///etc/passwd")


def test_an_unknown_field_is_refused() -> None:
    """A misspelled field is an error, not something to ignore quietly."""
    with pytest.raises(ValidationError):
        Request(audio_url="https://example.com/talk.flac", timestamp=True)


def test_the_weights_are_pinned_to_a_commit() -> None:
    """A moving reference is what puts unloadable weights on a worker."""
    assert len(REVISION) == 40
    assert REVISION != "main"


def test_a_word_carries_its_own_span() -> None:
    word = Word(text="hello", start=1.5, end=2.25)
    assert (word.start, word.end) == (1.5, 2.25)


def test_a_transcript_with_timestamps_off_has_no_words() -> None:
    empty = Transcript(text="hello", words=[], seconds=1.0, chunks=1)
    assert empty.words == []


def test_subword_pieces_are_grouped_into_words() -> None:
    """A piece that opens with a space opens a word, the rest extend the last one."""
    pieces = [
        {"token": "He", "start": 0.32, "end": 0.48},
        {"token": " h", "start": 0.48, "end": 0.64},
        {"token": "op", "start": 0.64, "end": 0.8},
        {"token": "ed", "start": 0.8, "end": 0.96},
    ]
    assert _words(pieces) == [("He", 0.32, 0.48), ("hoped", 0.48, 0.96)]


def test_a_word_keeps_the_span_of_all_its_pieces() -> None:
    """The start is the first piece's and the end is the last one's."""
    pieces = [
        {"token": " ste", "start": 1.52, "end": 1.68},
        {"token": "w", "start": 1.68, "end": 1.84},
    ]
    assert _words(pieces) == [("stew", 1.52, 1.84)]


def test_no_pieces_is_no_words() -> None:
    assert _words([]) == []
