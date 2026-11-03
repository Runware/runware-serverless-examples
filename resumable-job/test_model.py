"""Tests that run with neither torch nor boto3 installed."""

import pytest
from pydantic import ValidationError

from model import STEPS_PER_CALL, Progress, Request, ResumableJob


def test_a_job_name_is_required() -> None:
    with pytest.raises(ValidationError):
        Request()


def test_a_job_name_with_a_slash_is_refused() -> None:
    """The name becomes part of an object key, so it cannot carry a path."""
    with pytest.raises(ValidationError):
        Request(job="../etc/passwd")


def test_the_key_is_stable_for_a_job_name() -> None:
    assert ResumableJob()._key("nightly") == "checkpoints/nightly.pt"


def test_a_slice_is_bounded() -> None:
    """What bounds a slice is what a lost worker should cost, not a timeout."""
    assert 0 < STEPS_PER_CALL <= 1000


def test_progress_says_when_to_call_again() -> None:
    partial = Progress(
        job="nightly", steps_done=200, total_steps=1000,
        done=False, loss=0.5, resumed_from=0,
    )
    finished = Progress(
        job="nightly", steps_done=1000, total_steps=1000,
        done=True, loss=0.01, resumed_from=800,
    )

    assert partial.done is False
    assert finished.done is True


def test_the_store_is_described_entirely_by_the_environment() -> None:
    """Nothing about which store this is belongs in the file."""
    import model

    for name in ("BUCKET", "ENDPOINT", "REGION"):
        assert isinstance(getattr(model, name), str)
