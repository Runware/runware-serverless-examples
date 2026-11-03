"""Tests that run with neither torch nor transformers installed."""

import pytest
from pydantic import ValidationError

from model import MAX_RESIDUES, REVISION, Request

PEPTIDE = "MKTAYIAKQRQISFVKSHFSRQ"


def test_a_sequence_is_accepted() -> None:
    assert Request(sequence=PEPTIDE).sequence == PEPTIDE


def test_a_non_amino_acid_letter_is_refused() -> None:
    """X, U, B and Z are not standard residues and the tokenizer has no id for them."""
    with pytest.raises(ValidationError):
        Request(sequence="MKTAYIAKQRQISFVKSHFSRX")


def test_lowercase_is_refused() -> None:
    with pytest.raises(ValidationError):
        Request(sequence=PEPTIDE.lower())


def test_a_sequence_over_the_ceiling_is_refused() -> None:
    with pytest.raises(ValidationError):
        Request(sequence="A" * (MAX_RESIDUES + 1))


def test_an_unknown_field_is_refused() -> None:
    with pytest.raises(ValidationError):
        Request(sequence=PEPTIDE, chains=2)


def test_the_weights_are_pinned_to_a_revision() -> None:
    """A moving reference is how you get weights the installed version cannot load."""
    assert REVISION != "main"
    assert len(REVISION) == 40


def test_the_confidence_field_promises_the_conventional_scale() -> None:
    """pLDDT is 0 to 100 everywhere it is read, so that is what this publishes."""
    from model import Folded

    assert "0 to 100" in (Folded.model_fields["confidence"].description or "")
