"""The shard arithmetic is the part a test can reach without a second GPU."""

from __future__ import annotations

from model import DEFAULT_SIZE, MultiGPU, _slice


def test_the_file_imports_without_the_model_s_own_dependencies() -> None:
    """The build imports this file with none of the app's requirements installed.

    Importing the test module is the same position, since torch is absent here
    too. The assertion names what that proves.
    """
    assert MultiGPU.reduce.__name__ == "reduce"


def test_requirements_are_declared_and_pinned() -> None:
    assert all("==" in line for line in MultiGPU.requirements)


def test_one_rank_takes_the_whole_range() -> None:
    assert _slice(10, 0, 1) == (0, 10)


def test_the_shares_are_contiguous_and_cover_everything() -> None:
    for size in (1, 2, 7, 10, 64, DEFAULT_SIZE):
        for world in (1, 2, 4, 8):
            shares = [_slice(size, rank, world) for rank in range(world)]

            assert shares[0][0] == 0
            assert shares[-1][1] == size
            assert all(a[1] == b[0] for a, b in zip(shares, shares[1:]))


def test_no_share_is_more_than_one_wider_than_another() -> None:
    widths = {end - start for start, end in (_slice(10, rank, 3) for rank in range(3))}

    assert max(widths) - min(widths) <= 1


def test_the_remainder_goes_to_the_early_ranks() -> None:
    assert [_slice(10, rank, 3) for rank in range(3)] == [(0, 4), (4, 7), (7, 10)]


def test_every_rank_reaching_the_collective_is_what_the_split_guarantees() -> None:
    """A rank with an empty share still has to arrive at the all-reduce.

    More ranks than values is the case that would tempt a handler to return
    early, which would hang every rank that joined the collective.
    """
    shares = [_slice(2, rank, 8) for rank in range(8)]

    assert sum(end - start for start, end in shares) == 2
    assert len(shares) == 8
