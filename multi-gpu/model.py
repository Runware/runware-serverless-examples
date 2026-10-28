"""One serving process per GPU, and a handler that every one of them runs."""

from __future__ import annotations

from runware_serverless import endpoint, serve

DEFAULT_SIZE = 1 << 22


@serve
class MultiGPU:
    requirements = ["torch==2.14.1"]

    def load(self) -> None:
        # Nothing here joins the world. @serve runs this inside the platform's
        # initialization, which sets this rank's device and brings the process
        # group up before load, then waits on a barrier after it. A second
        # init_process_group raises rather than being ignored.
        import torch
        import torch.distributed as dist

        self.world = dist.get_world_size() if dist.is_initialized() else 1
        self.rank = dist.get_rank() if dist.is_initialized() else 0
        self.device = (
            torch.device("cuda", torch.cuda.current_device())
            if torch.cuda.is_available()
            else torch.device("cpu")
        )

    @endpoint
    def world_view(self) -> dict[str, object]:
        import torch
        import torch.distributed as dist

        mine = {
            "rank": self.rank,
            "device": str(self.device),
            "card": torch.cuda.get_device_name(self.device)
            if torch.cuda.is_available()
            else "cpu",
        }
        if self.world == 1:
            return {"worldSize": 1, "ranks": [mine]}

        gathered: list[object] = [None] * self.world
        dist.all_gather_object(gathered, mine)

        return {"worldSize": self.world, "ranks": gathered}

    @endpoint
    def reduce(self, size: int = DEFAULT_SIZE) -> dict[str, object]:
        import torch
        import torch.distributed as dist

        size = DEFAULT_SIZE if size is None else size
        start, end = _slice(size, self.rank, self.world)

        # On this rank's own card, and only this rank's slice of the range. The
        # whole point of the world is that no single card holds all of it.
        mine = torch.arange(start, end, dtype=torch.float64, device=self.device)
        partial = mine.sum()

        if self.world == 1:
            return {"total": partial.item(), "worldSize": 1, "size": size}

        # Every rank reaches this, because every rank ran the same request. A rank
        # that returned early would hang the ones that arrived here.
        total = partial.clone()
        dist.all_reduce(total)

        return {
            "total": total.item(),
            "worldSize": self.world,
            "size": size,
            "partials": _partials(partial, self.world),
        }


def _slice(size: int, rank: int, world: int) -> tuple[int, int]:
    """This rank's half-open share of ``range(size)``, the remainder spread early."""
    share, remainder = divmod(size, world)
    start = rank * share + min(rank, remainder)

    return start, start + share + (1 if rank < remainder else 0)


def _partials(partial: object, world: int) -> list[object]:
    import torch.distributed as dist

    gathered: list[object] = [None] * world
    dist.all_gather_object(gathered, partial.item())  # pyright: ignore[reportAttributeAccessIssue]

    return gathered
