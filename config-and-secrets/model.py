"""What a worker can see of its own configuration, and what it must never answer with."""

from __future__ import annotations

import os

from pydantic import BaseModel, Field

from runware_serverless import serve

EXPECTED = ("LOG_LEVEL", "CACHE_DIR", "API_TOKEN")
"""The names this app reads. Two are plain configuration and one is a secret, and
nothing below can tell which is which."""


class Binding(BaseModel):
    name: str
    present: bool = Field(description="Whether the name reached the worker. Never the value.")


class Report(BaseModel):
    bindings: list[Binding]
    missing: list[str] = Field(description="The expected names that did not arrive.")


@serve
class ConfigAndSecrets:
    def load(self) -> None:
        # Read once, not per request. Changing a variable or an attachment records a
        # new version and rolls the app, so a running worker never holds a stale
        # value: it is replaced by one that read the new set here.
        self.present = {name: name in os.environ for name in EXPECTED}

    def predict(self) -> Report:
        return Report(
            bindings=[Binding(name=name, present=ok) for name, ok in self.present.items()],
            missing=[name for name, ok in self.present.items() if not ok],
        )
