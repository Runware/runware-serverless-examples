"""The smallest app that works: no weights, no GPU work, two endpoints."""

from __future__ import annotations

from runware_serverless import endpoint, serve


@serve
class EchoModel:
    def load(self) -> None:
        pass

    @endpoint
    def echo(self, message: str) -> dict[str, object]:
        return {
            "message": message,
            "uppercase": message.upper(),
            "length": len(message),
        }

    @endpoint
    def reverse_message(self, message: str, repeat: int = 1) -> dict[str, object]:
        repeat = 1 if repeat is None else repeat
        return {
            "message": message,
            "reversed": message[::-1] * repeat,
        }
