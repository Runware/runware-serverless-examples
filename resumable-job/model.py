"""A job that outlives one request: do a slice, checkpoint, answer, resume.

The training here is a stand-in, deliberately small, because the subject is the
checkpoint and where it goes. Swap the three lines in ``_step`` for real work and
nothing else about the shape changes.
"""

from __future__ import annotations

import io
import os

from pydantic import BaseModel, Field

from runware_serverless import serve

STEPS_PER_CALL = 200
"""Small enough that every call returns progress you can watch, and that a worker
lost between two calls costs one slice rather than the whole job.

A code app does not apply a request timeout, so this is not a limit being worked
around. It is a choice about what a failure costs."""

BUCKET = os.environ.get("S3_BUCKET", "")
ENDPOINT = os.environ.get("S3_ENDPOINT", "")
REGION = os.environ.get("S3_REGION", "")
"""Any S3-compatible store. All three come from secrets, which the platform puts
in the worker's environment, so nothing about the store is in this file.

Leave ``S3_ENDPOINT`` and ``S3_REGION`` unset for AWS and boto3 resolves both the
usual way. Other stores need the endpoint, and some, Cloudflare R2 among them,
want the region set to ``auto``."""


class Request(BaseModel):
    job: str = Field(
        description="Names the checkpoint. The same name resumes, a new one starts over.",
        min_length=1,
        max_length=64,
        pattern=r"^[A-Za-z0-9._-]+$",
    )
    total_steps: int = Field(default=1000, ge=1, le=1_000_000)


class Progress(BaseModel):
    job: str
    steps_done: int
    total_steps: int
    done: bool = Field(description="False means call again with the same job name.")
    loss: float
    resumed_from: int = Field(description="The step the checkpoint this call started from held.")


@serve
class ResumableJob:
    requirements = ["torch==2.14.1", "boto3==1.43.123"]

    def load(self) -> None:
        if hasattr(self, "store"):
            return

        import boto3
        import torch

        if not BUCKET:
            raise RuntimeError(
                "S3_BUCKET is unset. Attach the secrets before invoking: "
                "runware serverless secrets attach resumable-job --secret s3"
            )

        self.torch = torch
        self.store = boto3.client(
            "s3",
            endpoint_url=ENDPOINT or None,
            region_name=REGION or None,
            aws_access_key_id=os.environ["S3_ACCESS_KEY_ID"],
            aws_secret_access_key=os.environ["S3_SECRET_ACCESS_KEY"],
        )

    def predict(self, request: Request) -> Progress:
        state = self._restore(request.job)
        started_at = state["steps_done"]

        model, optimizer = self._build(state)
        loss = 0.0

        remaining = min(STEPS_PER_CALL, request.total_steps - started_at)
        for _ in range(max(remaining, 0)):
            loss = self._step(model, optimizer)

        steps_done = started_at + max(remaining, 0)
        self._save(request.job, model, optimizer, steps_done)

        return Progress(
            job=request.job,
            steps_done=steps_done,
            total_steps=request.total_steps,
            done=steps_done >= request.total_steps,
            loss=round(loss, 6),
            resumed_from=started_at,
        )

    def _key(self, job: str) -> str:
        return f"checkpoints/{job}.pt"

    def _restore(self, job: str) -> dict[str, object]:
        try:
            raw = self.store.get_object(Bucket=BUCKET, Key=self._key(job))["Body"].read()
        except self.store.exceptions.NoSuchKey:
            return {"steps_done": 0}

        # weights_only is the default from torch 2.6 and this is our own file, but
        # a checkpoint read back from object storage is still input, so say so.
        return self.torch.load(io.BytesIO(raw), weights_only=True)

    def _build(self, state: dict[str, object]) -> tuple[object, object]:
        torch = self.torch

        torch.manual_seed(0)
        model = torch.nn.Sequential(
            torch.nn.Linear(64, 256), torch.nn.GELU(), torch.nn.Linear(256, 1)
        ).to("cuda")
        optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)

        if "model" in state:
            model.load_state_dict(state["model"])
            optimizer.load_state_dict(state["optimizer"])

        return model, optimizer

    def _step(self, model: object, optimizer: object) -> float:
        torch = self.torch

        inputs = torch.randn(256, 64, device="cuda")
        targets = inputs.sum(dim=1, keepdim=True)

        loss = torch.nn.functional.mse_loss(model(inputs), targets)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        return float(loss.detach())

    def _save(self, job: str, model: object, optimizer: object, steps_done: int) -> None:
        buffer = io.BytesIO()
        self.torch.save(
            {
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "steps_done": steps_done,
            },
            buffer,
        )
        self.store.put_object(
            Bucket=BUCKET, Key=self._key(job), Body=buffer.getvalue()
        )
