"""A language model behind an endpoint whose shape you decide."""

from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from runware_serverless import serve

MODEL = "Qwen/Qwen3-4B-Instruct-2507"
"""Apache 2.0 and ungated, so this deploys with no token of any kind."""

CACHE = "/weights"
"""Mount a volume here, the same as any other app with weights."""

DEFAULT_TOKENS = 512
DEFAULT_TEMPERATURE = 0.7


class Message(BaseModel):
    role: Literal["system", "user", "assistant"]
    content: str


class Request(BaseModel):
    messages: list[Message] = Field(description="The conversation so far, oldest first.")
    max_tokens: int = Field(default=DEFAULT_TOKENS, ge=1, le=4096)
    temperature: float = Field(default=DEFAULT_TEMPERATURE, ge=0, le=2)


class Reply(BaseModel):
    content: str
    tokens: int = Field(description="Tokens generated, not counting the prompt.")


@serve
class ChatEndpoint:
    requirements = [
        "torch==2.14.1",
        "transformers==5.21.0",
        "accelerate==1.15.0",
    ]

    def load(self) -> None:
        if hasattr(self, "model"):
            return

        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=CACHE)
        self.model = AutoModelForCausalLM.from_pretrained(
            MODEL, torch_dtype=torch.bfloat16, cache_dir=CACHE
        ).to("cuda")
        self.model.eval()

    def predict(self, request: Request) -> Reply:
        import torch

        # The template is the model's own, shipped with its tokenizer. The turn
        # markers a model was trained on are not yours to guess at, and a prompt
        # assembled by hand is how an instruct model ends up answering nothing.
        prompt = self.tokenizer.apply_chat_template(
            [message.model_dump() for message in request.messages],
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = self.tokenizer([prompt], return_tensors="pt").to(self.model.device)

        with torch.inference_mode():
            produced = self.model.generate(
                **inputs,
                max_new_tokens=request.max_tokens,
                temperature=request.temperature,
                do_sample=request.temperature > 0,
            )

        # generate returns the prompt back with the answer appended, so the answer
        # is what sits past the input length.
        answer = produced[0][inputs.input_ids.shape[-1] :]

        return Reply(
            content=self.tokenizer.decode(answer, skip_special_tokens=True),
            tokens=len(answer),
        )
