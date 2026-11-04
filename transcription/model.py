"""Transcribing speech: an audio file in, text with timings out."""

from __future__ import annotations

import io
import urllib.request

from pydantic import BaseModel, ConfigDict, Field

from runware_serverless import serve

MODEL = "nvidia/parakeet-tdt-0.6b-v3"
"""CC-BY-4.0 and ungated, so this deploys with no token. The attribution the
license asks for falls on whoever deploys it, not on this file."""

REVISION = "541d1f99c6b0c3cd0b11a95167540bb8edefd82b"
"""Pinned, because a moving `main` is how you end up with weights the installed
version cannot load."""

CACHE = "/weights"

SAMPLE_RATE = 16000
"""What the feature extractor expects. Anything else is resampled on the way in."""

CHUNK_SECONDS = 300
"""How much audio crosses the GPU at once. Attention over the whole file is what
runs a worker out of memory, and a chunk boundary is the price: a word spoken
across one is split in two."""

TOKENS_PER_SECOND = 24
"""Headroom for `max_new_tokens`. Measured at about six on ordinary speech."""

MAX_SECONDS = 7200
MAX_BYTES = 512 * 1024 * 1024
"""Ceilings this app enforces. The platform checks the types it published and
not a value constraint, so a file past either of these is refused here, after a
worker has woken."""

FETCH_TIMEOUT = 120


class Request(BaseModel):
    # Without this pydantic ignores a field it does not know, so a caller who
    # misspells one gets a silent success. An unknown field is an error here.
    model_config = ConfigDict(extra="forbid")

    audio_url: str = Field(
        pattern=r"^https://",
        description="The audio to transcribe. The worker fetches it, so the file "
        "never crosses the request body.",
    )
    timestamps: bool = Field(
        default=True,
        description="Return a span per word, each with its start and end.",
    )


class Word(BaseModel):
    text: str
    start: float = Field(description="Seconds from the start of the file.")
    end: float


class Transcript(BaseModel):
    text: str = Field(description="The whole transcript, chunks joined in order.")
    words: list[Word] = Field(description="Empty when timestamps are off.")
    seconds: float = Field(description="Duration of the audio, not of the request.")
    chunks: int


@serve
class Transcription:
    requirements = [
        "torch==2.14.1",
        "transformers==5.22.1",
        "librosa==1.0.0",
        "soundfile==0.14.0",
        "soxr==1.1.0",
        "numpy==2.5.4",
    ]

    def load(self) -> None:
        if hasattr(self, "model"):
            return

        # Here and not at the top of the file. The build imports this file with
        # the platform's own dependencies, not yours, so nothing in requirements
        # exists until a worker runs.
        import numpy as np
        import torch
        from transformers import AutoModelForTDT, AutoProcessor

        self.torch = torch
        self.processor = AutoProcessor.from_pretrained(
            MODEL, revision=REVISION, cache_dir=CACHE
        )
        self.model = AutoModelForTDT.from_pretrained(
            MODEL, revision=REVISION, cache_dir=CACHE, dtype="auto"
        ).to("cuda")
        self.model.eval()

        print(f"warming up: transcribing one second of silence", flush=True)
        self._run(np.zeros(SAMPLE_RATE, dtype=np.float32))
        print("ready", flush=True)

    def predict(self, request: Request) -> Transcript:
        audio = self._fetch(request.audio_url)
        seconds = len(audio) / SAMPLE_RATE
        if seconds > MAX_SECONDS:
            raise ValueError(
                f"audio is {seconds:.0f}s, and this app accepts {MAX_SECONDS}s"
            )

        span = CHUNK_SECONDS * SAMPLE_RATE
        pieces = [audio[at : at + span] for at in range(0, len(audio), span)] or [audio]

        texts: list[str] = []
        words: list[Word] = []
        for index, piece in enumerate(pieces):
            text, spans = self._run(piece, want_spans=request.timestamps)
            texts.append(text)
            offset = index * CHUNK_SECONDS
            words.extend(
                Word(text=t, start=round(s + offset, 2), end=round(e + offset, 2))
                for t, s, e in spans
            )

        return Transcript(
            text=" ".join(t for t in texts if t).strip(),
            words=words,
            seconds=round(seconds, 2),
            chunks=len(pieces),
        )

    def _fetch(self, url: str):
        import numpy as np
        import soundfile
        import soxr

        with urllib.request.urlopen(url, timeout=FETCH_TIMEOUT) as answer:
            raw = answer.read(MAX_BYTES + 1)
        if len(raw) > MAX_BYTES:
            raise ValueError(f"the file is larger than {MAX_BYTES // (1024 * 1024)}MB")

        audio, rate = soundfile.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        audio = audio.mean(axis=1)
        if rate != SAMPLE_RATE:
            audio = soxr.resample(audio, rate, SAMPLE_RATE)
        return np.ascontiguousarray(audio, dtype=np.float32)

    def _run(self, audio, want_spans: bool = False):
        seconds = len(audio) / SAMPLE_RATE
        inputs = self.processor([audio], sampling_rate=SAMPLE_RATE)
        inputs = inputs.to(self.model.device, dtype=self.model.dtype)
        with self.torch.no_grad():
            output = self.model.generate(
                **inputs,
                return_dict_in_generate=True,
                # Without this, generation stops at a model-agnostic 1310 and a
                # long chunk comes back quietly cut off. Measured at roughly six
                # tokens a second, so this leaves room for fast speech.
                max_new_tokens=int(seconds * TOKENS_PER_SECOND) + 64,
            )

        if not want_spans:
            return _one(self.processor.decode(output.sequences, skip_special_tokens=True)), []

        texts, spans = self.processor.decode(
            output.sequences, durations=output.durations, skip_special_tokens=True
        )
        return _one(texts), _words(_one(spans))


def _one(decoded):
    """`decode` answers for a batch, and this app sends one item in it."""
    return decoded[0] if isinstance(decoded, (list, tuple)) and decoded else decoded


def _words(tokens):
    """Group the token spans into words.

    The transformers path returns one span per subword piece, ``He``, `` h``,
    ``op``, ``ed``, not the word and segment spans the NeMo API gives. A piece
    that opens with a space opens a word, which is what this tokenizer's own
    spelling already encodes.
    """
    words: list[tuple[str, float, float]] = []
    for piece in tokens or []:
        text, start, end = piece["token"], piece["start"], piece["end"]
        if not words or text.startswith(" "):
            words.append((text.strip(), start, end))
            continue
        was, began, _ = words[-1]
        words[-1] = (was + text, began, end)
    return [(text, start, end) for text, start, end in words if text]
