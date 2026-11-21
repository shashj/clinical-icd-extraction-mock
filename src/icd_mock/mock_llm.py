"""A stand-in for the model provider: it returns recorded text, never calls a model."""

from __future__ import annotations

import json
import math
import random
from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Completion:
    text: str
    usage: dict[str, int]
    first_token_ms: int
    latency_ms: int


def _tokens(text: str) -> int:
    # Roughly four characters per token.
    return max(1, math.ceil(len(text) / 4))


class MockLLM:
    def __init__(self, seed: int) -> None:
        self._seed = seed

    def complete(
        self, step: str, messages: list[dict[str, Any]], text: str, key: str
    ) -> Completion:
        """Return `text` with usage and latency derived from its length and a seeded jitter."""
        rng = random.Random(f"{self._seed}:{key}:{step}")
        prompt_tokens = round(_tokens(json.dumps(messages)) * rng.uniform(0.94, 1.06))
        output_tokens = round(_tokens(text) * rng.uniform(0.94, 1.06))
        first_token_ms = rng.randint(120, 520)
        latency_ms = first_token_ms + output_tokens * rng.randint(9, 16)
        return Completion(
            text=text,
            usage={
                "input": prompt_tokens,
                "output": output_tokens,
                "total": prompt_tokens + output_tokens,
            },
            first_token_ms=first_token_ms,
            latency_ms=latency_ms,
        )
