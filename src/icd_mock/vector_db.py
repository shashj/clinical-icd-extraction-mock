"""A stand-in for the ICD-10-CM vector index: it returns the recorded candidates."""

from __future__ import annotations

from typing import Any

from .dataset import Case, load_code_descriptions

INDEX_NAME = "icd10cm-fy2027-mock"
SIMILARITY_THRESHOLD = 0.6


class VectorDBTimeout(Exception):
    """Raised for an injected timeout, like a slow index in production."""


class MockVectorDB:
    def __init__(self) -> None:
        self._descriptions = load_code_descriptions()

    def search(
        self, case: Case, query: str, *, miss: bool = False, timeout: bool = False, top_k: int = 5
    ) -> list[dict[str, Any]]:
        if timeout:
            raise VectorDBTimeout("vector search timed out after 5000 ms")
        if miss:
            return []
        return [
            {
                "code": candidate.code,
                "description": self._descriptions[candidate.code],
                "score": candidate.score,
            }
            for candidate in case.candidates[:top_k]
        ]
