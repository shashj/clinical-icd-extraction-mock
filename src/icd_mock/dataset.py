"""The hand-written dataset: 30 synthetic clinical cases and their recorded step outputs."""

from __future__ import annotations

import json
import os
from functools import cache
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, Field, model_validator

DATASET_VERSION = "2027.1"
ICD_RELEASE = "ICD-10-CM FY2027"


def data_dir() -> Path:
    override = os.environ.get("ICD_MOCK_DATA_DIR")
    if override:
        return Path(override)
    return Path(__file__).resolve().parents[2] / "data"


class Patient(BaseModel):
    age: int = Field(ge=0, le=120)
    sex: Literal["female", "male"]


class TreatedItem(BaseModel):
    condition: str
    treatment: str


class Candidate(BaseModel):
    code: str
    score: float = Field(ge=0, le=1)


class Selection(BaseModel):
    code: str
    rationale: str


class Validation(BaseModel):
    verdict: Literal["accept", "reject"]
    reason: str


class Case(BaseModel):
    id: str
    specialty: str
    patient: Patient
    paragraphs: list[str] = Field(min_length=5, max_length=5)
    relevant: list[int]
    diseases: list[str] = Field(min_length=1)
    treated: list[TreatedItem] = Field(min_length=1)
    query: str
    # Set when the first vector search returns nothing and the query is rewritten.
    retry_query: str | None = None
    candidates: list[Candidate] = Field(min_length=5, max_length=5)
    expected_code: str
    # Set when the first selection is wrong and validation rejects it.
    first_selection: Selection | None = None
    first_validation_reason: str | None = None
    selection: Selection
    validation: Validation

    @model_validator(mode="after")
    def _consistent(self) -> Case:
        codes = [c.code for c in self.candidates]
        scores = [c.score for c in self.candidates]
        if scores != sorted(scores, reverse=True):
            raise ValueError(f"{self.id}: candidates must be sorted by descending score")
        if len(set(codes)) != len(codes):
            raise ValueError(f"{self.id}: duplicate candidate codes")
        if self.expected_code not in codes:
            raise ValueError(f"{self.id}: expected code is not among the candidates")
        if self.selection.code not in codes:
            raise ValueError(f"{self.id}: final selection is not among the candidates")
        if any(i < 0 or i >= len(self.paragraphs) for i in self.relevant):
            raise ValueError(f"{self.id}: relevant paragraph index out of range")
        if self.validation.verdict != "accept":
            raise ValueError(f"{self.id}: the final validation must accept")
        if (self.first_selection is None) != (self.first_validation_reason is None):
            raise ValueError(f"{self.id}: first_selection and first_validation_reason go together")
        if self.first_selection is not None:
            if self.first_selection.code not in codes:
                raise ValueError(f"{self.id}: first selection is not among the candidates")
            if self.first_selection.code == self.selection.code:
                raise ValueError(f"{self.id}: first selection must differ from the final one")
        return self

    # Derived views of the recorded steps.

    @property
    def body(self) -> str:
        return "\n\n".join(p.strip() for p in self.paragraphs)

    @property
    def raw_note(self) -> str:
        sex = self.patient.sex[0].upper()
        header = f"CLINIC NOTE {self.id.upper()} | {self.specialty} | {self.patient.age}{sex}"
        return f"{header}\n\n{self.body}"

    @property
    def cleaned_note(self) -> str:
        return self.body

    @property
    def relevant_paragraphs(self) -> list[str]:
        return [self.paragraphs[i].strip() for i in self.relevant]

    @property
    def is_loop(self) -> bool:
        return self.first_selection is not None

    @property
    def is_retrieval_miss(self) -> bool:
        return self.retry_query is not None

    @property
    def is_wrong_final(self) -> bool:
        return self.selection.code != self.expected_code


@cache
def load_cases() -> tuple[Case, ...]:
    cases: list[Case] = []
    for path in sorted((data_dir() / "cases").glob("*.json")):
        for raw in json.loads(path.read_text(encoding="utf-8")):
            cases.append(Case.model_validate(raw))
    ids = [c.id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError("duplicate case ids")
    return tuple(cases)


@cache
def load_code_descriptions() -> dict[str, str]:
    """Official ICD-10-CM descriptions for every code the dataset uses."""
    subset = json.loads((data_dir() / "icd10cm_subset.json").read_text(encoding="utf-8"))
    return dict(subset["codes"])


def used_codes(cases: tuple[Case, ...] | None = None) -> set[str]:
    codes: set[str] = set()
    for case in cases or load_cases():
        codes.update(c.code for c in case.candidates)
        codes.add(case.expected_code)
        codes.add(case.selection.code)
        if case.first_selection:
            codes.add(case.first_selection.code)
    return codes
