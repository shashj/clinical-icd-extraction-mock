"""The six prompts, kept in Langfuse prompt management and fetched on every call."""

from __future__ import annotations

import re
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from langfuse import Langfuse

PRODUCTION_LABEL = "production"


@dataclass(frozen=True)
class PromptSpec:
    name: str
    messages: tuple[tuple[str, str], ...]  # (role, content)
    config: dict[str, Any]


SPECS: tuple[PromptSpec, ...] = (
    PromptSpec(
        name="icd-note-cleaning",
        messages=(
            (
                "system",
                "You prepare clinical notes for diagnosis coding. Remove the header line and any "
                "identifiers, expand abbreviations, and keep every clinical fact unchanged. "
                "Return only the cleaned note.",
            ),
            ("user", "{{raw_note}}"),
        ),
        config={"model": "gpt-4o-mini", "temperature": 0},
    ),
    PromptSpec(
        name="icd-paragraph-extraction",
        messages=(
            (
                "system",
                "You select the paragraphs of a clinical note that matter for assigning diagnosis "
                "codes: the history of the presenting problem, objective findings, and the "
                "assessment or plan. Ignore administrative, scheduling and social details. Return "
                "a JSON array containing the selected paragraphs verbatim.",
            ),
            ("user", "{{cleaned_note}}"),
        ),
        config={"model": "gemini-2.5-flash", "temperature": 0},
    ),
    PromptSpec(
        name="icd-disease-treatment-extraction",
        messages=(
            (
                "system",
                "Extract every disease or condition the clinician diagnoses and what is being "
                "treated and how. Return JSON with the keys diseases (a list of names) and "
                "treated (a list of objects with condition and treatment). Do not infer diagnoses "
                "that are not stated.",
            ),
            ("user", "{{paragraphs}}"),
        ),
        config={"model": "claude-sonnet-4-5-20250929", "temperature": 0},
    ),
    PromptSpec(
        name="icd-query-builder",
        messages=(
            (
                "system",
                "Write one short search query for an ICD-10-CM vector index from the extracted "
                "diagnoses. Prefer the clinician's wording and include qualifiers such as "
                "laterality, severity, acuity and complications. Return JSON with the key query.",
            ),
            ("user", "Diagnoses: {{diseases}}\nTreatments: {{treated}}"),
        ),
        config={"model": "gpt-4o-mini", "temperature": 0},
    ),
    PromptSpec(
        name="icd-code-selection",
        messages=(
            (
                "system",
                "You are a certified clinical coder. Choose the single best ICD-10-CM code from "
                "the candidates for the documented diagnosis. Choose the most specific code that "
                "the documentation supports and never assume details that are not written. "
                "Return JSON with the keys code and rationale.",
            ),
            (
                "user",
                "Relevant paragraphs:\n{{paragraphs}}\n\nDiagnoses and treatments:\n{{extraction}}"
                "\n\nCandidates:\n{{candidates}}\n\nFeedback on a previous attempt: {{feedback}}",
            ),
        ),
        config={"model": "gpt-4.1", "temperature": 0},
    ),
    PromptSpec(
        name="icd-code-validation",
        messages=(
            (
                "system",
                "You audit ICD-10-CM code assignments. Check the chosen code against the clinical "
                "note and the candidate list. Accept only if the documentation supports the code "
                "and no more specific candidate applies. Return JSON with the keys verdict "
                "(accept or reject) and reason.",
            ),
            (
                "user",
                "Relevant paragraphs:\n{{paragraphs}}\n\nChosen code: {{code}} {{description}}"
                "\n\nCandidates:\n{{candidates}}",
            ),
        ),
        config={"model": "o3-mini", "max_tokens": 1000},
    ),
)

SPECS_BY_NAME = {spec.name: spec for spec in SPECS}


def _substitute(template: str, variables: dict[str, str]) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(variables.get(m.group(1), m.group(0))), template)


@dataclass(frozen=True)
class PromptHandle:
    name: str
    config: dict[str, Any]
    compile: Callable[..., list[dict[str, Any]]]
    # The Langfuse prompt client to link a generation to, or None offline.
    client: Any | None = None


class LocalPromptStore:
    """Compiles the prompts defined in this file. Used for dry runs and tests."""

    def get(self, name: str) -> PromptHandle:
        spec = SPECS_BY_NAME[name]

        def compile_messages(**variables: str) -> list[dict[str, Any]]:
            return [
                {"role": role, "content": _substitute(content, variables)}
                for role, content in spec.messages
            ]

        return PromptHandle(name=name, config=dict(spec.config), compile=compile_messages)


class LangfusePromptStore:
    """Fetches the production version of each prompt from Langfuse prompt management."""

    def __init__(self, langfuse: Langfuse) -> None:
        self._langfuse = langfuse

    def get(self, name: str) -> PromptHandle:
        client = self._langfuse.get_prompt(name, label=PRODUCTION_LABEL, type="chat")
        return PromptHandle(
            name=name,
            config=dict(client.config or {}),
            compile=client.compile,
            client=client,
        )


def _existing_messages(client: Any) -> list[tuple[str, str]]:
    return [
        (m["role"], m["content"]) for m in client.prompt if m.get("type", "message") == "message"
    ]


def sync_prompts(langfuse: Langfuse) -> dict[str, str]:
    """Create or update the six prompts in Langfuse. Returns name -> created, updated, unchanged."""
    result: dict[str, str] = {}
    for spec in SPECS:
        status = "created"
        try:
            existing = langfuse.get_prompt(
                spec.name,
                label=PRODUCTION_LABEL,
                type="chat",
                cache_ttl_seconds=0,
                max_retries=0,
            )
        except Exception:  # noqa: BLE001 - a missing prompt raises, anything else fails on create
            existing = None
        if existing is not None:
            same = _existing_messages(existing) == list(spec.messages) and (
                dict(existing.config or {}) == spec.config
            )
            if same:
                result[spec.name] = "unchanged"
                continue
            status = "updated"
        langfuse.create_prompt(
            name=spec.name,
            type="chat",
            prompt=[{"role": role, "content": content} for role, content in spec.messages],
            labels=[PRODUCTION_LABEL],
            tags=["icd-extraction"],
            config=spec.config,
            commit_message="Synced from clinical-icd-extraction-mock",
        )
        result[spec.name] = status
    return result
