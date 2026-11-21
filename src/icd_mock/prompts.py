"""The six prompts. Langfuse owns them: the service fetches each one, with its model, before a run.

`REFERENCE_PROMPTS` below is a built-in copy used only for dry runs, the tests and
`icd-mock prompts show`, which prints it so you can create the prompts in Langfuse.
"""

from __future__ import annotations

import json
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


REFERENCE_PROMPTS: tuple[PromptSpec, ...] = (
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

REFERENCE_BY_NAME = {spec.name: spec for spec in REFERENCE_PROMPTS}
PROMPT_NAMES = tuple(spec.name for spec in REFERENCE_PROMPTS)


def _substitute(template: str, variables: dict[str, str]) -> str:
    return re.sub(r"\{\{(\w+)\}\}", lambda m: str(variables.get(m.group(1), m.group(0))), template)


def prompt_variables(spec: PromptSpec) -> list[str]:
    found = {name for _, content in spec.messages for name in re.findall(r"\{\{(\w+)\}\}", content)}
    return sorted(found)


class PromptSetupError(Exception):
    """A prompt is missing in Langfuse or has no model. The message says how to fix it."""


@dataclass(frozen=True)
class PromptModel:
    """The model a prompt version runs on, as resolved from the Langfuse model registry."""

    provider: str
    model_id: str
    params: dict[str, Any]
    registry_model_id: str | None = None

    @property
    def label(self) -> str:
        return f"{self.provider}/{self.model_id}"


@dataclass(frozen=True)
class PromptHandle:
    name: str
    version: int | None
    label: str
    model: PromptModel
    compile: Callable[..., list[dict[str, Any]]]
    # The Langfuse prompt client to link a generation to, or None offline.
    client: Any | None = None


def parse_prompt_model(name: str, version: int | None, raw: dict[str, Any]) -> PromptModel:
    """Read the registry model from a prompt API response. Raises if the prompt has none."""
    model = raw.get("model")
    if not isinstance(model, dict) or not model.get("modelId"):
        raise PromptSetupError(
            f"Prompt {name!r} (version {version}) has no model. Attach a model from the model "
            "registry to the prompt version in Langfuse. This needs a Langfuse with the model "
            "registry."
        )
    return PromptModel(
        provider=str(model.get("provider", "")),
        model_id=str(model["modelId"]),
        params=dict(model.get("params") or {}),
        registry_model_id=raw.get("registryModelId"),
    )


class LocalPromptStore:
    """The built-in reference prompts. Used for dry runs and tests, never for a real run."""

    def get(self, name: str) -> PromptHandle:
        spec = REFERENCE_BY_NAME[name]

        def compile_messages(**variables: str) -> list[dict[str, Any]]:
            return [
                {"role": role, "content": _substitute(content, variables)}
                for role, content in spec.messages
            ]

        params = {k: v for k, v in spec.config.items() if k in ("temperature", "max_tokens")}
        return PromptHandle(
            name=name,
            version=None,
            label="reference",
            model=PromptModel(provider="reference", model_id=spec.config["model"], params=params),
            compile=compile_messages,
        )

    def versions(self) -> dict[str, int | None]:
        return {name: None for name in PROMPT_NAMES}


def _as_dict(raw: Any) -> dict[str, Any]:
    dump = getattr(raw, "model_dump", None) or raw.dict
    return dict(dump(by_alias=True))


class LangfusePromptStore:
    """Fetches the production version of each prompt, with its model, from Langfuse.

    Call `load()` once before a run. The run then uses exactly those versions throughout.
    """

    def __init__(self, langfuse: Langfuse, names: tuple[str, ...] = PROMPT_NAMES) -> None:
        self._langfuse = langfuse
        self._names = names
        self._handles: dict[str, PromptHandle] = {}

    def load(self) -> dict[str, PromptHandle]:
        missing: list[str] = []
        problems: list[str] = []
        handles: dict[str, PromptHandle] = {}
        for name in self._names:
            try:
                client = self._langfuse.get_prompt(
                    name, label=PRODUCTION_LABEL, type="chat", cache_ttl_seconds=0
                )
                raw = _as_dict(self._langfuse.api.prompts.get(name, label=PRODUCTION_LABEL))
            except Exception:  # noqa: BLE001 - a missing prompt raises, anything else surfaces below
                missing.append(name)
                continue
            try:
                model = parse_prompt_model(name, client.version, raw)
            except PromptSetupError as exc:
                problems.append(str(exc))
                continue
            handles[name] = PromptHandle(
                name=name,
                version=client.version,
                label=PRODUCTION_LABEL,
                model=model,
                compile=client.compile,
                client=client,
            )
        if missing:
            problems.insert(
                0,
                "Not found in Langfuse with the 'production' label: "
                + ", ".join(missing)
                + ". Create them in Langfuse (run `icd-mock prompts show` for the reference text).",
            )
        if problems:
            raise PromptSetupError("\n".join(problems))
        self._handles = handles
        return handles

    def get(self, name: str) -> PromptHandle:
        return self._handles[name]

    def versions(self) -> dict[str, int | None]:
        return {name: handle.version for name, handle in self._handles.items()}


def reference_prompts_json() -> str:
    """The reference prompts as JSON, to create them in Langfuse."""
    return json.dumps(
        [
            {
                "name": spec.name,
                "type": "chat",
                "labels": [PRODUCTION_LABEL],
                "variables": prompt_variables(spec),
                "prompt": [{"role": role, "content": content} for role, content in spec.messages],
                "suggested_model": spec.config["model"],
            }
            for spec in REFERENCE_PROMPTS
        ],
        indent=2,
    )
