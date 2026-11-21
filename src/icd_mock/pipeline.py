"""The six-prompt clinical ICD extraction pipeline, with every model call replaced by a mock.

One trace per clinical note:

    icd-extraction (chain)
    |- clean-note                   generation   icd-note-cleaning
    |- extract-paragraphs           generation   icd-paragraph-extraction
    |- extract-diseases-treatments  generation   icd-disease-treatment-extraction
    |- build-query                  generation   icd-query-builder
    |- vector-search                retriever    (ICD-10-CM candidates; may time out or be empty)
    |- select-code                  generation   icd-code-selection
    |- validate-code                generation   icd-code-validation
    |- (select-code, validate-code again when validation rejects the first choice)
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

from langfuse import Langfuse

from .dataset import DATASET_VERSION, Case, Selection, load_code_descriptions
from .mock_llm import MockLLM
from .prompts import LangfusePromptStore, LocalPromptStore
from .vector_db import INDEX_NAME, SIMILARITY_THRESHOLD, MockVectorDB, VectorDBTimeout

TIMEOUT_RATE = 0.03
# Langfuse orders observations by start time, which has millisecond precision, so every step
# waits a moment before it starts. Without this, steps of a fast mock run tie and appear shuffled.
MIN_STEP_GAP_SECONDS = 0.003
SEARCH_LATENCY_MS = 90
TOP_K = 5
CLINICIANS = 12
MODEL_PARAMETER_KEYS = ("temperature", "max_tokens", "top_p")


def _json(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False)


@dataclass
class StepRecord:
    name: str
    kind: str  # generation, retriever or chain
    output: Any
    level: str = "DEFAULT"


@dataclass
class PipelineResult:
    case_id: str
    trace_id: str | None
    final_code: str
    expected_code: str
    selection_attempts: int
    retrieval_attempts: int
    timeout_injected: bool
    steps: list[StepRecord] = field(default_factory=list)

    @property
    def correct(self) -> bool:
        return self.final_code == self.expected_code

    def count(self, kind: str) -> int:
        return sum(1 for step in self.steps if step.kind == kind)


class IcdPipeline:
    def __init__(
        self,
        langfuse: Langfuse,
        prompts: LangfusePromptStore | LocalPromptStore,
        *,
        seed: int,
        run_id: str,
        latency_scale: float = 0.0,
    ) -> None:
        self._lf = langfuse
        self._prompts = prompts
        self._seed = seed
        self._run_id = run_id
        self._llm = MockLLM(seed)
        self._vector_db = MockVectorDB()
        self._descriptions = load_code_descriptions()
        self._latency_scale = latency_scale

    def run_case(self, case: Case, index: int) -> PipelineResult:
        rng = random.Random(f"{self._seed}:trace:{index}")
        timeout_injected = rng.random() < TIMEOUT_RATE
        user_id = f"clinician-{rng.randint(1, CLINICIANS):02d}"
        session_id = f"encounter-{self._run_id}-{index // 3}"
        key = str(index)
        steps: list[StepRecord] = []
        tags = ["icd-extraction", case.specialty.replace(" ", "-")]
        metadata = {
            "case_id": case.id,
            "specialty": case.specialty,
            "patient_age": str(case.patient.age),
            "dataset_version": DATASET_VERSION,
            "run_seed": str(self._seed),
            "run_id": self._run_id,
            "prompt_versions": json.dumps(self._prompts.versions(), sort_keys=True),
        }
        note_input = {"case_id": case.id, "note": case.raw_note}

        # No trace_context here: passing only a trace id gives the root a parent span that does not
        # exist, so the trace would have no real root. The SDK creates the trace from this span.
        with self._lf.start_as_current_observation(
            name="icd-extraction",
            as_type="chain",
            input=note_input,
            metadata=metadata,
        ) as root:
            root.update_trace(
                name="icd-extraction",
                user_id=user_id,
                session_id=session_id,
                tags=tags,
                metadata=metadata,
                input=note_input,
            )

            cleaned = self._generation(
                steps, "clean-note", "icd-note-cleaning",
                {"raw_note": case.raw_note}, case.cleaned_note, key,
            )  # fmt: skip
            paragraphs_text = _json(case.relevant_paragraphs)
            self._generation(
                steps, "extract-paragraphs", "icd-paragraph-extraction",
                {"cleaned_note": cleaned}, paragraphs_text, key,
            )  # fmt: skip
            treated = [item.model_dump() for item in case.treated]
            extraction_text = _json({"diseases": case.diseases, "treated": treated})
            self._generation(
                steps, "extract-diseases-treatments", "icd-disease-treatment-extraction",
                {"paragraphs": paragraphs_text}, extraction_text, key,
            )  # fmt: skip
            self._generation(
                steps, "build-query", "icd-query-builder",
                {"diseases": _json(case.diseases), "treated": _json(treated)},
                _json({"query": case.query}), key,
            )  # fmt: skip

            hits, retrieval_attempts = self._retrieve(steps, case, timeout_injected)
            candidates_text = _json(hits)

            selection_attempts = 0
            feedback = "none"
            attempts: list[tuple[Selection, str | None]] = []
            if case.first_selection is not None:
                attempts.append((case.first_selection, case.first_validation_reason))
            attempts.append((case.selection, None))
            for selection, reject_reason in attempts:
                selection_attempts += 1
                attempt_key = f"{key}:{selection_attempts}"
                self._generation(
                    steps, "select-code", "icd-code-selection",
                    {
                        "paragraphs": paragraphs_text,
                        "extraction": extraction_text,
                        "candidates": candidates_text,
                        "feedback": feedback,
                    },
                    _json({"code": selection.code, "rationale": selection.rationale}),
                    attempt_key,
                )  # fmt: skip
                verdict = (
                    {"verdict": "reject", "reason": reject_reason}
                    if reject_reason
                    else case.validation.model_dump()
                )
                self._generation(
                    steps, "validate-code", "icd-code-validation",
                    {
                        "paragraphs": paragraphs_text,
                        "code": selection.code,
                        "description": self._descriptions[selection.code],
                        "candidates": candidates_text,
                    },
                    _json(verdict), attempt_key,
                )  # fmt: skip
                if verdict["verdict"] == "reject":
                    feedback = f"Rejected. {verdict['reason']}"

            final_code = case.selection.code
            output = {
                "icd_code": final_code,
                "description": self._descriptions[final_code],
                "validated": True,
                "selection_attempts": selection_attempts,
            }
            root.update(output=output)
            root.update_trace(output=output)
            root.score_trace(
                name="icd_correct",
                value=1.0 if final_code == case.expected_code else 0.0,
                data_type="BOOLEAN",
                comment=f"expected {case.expected_code}",
            )
            root.score_trace(name="validation_passed", value=1.0, data_type="BOOLEAN")
            root.score_trace(
                name="selection_attempts",
                value=float(selection_attempts),
                data_type="NUMERIC",
            )
            steps.append(StepRecord("icd-extraction", "chain", output))
            result_trace_id = root.trace_id

        return PipelineResult(
            case_id=case.id,
            trace_id=result_trace_id,
            final_code=final_code,
            expected_code=case.expected_code,
            selection_attempts=selection_attempts,
            retrieval_attempts=retrieval_attempts,
            timeout_injected=timeout_injected,
            steps=steps,
        )

    def _generation(
        self,
        steps: list[StepRecord],
        name: str,
        prompt_name: str,
        variables: dict[str, str],
        recorded_output: str,
        key: str,
    ) -> str:
        time.sleep(MIN_STEP_GAP_SECONDS)
        handle = self._prompts.get(prompt_name)
        messages = handle.compile(**variables)
        # The model and its parameters come from the model registry entry on the prompt version.
        parameters = {k: v for k, v in handle.model.params.items() if k in MODEL_PARAMETER_KEYS}
        # `prompt=` links the generation to the prompt version in Langfuse. The same facts go into
        # metadata so they can be read and filtered without opening the prompt.
        with self._lf.start_as_current_observation(
            name=name,
            as_type="generation",
            model=handle.model.model_id,
            model_parameters=parameters or None,
            input=messages,
            prompt=handle.client,
            metadata={
                "prompt_name": handle.name,
                "prompt_version": "reference" if handle.version is None else str(handle.version),
                "prompt_label": handle.label,
                "model_provider": handle.model.provider,
                "registry_model_id": handle.model.registry_model_id or "",
            },
        ) as generation:
            completion = self._llm.complete(name, messages, recorded_output, key)
            first_token_at = None
            if self._latency_scale > 0:
                time.sleep(completion.first_token_ms / 1000 * self._latency_scale)
                first_token_at = datetime.now(UTC)
                remaining = completion.latency_ms - completion.first_token_ms
                time.sleep(remaining / 1000 * self._latency_scale)
            generation.update(
                output=completion.text,
                usage_details=completion.usage,
                completion_start_time=first_token_at,
            )
        steps.append(StepRecord(name, "generation", completion.text))
        return completion.text

    def _retrieve(
        self, steps: list[StepRecord], case: Case, timeout_injected: bool
    ) -> tuple[list[dict[str, Any]], int]:
        attempts = 0
        query = case.query
        if timeout_injected:
            attempts += 1
            self._search(steps, case, query, attempts, timeout=True)
        if case.is_retrieval_miss:
            attempts += 1
            self._search(steps, case, query, attempts, miss=True)
            query = case.retry_query or query
        attempts += 1
        hits = self._search(steps, case, query, attempts)
        return hits or [], attempts

    def _search(
        self,
        steps: list[StepRecord],
        case: Case,
        query: str,
        attempt: int,
        *,
        miss: bool = False,
        timeout: bool = False,
    ) -> list[dict[str, Any]] | None:
        time.sleep(MIN_STEP_GAP_SECONDS)
        with self._lf.start_as_current_observation(
            name="vector-search",
            as_type="retriever",
            input={"query": query, "top_k": TOP_K, "index": INDEX_NAME, "attempt": attempt},
            metadata={"vector_db": "mock-vector-index", "threshold": str(SIMILARITY_THRESHOLD)},
        ) as span:
            time.sleep(SEARCH_LATENCY_MS / 1000 * self._latency_scale)
            try:
                hits = self._vector_db.search(case, query, miss=miss, timeout=timeout, top_k=TOP_K)
            except VectorDBTimeout as exc:
                span.update(level="ERROR", status_message=str(exc), output={"error": "timeout"})
                steps.append(StepRecord("vector-search", "retriever", None, level="ERROR"))
                return None
            if not hits:
                span.update(
                    level="WARNING",
                    status_message=f"No candidates above similarity {SIMILARITY_THRESHOLD}",
                    output=[],
                )
                steps.append(StepRecord("vector-search", "retriever", [], level="WARNING"))
                return []
            span.update(output=hits)
        steps.append(StepRecord("vector-search", "retriever", hits))
        return hits
