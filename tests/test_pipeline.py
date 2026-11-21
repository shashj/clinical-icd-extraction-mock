from dataclasses import replace

import pytest
from langfuse import Langfuse
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import SimpleSpanProcessor
from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

from icd_mock.config import make_offline_client
from icd_mock.dataset import load_cases
from icd_mock.pipeline import IcdPipeline
from icd_mock.prompts import REFERENCE_PROMPTS, LocalPromptStore

CASES = {case.id: case for case in load_cases()}
INDEX = {case.id: i for i, case in enumerate(load_cases())}


class RecordingStore(LocalPromptStore):
    """Remembers every compiled message so a test can check no variable was left unfilled."""

    def __init__(self) -> None:
        self.compiled: list[str] = []

    def get(self, name):
        handle = super().get(name)

        def compile_and_record(**variables):
            messages = handle.compile(**variables)
            self.compiled.extend(m["content"] for m in messages)
            return messages

        return replace(handle, compile=compile_and_record)


@pytest.fixture(scope="module")
def pipeline():
    return IcdPipeline(make_offline_client(), LocalPromptStore(), seed=42, run_id="test")


def names(result):
    return [step.name for step in result.steps]


def test_plain_case_runs_all_six_prompts_once(pipeline):
    result = pipeline.run_case(CASES["case-01"], INDEX["case-01"])
    assert names(result) == [
        "clean-note",
        "extract-paragraphs",
        "extract-diseases-treatments",
        "build-query",
        "vector-search",
        "select-code",
        "validate-code",
        "icd-extraction",
    ]
    assert result.final_code == "E11.9" and result.correct


def test_rejected_selection_is_retried(pipeline):
    result = pipeline.run_case(CASES["case-03"], INDEX["case-03"])
    assert names(result).count("select-code") == 2
    assert names(result).count("validate-code") == 2
    first_verdict = [s.output for s in result.steps if s.name == "validate-code"][0]
    assert '"reject"' in first_verdict
    assert result.final_code == "J18.9"


def test_empty_vector_search_is_retried_with_a_warning(pipeline):
    result = pipeline.run_case(CASES["case-07"], INDEX["case-07"])
    searches = [s for s in result.steps if s.name == "vector-search"]
    assert [s.level for s in searches] == ["WARNING", "DEFAULT"]
    assert searches[0].output == [] and searches[1].output


def test_wrong_final_selection_is_scored_incorrect(pipeline):
    result = pipeline.run_case(CASES["case-10"], INDEX["case-10"])
    assert result.final_code == "M54.59" and not result.correct


def test_injected_timeout_is_an_error_then_a_retry(pipeline):
    results = (pipeline.run_case(CASES["case-01"], i) for i in range(0, 300, 30))
    timed_out = next(r for r in results if r.timeout_injected)
    searches = [s for s in timed_out.steps if s.name == "vector-search"]
    assert searches[0].level == "ERROR" and searches[-1].level == "DEFAULT"


def test_same_index_gives_the_same_outputs(pipeline):
    first = pipeline.run_case(CASES["case-12"], 11)
    second = pipeline.run_case(CASES["case-12"], 11)
    assert [s.output for s in first.steps] == [s.output for s in second.steps]


def test_each_request_is_one_trace_with_a_real_root():
    # A root that has a parent id no span carries leaves the trace without a root in Langfuse.
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key="pk-root-test",
        secret_key="sk-root-test",
        host="http://localhost:1",
        tracer_provider=provider,
    )
    pipeline = IcdPipeline(client, LocalPromptStore(), seed=42, run_id="test")
    pipeline.run_case(CASES["case-03"], INDEX["case-03"])

    spans = exporter.get_finished_spans()
    roots = [s for s in spans if s.parent is None]
    assert [r.name for r in roots] == ["icd-extraction"]
    assert len({s.context.trace_id for s in spans}) == 1
    span_ids = {s.context.span_id for s in spans}
    assert all(s.parent.span_id in span_ids for s in spans if s.parent is not None)


def test_every_prompt_variable_is_filled_for_every_case():
    store = RecordingStore()
    pipeline = IcdPipeline(make_offline_client(), store, seed=1, run_id="test")
    for i, case in enumerate(load_cases()):
        pipeline.run_case(case, i)
    assert store.compiled and not any("{{" in text for text in store.compiled)


def test_six_reference_prompts_each_name_a_model():
    assert len(REFERENCE_PROMPTS) == 6
    assert all(spec.config.get("model") for spec in REFERENCE_PROMPTS)


class LinkedStore(LocalPromptStore):
    """Pretends every prompt came from Langfuse as version 7, with a linkable prompt client."""

    def get(self, name):
        client = type("PromptClient", (), {"name": name, "version": 7, "is_fallback": False})()
        return replace(super().get(name), version=7, label="production", client=client)

    def versions(self):
        return {name: 7 for name in super().versions()}


def test_traces_record_which_prompt_version_and_model_ran():
    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    client = Langfuse(
        public_key="pk-prompt-test",
        secret_key="sk-prompt-test",
        host="http://localhost:1",
        tracer_provider=provider,
    )
    IcdPipeline(client, LinkedStore(), seed=42, run_id="test").run_case(CASES["case-01"], 0)

    spans = exporter.get_finished_spans()
    generations = [
        s for s in spans if s.attributes.get("langfuse.observation.type") == "generation"
    ]
    assert len(generations) == 6
    for span in generations:
        attrs = span.attributes
        assert attrs["langfuse.observation.prompt.name"].startswith("icd-")
        assert attrs["langfuse.observation.prompt.version"] == 7
        assert attrs["langfuse.observation.metadata.prompt_version"] == "7"
        assert attrs["langfuse.observation.metadata.prompt_label"] == "production"
        assert attrs["langfuse.observation.model.name"]
    linked = {s.attributes["langfuse.observation.prompt.name"] for s in generations}
    assert linked == {spec.name for spec in REFERENCE_PROMPTS}
    root = next(s for s in spans if s.parent is None)
    assert '"icd-code-selection": 7' in root.attributes["langfuse.trace.metadata.prompt_versions"]
