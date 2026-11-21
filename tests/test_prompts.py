import pytest

from icd_mock.prompts import (
    PROMPT_NAMES,
    LangfusePromptStore,
    PromptSetupError,
    parse_prompt_model,
)


def test_a_registry_model_is_read_from_the_prompt_response():
    model = parse_prompt_model(
        "icd-code-selection",
        3,
        {
            "registryModelId": "reg-1",
            "model": {"provider": "openai", "modelId": "gpt-4.1", "params": {"temperature": 0}},
        },
    )
    assert (model.provider, model.model_id, model.params) == (
        "openai",
        "gpt-4.1",
        {"temperature": 0},
    )
    assert model.registry_model_id == "reg-1"
    assert model.label == "openai/gpt-4.1"


def test_a_prompt_without_a_model_is_refused_with_the_fix():
    with pytest.raises(PromptSetupError, match=r"icd-code-selection.*no model.*registry"):
        parse_prompt_model("icd-code-selection", 3, {"model": None})


class FakeApiPrompt:
    def __init__(self, data):
        self._data = data

    def model_dump(self, **_):
        return self._data


class FakeLangfuse:
    """Serves some prompts, with or without a model, and has no others."""

    def __init__(self, with_model: set[str], without_model: set[str]):
        self.with_model, self.without_model = with_model, without_model
        self.api = type("Api", (), {"prompts": self})()

    def get_prompt(self, name, **_):
        if name not in self.with_model | self.without_model:
            raise LookupError(name)
        return type("Client", (), {"name": name, "version": 2, "compile": lambda **_: []})()

    def get(self, name, **_):
        model = {"provider": "openai", "modelId": "gpt-4o-mini", "params": {}}
        return FakeApiPrompt({"model": model if name in self.with_model else None})


def test_load_reports_every_missing_and_model_less_prompt_at_once():
    langfuse = FakeLangfuse(with_model={PROMPT_NAMES[0]}, without_model={PROMPT_NAMES[1]})
    with pytest.raises(PromptSetupError) as raised:
        LangfusePromptStore(langfuse).load()
    message = str(raised.value)
    assert PROMPT_NAMES[1] in message and "no model" in message
    assert all(name in message for name in PROMPT_NAMES[2:])
    assert "prompts show" in message


def test_load_returns_a_handle_with_version_and_model_for_each_prompt():
    langfuse = FakeLangfuse(with_model=set(PROMPT_NAMES), without_model=set())
    store = LangfusePromptStore(langfuse)
    handles = store.load()
    assert set(handles) == set(PROMPT_NAMES)
    assert store.versions() == {name: 2 for name in PROMPT_NAMES}
    assert all(h.model.label == "openai/gpt-4o-mini" for h in handles.values())
