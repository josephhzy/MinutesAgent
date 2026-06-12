"""
Unit tests for minute_writer.

These tests never call OpenAI. They inject fake LLM / agent objects so the
prompt-building, input-validation, error-handling and orchestration logic can
be verified offline, with no API key and no network.

Run with:  pytest -v
"""

import pytest

import minute_writer as mw


# --- Test doubles -----------------------------------------------------------

class FakeMessage:
    """Stand-in for a LangChain message object (just needs `.content`)."""

    def __init__(self, content):
        self.content = content


class FakeLLM:
    """Records the prompts it receives and returns a canned reply."""

    def __init__(self, reply="FAKE LLM OUTPUT"):
        self.reply = reply
        self.prompts = []

    def invoke(self, prompt):
        self.prompts.append(prompt)
        return FakeMessage(self.reply)


class BoomLLM:
    """An LLM whose call always fails, to test error handling."""

    def invoke(self, prompt):
        raise RuntimeError("simulated provider outage")


class FakeAgent:
    """Stand-in for the agent graph returned by create_agent."""

    def __init__(self, final="## Meeting Summary\n...\n## Action Items\n1. ..."):
        self.final = final
        self.received = None

    def invoke(self, payload):
        self.received = payload
        # Mirror the real shape: result["messages"][-1].content is the answer.
        return {"messages": [FakeMessage("intermediate"), FakeMessage(self.final)]}


# --- validate_notes ---------------------------------------------------------

@pytest.mark.parametrize("bad", ["", "   ", "\n\t  "])
def test_validate_notes_rejects_empty(bad):
    with pytest.raises(ValueError):
        mw.validate_notes(bad)


@pytest.mark.parametrize("bad", [None, 123, ["notes"]])
def test_validate_notes_rejects_non_string(bad):
    with pytest.raises(ValueError):
        mw.validate_notes(bad)


def test_validate_notes_strips_whitespace():
    assert mw.validate_notes("  hello world  ") == "hello world"


# --- load_api_key -----------------------------------------------------------

def test_load_api_key_missing(monkeypatch):
    # Stop the real .env from being read, then clear the variable.
    monkeypatch.setattr(mw, "load_dotenv", lambda *a, **k: None)
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError):
        mw.load_api_key()


def test_load_api_key_rejects_placeholder(monkeypatch):
    monkeypatch.setattr(mw, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-your-actual-key-here")
    with pytest.raises(RuntimeError):
        mw.load_api_key()


def test_load_api_key_accepts_real_key(monkeypatch):
    monkeypatch.setattr(mw, "load_dotenv", lambda *a, **k: None)
    monkeypatch.setenv("OPENAI_API_KEY", "sk-real-abc123")
    assert mw.load_api_key() == "sk-real-abc123"


# --- summarize_notes / extract_items ---------------------------------------

def test_summarize_notes_calls_llm_with_notes_in_prompt():
    fake = FakeLLM("SUMMARY TEXT")
    out = mw.summarize_notes("we discussed the Q1 roadmap", llm=fake)
    assert out == "SUMMARY TEXT"
    # The raw notes must be embedded in the prompt sent to the LLM.
    assert "we discussed the Q1 roadmap" in fake.prompts[0]
    # And the summary instructions, not the action-item ones, were used.
    assert "structured meeting summary" in fake.prompts[0]


def test_extract_items_calls_llm_with_action_item_template():
    fake = FakeLLM("1. Mike - send timeline - Friday")
    out = mw.extract_items("Mike to send the timeline by Friday", llm=fake)
    assert "Mike" in out
    assert "[Owner] - [Task] - [Deadline]" in fake.prompts[0]


def test_summarize_notes_rejects_empty_before_calling_llm():
    fake = FakeLLM()
    with pytest.raises(ValueError):
        mw.summarize_notes("   ", llm=fake)
    assert fake.prompts == []  # LLM must not be called for invalid input


def test_llm_failure_is_wrapped_in_runtime_error():
    with pytest.raises(RuntimeError, match="language model call failed"):
        mw.summarize_notes("some real notes", llm=BoomLLM())


# --- tool wrappers ----------------------------------------------------------

def test_tool_wrapper_routes_through_get_llm(monkeypatch):
    fake = FakeLLM("TOOL OUTPUT")
    monkeypatch.setattr(mw, "get_llm", lambda: fake)
    # @tool objects are invoked with a dict of their arguments.
    result = mw.summarize_meeting_notes.invoke({"notes": "kickoff meeting notes"})
    assert result == "TOOL OUTPUT"
    assert "kickoff meeting notes" in fake.prompts[0]


def test_both_tools_are_registered_with_expected_names():
    names = {mw.summarize_meeting_notes.name, mw.extract_action_items.name}
    assert names == {"summarize_meeting_notes", "extract_action_items"}


# --- run_minute_writer ------------------------------------------------------

def test_run_minute_writer_returns_final_message():
    agent = FakeAgent(final="## Meeting Summary\nDone.")
    out = mw.run_minute_writer("real meeting notes go here", agent=agent)
    assert out == "## Meeting Summary\nDone."


def test_run_minute_writer_passes_notes_as_human_message():
    agent = FakeAgent()
    mw.run_minute_writer("notes for the agent", agent=agent)
    sent = agent.received["messages"][0]
    assert sent.content == "notes for the agent"


def test_run_minute_writer_rejects_empty_input():
    agent = FakeAgent()
    with pytest.raises(ValueError):
        mw.run_minute_writer("", agent=agent)
    assert agent.received is None  # agent must not run on invalid input
