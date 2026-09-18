"""Host workflow tests use synthetic discovery and SDK results without live LLM calls."""

import asyncio
from copy import deepcopy
from dataclasses import replace
from unittest.mock import AsyncMock

import pytest
from mcp import types

from host.capabilities import CapabilityCatalog, PromptCapability, PromptArgument
from host.llm import LLMResponse, LLMToolCall
from host.orchestrator import OrchestrationResult, OrchestrationError
from host.prompt_workflow import PromptWorkflow, PromptWorkflowError
from tests.test_host_orchestrator import FakeLLM, FakeMCP, make_orchestrator, resource_selection, selection


def catalog():
    """Different argument names prove validation is discovery-driven."""
    return CapabilityCatalog(tools=(), resources=(), resource_templates=(), prompts=(
        PromptCapability("selected", "Workflow", (
            PromptArgument("item", None, True), PromptArgument("context", None, False),
        )),
        PromptCapability("empty", None, ()),
    ))


def prompt_result():
    """Use multiple roles so workflow tests detect reordering or flattening."""
    return types.GetPromptResult(messages=[
        types.PromptMessage(role="assistant", content=types.TextContent(type="text", text="Context")),
        types.PromptMessage(role="user", content=types.TextContent(type="text", text="Explain JOB-005")),
    ])


@pytest.mark.parametrize("name,args", [
    ("selected", {"item": "JOB-005"}),
    ("selected", {"item": "JOB-005", "context": "extra"}),
    ("empty", None), ("empty", {}),
])
def test_generic_discovered_prompt_execution(name, args):
    """Required, optional, and absent arguments reach MCP exactly as supplied."""
    client, engine = AsyncMock(), AsyncMock()
    client.get_prompt.return_value = prompt_result()
    engine.run.return_value = OrchestrationResult("Answer", ())
    result = asyncio.run(PromptWorkflow(client, catalog(), engine).run_prompt(name, args))
    client.get_prompt.assert_awaited_once_with(name, args)
    engine.run.assert_awaited_once_with([
        {"role": "assistant", "content": "Context"}, {"role": "user", "content": "Explain JOB-005"},
    ])
    assert result.answer == "Answer"


@pytest.mark.parametrize("name,args,reason", [
    ("missing", {}, "Unknown"), ("selected", {}, "Missing required"),
    ("selected", {"item": "x", "wrong": "y"}, "Unexpected"),
    ("selected", {"item": 5}, "string values"),
])
def test_invalid_selection_rejected_before_retrieval(name, args, reason):
    """Invalid user selection cannot start either MCP retrieval or LLM execution."""
    client, engine = AsyncMock(), AsyncMock()
    with pytest.raises(PromptWorkflowError, match=reason):
        asyncio.run(PromptWorkflow(client, catalog(), engine).run_prompt(name, args))
    client.get_prompt.assert_not_awaited()
    engine.run.assert_not_awaited()


def test_prompt_uses_existing_resource_and_scoring_engine():
    """Retrieved instructions enter the real bounded loop with both dispatch paths."""
    llm = FakeLLM(resource_selection(), selection("score_job_match"), LLMResponse("Evidence-based guidance", ()))
    mcp = FakeMCP()
    mcp.get_prompt = AsyncMock(return_value=prompt_result())
    engine = make_orchestrator(llm, mcp)
    result = asyncio.run(PromptWorkflow(mcp, catalog(), engine).run_prompt("selected", {"item": "JOB-005"}))
    assert [call.name for call in result.executed_tool_calls] == ["read_mcp_resource", "score_job_match"]
    assert mcp.reads == ["candidate://profile"]
    assert mcp.calls == [("score_job_match", {"job_id": "JOB-005"})]
    for messages, options in llm.requests:
        assert messages[:2] == [{"role": "assistant", "content": "Context"}, {"role": "user", "content": "Explain JOB-005"}]
        assert "save_application" not in [tool["function"]["name"] for tool in options["tools"]]


def test_run_does_not_mutate_input_history():
    """Appending workflow evidence must not modify reusable caller-owned messages."""
    history = [{"role": "user", "content": "Read context"}]
    original = deepcopy(history)
    engine = make_orchestrator(FakeLLM(resource_selection(), LLMResponse("Done", ())), FakeMCP())
    asyncio.run(engine.run(history))
    assert history == original


def test_prompt_cannot_enable_disallowed_execution():
    """Prompt instructions cannot expand the Host automatic Tool policy."""
    mcp = FakeMCP()
    mcp.get_prompt = AsyncMock(return_value=prompt_result())
    engine = make_orchestrator(FakeLLM(selection("save_application")), mcp)
    with pytest.raises(OrchestrationError, match="disallowed"):
        asyncio.run(PromptWorkflow(mcp, catalog(), engine).run_prompt("empty"))
    assert mcp.calls == []
