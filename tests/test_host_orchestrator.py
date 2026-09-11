"""Host policy and conversation tests, independent of network and server logic."""

import asyncio
import json
from copy import deepcopy

import pytest
from mcp import types
from mcp.shared.exceptions import MCPError

from host.capabilities import CapabilityCatalog, ToolCapability
from host.llm import LLMResponse, LLMToolCall
from host.orchestrator import AUTO_EXECUTABLE_TOOLS, JobPilotOrchestrator, OrchestrationError


class FakeLLM:
    """Capture exact continuation messages without contacting a provider."""

    def __init__(self, *responses):
        self.responses = iter(responses)
        self.requests = []

    async def complete(self, messages, **kwargs):
        self.requests.append((deepcopy(messages), deepcopy(kwargs)))
        return next(self.responses)


class FakeMCP:
    """Expose only generic execution so tests cannot bypass the MCP boundary."""

    def __init__(self, result=None, error=None):
        self.calls = []
        self.result = result if result is not None else types.CallToolResult(
            content=[], structuredContent={"score": 60.0},
        )
        self.error = error

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        if self.error:
            raise self.error
        return self.result


def make_orchestrator(llm, mcp):
    """Include a disallowed discovered Tool to exercise both policy boundaries."""
    catalog = CapabilityCatalog(
        tools=tuple(ToolCapability(name, name, {
            "type": "object", "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        }) for name in ("search_jobs", "score_job_match", "save_application")),
        resources=(), resource_templates=(), prompts=(),
    )
    return JobPilotOrchestrator(llm, mcp, catalog, allowed_tools=AUTO_EXECUTABLE_TOOLS)


def selection(*names):
    """Model selections carry correlation IDs and JSON-native arguments."""
    return LLMResponse(None, tuple(
        LLMToolCall(str(index), name, {"job_id": "JOB-005"})
        for index, name in enumerate(names)
    ))


def test_plain_answer_and_filtered_discovered_schemas():
    """Text-only answers stop immediately; schemas preserve discovery and policy."""
    llm, mcp = FakeLLM(LLMResponse("Hello", ())), FakeMCP()
    result = asyncio.run(make_orchestrator(llm, mcp).ask("Hi"))
    assert result.answer == "Hello" and result.executed_tool_calls == ()
    assert mcp.calls == [] and len(llm.requests) == 1
    options = llm.requests[0][1]
    assert options["tool_choice"] == "auto"
    assert [tool["function"]["name"] for tool in options["tools"]] == [
        "search_jobs", "score_job_match",
    ]
    assert options["tools"][0]["function"]["parameters"]["required"] == ["job_id"]


@pytest.mark.parametrize("structured,is_error", [(True, False), (False, False), (False, True)])
def test_result_continuation_and_terminal_answer(structured, is_error):
    """Structured data wins; text fallback and Tool failure flags survive JSON encoding."""
    raw = types.CallToolResult(
        content=[types.TextContent(type="text", text="fallback")],
        structuredContent={"score": 60.0} if structured else None,
        isError=is_error,
    )
    llm = FakeLLM(selection("score_job_match"), LLMResponse("Final answer", ()))
    mcp = FakeMCP(raw)
    result = asyncio.run(make_orchestrator(llm, mcp).ask("Score this job"))
    assert mcp.calls == [("score_job_match", {"job_id": "JOB-005"})]
    assert result.answer == "Final answer"
    messages, options = llm.requests[1]
    assert options == {}  # No schemas or Tool-selection options in round two.
    assert messages[0] == {"role": "user", "content": "Score this job"}
    call = messages[1]["tool_calls"][0]
    assert json.loads(call["function"]["arguments"]) == {"job_id": "JOB-005"}
    assert messages[2]["tool_call_id"] == call["id"]
    value = {"score": 60.0} if structured else "fallback"
    assert json.loads(messages[2]["content"]) == {"is_error": is_error, "result": value}
    assert result.executed_tool_calls[0].result == value
    assert result.executed_tool_calls[0].is_error is is_error


@pytest.mark.parametrize("name,reason", [("unknown", "unknown"), ("save_application", "disallowed")])
def test_invalid_batch_is_rejected_before_any_execution(name, reason):
    """A later forbidden call must prevent even the earlier allowed call from running."""
    llm, mcp = FakeLLM(selection("search_jobs", name)), FakeMCP()
    with pytest.raises(OrchestrationError, match=reason):
        asyncio.run(make_orchestrator(llm, mcp).ask("Find"))
    assert mcp.calls == [] and len(llm.requests) == 1


def test_multiple_calls_keep_provider_order_and_result_ids():
    """Every first-round call gets exactly one matching continuation message."""
    llm = FakeLLM(selection("search_jobs", "score_job_match"), LLMResponse("Done", ()))
    mcp = FakeMCP()
    result = asyncio.run(make_orchestrator(llm, mcp).ask("Find and score"))
    assert [name for name, _ in mcp.calls] == ["search_jobs", "score_job_match"]
    assert [call.id for call in result.executed_tool_calls] == ["0", "1"]
    assert [message["tool_call_id"] for message in llm.requests[1][0][2:]] == ["0", "1"]


def test_second_round_calls_never_execute():
    """The terminal completion cannot reopen execution, even for an allowed Tool."""
    llm, mcp = FakeLLM(selection("search_jobs"), selection("score_job_match")), FakeMCP()
    with pytest.raises(OrchestrationError, match="another Tool round"):
        asyncio.run(make_orchestrator(llm, mcp).ask("Find"))
    assert len(mcp.calls) == 1 and len(llm.requests) == 2


def test_protocol_failure_stops_before_summary():
    """Protocol failures are Host errors, not invented successful Tool results."""
    llm = FakeLLM(selection("search_jobs"))
    mcp = FakeMCP(error=MCPError(-32000, "failure"))
    with pytest.raises(OrchestrationError, match="MCP execution failed"):
        asyncio.run(make_orchestrator(llm, mcp).ask("Find"))
    assert len(llm.requests) == 1


def test_unsupported_mcp_result_is_rejected():
    """Unknown protocol result variants cannot silently masquerade as success."""
    with pytest.raises(OrchestrationError, match="unsupported Tool result"):
        asyncio.run(make_orchestrator(
            FakeLLM(selection("search_jobs")), FakeMCP(object()),
        ).ask("Find"))
