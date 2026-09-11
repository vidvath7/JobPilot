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


def make_orchestrator(llm, mcp, max_tool_rounds=3):
    """Include a disallowed discovered Tool to exercise both policy boundaries."""
    catalog = CapabilityCatalog(
        tools=tuple(ToolCapability(name, name, {
            "type": "object", "properties": {"job_id": {"type": "string"}},
            "required": ["job_id"],
        }) for name in ("search_jobs", "score_job_match", "save_application")),
        resources=(), resource_templates=(), prompts=(),
    )
    return JobPilotOrchestrator(
        llm, mcp, catalog, allowed_tools=AUTO_EXECUTABLE_TOOLS,
        max_tool_rounds=max_tool_rounds,
    )


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
    assert options == llm.requests[0][1]  # Permitted schemas remain available.
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


def search_response(call_id, **arguments):
    """Give each round a distinct protocol ID, independently of search arguments."""
    return LLMResponse(None, (LLMToolCall(call_id, "search_jobs", arguments),))


@pytest.mark.parametrize("round_count", [2, 3])
def test_multiple_rounds_preserve_history_policy_and_execution_order(round_count):
    """Different literal search arguments may recover while every result stays in history."""
    roles = ["GenAI", "AI Engineer", "Applied AI Engineer"][:round_count]
    llm = FakeLLM(
        *(search_response(str(i), role=role) for i, role in enumerate(roles)),
        LLMResponse("Found jobs", ()),
    )
    mcp = FakeMCP()
    result = asyncio.run(make_orchestrator(llm, mcp).ask("Find jobs"))
    assert result.answer == "Found jobs"
    assert mcp.calls == [("search_jobs", {"role": role}) for role in roles]
    assert [call.arguments["role"] for call in result.executed_tool_calls] == roles
    assert len(llm.requests) == round_count + 1
    for index, (messages, options) in enumerate(llm.requests):
        assert len(messages) == 1 + index * 2
        assert messages[0] == {"role": "user", "content": "Find jobs"}
        assert {tool["function"]["name"] for tool in options["tools"]} == AUTO_EXECUTABLE_TOOLS
        assert options["tool_choice"] == "auto"
        if index:
            previous = llm.requests[index - 1][0]
            assert messages[:len(previous)] == previous
            assert messages[-1]["tool_call_id"] == str(index - 1)
            assert json.loads(messages[-1]["content"])["result"] == {"score": 60.0}


@pytest.mark.parametrize("limit", [1, 3])
def test_round_limit_rejects_extra_batch_before_execution(limit):
    """A response after the configured budget may answer but cannot execute more Tools."""
    llm = FakeLLM(*(search_response(str(i), role=str(i)) for i in range(limit + 1)))
    mcp = FakeMCP()
    with pytest.raises(OrchestrationError, match=f"Maximum Tool rounds exceeded \\({limit}\\)"):
        asyncio.run(make_orchestrator(llm, mcp, limit).ask("Find"))
    assert len(mcp.calls) == limit
    assert len(llm.requests) == limit + 1


def test_identical_call_rejected_despite_argument_key_order():
    """Canonical JSON recognizes exact repeats without treating new arguments as aliases."""
    llm = FakeLLM(
        search_response("one", role="GenAI", location="Germany"),
        search_response("two", location="Germany", role="GenAI"),
    )
    mcp = FakeMCP()
    with pytest.raises(OrchestrationError, match="Identical Tool call repeated"):
        asyncio.run(make_orchestrator(llm, mcp).ask("Find"))
    assert len(mcp.calls) == 1


def test_multiple_calls_count_as_one_round():
    """The budget counts model batches, not the number of operations in a batch."""
    llm = FakeLLM(selection("search_jobs", "score_job_match"), LLMResponse("Done", ()))
    mcp = FakeMCP()
    result = asyncio.run(make_orchestrator(llm, mcp, 1).ask("Find"))
    assert len(result.executed_tool_calls) == 2
    assert result.answer == "Done"


def test_later_disallowed_call_still_rejected():
    """Successful earlier rounds never broaden automatic Tool permission."""
    llm = FakeLLM(search_response("one", role="AI"), selection("save_application"))
    mcp = FakeMCP()
    with pytest.raises(OrchestrationError, match="disallowed"):
        asyncio.run(make_orchestrator(llm, mcp).ask("Find"))
    assert len(mcp.calls) == 1


def test_duplicate_call_ids_across_history_rejected():
    """Even different operations must have unambiguous result correlation IDs."""
    llm = FakeLLM(search_response("same", role="AI"), search_response("same", role="ML"))
    mcp = FakeMCP()
    with pytest.raises(OrchestrationError, match="duplicate Tool-call IDs"):
        asyncio.run(make_orchestrator(llm, mcp).ask("Find"))
    assert len(mcp.calls) == 1


@pytest.mark.parametrize("limit", [0, -1, True, 1.5])
def test_invalid_round_limits_rejected(limit):
    """Invalid configuration must fail before any provider or MCP activity."""
    with pytest.raises(ValueError, match="positive integer"):
        make_orchestrator(FakeLLM(), FakeMCP(), limit)


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
