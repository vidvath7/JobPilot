"""Per-action approval tests with controlled models; no live provider requests."""

import asyncio
import json
from pathlib import Path
from unittest.mock import AsyncMock

import pytest

from host.llm import LLMResponse, LLMToolCall
from host.main import confirm_action, run_repl
from host.mcp_client import JobPilotMCPClient
from host.orchestrator import (
    AUTO_EXECUTABLE_TOOLS, CONFIRMATION_REQUIRED_TOOLS, JobPilotOrchestrator, OrchestrationError,
)
from tests.test_host_orchestrator import FakeLLM, FakeMCP, make_orchestrator, selection, resource_selection
from host.capabilities import CapabilityCatalog


def save(call_id="save1", job="JOB-005"):
    """A normalized model request independent of the displayed confirmation UI."""
    return LLMResponse(None, (LLMToolCall(call_id, "save_application", {"job_id": job}),))


def engine(llm, mcp, approval):
    """Use the explicit production policy while injecting the approval mechanism."""
    return make_orchestrator(llm, mcp, confirmation_required_tools=CONFIRMATION_REQUIRED_TOOLS, approval_handler=approval)


@pytest.mark.parametrize("response", [selection("search_jobs"), resource_selection()])
def test_read_only_operations_never_request_approval(response):
    """Both read paths stay automatic even when write capabilities are visible."""
    approval = AsyncMock(return_value=True)
    asyncio.run(engine(FakeLLM(response, LLMResponse("Done", ())), FakeMCP(), approval).ask("Read"))
    approval.assert_not_awaited()


def test_approval_precedes_execution_and_uses_exact_arguments():
    """Approval sees a safe copy; callback mutation cannot alter the actual action."""
    mcp = FakeMCP()
    observed = []

    async def approve(name, arguments):
        assert mcp.calls == []
        observed.append((name, dict(arguments)))
        arguments["job_id"] = "MUTATED"
        return True

    llm = FakeLLM(save(), LLMResponse("Saved", ()))
    result = asyncio.run(engine(llm, mcp, approve).ask("Save"))
    assert observed == [("save_application", {"job_id": "JOB-005"})]
    assert mcp.calls == observed
    assert len(result.executed_tool_calls) == 1
    assert json.loads(llm.requests[1][0][-1]["content"])["is_error"] is False
    tools = {item["function"]["name"]: item["function"] for item in llm.requests[0][1]["tools"]}
    assert set(tools) == {"search_jobs", "score_job_match", "save_application", "read_mcp_resource"}
    assert tools["save_application"]["parameters"]["required"] == ["job_id"]


@pytest.mark.parametrize("decision", [False, None, "yes", 1])
def test_decline_never_executes_and_returns_normal_final_answer(decision):
    """Only the boolean True grants approval; declined calls are not execution traces."""
    llm, mcp = FakeLLM(save(), LLMResponse("Nothing saved", ())), FakeMCP()
    result = asyncio.run(engine(llm, mcp, AsyncMock(return_value=decision)).ask("Save"))
    assert mcp.calls == [] and result.executed_tool_calls == ()
    assert result.answer == "Nothing saved"
    message = llm.requests[1][0][-1]
    assert message["tool_call_id"] == "save1"
    assert json.loads(message["content"]) == {
        "approved": False, "executed": False, "reason": "User declined the requested action.",
    }


@pytest.mark.parametrize("same_batch", [False, True])
def test_declined_duplicate_does_not_prompt_twice(same_batch):
    """Decline protection covers repeated requests within a batch and across rounds."""
    first, second = save(), save("save2")
    responses = [LLMResponse(None, first.tool_calls + second.tool_calls)] if same_batch else [first, second]
    approval, mcp = AsyncMock(return_value=False), FakeMCP()
    with pytest.raises(OrchestrationError, match="declined Tool request repeated"):
        asyncio.run(engine(FakeLLM(*responses), mcp, approval).ask("Save"))
    assert approval.await_count == 1 and mcp.calls == []


def test_distinct_actions_and_new_requests_require_new_approval():
    """No approval carries between actions, rounds, or separate run invocations."""
    approval, mcp = AsyncMock(return_value=True), FakeMCP()
    llm = FakeLLM(save(), save("save2", "JOB-001"), LLMResponse("Done", ()), save(), LLMResponse("Done", ()))
    orchestrator = engine(llm, mcp, approval)

    async def run():
        await orchestrator.ask("Save two")
        await orchestrator.ask("Save again")

    asyncio.run(run())
    assert approval.await_count == 3 and len(mcp.calls) == 3


def test_missing_or_failed_approval_fails_closed():
    """Unavailable UI is never interpreted as user consent."""
    for approval in (None, AsyncMock(side_effect=RuntimeError("UI failed"))):
        mcp = FakeMCP()
        with pytest.raises(OrchestrationError, match="approval|Approval"):
            asyncio.run(engine(FakeLLM(save()), mcp, approval).ask("Save"))
        assert mcp.calls == []


@pytest.mark.parametrize("answer,expected", [("y", True), ("YES", True), ("Yes", True), ("", False), ("n", False), ("sure", False)])
def test_confirmation_displays_action_before_reading_input(answer, expected):
    """Only y/yes approves after the exact name and arguments have been displayed."""
    output = []

    def read(prompt):
        assert "Tool: save_application" in output
        assert '"job_id": "JOB-005"' in "\n".join(output)
        assert prompt == "Execute this action? [y/N]: "
        return answer

    assert asyncio.run(confirm_action("save_application", {"job_id": "JOB-005"}, input_function=read, output_function=output.append)) is expected


def test_decline_keeps_repl_alive():
    """A real CLI callback decline returns control to ask and the next REPL command."""
    output, mcp = [], FakeMCP()

    async def approve(name, arguments):
        return await confirm_action(name, arguments, input_function=lambda _: "", output_function=output.append)

    orchestrator = engine(FakeLLM(save(), LLMResponse("Not saved", ())), mcp, approve)
    commands = iter(["ask Save JOB-005", "help", "quit"])
    asyncio.run(run_repl(CapabilityCatalog((), (), (), ()), mcp, orchestrator=orchestrator,
                         input_function=lambda _: next(commands), output_function=output.append))
    assert mcp.calls == []
    assert "Not saved" in output and "Available commands:" in "\n".join(output)


def test_real_stdio_approval_persists_only_to_temporary_store(tmp_path):
    """Prove the child environment override and approval gate through real MCP.

    Model selections and user decisions are synthetic; persistence uses the real
    server. Production data is checked byte-for-byte after both outcomes.
    """
    store = tmp_path / "applications.json"
    store.write_text("[]", encoding="utf-8")
    production = Path(__file__).resolve().parents[1] / "data" / "applications.json"
    before = production.read_bytes()

    async def run():
        async with JobPilotMCPClient(server_environment={"JOBPILOT_APPLICATIONS_PATH": str(store)}) as client:
            catalog = await client.discover_capabilities()
            for decision in (False, True):
                orchestrator = JobPilotOrchestrator(
                    FakeLLM(save(), LLMResponse("Done", ())), client, catalog,
                    allowed_tools=AUTO_EXECUTABLE_TOOLS,
                    confirmation_required_tools=CONFIRMATION_REQUIRED_TOOLS,
                    approval_handler=AsyncMock(return_value=decision),
                )
                await orchestrator.ask("Save")
                records = json.loads(store.read_text(encoding="utf-8"))
                if not decision:
                    assert records == []
                else:
                    assert len(records) == 1
                    assert records[0]["application_id"] == "APP-001"
                    assert records[0]["job_id"] == "JOB-005"
                    assert records[0]["status"] == "applied"

    try:
        asyncio.run(run())
    finally:
        assert production.read_bytes() == before
