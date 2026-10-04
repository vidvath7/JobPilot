"""Offline A3 acceptance: fake models, real manager routing, no external I/O."""

import asyncio
import json
from copy import deepcopy

import pytest
from mcp import types

from host.capabilities import CapabilityCatalog, ToolCapability
from host.llm import LLMResponse, LLMToolCall
from host.mcp_server_manager import MCPServerManager
from host.multi_server_orchestrator import (
    EXTERNAL_AUTO_TOOLS, ModelToolRegistry, MultiServerOrchestrator, model_tool_name,
)
from host.orchestrator import OrchestrationError


SCHEMA = {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}


class Client:
    """Advertise unsafe decoys to verify policy, not discovery, grants model access."""

    def __init__(self, server_id):
        self.server_id, self.is_connected, self.calls = server_id, False, []

    async def connect(self):
        self.is_connected = True

    async def close(self):
        self.is_connected = False

    async def discover_capabilities(self):
        names = ("search_jobs", "get_job_details", "get_related_jobs", "update_account", "save_application")
        if self.server_id == "jobpilot":
            names = ("search_jobs", "score_job_match", "save_application")
        return CapabilityCatalog(tuple(ToolCapability(n, "Discovered description", deepcopy(SCHEMA))
                                       for n in names), (), (), ())

    async def call_tool(self, name, arguments=None):
        self.calls.append((name, arguments))
        return types.CallToolResult(content=[], structured_content={"jobs": [{"title": "AI Engineer"}]})


class LLM:
    """Capture entire requests to check schema safety and chronological evidence."""

    def __init__(self, responses):
        self.responses = iter(responses)
        self.requests = []

    async def complete(self, messages, **kwargs):
        self.requests.append((deepcopy(messages), deepcopy(kwargs)))
        return next(self.responses)


def call(name="himalayas__search_jobs", arguments=None, id="1"):
    return LLMResponse(None, (LLMToolCall(id, name, arguments or {"query": "AI"}),))


async def exercise(responses, *, approve=None, rounds=3):
    manager = MCPServerManager()
    local, remote = Client("jobpilot"), Client("himalayas")
    manager.register(local)
    manager.register(remote)
    async with manager:
        catalog = await manager.discover_capabilities()
        llm = LLM(responses)
        orchestrator = MultiServerOrchestrator(llm, manager, catalog,
                                              approval_handler=approve, max_tool_rounds=rounds)
        result = await orchestrator.ask("Find AI remote jobs")
        return result, llm, local, remote, catalog


def test_alias_and_reverse_resolution_preserve_collision():
    """Both search Tools are separately addressable without exposing local search by default."""
    async def run():
        result, _, _, _, catalog = await exercise([call(), LLMResponse("hello", ())])
        registry = ModelToolRegistry(catalog, {"jobpilot::search_jobs", "himalayas::search_jobs"})
        assert model_tool_name("himalayas::search_jobs") == "himalayas__search_jobs"
        assert registry.resolve("himalayas__search_jobs") == "himalayas::search_jobs"
        assert registry.resolve("jobpilot__search_jobs") == "jobpilot::search_jobs"
        for bad in ("search_jobs", "himalayas::search_jobs", "missing__search_jobs"):
            with pytest.raises(OrchestrationError):
                registry.resolve(bad)
        with pytest.raises(ValueError, match="Duplicate"):
            ModelToolRegistry(type(catalog)(catalog.tools + catalog.tools, (), (), ()), EXTERNAL_AUTO_TOOLS)
        assert result.answer == "hello"
    asyncio.run(run())


@pytest.mark.parametrize("qualified", ["a__b::c", "a::b__c", "a::b.c", "search_jobs", "a::" + "b" * 64])
def test_ambiguous_or_unsafe_alias_rejected(qualified):
    """Reject collisions and invalid function identifiers instead of sanitizing names."""
    with pytest.raises(ValueError):
        model_tool_name(qualified)


def test_dynamic_policy_schema_routing_and_multi_round_history():
    """Search then details routes to the remote owner with original names and JSON evidence."""
    async def run():
        result, llm, local, remote, catalog = await exercise([
            call(), call("himalayas__get_job_details", {"company_slug": "test", "job_slug": "ai"}, "2"),
            LLMResponse("Shortlist", ()),
        ])
        assert not local.calls
        assert [name for name, _ in remote.calls] == ["search_jobs", "get_job_details"]
        assert remote.calls[0][1] == {"query": "AI"}
        assert [c.name for c in result.executed_tool_calls] == ["himalayas__search_jobs", "himalayas__get_job_details"]
        for messages, request in llm.requests:
            definitions = {t["function"]["name"]: t["function"] for t in request["tools"]}
            assert set(definitions) == {
                "himalayas__search_jobs", "himalayas__get_job_details", "himalayas__get_related_jobs",
                "jobpilot__score_job_match", "jobpilot__save_application", "read_mcp_resource",
            }
            assert definitions["himalayas__search_jobs"]["parameters"] == SCHEMA
            assert definitions["himalayas__search_jobs"]["description"] == "Discovered description"
            json.dumps(messages)  # Provider history must contain no MCP SDK instances.
        history = llm.requests[-1][0]
        assert [m["tool_call_id"] for m in history if m["role"] == "tool"] == ["1", "2"]
        assert json.loads(history[3]["content"])["result"]["jobs"][0]["title"] == "AI Engineer"
        llm.requests[0][1]["tools"][0]["function"]["parameters"]["required"].clear()
        assert catalog.tools[0].capability.input_schema == SCHEMA
    asyncio.run(run())


@pytest.mark.parametrize("responses,rounds,match", [
    ([call(), call(id="2")], 3, "repeated"),
    ([call(), call(arguments={"query": "ML"}, id="2")], 1, "Maximum Tool rounds"),
    ([call("himalayas__update_account")], 3, "unknown Tool"),
    ([call("invented__tool")], 3, "unknown Tool"),
])
def test_existing_engine_safety_applies_to_external_calls(responses, rounds, match):
    """Reuse repeat/round/name enforcement; no second orchestration loop exists."""
    with pytest.raises(OrchestrationError, match=match):
        asyncio.run(exercise(responses, rounds=rounds))


@pytest.mark.parametrize("approved", [True, False])
def test_local_save_requires_fresh_approval(approved):
    """Namespacing must never bypass the existing state-changing approval gate."""
    approvals = []

    async def approve(name, arguments):
        approvals.append((name, arguments))
        return approved

    async def run():
        result, llm, local, remote, _ = await exercise([
            call(), call("jobpilot__save_application", {"job_id": "JOB-005"}, "2"), LLMResponse("Done", ()),
        ], approve=approve)
        assert approvals == [("jobpilot::save_application", {"job_id": "JOB-005"})]
        assert local.calls == ([("save_application", {"job_id": "JOB-005"})] if approved else [])
        assert remote.calls == [("search_jobs", {"query": "AI"})]
        assert len(result.executed_tool_calls) == 1 + int(approved)
        if not approved:
            assert json.loads(llm.requests[-1][0][-1]["content"])["executed"] is False
    asyncio.run(run())


def test_live_cli_routes_ask_but_keeps_manual_and_prompt_paths_local(monkeypatch):
    """Exercise opt-in startup wiring without real transports, credentials, or a terminal."""
    import host.main as main
    import host.himalayas as himalayas

    local, remote = Client("jobpilot"), Client("himalayas")
    llm = LLM([call(), LLMResponse("Live shortlist", ())])
    monkeypatch.setattr(main, "JobPilotMCPClient", lambda: local)
    monkeypatch.setattr(himalayas, "create_himalayas_client", lambda: remote)
    monkeypatch.setattr(main, "NVIDIALLMClient", lambda: llm)
    real_repl = main.run_repl
    output = []

    async def repl(catalog, client, *, orchestrator, prompt_workflow):
        assert client is local
        assert {tool.name for tool in catalog.tools} == {"search_jobs", "score_job_match", "save_application"}
        commands = iter(["ask Find AI jobs", "quit"])
        await real_repl(catalog, client, orchestrator=orchestrator, prompt_workflow=prompt_workflow,
                        input_function=lambda _: next(commands), output_function=output.append)

    monkeypatch.setattr(main, "run_repl", repl)
    asyncio.run(main.run_host(live_jobs=True))
    assert remote.calls == [("search_jobs", {"query": "AI"})]
    assert not local.calls
    assert not local.is_connected and not remote.is_connected
    assert any("himalayas__search_jobs" in line for line in output)
    assert "Live shortlist" in output
