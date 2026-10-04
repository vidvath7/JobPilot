"""A3 provenance and diagnostic regressions without network or real credentials."""

import asyncio
import json

import pytest
from mcp import types

from host.mcp_server_manager import MCPServerManager
from host.multi_server_orchestrator import MultiServerOrchestrator
from host.orchestrator import OrchestrationError
from host.llm import LLMResponse
from scripts.smoke_himalayas_search import print_execution, report_error, slug_pair
from tests.test_multi_server_orchestrator import Client, LLM, call, exercise


def test_final_answer_without_external_execution_rejected():
    """Polished model-only job listings are not accepted as live search evidence."""
    with pytest.raises(OrchestrationError, match="without external Tool execution"):
        asyncio.run(exercise([LLMResponse("Here are ten live Himalayas jobs", ())]))


@pytest.mark.parametrize("tool_error", [False, True])
def test_external_fields_errors_and_trace_survive_serialization(tool_error):
    """Keep every original structured field, including real slug fields, in model history."""
    payload = {"jobs": [{"title": "AI Engineer", "companyName": "Synthetic",
                         "company_slug": "synthetic", "job_slug": "actual-returned-id",
                         "applicationLink": "https://example.test/apply", "extra": {"keep": [1, 2]}}]}
    observed = []
    llm = LLM([call(), LLMResponse("Grounded answer", ())])

    class ResultClient(Client):
        async def call_tool(self, name, arguments=None):
            return types.CallToolResult(content=[], structured_content=payload, is_error=tool_error)

    async def run():
        manager = MCPServerManager()
        manager.register(ResultClient("himalayas"))
        async with manager:
            catalog = await manager.discover_capabilities()
            orchestrator = MultiServerOrchestrator(llm, manager, catalog, execution_observer=observed.append)
            if tool_error:
                with pytest.raises(OrchestrationError, match="without external Tool execution"):
                    await orchestrator.ask("Find live jobs")
            else:
                result = await orchestrator.ask("Find live jobs")
                assert result.executed_tool_calls == tuple(observed)
                assert result.answer == "Grounded answer"
            message = json.loads(llm.requests[-1][0][-1]["content"])
            assert message == {"is_error": tool_error, "result": payload}
            assert observed[0].name == "himalayas__search_jobs"
            assert slug_pair(message["result"]) == ("synthetic", "actual-returned-id")
            output = []
            print_execution(observed[0], orchestrator.registry, output.append)
            assert "himalayas::search_jobs" in "\n".join(output)
            assert "actual-returned-id" in "\n".join(output)
    asyncio.run(run())


def test_completed_trace_visible_even_when_later_round_fails():
    """Immediate observation prevents a round-limit/repetition error hiding execution."""
    observed = []

    async def run():
        manager = MCPServerManager()
        manager.register(Client("himalayas"))
        async with manager:
            orchestrator = MultiServerOrchestrator(
                LLM([call(), call(id="2")]), manager, await manager.discover_capabilities(),
                execution_observer=observed.append,
            )
            with pytest.raises(OrchestrationError, match="repeated"):
                await orchestrator.ask("Find jobs")
        assert len(observed) == 1
    asyncio.run(run())


def test_safe_smoke_error_message_and_no_provider_payload():
    """Expose Host errors without printing chained exceptions or raw provider data."""
    output = []
    error = OrchestrationError("Maximum Tool rounds exceeded (3).")
    error.__cause__ = RuntimeError("private provider payload")
    report_error(error, output.append)
    assert output == ["Experiment failed (OrchestrationError): Maximum Tool rounds exceeded (3)."]
    report_error(RuntimeError("private provider payload"), output.append)
    assert "private provider payload" not in "\n".join(output)


def test_slugs_are_never_invented_from_titles():
    """JSON text fallback retains supplied identifiers, but titles alone are insufficient."""
    assert slug_pair({"title": "AI Engineer", "companyName": "Example"}) is None
    assert slug_pair(json.dumps({"company_slug": "given-company", "job_slug": "given-job"})) == (
        "given-company", "given-job")


def test_observed_markdown_response_preserves_exact_url_slugs():
    """The real endpoint uses text job URLs, not guaranteed structured slug fields."""
    from host.orchestrator import _tool_result_value
    text = "Found 1 jobs\nApply: https://himalayas.app/companies/example-inc/jobs/exact-job-123?utm_source=mcp"
    raw = types.CallToolResult(content=[types.TextContent(type="text", text=text)])
    value = _tool_result_value(raw)
    assert value == text
    assert json.loads(json.dumps(value)) == text
    assert slug_pair(value) == ("example-inc", "exact-job-123")
    assert slug_pair("https://other.test/companies/example/jobs/fake") is None
