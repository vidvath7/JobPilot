"""Milestone 6 acceptance through the real Host-side MCP connection.

These tests use the real stdio server. They validate lifecycle and discovery at
the Host boundary with isolated persistence and no live provider requests.
"""

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import types

from host.mcp_client import JobPilotMCPClient
import host.mcp_client as client_module


def test_host_client_discovers_server_capabilities_and_closes(tmp_path, monkeypatch) -> None:
    """Accept discovery, execution, persistence, and cleanup in one real session."""
    store = tmp_path / "applications.json"
    store.write_text("[]", encoding="utf-8")
    production = Path(__file__).resolve().parents[1] / "data" / "applications.json"
    before = production.read_bytes()
    assert json.loads(before) == []
    real_stdio = client_module.stdio_client

    @asynccontextmanager
    async def guarded_stdio(parameters):
        # Fail before launch if isolation regresses; the real SDK still owns all
        # transport and server operations after this observation-only check.
        assert parameters.env["JOBPILOT_APPLICATIONS_PATH"] == str(store)
        async with real_stdio(parameters) as streams:
            yield streams

    monkeypatch.setattr(client_module, "stdio_client", guarded_stdio)
    try:
        asyncio.run(_exercise_host_client_lifecycle(store))
    finally:
        assert production.read_bytes() == before


async def _exercise_host_client_lifecycle(store: Path) -> None:
    """Verify connection state, capability metadata, and bounded shutdown."""
    client = JobPilotMCPClient(server_environment={"JOBPILOT_APPLICATIONS_PATH": str(store)})
    assert client.is_connected is False

    async with client:
        assert client.is_connected is True

        # Every discovery request crosses stdio through the same initialized
        # ClientSession; no server implementation is imported by the Host.
        tools = await client.list_tools()
        resources = await client.list_resources()
        resource_templates = await client.list_resource_templates()
        prompts = await client.list_prompts()

        assert len(tools) == 3
        assert {tool.name for tool in tools} == {
            "search_jobs",
            "score_job_match",
            "save_application",
        }
        assert len(resources) == 2
        assert {str(resource.uri) for resource in resources} == {
            "candidate://profile",
            "applications://all",
        }
        assert len(resource_templates) == 1
        assert {
            template.uri_template for template in resource_templates
        } == {"jobs://job/{job_id}"}

        assert len(prompts) == 1
        prompt = prompts[0]
        assert prompt.name == "prepare_application"
        assert prompt.arguments is not None
        assert [(argument.name, argument.required) for argument in prompt.arguments] == [
            ("job_id", True)
        ]

        catalog = await client.discover_capabilities()
        assert len(catalog.tools) == 3
        assert {tool.name for tool in catalog.tools} == {
            "search_jobs",
            "score_job_match",
            "save_application",
        }
        assert len(catalog.resources) == 2
        assert {resource.uri for resource in catalog.resources} == {
            "candidate://profile",
            "applications://all",
        }
        assert len(catalog.resource_templates) == 1
        assert catalog.resource_templates[0].uri_template == "jobs://job/{job_id}"
        assert len(catalog.prompts) == 1

        normalized_prompt = catalog.prompts[0]
        assert normalized_prompt.name == "prepare_application"
        assert "application-preparation workflow" in (
            normalized_prompt.description or ""
        ).casefold()
        assert [
            (argument.name, argument.required)
            for argument in normalized_prompt.arguments
        ] == [("job_id", True)]

        score_tool = next(
            tool for tool in catalog.tools if tool.name == "score_job_match"
        )
        assert set(score_tool.input_schema["properties"]) == {"job_id"}
        assert score_tool.input_schema["required"] == ["job_id"]

        save_tool = next(
            tool for tool in catalog.tools if tool.name == "save_application"
        )
        assert set(save_tool.input_schema["properties"]) == {
            "job_id",
            "status",
            "notes",
        }
        assert save_tool.input_schema["required"] == ["job_id"]

        # These generic calls prove the Host can execute discovered capabilities
        # without importing server adapters or adding JobPilot-specific methods.
        search_result = await client.call_tool(
            "search_jobs",
            {"role": "AI Engineer"},
        )
        assert isinstance(search_result, types.CallToolResult)
        assert search_result.is_error is False
        assert isinstance(search_result.structured_content, dict)
        assert search_result.structured_content["result"]

        score_result = await client.call_tool(
            "score_job_match",
            {"job_id": "JOB-005"},
        )
        assert isinstance(score_result, types.CallToolResult)
        assert score_result.is_error is False
        assert score_result.structured_content is not None
        assert score_result.structured_content["score"] == 60.0

        resource_result = await client.read_resource("candidate://profile")
        assert isinstance(resource_result, types.ReadResourceResult)
        resource_content = resource_result.contents[0]
        assert isinstance(resource_content, types.TextResourceContents)
        assert isinstance(json.loads(resource_content.text), dict)

        job_result = await client.read_resource("jobs://job/JOB-005")
        assert isinstance(job_result, types.ReadResourceResult)
        job = json.loads(job_result.contents[0].text)
        assert job["id"] == "JOB-005"
        assert "required_skills" in job and "description" in job

        prompt_result = await client.get_prompt(
            "prepare_application",
            {"job_id": "JOB-005"},
        )
        assert isinstance(prompt_result, types.GetPromptResult)
        assert prompt_result.messages
        prompt_content = prompt_result.messages[0].content
        assert isinstance(prompt_content, types.TextContent)
        assert "JOB-005" in prompt_content.text

        # This explicit generic MCP call tests persistence, not model consent;
        # the separate approval tests cover the orchestrator's per-action gate.
        saved = await client.call_tool("save_application", {"job_id": "JOB-005"})
        assert isinstance(saved, types.CallToolResult) and not saved.is_error
        records = json.loads(store.read_text(encoding="utf-8"))
        assert len(records) == 1
        assert records[0]["application_id"] == "APP-001"
        assert records[0]["job_id"] == "JOB-005"
        history = await client.read_resource("applications://all")
        assert isinstance(history, types.ReadResourceResult)
        assert json.loads(history.contents[0].text) == records

    # Returning from the context proves subprocess/session cleanup completed; the
    # disconnected guard also prevents accidental reuse of a closed SDK session.
    assert client.is_connected is False
    with pytest.raises(RuntimeError, match="not connected"):
        await client.list_tools()
    with pytest.raises(RuntimeError, match="not connected"):
        await client.discover_capabilities()
    with pytest.raises(RuntimeError, match="not connected"):
        await client.call_tool("any_tool")
    with pytest.raises(RuntimeError, match="not connected"):
        await client.read_resource("any://resource")
    with pytest.raises(RuntimeError, match="not connected"):
        await client.get_prompt("any_prompt")
