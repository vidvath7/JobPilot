"""Integration tests for the Host-side MCP connection foundation.

These tests use the real stdio server. They validate lifecycle and discovery at
the Host boundary without importing or invoking server capability handlers.
"""

import asyncio

import pytest

from host.mcp_client import JobPilotMCPClient


def test_host_client_discovers_server_capabilities_and_closes() -> None:
    """Discover the complete MCP surface through one reusable client session."""
    asyncio.run(_exercise_host_client_lifecycle())


async def _exercise_host_client_lifecycle() -> None:
    """Verify connection state, capability metadata, and bounded shutdown."""
    client = JobPilotMCPClient()
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

    # Returning from the context proves subprocess/session cleanup completed; the
    # disconnected guard also prevents accidental reuse of a closed SDK session.
    assert client.is_connected is False
    with pytest.raises(RuntimeError, match="not connected"):
        await client.list_tools()
