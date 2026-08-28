"""Component tests for the MCP adapter and in-process server registry.

These tests intentionally avoid stdio: direct calls and registry inspection make
adapter or registration failures faster to isolate from protocol transport issues.
"""

import asyncio
from unittest.mock import Mock

import server.tools.search_jobs as search_jobs_module
from server.main import mcp


def test_search_jobs_tool_delegates_to_job_service(monkeypatch) -> None:
    """Verify the MCP adapter forwards arguments without reimplementing search."""
    expected_result = [
        {
            "id": "JOB-001",
            "title": "AI Engineer",
            "company": "Lumen Forge AI",
            "location": "Berlin, Germany",
            "experience_level": "Mid-level",
        }
    ]
    job_service = Mock()
    job_service.search_jobs.return_value = expected_result
    # Replacing the service isolates the adapter boundary: this test should fail
    # only if MCP-facing argument forwarding or result passthrough changes.
    monkeypatch.setattr(search_jobs_module, "_job_service", job_service)

    result = search_jobs_module.search_jobs(
        role="AI Engineer",
        location="Berlin",
        experience_level="Mid-level",
    )

    job_service.search_jobs.assert_called_once_with(
        role="AI Engineer",
        location="Berlin",
        experience_level="Mid-level",
    )
    assert result is expected_result


def test_server_registers_search_jobs_tool() -> None:
    """Verify discovery metadata is present without starting a transport client."""
    # MCPServer's registry API is asynchronous even for this in-process check.
    registered_tools = asyncio.run(mcp.list_tools())

    assert [tool.name for tool in registered_tools] == ["search_jobs"]
    assert "optional role, location, and experience-level filters" in (
        registered_tools[0].description or ""
    )


def test_server_registers_static_candidate_profile_resource() -> None:
    """Verify Resource metadata separately from file loading and stdio transport."""
    registered_resources = asyncio.run(mcp.list_resources())
    resource_templates = asyncio.run(mcp.list_resource_templates())

    assert len(registered_resources) == 1
    candidate_profile = registered_resources[0]
    assert str(candidate_profile.uri) == "candidate://profile"
    assert candidate_profile.mime_type == "application/json"
    assert "candidate profile" in (candidate_profile.description or "").casefold()
    assert resource_templates == []
