"""Component tests for the MCP adapter and in-process server registry.

These tests intentionally avoid stdio: direct calls and registry inspection make
adapter or registration failures faster to isolate from protocol transport issues.
"""

import asyncio
from unittest.mock import Mock

import server.resources.applications as applications_module
import server.tools.save_application as save_application_module
import server.tools.score_job_match as score_job_match_module
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


def test_score_job_match_tool_delegates_to_matching_service(monkeypatch) -> None:
    """Verify the MCP adapter preserves the frozen matching result unchanged."""
    expected_result = {"job_id": "JOB-005", "score": 60.0}
    matching_service = Mock()
    matching_service.score_job_match.return_value = expected_result
    monkeypatch.setattr(
        score_job_match_module, "_matching_service", matching_service
    )

    result = score_job_match_module.score_job_match("JOB-005")

    matching_service.score_job_match.assert_called_once_with("JOB-005")
    assert result is expected_result


def test_save_application_tool_delegates_to_application_service(monkeypatch) -> None:
    """Verify the state-changing adapter forwards and returns domain values."""
    expected_record = {
        "application_id": "APP-001",
        "job_id": "JOB-005",
        "status": "interview",
        "applied_at": "2026-08-31T10:00:00+00:00",
        "notes": "Technical interview scheduled.",
    }
    application_service = Mock()
    application_service.save_application.return_value = expected_record
    monkeypatch.setattr(
        save_application_module,
        "_application_service",
        application_service,
    )

    result = save_application_module.save_application(
        job_id="JOB-005",
        status="interview",
        notes="Technical interview scheduled.",
    )

    application_service.save_application.assert_called_once_with(
        job_id="JOB-005",
        status="interview",
        notes="Technical interview scheduled.",
    )
    assert result is expected_record


def test_applications_resource_delegates_to_application_service(monkeypatch) -> None:
    """Verify the Resource passes through the service's persisted snapshot."""
    expected_history = [{"application_id": "APP-001", "job_id": "JOB-005"}]
    application_service = Mock()
    application_service.get_applications.return_value = expected_history
    monkeypatch.setattr(
        applications_module,
        "_application_service",
        application_service,
    )

    result = applications_module.applications()

    application_service.get_applications.assert_called_once_with()
    assert result is expected_history


def test_server_registers_search_jobs_tool() -> None:
    """Verify discovery metadata is present without starting a transport client."""
    # MCPServer's registry API is asynchronous even for this in-process check.
    registered_tools = asyncio.run(mcp.list_tools())
    tools_by_name = {tool.name: tool for tool in registered_tools}

    assert set(tools_by_name) == {
        "search_jobs",
        "score_job_match",
        "save_application",
    }
    assert "optional role, location, and experience-level filters" in (
        tools_by_name["search_jobs"].description or ""
    )


def test_server_registers_score_job_match_tool() -> None:
    """Verify the score Tool's required input and discovery description."""
    registered_tools = asyncio.run(mcp.list_tools())
    score_tool = next(
        tool for tool in registered_tools if tool.name == "score_job_match"
    )

    assert set(score_tool.input_schema["properties"]) == {"job_id"}
    assert score_tool.input_schema["required"] == ["job_id"]
    description = (score_tool.description or "").casefold()
    assert "deterministic weighted scoring" in description
    assert "component scores and evidence" in description


def test_server_registers_save_application_tool() -> None:
    """Verify discovery describes the Tool's inputs and persistent side effect."""
    registered_tools = asyncio.run(mcp.list_tools())
    save_tool = next(
        tool for tool in registered_tools if tool.name == "save_application"
    )

    assert set(save_tool.input_schema["properties"]) == {
        "job_id",
        "status",
        "notes",
    }
    assert save_tool.input_schema["required"] == ["job_id"]
    description = (save_tool.description or "").casefold()
    assert "persist" in description
    assert "status" in description and "defaults to" in description
    assert "notes" in description and "optional" in description
    assert "application id" in description
    assert "utc timestamp" in description
    assert "generated" in description and "internally" in description


def test_server_registers_static_candidate_profile_resource() -> None:
    """Verify Resource metadata separately from file loading and stdio transport."""
    registered_resources = asyncio.run(mcp.list_resources())
    resources_by_uri = {
        str(resource.uri): resource for resource in registered_resources
    }

    assert len(registered_resources) == 2
    assert set(resources_by_uri) == {
        "candidate://profile",
        "applications://all",
    }
    candidate_profile = resources_by_uri["candidate://profile"]
    assert candidate_profile.mime_type == "application/json"
    assert "candidate profile" in (candidate_profile.description or "").casefold()


def test_server_registers_applications_resource() -> None:
    """Verify application history is registered as persisted readable context."""
    registered_resources = asyncio.run(mcp.list_resources())
    resources_by_uri = {
        str(resource.uri): resource for resource in registered_resources
    }

    assert len(registered_resources) == 2
    assert set(resources_by_uri) == {
        "candidate://profile",
        "applications://all",
    }
    applications_resource = resources_by_uri["applications://all"]
    assert applications_resource.mime_type == "application/json"
    description = (applications_resource.description or "").casefold()
    assert "application history" in description
    assert "persisted" in description


def test_server_registers_job_details_resource_template() -> None:
    """Verify the parameterized URI is discoverable only as a template."""
    resource_templates = asyncio.run(mcp.list_resource_templates())

    assert len(resource_templates) == 1
    job_details = resource_templates[0]
    assert job_details.uri_template == "jobs://job/{job_id}"
    assert job_details.mime_type == "application/json"
    assert "full details" in (job_details.description or "").casefold()


def test_server_registers_prepare_application_prompt() -> None:
    """Verify Prompt discovery exposes one required workflow argument."""
    registered_prompts = asyncio.run(mcp.list_prompts())

    assert len(registered_prompts) == 1
    prompt = registered_prompts[0]
    assert prompt.name == "prepare_application"
    assert "application-preparation workflow" in (
        prompt.description or ""
    ).casefold()
    assert prompt.arguments is not None
    assert [(argument.name, argument.required) for argument in prompt.arguments] == [
        ("job_id", True)
    ]

    # Prompt registration must not change the established operational surface.
    assert {
        tool.name for tool in asyncio.run(mcp.list_tools())
    } == {"search_jobs", "score_job_match", "save_application"}
    assert {
        str(resource.uri) for resource in asyncio.run(mcp.list_resources())
    } == {"candidate://profile", "applications://all"}
    assert {
        template.uri_template
        for template in asyncio.run(mcp.list_resource_templates())
    } == {"jobs://job/{job_id}"}
