"""End-to-end protocol test for the MCP Client ↔ JobPilot Server stdio slice.

Unlike component tests, this module never imports the Tool adapter or JobService.
It discovers and invokes ``search_jobs`` exactly as an external MCP client would,
so failures expose initialization, schema, serialization, or transport problems.
"""

import asyncio
import json
import os
import sys
from pathlib import Path

import pytest
from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client
from mcp.shared.exceptions import MCPError


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FIELDS = {"id", "title", "company", "location", "experience_level"}
FULL_JOB_FIELDS = SUMMARY_FIELDS | {"required_skills", "description", "url"}


def test_search_jobs_over_mcp_stdio() -> None:
    """Exercise discovery and Tool calls across a real MCP stdio session."""
    asyncio.run(_exercise_search_jobs_over_mcp_stdio())


async def _exercise_search_jobs_over_mcp_stdio() -> None:
    """Run the MCP lifecycle while ensuring all async resources close cleanly."""
    # ``sys.executable`` is pytest's project-interpreter path, avoiding reliance on
    # a globally installed Python or MCP command—especially important on Windows.
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.main"],
        cwd=PROJECT_ROOT,
    )

    # stdio_client owns the server subprocess and byte streams; ClientSession owns
    # MCP request/response state. Exiting both contexts shuts the process down.
    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=10.0,
        ) as session:
            # Initialization negotiates protocol capabilities before either peer
            # may perform normal discovery or invocation requests.
            initialization = await session.initialize()
            assert initialization.server_info.name == "jobpilot"

            # list_tools() tests protocol discovery—not the server's Python
            # registry—and exposes the generated description and input schema.
            tools_result = await session.list_tools()
            tools_by_name = {tool.name: tool for tool in tools_result.tools}
            assert set(tools_by_name) == {
                "search_jobs",
                "score_job_match",
                "save_application",
            }

            search_tool = tools_by_name["search_jobs"]
            assert "optional role, location, and experience-level filters" in (
                search_tool.description or ""
            )
            assert set(search_tool.input_schema["properties"]) == {
                "role",
                "location",
                "experience_level",
            }
            assert search_tool.input_schema.get("required", []) == []

            # call_tool() crosses stdio and dispatches by protocol name. A direct
            # Python call here would bypass the lifecycle this test exists to prove.
            filtered_result = await session.call_tool(
                "search_jobs",
                arguments={"role": "AI"},
            )
            filtered_jobs = _jobs_from_result(filtered_result)
            assert filtered_jobs
            assert all(set(job) == SUMMARY_FIELDS for job in filtered_jobs)
            assert all("ai" in job["title"].casefold() for job in filtered_jobs)

            all_jobs_result = await session.call_tool("search_jobs", arguments={})
            assert len(_jobs_from_result(all_jobs_result)) == 10


def _jobs_from_result(result: object) -> list[dict[str, str]]:
    """Extract jobs from MCP v2 structured output after checking success fields."""
    # MCP v2 provides structured_content for typed Tool results, so the test does
    # not parse display-oriented text content or assume a legacy response shape.
    assert isinstance(result, types.CallToolResult)
    assert result.result_type == "complete"
    assert result.is_error is False
    assert isinstance(result.structured_content, dict)

    jobs = result.structured_content.get("result")
    assert isinstance(jobs, list)
    assert all(isinstance(job, dict) for job in jobs)
    return jobs


def test_candidate_profile_resource_over_mcp_stdio() -> None:
    """Discover and read the candidate profile through a real MCP session."""
    asyncio.run(_exercise_candidate_profile_resource_over_mcp_stdio())


async def _exercise_candidate_profile_resource_over_mcp_stdio() -> None:
    """Validate the Resource lifecycle without importing its Python adapter."""
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.main"],
        cwd=PROJECT_ROOT,
    )

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=10.0,
        ) as session:
            await session.initialize()

            # Resource discovery proves the metadata crossed the protocol rather
            # than being inspected from the in-process MCPServer registry.
            resources_result = await session.list_resources()
            assert len(resources_result.resources) == 1
            resource = resources_result.resources[0]
            assert str(resource.uri) == "candidate://profile"
            assert resource.mime_type == "application/json"

            # read_resource() is the MCP operation for retrieving addressable
            # context; it is deliberately distinct from invoking a Tool.
            read_result = await session.read_resource("candidate://profile")
            assert isinstance(read_result, types.ReadResourceResult)
            assert read_result.result_type == "complete"
            assert len(read_result.contents) == 1

            content = read_result.contents[0]
            assert isinstance(content, types.TextResourceContents)
            assert str(content.uri) == "candidate://profile"
            assert content.mime_type == "application/json"

            profile = json.loads(content.text)
            assert isinstance(profile, dict)
            assert {
                "name",
                "summary",
                "skills",
                "experience",
                "education",
                "preferred_roles",
                "preferred_locations",
                "preferred_experience_levels",
            } <= profile.keys()


def test_job_details_resource_template_over_mcp_stdio() -> None:
    """Discover the template and read known and unknown jobs through MCP."""
    asyncio.run(_exercise_job_details_resource_template_over_mcp_stdio())


async def _exercise_job_details_resource_template_over_mcp_stdio() -> None:
    """Exercise template discovery, URI binding, and protocol error mapping."""
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.main"],
        cwd=PROJECT_ROOT,
    )

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=10.0,
        ) as session:
            await session.initialize()

            static_resources = await session.list_resources()
            assert [str(resource.uri) for resource in static_resources.resources] == [
                "candidate://profile"
            ]

            templates_result = await session.list_resource_templates()
            assert len(templates_result.resource_templates) == 1
            template = templates_result.resource_templates[0]
            assert template.uri_template == "jobs://job/{job_id}"
            assert template.mime_type == "application/json"
            assert "full details" in (template.description or "").casefold()

            read_result = await session.read_resource("jobs://job/JOB-001")
            assert isinstance(read_result, types.ReadResourceResult)
            assert read_result.result_type == "complete"
            assert len(read_result.contents) == 1

            content = read_result.contents[0]
            assert isinstance(content, types.TextResourceContents)
            assert str(content.uri) == "jobs://job/JOB-001"
            assert content.mime_type == "application/json"

            job = json.loads(content.text)
            assert isinstance(job, dict)
            assert job["id"] == "JOB-001"
            assert set(job) == FULL_JOB_FIELDS

            # Handler exceptions cross the JSON-RPC boundary as MCPError. Assert
            # the stable protocol shape without coupling to server traceback text.
            with pytest.raises(MCPError) as unknown_job_error:
                await session.read_resource("jobs://job/JOB-999")

            assert isinstance(unknown_job_error.value.code, int)
            assert unknown_job_error.value.message


def test_score_job_match_over_mcp_stdio() -> None:
    """Discover and invoke deterministic matching through the MCP Tool boundary."""
    asyncio.run(_exercise_score_job_match_over_mcp_stdio())


async def _exercise_score_job_match_over_mcp_stdio() -> None:
    """Verify Tool schema, structured output, and stable error signaling."""
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.main"],
        cwd=PROJECT_ROOT,
    )

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=10.0,
        ) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools_by_name = {tool.name: tool for tool in tools_result.tools}
            assert set(tools_by_name) == {
                "search_jobs",
                "score_job_match",
                "save_application",
            }

            score_tool = tools_by_name["score_job_match"]
            assert set(score_tool.input_schema["properties"]) == {"job_id"}
            assert score_tool.input_schema["required"] == ["job_id"]

            result = await session.call_tool(
                "score_job_match",
                arguments={"job_id": "JOB-005"},
            )
            assert isinstance(result, types.CallToolResult)
            assert result.result_type == "complete"
            assert result.is_error is False
            assert isinstance(result.structured_content, dict)

            match = result.structured_content
            assert set(match) == {
                "job_id",
                "job_title",
                "company",
                "score",
                "weights",
                "components",
                "evidence",
            }
            assert match["job_id"] == "JOB-005"
            assert match["score"] == 60.0
            assert match["weights"] == {
                "skills": 0.50,
                "role": 0.20,
                "experience_level": 0.20,
                "location": 0.10,
            }
            assert match["components"] == {
                "skills": 80.0,
                "role": 0.0,
                "experience_level": 100.0,
                "location": 0.0,
            }

            evidence = match["evidence"]
            assert evidence["matched_required_skills"] == [
                "Python",
                "pandas",
                "SQL",
                "scikit-learn",
            ]
            assert evidence["missing_required_skills"] == ["Statistics"]
            assert set(evidence["role_match"]) == {
                "job_role",
                "matched_preference",
                "match_type",
            }
            assert set(evidence["experience_match"]) == {
                "job_level",
                "candidate_preferences",
                "matched",
            }
            assert set(evidence["location_match"]) == {
                "job_location",
                "candidate_preferences",
                "matched_preference",
                "matched",
            }

            # Tool execution failures are represented as a completed
            # CallToolResult with is_error=True, unlike Resource read failures,
            # which arrive as MCPError exceptions in this SDK version.
            unknown_job = await session.call_tool(
                "score_job_match",
                arguments={"job_id": "JOB-999"},
            )
            assert isinstance(unknown_job, types.CallToolResult)
            assert unknown_job.result_type == "complete"
            assert unknown_job.is_error is True
            assert unknown_job.content


def test_save_application_over_mcp_stdio(tmp_path: Path) -> None:
    """Persist through real MCP while isolating the production application store."""
    applications_path = tmp_path / "applications.json"
    applications_path.write_text("[]\n", encoding="utf-8")

    asyncio.run(
        _exercise_save_application_over_mcp_stdio(applications_path)
    )


async def _exercise_save_application_over_mcp_stdio(
    applications_path: Path,
) -> None:
    """Verify discovery, state change, and duplicate error across stdio."""
    # StdioServerParameters passes this environment to the child server process.
    # Copying the current environment preserves Python/runtime configuration while
    # redirecting only JobPilot's mutable application store.
    server_environment = os.environ.copy()
    server_environment["JOBPILOT_APPLICATIONS_PATH"] = str(applications_path)
    server_parameters = StdioServerParameters(
        command=sys.executable,
        args=["-m", "server.main"],
        env=server_environment,
        cwd=PROJECT_ROOT,
    )

    async with stdio_client(server_parameters) as (read_stream, write_stream):
        async with ClientSession(
            read_stream,
            write_stream,
            read_timeout_seconds=10.0,
        ) as session:
            await session.initialize()

            tools_result = await session.list_tools()
            tools_by_name = {tool.name: tool for tool in tools_result.tools}
            assert set(tools_by_name) == {
                "search_jobs",
                "score_job_match",
                "save_application",
            }

            save_tool = tools_by_name["save_application"]
            assert set(save_tool.input_schema["properties"]) == {
                "job_id",
                "status",
                "notes",
            }
            assert save_tool.input_schema["required"] == ["job_id"]

            # This call crosses the complete MCP lifecycle and performs a real
            # write in the injected store; no adapter or service is imported here.
            result = await session.call_tool(
                "save_application",
                arguments={"job_id": "JOB-005"},
            )
            assert isinstance(result, types.CallToolResult)
            assert result.result_type == "complete"
            assert result.is_error is False
            assert isinstance(result.structured_content, dict)

            record = result.structured_content
            assert record["application_id"] == "APP-001"
            assert record["job_id"] == "JOB-005"
            assert record["status"] == "applied"
            assert isinstance(record["applied_at"], str)
            assert record["applied_at"]
            assert record["notes"] is None

            # Tool exceptions are converted into completed error results by the
            # SDK, keeping domain tracebacks behind the protocol boundary.
            duplicate = await session.call_tool(
                "save_application",
                arguments={"job_id": "JOB-005"},
            )
            assert isinstance(duplicate, types.CallToolResult)
            assert duplicate.result_type == "complete"
            assert duplicate.is_error is True
            assert duplicate.content

    persisted_records = json.loads(applications_path.read_text(encoding="utf-8"))
    assert persisted_records == [record]
