"""End-to-end protocol test for the MCP Client ↔ JobPilot Server stdio slice.

Unlike component tests, this module never imports the Tool adapter or JobService.
It discovers and invokes ``search_jobs`` exactly as an external MCP client would,
so failures expose initialization, schema, serialization, or transport problems.
"""

import asyncio
import sys
from pathlib import Path

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


PROJECT_ROOT = Path(__file__).resolve().parents[1]
SUMMARY_FIELDS = {"id", "title", "company", "location", "experience_level"}


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
            assert [tool.name for tool in tools_result.tools] == ["search_jobs"]

            search_tool = tools_result.tools[0]
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
