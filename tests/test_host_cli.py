"""Unit tests for deterministic Host CLI commands and MCP-aware rendering.

These tests inject a controlled catalog and terminal functions, so they validate
the human-facing Host layer without starting stdio or requiring manual input.
"""

import asyncio

import pytest
from mcp import types
from mcp.shared.exceptions import MCPError

from host.capabilities import (
    CapabilityCatalog,
    PromptArgument,
    PromptCapability,
    ResourceCapability,
    ResourceTemplateCapability,
    ToolCapability,
)
from host.main import (
    UNKNOWN_COMMAND_MESSAGE,
    format_capabilities,
    format_help,
    format_prompts,
    run_repl,
)


class FakeMCPClient:
    """Record generic CLI routing and return controlled MCP SDK results."""

    def __init__(self) -> None:
        self.tool_calls: list[tuple[str, dict[str, object] | None]] = []
        self.resource_reads: list[str] = []
        self.prompt_requests: list[tuple[str, dict[str, str] | None]] = []
        self.tool_result = types.CallToolResult(
            content=[],
            structuredContent={"result": "structured value"},
        )

    async def call_tool(
        self,
        name: str,
        arguments: dict[str, object] | None = None,
    ) -> types.CallToolResult:
        self.tool_calls.append((name, arguments))
        if name == "protocol_failure":
            raise MCPError(-32000, "Tool protocol failure")
        return self.tool_result

    async def read_resource(self, uri: str) -> types.ReadResourceResult:
        self.resource_reads.append(uri)
        if uri == "failure://resource":
            raise MCPError(-32000, "Resource protocol failure")
        return types.ReadResourceResult(
            contents=[
                types.TextResourceContents(
                    uri=uri,
                    mimeType="application/json",
                    text='{"resource": "text value"}',
                )
            ]
        )

    async def get_prompt(
        self,
        name: str,
        arguments: dict[str, str] | None = None,
    ) -> types.GetPromptResult:
        self.prompt_requests.append((name, arguments))
        if name == "protocol_failure":
            raise MCPError(-32000, "Prompt protocol failure")
        return types.GetPromptResult(
            messages=[
                types.PromptMessage(
                    role="user",
                    content=types.TextContent(
                        type="text",
                        text="Rendered prompt instructions.",
                    ),
                )
            ]
        )


@pytest.fixture
def catalog() -> CapabilityCatalog:
    """Provide synthetic metadata so CLI tests cannot hardcode server contents."""
    return CapabilityCatalog(
        tools=(
            ToolCapability(
                name="synthetic_tool",
                description="Performs a synthetic operation.",
                input_schema={"type": "object", "properties": {}},
            ),
        ),
        resources=(
            ResourceCapability(
                uri="synthetic://context",
                name="synthetic_context",
                description="Provides synthetic context.",
                mime_type="application/json",
            ),
        ),
        resource_templates=(
            ResourceTemplateCapability(
                uri_template="synthetic://item/{item_id}",
                name="synthetic_item",
                description="Provides one synthetic item.",
                mime_type="application/json",
            ),
        ),
        prompts=(
            PromptCapability(
                name="prepare_application",
                description="Prepare an application workflow.",
                arguments=(
                    PromptArgument(
                        name="job_id",
                        description="Selected JobPilot job.",
                        required=True,
                    ),
                ),
            ),
            PromptCapability(
                name="second_workflow",
                description="A newly discovered workflow.",
                arguments=(
                    PromptArgument(
                        name="optional_context",
                        description=None,
                        required=False,
                    ),
                ),
            ),
        ),
    )


def test_help_lists_exact_supported_commands() -> None:
    """Keep the documented command surface aligned with the deterministic parser."""
    help_output = format_help()

    for command in ("help", "capabilities", "prompts", "quit", "exit"):
        assert command in help_output


def test_capabilities_output_comes_from_catalog(catalog: CapabilityCatalog) -> None:
    """Render every category using discovered values rather than JobPilot names."""
    output = format_capabilities(catalog)

    assert "Tools:" in output and "synthetic_tool" in output
    assert "Resources:" in output and "synthetic://context" in output
    assert "Resource Templates:" in output
    assert "synthetic://item/{item_id}" in output
    assert "Prompts:" in output and "prepare_application" in output
    assert "Performs a synthetic operation." in output


def test_prompts_show_dynamic_workflows_and_argument_requirements(
    catalog: CapabilityCatalog,
) -> None:
    """Surface every discovered Prompt without adding command-specific code."""
    output = format_prompts(catalog.prompts)

    assert "prepare_application" in output
    assert "Prepare an application workflow." in output
    assert "job_id (required)" in output
    assert "second_workflow" in output
    assert "A newly discovered workflow." in output
    assert "optional_context (optional)" in output


def test_unknown_command_prints_guidance(catalog: CapabilityCatalog) -> None:
    """Reject unsupported input without fuzzy or natural-language interpretation."""
    output: list[str] = []
    commands = iter(["do something", "quit"])

    asyncio.run(
        run_repl(
            catalog,
            FakeMCPClient(),
            input_function=lambda _: next(commands),
            output_function=output.append,
        )
    )

    assert UNKNOWN_COMMAND_MESSAGE in output


def test_repl_routes_display_commands_to_cached_catalog(
    catalog: CapabilityCatalog,
) -> None:
    """Exercise all display commands without triggering another discovery pass."""
    output: list[str] = []
    commands = iter(["help", "capabilities", "prompts", "quit"])

    asyncio.run(
        run_repl(
            catalog,
            FakeMCPClient(),
            input_function=lambda _: next(commands),
            output_function=output.append,
        )
    )

    rendered_output = "\n".join(output)
    assert "Available commands:" in rendered_output
    assert "synthetic_tool" in rendered_output
    assert "prepare_application" in rendered_output
    assert "job_id (required)" in rendered_output


@pytest.mark.parametrize("command", ["quit", "exit"])
def test_quit_commands_terminate_loop(
    command: str,
    catalog: CapabilityCatalog,
) -> None:
    """Both approved exit commands return without requesting more input."""
    calls = 0

    def input_function(_: str) -> str:
        nonlocal calls
        calls += 1
        return command

    asyncio.run(
        run_repl(
            catalog,
            FakeMCPClient(),
            input_function=input_function,
            output_function=lambda _: None,
        )
    )

    assert calls == 1


def test_rendering_handles_empty_capability_categories() -> None:
    """Display useful empty states instead of assuming discovery returned entries."""
    empty_catalog = CapabilityCatalog(
        tools=(),
        resources=(),
        resource_templates=(),
        prompts=(),
    )

    output = format_capabilities(empty_catalog)
    assert output.count("None discovered") == 4
    assert format_prompts(empty_catalog.prompts).endswith("None discovered")


def test_call_parses_object_and_renders_structured_content(
    catalog: CapabilityCatalog,
) -> None:
    """Route JSON object arguments and prefer Tool structured content."""
    client = FakeMCPClient()
    output = _run_commands(
        catalog,
        client,
        ['call synthetic_tool {"count": 2}', "quit"],
    )

    assert client.tool_calls == [("synthetic_tool", {"count": 2})]
    assert "Tool result (success):" in output
    assert '"result": "structured value"' in output


def test_call_renders_tool_error_result(catalog: CapabilityCatalog) -> None:
    """Distinguish a completed Tool error result from successful execution."""
    client = FakeMCPClient()
    client.tool_result = types.CallToolResult(
        content=[types.TextContent(type="text", text="Domain validation failed.")],
        isError=True,
    )

    output = _run_commands(catalog, client, ["call synthetic_tool", "quit"])

    assert "Tool result (error):" in output
    assert "Domain validation failed." in output


def test_read_routes_uri_and_renders_text(catalog: CapabilityCatalog) -> None:
    """Pass a Resource URI unchanged and display returned text content."""
    client = FakeMCPClient()

    output = _run_commands(
        catalog,
        client,
        ["read synthetic://context", "quit"],
    )

    assert client.resource_reads == ["synthetic://context"]
    assert "Resource result:" in output
    assert '"resource": "text value"' in output


def test_prompt_parses_arguments_and_renders_messages(
    catalog: CapabilityCatalog,
) -> None:
    """Route string Prompt arguments and label returned message roles."""
    client = FakeMCPClient()

    output = _run_commands(
        catalog,
        client,
        ['prompt prepare_application {"job_id": "JOB-005"}', "quit"],
    )

    assert client.prompt_requests == [
        ("prepare_application", {"job_id": "JOB-005"})
    ]
    assert "Prompt result:" in output
    assert "user:" in output
    assert "Rendered prompt instructions." in output


def test_argument_errors_are_recoverable(catalog: CapabilityCatalog) -> None:
    """Reject malformed inputs and continue processing later valid commands."""
    client = FakeMCPClient()

    output = _run_commands(
        catalog,
        client,
        [
            "call",
            "call synthetic_tool {invalid}",
            "call synthetic_tool [1, 2]",
            'call synthetic_tool {"valid": true}',
            "quit",
        ],
    )

    assert "Usage: call" in output
    assert "Invalid JSON arguments:" in output
    assert "JSON arguments must be an object." in output
    assert client.tool_calls == [("synthetic_tool", {"valid": True})]
    assert "Tool result (success):" in output


def test_mcp_errors_are_concise_and_repl_survives(
    catalog: CapabilityCatalog,
) -> None:
    """Render protocol failures without tracebacks and continue to the next command."""
    client = FakeMCPClient()

    output = _run_commands(
        catalog,
        client,
        [
            "call protocol_failure",
            "read failure://resource",
            'prompt protocol_failure {"id": "123"}',
            "help",
            "quit",
        ],
    )

    assert "MCP error: Tool protocol failure" in output
    assert "MCP error: Resource protocol failure" in output
    assert "MCP error: Prompt protocol failure" in output
    assert "Traceback" not in output
    assert "Available commands:" in output


def _run_commands(
    catalog: CapabilityCatalog,
    client: FakeMCPClient,
    commands: list[str],
) -> str:
    """Run scripted input through the async REPL and combine captured output."""
    command_iterator = iter(commands)
    output: list[str] = []
    asyncio.run(
        run_repl(
            catalog,
            client,
            input_function=lambda _: next(command_iterator),
            output_function=output.append,
        )
    )
    return "\n".join(output)
