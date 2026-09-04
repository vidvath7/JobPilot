"""Minimal interactive Host for viewing dynamically discovered MCP capabilities.

The CLI makes MCP discovery visible to a human without executing capabilities.
It performs one catalog discovery at startup, then renders that immutable snapshot
for the lifetime of the REPL session.
"""

import asyncio
import json
from collections.abc import Callable, Sequence
from pathlib import Path

from mcp import types
from mcp.shared.exceptions import MCPError
from dotenv import load_dotenv

from host.capabilities import (
    CapabilityCatalog,
    PromptCapability,
)
from host.mcp_client import JobPilotMCPClient


WELCOME_MESSAGE = 'JobPilot Host connected. Type "help" for available commands.'
UNKNOWN_COMMAND_MESSAGE = 'Unknown command. Type "help" for available commands.'
PROJECT_ROOT = Path(__file__).resolve().parents[1]


def _load_local_environment(repository_root: Path = PROJECT_ROOT) -> None:
    """Load root ``.env`` values without replacing the process environment.

    Configuration loading belongs at the executable Host boundary. Provider
    adapters such as ``NVIDIALLMClient`` remain unaware of files and continue to
    consume only the resulting environment variables.
    """
    load_dotenv(repository_root / ".env", override=False)


def format_help() -> str:
    """Describe only the commands supported by this intentionally small REPL."""
    return """Available commands:
  help          Show this command list.
  capabilities  Show all capabilities discovered from the MCP server.
  prompts       Show discovered Prompt workflows and their arguments.
  call <tool_name> [<json_object>]
                Invoke a Tool, for example: call example_tool {"key":"value"}
  read <resource_uri>
                Read a Resource, for example: read example://context
  prompt <prompt_name> [<json_object>]
                Retrieve a Prompt, for example: prompt example_prompt {"id":"123"}
  quit          Exit JobPilot.
  exit          Alias for quit."""


def format_capabilities(catalog: CapabilityCatalog) -> str:
    """Render every catalog category without hardcoding capability identities."""
    sections = [
        _format_section(
            "Tools",
            [
                _format_capability(tool.name, tool.description)
                for tool in catalog.tools
            ],
        ),
        _format_section(
            "Resources",
            [
                _format_capability(resource.uri, resource.description)
                for resource in catalog.resources
            ],
        ),
        _format_section(
            "Resource Templates",
            [
                _format_capability(template.uri_template, template.description)
                for template in catalog.resource_templates
            ],
        ),
        _format_section(
            "Prompts",
            [
                _format_capability(prompt.name, prompt.description)
                for prompt in catalog.prompts
            ],
        ),
    ]
    return "\n\n".join(sections)


def format_prompts(prompts: Sequence[PromptCapability]) -> str:
    """Present discovered Prompts as user-visible workflows with arguments."""
    if not prompts:
        return "Available Prompt workflows:\n- None discovered"

    lines = ["Available Prompt workflows:"]
    for prompt in prompts:
        lines.append(_format_capability(prompt.name, prompt.description))
        if not prompt.arguments:
            lines.append("  Arguments: none")
            continue

        lines.append("  Arguments:")
        for argument in prompt.arguments:
            requirement = "required" if argument.required is True else "optional"
            description = (
                f" - {argument.description}" if argument.description else ""
            )
            lines.append(f"    - {argument.name} ({requirement}){description}")
    return "\n".join(lines)


async def run_repl(
    catalog: CapabilityCatalog,
    client: JobPilotMCPClient,
    *,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> None:
    """Process explicit commands through one client and cached capability catalog."""
    output_function(WELCOME_MESSAGE)

    while True:
        try:
            command_line = input_function("JobPilot> ").strip()
        except (EOFError, KeyboardInterrupt):
            # Terminal end-of-input and Ctrl+C use the same normal return path as
            # quit, allowing the surrounding async client context to close.
            output_function("")
            return

        parts = command_line.split(maxsplit=2)
        command = parts[0].casefold() if parts else ""

        if command == "help" and len(parts) == 1:
            output_function(format_help())
        elif command == "capabilities" and len(parts) == 1:
            output_function(format_capabilities(catalog))
        elif command == "prompts" and len(parts) == 1:
            output_function(format_prompts(catalog.prompts))
        elif command in {"quit", "exit"} and len(parts) == 1:
            return
        elif command == "call":
            await _handle_call(parts, client, output_function)
        elif command == "read":
            await _handle_read(parts, client, output_function)
        elif command == "prompt":
            await _handle_prompt(parts, client, output_function)
        else:
            # Deliberately avoid fuzzy or natural-language interpretation: command
            # selection remains deterministic until an LLM milestone approves it.
            output_function(UNKNOWN_COMMAND_MESSAGE)


async def run_host() -> None:
    """Connect, discover once, run the REPL, and close the MCP client normally."""
    async with JobPilotMCPClient() as client:
        catalog = await client.discover_capabilities()
        await run_repl(catalog, client)


async def _handle_call(
    parts: list[str],
    client: JobPilotMCPClient,
    output_function: Callable[[str], None],
) -> None:
    """Parse one generic Tool command and render its raw MCP result."""
    if len(parts) < 2:
        output_function("Usage: call <tool_name> [<json_arguments>]")
        return

    arguments = _parse_json_object(parts[2] if len(parts) == 3 else None)
    if isinstance(arguments, str):
        output_function(arguments)
        return

    try:
        result = await client.call_tool(parts[1], arguments)
    except MCPError as error:
        output_function(_format_mcp_error(error))
        return
    output_function(format_tool_result(result))


async def _handle_read(
    parts: list[str],
    client: JobPilotMCPClient,
    output_function: Callable[[str], None],
) -> None:
    """Route one exact Resource URI through the generic Host client."""
    if len(parts) != 2:
        output_function("Usage: read <resource_uri>")
        return

    try:
        result = await client.read_resource(parts[1])
    except MCPError as error:
        output_function(_format_mcp_error(error))
        return
    output_function(format_resource_result(result))


async def _handle_prompt(
    parts: list[str],
    client: JobPilotMCPClient,
    output_function: Callable[[str], None],
) -> None:
    """Parse generic Prompt arguments and render returned MCP messages."""
    if len(parts) < 2:
        output_function("Usage: prompt <prompt_name> [<json_arguments>]")
        return

    arguments = _parse_json_object(parts[2] if len(parts) == 3 else None)
    if isinstance(arguments, str):
        output_function(arguments)
        return
    if arguments is not None and not all(
        isinstance(value, str) for value in arguments.values()
    ):
        output_function("Prompt argument values must be strings.")
        return

    try:
        result = await client.get_prompt(parts[1], arguments)
    except MCPError as error:
        output_function(_format_mcp_error(error))
        return
    output_function(format_prompt_result(result))


def _parse_json_object(raw_arguments: str | None) -> dict[str, object] | str | None:
    """Decode optional command arguments and return concise validation errors."""
    if raw_arguments is None:
        return None
    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        return f"Invalid JSON arguments: {error.msg}."
    if not isinstance(arguments, dict):
        return "JSON arguments must be an object."
    return arguments


def format_tool_result(result: object) -> str:
    """Render MCP Tool success/error state and its most useful available content."""
    if not isinstance(result, types.CallToolResult):
        return _format_unexpected_result("Tool", result)

    status = "error" if result.is_error is True else "success"
    lines = [f"Tool result ({status}):"]
    if result.structured_content is not None:
        lines.append(_format_json(result.structured_content))
    else:
        lines.extend(_format_content_block(content) for content in result.content)
    return "\n".join(lines)


def format_resource_result(result: object) -> str:
    """Render text Resource contents while safely describing other content types."""
    if not isinstance(result, types.ReadResourceResult):
        return _format_unexpected_result("Resource", result)

    lines = ["Resource result:"]
    for content in result.contents:
        if isinstance(content, types.TextResourceContents):
            lines.append(content.text)
        elif isinstance(content, types.BlobResourceContents):
            lines.append(
                f"[binary resource: {content.mime_type or 'unknown MIME type'}]"
            )
        else:
            lines.append(_format_model(content))
    return "\n".join(lines)


def format_prompt_result(result: object) -> str:
    """Render MCP Prompt roles and message content without provider conversion."""
    if not isinstance(result, types.GetPromptResult):
        return _format_unexpected_result("Prompt", result)

    lines = ["Prompt result:"]
    for message in result.messages:
        lines.append(f"{message.role}:")
        lines.append(_format_content_block(message.content))
    return "\n".join(lines)


def _format_content_block(content: object) -> str:
    """Prefer human-readable MCP text and safely serialize other content blocks."""
    if isinstance(content, types.TextContent):
        return content.text
    return _format_model(content)


def _format_model(value: object) -> str:
    """Serialize an MCP model when no dedicated human-readable rendering exists."""
    if hasattr(value, "model_dump"):
        return _format_json(value.model_dump(by_alias=True, mode="json"))
    return str(value)


def _format_json(value: object) -> str:
    """Pretty-print JSON-compatible MCP structured data."""
    return json.dumps(value, indent=2, ensure_ascii=False)


def _format_unexpected_result(operation: str, result: object) -> str:
    """Represent future MCP result variants without crashing the REPL."""
    return f"{operation} returned an unsupported result type: {type(result).__name__}."


def _format_mcp_error(error: MCPError) -> str:
    """Expose concise protocol error information without a traceback."""
    return f"MCP error: {error.message}"


def _format_section(title: str, entries: Sequence[str]) -> str:
    """Render an empty-safe capability category."""
    rendered_entries = entries or ["- None discovered"]
    return "\n".join([f"{title}:", *rendered_entries])


def _format_capability(identifier: str, description: str | None) -> str:
    """Render protocol-provided identity and optional description without inference."""
    # SDK descriptions may originate from multiline docstrings. Collapsing only
    # display whitespace keeps terminal output compact without changing catalog data.
    rendered_description = " ".join(description.split()) if description else None
    return f"- {identifier}" + (
        f" - {rendered_description}" if rendered_description else ""
    )


def main() -> None:
    """Run the asynchronous Host lifecycle from ``python -m host.main``."""
    _load_local_environment()
    asyncio.run(run_host())


if __name__ == "__main__":
    main()
