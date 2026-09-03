"""Minimal interactive Host for viewing dynamically discovered MCP capabilities.

The CLI makes MCP discovery visible to a human without executing capabilities.
It performs one catalog discovery at startup, then renders that immutable snapshot
for the lifetime of the REPL session.
"""

import asyncio
from collections.abc import Callable, Sequence

from host.capabilities import (
    CapabilityCatalog,
    PromptCapability,
)
from host.mcp_client import JobPilotMCPClient


WELCOME_MESSAGE = 'JobPilot Host connected. Type "help" for available commands.'
UNKNOWN_COMMAND_MESSAGE = 'Unknown command. Type "help" for available commands.'


def format_help() -> str:
    """Describe only the commands supported by this intentionally small REPL."""
    return """Available commands:
  help          Show this command list.
  capabilities  Show all capabilities discovered from the MCP server.
  prompts       Show discovered Prompt workflows and their arguments.
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


def run_repl(
    catalog: CapabilityCatalog,
    *,
    input_function: Callable[[str], str] = input,
    output_function: Callable[[str], None] = print,
) -> None:
    """Process explicit commands against one previously discovered catalog."""
    output_function(WELCOME_MESSAGE)

    while True:
        try:
            command = input_function("JobPilot> ").strip().casefold()
        except (EOFError, KeyboardInterrupt):
            # Terminal end-of-input and Ctrl+C use the same normal return path as
            # quit, allowing the surrounding async client context to close.
            output_function("")
            return

        if command == "help":
            output_function(format_help())
        elif command == "capabilities":
            output_function(format_capabilities(catalog))
        elif command == "prompts":
            output_function(format_prompts(catalog.prompts))
        elif command in {"quit", "exit"}:
            return
        else:
            # Deliberately avoid fuzzy or natural-language interpretation: command
            # selection remains deterministic until an LLM milestone approves it.
            output_function(UNKNOWN_COMMAND_MESSAGE)


async def run_host() -> None:
    """Connect, discover once, run the REPL, and close the MCP client normally."""
    async with JobPilotMCPClient() as client:
        catalog = await client.discover_capabilities()
        run_repl(catalog)


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
    asyncio.run(run_host())


if __name__ == "__main__":
    main()
