"""Unit tests for deterministic Host CLI commands and catalog rendering.

These tests inject a controlled catalog and terminal functions, so they validate
the human-facing Host layer without starting stdio or requiring manual input.
"""

import pytest

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

    run_repl(
        catalog,
        input_function=lambda _: next(commands),
        output_function=output.append,
    )

    assert UNKNOWN_COMMAND_MESSAGE in output


def test_repl_routes_display_commands_to_cached_catalog(
    catalog: CapabilityCatalog,
) -> None:
    """Exercise all display commands without triggering another discovery pass."""
    output: list[str] = []
    commands = iter(["help", "capabilities", "prompts", "quit"])

    run_repl(
        catalog,
        input_function=lambda _: next(commands),
        output_function=output.append,
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

    run_repl(catalog, input_function=input_function, output_function=lambda _: None)

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
