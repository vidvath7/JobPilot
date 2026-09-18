"""Unit tests for Host normalization below the MCP transport boundary.

Actual MCP SDK metadata models provide controlled inputs. This isolates catalog
mapping failures from stdio lifecycle and server discovery failures.
"""

from dataclasses import FrozenInstanceError

import pytest
from mcp import types

from host.capabilities import (
    PromptArgument,
    PromptCapability,
    ResourceCapability,
    ResourceTemplateCapability,
    ToolCapability,
    normalize_capabilities,
)


def test_normalizes_sdk_metadata_without_interpretation() -> None:
    """Preserve protocol fields while returning independent Host model types."""
    input_schema = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }
    sdk_tool = types.Tool(
        name="example_tool",
        description="Example description",
        input_schema=input_schema,
    )
    sdk_resource = types.Resource(
        uri="example://resource",
        name="example_resource",
        description=None,
        mime_type="application/json",
    )
    sdk_template = types.ResourceTemplate(
        uri_template="example://item/{item_id}",
        name="example_item",
        description="Returns one example item.",
        mime_type="application/json",
    )
    sdk_prompt = types.Prompt(
        name="example_prompt",
        description=None,
        arguments=[
            types.PromptArgument(
                name="item_id",
                description="Selected item identifier.",
                required=True,
            )
        ],
    )

    catalog = normalize_capabilities(
        tools=[sdk_tool],
        resources=[sdk_resource],
        resource_templates=[sdk_template],
        prompts=[sdk_prompt],
    )

    assert catalog.tools == (
        ToolCapability(
            name="example_tool",
            description="Example description",
            input_schema=input_schema,
        ),
    )
    assert catalog.resources == (
        ResourceCapability(
            uri="example://resource",
            name="example_resource",
            description=None,
            mime_type="application/json",
        ),
    )
    assert catalog.resource_templates == (
        ResourceTemplateCapability(
            uri_template="example://item/{item_id}",
            name="example_item",
            description="Returns one example item.",
            mime_type="application/json",
        ),
    )
    assert catalog.prompts == (
        PromptCapability(
            name="example_prompt",
            description=None,
            arguments=(
                PromptArgument(
                    name="item_id",
                    description="Selected item identifier.",
                    required=True,
                ),
            ),
        ),
    )
    prompt_argument = catalog.prompts[0].arguments[0]
    assert prompt_argument.name == "item_id"
    assert prompt_argument.description == "Selected item identifier."
    assert prompt_argument.required is True

    assert isinstance(catalog.tools[0], ToolCapability)
    assert not isinstance(catalog.tools[0], types.Tool)
    assert isinstance(catalog.resources[0], ResourceCapability)
    assert not isinstance(catalog.resources[0], types.Resource)
    assert isinstance(catalog.resource_templates[0], ResourceTemplateCapability)
    assert not isinstance(catalog.resource_templates[0], types.ResourceTemplate)
    assert isinstance(catalog.prompts[0], PromptCapability)
    assert not isinstance(catalog.prompts[0], types.Prompt)

    # The catalog owns a deep-copied schema and frozen model fields.
    assert catalog.tools[0].input_schema is not sdk_tool.input_schema
    sdk_tool.input_schema["required"].clear()
    assert catalog.tools[0].input_schema["required"] == ["job_id"]
    with pytest.raises(FrozenInstanceError):
        catalog.tools[0].name = "changed"  # type: ignore[misc]


def test_prompt_without_arguments_normalizes_to_empty_tuple() -> None:
    """Handle absent optional Prompt metadata without inventing arguments."""
    sdk_prompt = types.Prompt(
        name="argument_free_prompt",
        description=None,
        arguments=None,
    )

    catalog = normalize_capabilities(
        tools=[],
        resources=[],
        resource_templates=[],
        prompts=[sdk_prompt],
    )

    assert catalog.prompts[0].description is None
    assert catalog.prompts[0].arguments == ()
