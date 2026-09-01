"""Host-owned models and pure normalization for discovered MCP capabilities.

The MCP SDK remains the protocol boundary, while these small models give future
Host UI and orchestration code a stable application-facing catalog. Normalization
copies metadata only; it does not interpret capability purpose or side effects.
"""

from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass
from typing import Any

from mcp import types


@dataclass(frozen=True)
class ToolCapability:
    """Host view of a Tool's discoverable name, description, and input schema."""

    name: str
    description: str | None
    input_schema: dict[str, Any]


@dataclass(frozen=True)
class ResourceCapability:
    """Host view of one statically addressable MCP Resource."""

    uri: str
    name: str
    description: str | None
    mime_type: str | None


@dataclass(frozen=True)
class ResourceTemplateCapability:
    """Host view of a parameterized MCP Resource URI template."""

    uri_template: str
    name: str
    description: str | None
    mime_type: str | None


@dataclass(frozen=True)
class PromptArgument:
    """Host view of one Prompt argument exactly as advertised by MCP."""

    name: str
    description: str | None
    required: bool | None


@dataclass(frozen=True)
class PromptCapability:
    """Host view of a discoverable model-facing MCP Prompt."""

    name: str
    description: str | None
    arguments: tuple[PromptArgument, ...]


@dataclass(frozen=True)
class CapabilityCatalog:
    """Immutable grouping of all capability categories discovered in one pass."""

    tools: tuple[ToolCapability, ...]
    resources: tuple[ResourceCapability, ...]
    resource_templates: tuple[ResourceTemplateCapability, ...]
    prompts: tuple[PromptCapability, ...]


def normalize_tool(tool: types.Tool) -> ToolCapability:
    """Copy Tool discovery metadata without inferring operational semantics."""
    # A deep copy prevents later mutation of an SDK model's JSON schema from
    # silently changing the Host catalog. The schema itself stays JSON-native.
    return ToolCapability(
        name=tool.name,
        description=tool.description,
        input_schema=deepcopy(tool.input_schema),
    )


def normalize_resource(resource: types.Resource) -> ResourceCapability:
    """Preserve the exact Resource identity and optional display metadata."""
    return ResourceCapability(
        uri=str(resource.uri),
        name=resource.name,
        description=resource.description,
        mime_type=resource.mime_type,
    )


def normalize_resource_template(
    resource_template: types.ResourceTemplate,
) -> ResourceTemplateCapability:
    """Preserve a Resource Template without interpreting its URI variables."""
    return ResourceTemplateCapability(
        uri_template=resource_template.uri_template,
        name=resource_template.name,
        description=resource_template.description,
        mime_type=resource_template.mime_type,
    )


def normalize_prompt_argument(argument: types.PromptArgument) -> PromptArgument:
    """Preserve Prompt argument metadata, including an unspecified required flag."""
    return PromptArgument(
        name=argument.name,
        description=argument.description,
        required=argument.required,
    )


def normalize_prompt(prompt: types.Prompt) -> PromptCapability:
    """Convert a Prompt and its optional arguments into Host-owned values."""
    return PromptCapability(
        name=prompt.name,
        description=prompt.description,
        arguments=tuple(
            normalize_prompt_argument(argument)
            for argument in (prompt.arguments or [])
        ),
    )


def normalize_capabilities(
    *,
    tools: Iterable[types.Tool],
    resources: Iterable[types.Resource],
    resource_templates: Iterable[types.ResourceTemplate],
    prompts: Iterable[types.Prompt],
) -> CapabilityCatalog:
    """Build one Host catalog from unmodified MCP discovery result objects."""
    return CapabilityCatalog(
        tools=tuple(normalize_tool(tool) for tool in tools),
        resources=tuple(normalize_resource(resource) for resource in resources),
        resource_templates=tuple(
            normalize_resource_template(template)
            for template in resource_templates
        ),
        prompts=tuple(normalize_prompt(prompt) for prompt in prompts),
    )
