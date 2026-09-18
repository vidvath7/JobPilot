"""Host-local LLM function metadata and validation for MCP Resource reads.

This bridge is not a server Tool: it advertises discovered context to the model
and permits the Host to dispatch a validated URI through resources/read.
"""

import re

from mcp import types

from host.capabilities import CapabilityCatalog
from host.llm import LLMToolDefinition


RESOURCE_BRIDGE_NAME = "read_mcp_resource"


def resource_bridge_definition(catalog: CapabilityCatalog) -> LLMToolDefinition:
    """Describe available context dynamically without changing catalog.tools."""
    lines = [
        "Read an MCP Resource (read-only). The URI must match a discovered static "
        "Resource or a concrete instance of a discovered Resource Template.",
        "Static Resources:",
    ]
    lines.extend(
        f"- {item.uri}" + (f": {item.description}" if item.description else "")
        for item in catalog.resources
    )
    lines.append("Resource Templates (replace placeholders with concrete values):")
    lines.extend(
        f"- {item.uri_template}" + (f": {item.description}" if item.description else "")
        for item in catalog.resource_templates
    )
    return {
        "type": "function",
        "function": {
            "name": RESOURCE_BRIDGE_NAME,
            "description": "\n".join(lines),
            "parameters": {
                "type": "object",
                "properties": {"uri": {
                    "type": "string", "description": "The concrete discovered MCP Resource URI.",
                }},
                "required": ["uri"],
                "additionalProperties": False,
            },
        },
    }


def validate_resource_arguments(arguments: dict[str, object], catalog: CapabilityCatalog) -> str:
    """Accept exact static URIs or simple, single-segment template substitutions.

    Only {name} placeholders are supported. Literal text is regex-escaped and
    full matching prevents unrelated suffixes, paths, or query strings. This
    deliberately does not implement RFC URI-template expansion operators.
    """
    if set(arguments) != {"uri"} or not isinstance(arguments["uri"], str):
        raise ValueError("Resource bridge requires exactly one string argument: uri.")
    uri = arguments["uri"]
    if any(uri == item.uri for item in catalog.resources):
        return uri
    for item in catalog.resource_templates:
        parts = re.split(r"(\{[A-Za-z_][A-Za-z0-9_]*\})", item.uri_template)
        if len(parts) == 1 or any("{" in part or "}" in part for part in parts[::2]):
            continue
        pattern = "".join(
            r"[^/\s?#{}]+" if index % 2 else re.escape(part)
            for index, part in enumerate(parts)
        )
        if re.fullmatch(pattern, uri):
            return uri
    raise ValueError("Resource URI does not match discovered Resources or Resource Templates.")


def resource_result_value(result: types.ReadResourceResult) -> list[dict[str, object]]:
    """Preserve each text item and its identity in JSON-native continuation data."""
    contents = []
    for item in result.contents:
        if not isinstance(item, types.TextResourceContents):
            raise ValueError("Resource bridge supports text Resource contents only.")
        contents.append({"uri": str(item.uri), "mime_type": item.mime_type, "text": item.text})
    return contents
