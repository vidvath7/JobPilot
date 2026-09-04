"""Pure conversion from Host MCP Tool metadata to LLM function definitions.

Only Tools cross this boundary. Resources, Resource Templates, and Prompts have
different MCP semantics and are intentionally excluded from provider Tool input.
"""

from copy import deepcopy

from host.capabilities import CapabilityCatalog, ToolCapability
from host.llm import LLMToolDefinition


def tool_capability_to_llm_tool(tool: ToolCapability) -> LLMToolDefinition:
    """Map one discovered Tool without renaming or interpreting its behavior."""
    return {
        "type": "function",
        "function": {
            "name": tool.name,
            "description": tool.description,
            # Schemas are mutable dictionaries. Copying preserves JSON Schema
            # semantics while isolating the immutable Host catalog from callers.
            "parameters": deepcopy(tool.input_schema),
        },
    }


def catalog_tools_to_llm_tools(
    catalog: CapabilityCatalog,
) -> tuple[LLMToolDefinition, ...]:
    """Convert exactly ``catalog.tools`` and no other MCP capability category."""
    return tuple(tool_capability_to_llm_tool(tool) for tool in catalog.tools)
