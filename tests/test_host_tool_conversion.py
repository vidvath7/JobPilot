"""Unit tests for pure MCP Tool metadata to LLM schema conversion."""

from host.capabilities import CapabilityCatalog, ToolCapability
from host.tool_conversion import (
    catalog_tools_to_llm_tools,
    tool_capability_to_llm_tool,
)


def test_converts_tool_name_description_and_schema_exactly() -> None:
    """Preserve protocol metadata without adding aliases or interpretation."""
    schema = {
        "type": "object",
        "properties": {
            "job_id": {"type": "string"},
            "status": {"type": "string", "default": "applied"},
            "notes": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        },
        "required": ["job_id"],
    }
    capability = ToolCapability(
        name="save_application",
        description="Persist an application.",
        input_schema=schema,
    )

    converted = tool_capability_to_llm_tool(capability)

    assert converted == {
        "type": "function",
        "function": {
            "name": "save_application",
            "description": "Persist an application.",
            "parameters": schema,
        },
    }
    parameters = converted["function"]["parameters"]
    assert parameters["required"] == ["job_id"]
    assert set(parameters["properties"]) == {"job_id", "status", "notes"}
    assert "status" not in parameters["required"]
    assert "notes" not in parameters["required"]


def test_catalog_conversion_uses_only_tools_and_does_not_mutate_source() -> None:
    """Convert multiple Tools while leaving catalog schemas independently owned."""
    first_schema = {
        "type": "object",
        "properties": {"role": {"type": "string"}},
    }
    second_schema = {
        "type": "object",
        "properties": {"job_id": {"type": "string"}},
        "required": ["job_id"],
    }
    catalog = CapabilityCatalog(
        tools=(
            ToolCapability("search_jobs", None, first_schema),
            ToolCapability("score_job_match", "Score a job.", second_schema),
        ),
        resources=(),
        resource_templates=(),
        prompts=(),
    )

    converted = catalog_tools_to_llm_tools(catalog)

    assert len(converted) == 2
    assert [tool["function"]["name"] for tool in converted] == [
        "search_jobs",
        "score_job_match",
    ]
    assert converted[0]["function"]["description"] is None
    assert converted[1]["function"]["parameters"] == second_schema

    # Mutating provider-facing data must not change the source catalog.
    converted[1]["function"]["parameters"]["required"].append("extra")
    assert catalog.tools[1].input_schema["required"] == ["job_id"]
