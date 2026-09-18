"""Pure Host bridge tests: discovery metadata, URI boundaries, and text conversion."""

import json

import pytest
from mcp import types

from host.capabilities import CapabilityCatalog, ResourceCapability, ResourceTemplateCapability
from host.resource_bridge import resource_bridge_definition, validate_resource_arguments, resource_result_value


@pytest.fixture
def catalog():
    """Synthetic discovery proves the bridge does not hardcode JobPilot Resources."""
    return CapabilityCatalog(
        tools=(), prompts=(),
        resources=(ResourceCapability("synthetic://profile", "profile", "Synthetic context", "application/json"),),
        resource_templates=(ResourceTemplateCapability("synthetic://items/{item_id}", "item", "Synthetic details", "application/json"),),
    )


def test_dynamic_bridge_definition(catalog):
    """The Host function is separate from catalog Tools and preserves discovered descriptions."""
    function = resource_bridge_definition(catalog)["function"]
    assert function["name"] == "read_mcp_resource"
    assert function["parameters"] == {
        "type": "object", "properties": {"uri": {
            "type": "string", "description": "The concrete discovered MCP Resource URI.",
        }}, "required": ["uri"], "additionalProperties": False,
    }
    for text in ("read-only", "synthetic://profile", "Synthetic context", "synthetic://items/{item_id}", "Synthetic details"):
        assert text in function["description"]
    assert catalog.tools == ()


@pytest.mark.parametrize("uri", ["synthetic://profile", "synthetic://items/123"])
def test_discovered_uri_accepted(catalog, uri):
    """Static identity and concrete simple template values are permitted."""
    assert validate_resource_arguments({"uri": uri}, catalog) == uri


@pytest.mark.parametrize("arguments", [
    {}, {"uri": 4}, {"uri": "synthetic://profile", "extra": True},
    {"uri": "https://unrelated.example"}, {"uri": "synthetic://items/"},
    {"uri": "synthetic://items/123/extra"}, {"uri": "synthetic://items/123?extra"},
    {"uri": "synthetic://items/{item_id}"}, {"uri": "synthetic://profile/extra"},
])
def test_invalid_resource_arguments_rejected(catalog, arguments):
    """Full matching rejects missing values, wrong shapes, and unrelated paths."""
    with pytest.raises(ValueError):
        validate_resource_arguments(arguments, catalog)


def test_multiple_resource_contents_preserve_identity_and_text():
    """Multiple SDK content items become an ordered, JSON-serializable structure."""
    result = types.ReadResourceResult(contents=[
        types.TextResourceContents(uri=f"synthetic://item/{i}", text=f"text {i}")
        for i in range(2)
    ])
    converted = resource_result_value(result)
    assert [item["text"] for item in converted] == ["text 0", "text 1"]
    assert [item["uri"] for item in converted] == ["synthetic://item/0", "synthetic://item/1"]
    json.dumps(converted)
