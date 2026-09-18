"""Pure conversion tests isolate MCP message fidelity from provider execution."""

import json

import pytest
from mcp import types

from host.prompt_conversion import prompt_messages_to_llm, PromptConversionError


def test_text_roles_order_and_sdk_isolation():
    """Preserve each role and literal text without leaking SDK model objects."""
    messages = [types.PromptMessage(role=role, content=types.TextContent(type="text", text=text))
                for role, text in [("user", "First"), ("assistant", "Second"), ("user", "Third")]]
    converted = prompt_messages_to_llm(messages)
    assert converted == [{"role": m.role, "content": m.content.text} for m in messages]
    assert json.loads(json.dumps(converted)) == converted


def test_unsupported_content_is_rejected():
    """Image instructions cannot silently disappear from a text-only workflow."""
    message = types.PromptMessage(role="user", content=types.ImageContent(type="image", data="AA==", mimeType="image/png"))
    with pytest.raises(PromptConversionError, match="Only text"):
        prompt_messages_to_llm([message])


def test_empty_prompt_rejected():
    """No instructions is an explicit failure, not a blank provider request."""
    with pytest.raises(PromptConversionError, match="no messages"):
        prompt_messages_to_llm([])
