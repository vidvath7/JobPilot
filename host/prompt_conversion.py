"""Convert MCP instructions at the Host boundary without executing capabilities."""

from collections.abc import Sequence

from mcp import types

from host.llm import LLMMessage


class PromptConversionError(ValueError):
    """The Host cannot represent the returned Prompt content faithfully."""


def prompt_messages_to_llm(messages: Sequence[types.PromptMessage]) -> list[LLMMessage]:
    """Preserve text, roles, and ordering; fail rather than discard other content."""
    if not messages:
        raise PromptConversionError("Prompt returned no messages.")
    converted = []
    for message in messages:
        if not isinstance(message.content, types.TextContent):
            raise PromptConversionError("Only text Prompt content is supported.")
        converted.append({"role": message.role, "content": message.content.text})
    return converted
