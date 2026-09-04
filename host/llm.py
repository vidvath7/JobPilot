"""Provider-independent LLM boundary models for the JobPilot Host.

Future orchestration can depend on these small values without importing NVIDIA
or OpenAI SDK response classes. This module defines no execution loop.
"""

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Protocol


LLMMessage = dict[str, object]
LLMToolDefinition = dict[str, object]
LLMToolChoice = str | dict[str, object]


@dataclass(frozen=True)
class LLMToolCall:
    """Provider-neutral representation of one model-requested function call."""

    id: str
    name: str
    arguments: dict[str, object]


@dataclass(frozen=True)
class LLMResponse:
    """Provider-neutral assistant text and zero or more requested Tool calls."""

    content: str | None
    tool_calls: tuple[LLMToolCall, ...]


class LLMClient(Protocol):
    """Minimal async chat-completion boundary required by the future Host loop."""

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[LLMToolDefinition] | None = None,
        tool_choice: LLMToolChoice = "auto",
    ) -> LLMResponse:
        """Return normalized model output without executing requested Tools."""
