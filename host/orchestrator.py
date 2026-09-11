"""Host-owned, bounded coordination between LLM selection and MCP execution.

Discovery supplies schemas; explicit Host policy supplies permission. Neither
the provider adapter nor the MCP transport decides which operations may run.
"""

import json
from collections.abc import Iterable
from copy import deepcopy
from dataclasses import dataclass

from mcp import types

from host.capabilities import CapabilityCatalog
from host.llm import LLMClient, LLMMessage
from host.mcp_client import JobPilotMCPClient
from host.tool_conversion import tool_capability_to_llm_tool


# This is an explicit application policy, not a classification inferred from names.
AUTO_EXECUTABLE_TOOLS = frozenset({"search_jobs", "score_job_match"})


class OrchestrationError(RuntimeError):
    """A policy, protocol, or unsupported model state prevented completion."""


@dataclass(frozen=True)
class ExecutedToolCall:
    """Host trace containing JSON-native results rather than MCP SDK objects."""

    id: str
    name: str
    arguments: dict[str, object]
    result: object
    is_error: bool


@dataclass(frozen=True)
class OrchestrationResult:
    """Final model content and ordered evidence across all execution rounds."""

    answer: str | None
    executed_tool_calls: tuple[ExecutedToolCall, ...]


class JobPilotOrchestrator:
    """Reuse a connected MCP client within a finite Tool-round budget."""

    def __init__(
        self,
        llm_client: LLMClient,
        mcp_client: JobPilotMCPClient,
        catalog: CapabilityCatalog,
        *,
        allowed_tools: Iterable[str],
        max_tool_rounds: int = 3,
    ) -> None:
        if type(max_tool_rounds) is not int or max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be a positive integer.")
        self._max_tool_rounds = max_tool_rounds
        self._llm = llm_client
        self._mcp = mcp_client
        self._allowed = frozenset(allowed_tools)
        self._discovered = frozenset(tool.name for tool in catalog.tools)
        self._tools = tuple(
            tool_capability_to_llm_tool(tool)
            for tool in catalog.tools if tool.name in self._allowed
        )

    async def ask(self, user_message: str) -> OrchestrationResult:
        """Keep complete history while bounding execution and rejecting repeats."""
        if not user_message.strip():
            raise OrchestrationError("A non-empty user request is required.")
        messages: list[LLMMessage] = [{"role": "user", "content": user_message}]
        ids: set[str] = set()
        executed_keys: set[tuple[str, str]] = set()
        executed: list[ExecutedToolCall] = []

        # The extra completion can return the final answer after the last allowed
        # batch. A Tool request there is rejected before any further MCP operation.
        for rounds_used in range(self._max_tool_rounds + 1):
            response = await self._complete(
                messages, tools=self._tools, tool_choice="auto",
            )
            if not response.tool_calls:
                return OrchestrationResult(response.content, tuple(executed))
            if rounds_used == self._max_tool_rounds:
                raise OrchestrationError(
                    f"Maximum Tool rounds exceeded ({self._max_tool_rounds})."
                )

            # Validate the whole batch before execution. IDs remain unique across
            # history; repetition compares exact JSON values, not semantic intent.
            batch_keys = []
            for call in response.tool_calls:
                if call.name not in self._discovered:
                    raise OrchestrationError(f"Model requested unknown Tool: {call.name}")
                if call.name not in self._allowed:
                    raise OrchestrationError(f"Tool disallowed by automatic policy: {call.name}")
                if not isinstance(call.id, str) or not call.id.strip() or call.id in ids:
                    raise OrchestrationError("Model returned missing or duplicate Tool-call IDs.")
                ids.add(call.id)
                key = (call.name, json.dumps(
                    call.arguments, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ))
                if key in executed_keys:
                    raise OrchestrationError(f"Identical Tool call repeated: {call.name}.")
                batch_keys.append(key)

            messages.append({
                "role": "assistant",
                "content": response.content,
                "tool_calls": [
                    {"id": call.id, "type": "function", "function": {
                        "name": call.name,
                        "arguments": json.dumps(call.arguments),
                    }} for call in response.tool_calls
                ],
            })
            for call, key in zip(response.tool_calls, batch_keys):
                try:
                    raw = await self._mcp.call_tool(call.name, deepcopy(call.arguments))
                except Exception as error:
                    # Cancellation still propagates; ordinary protocol/transport
                    # failures become safe Host errors rather than fake success.
                    raise OrchestrationError(
                        f"MCP execution failed for {call.name} ({type(error).__name__})."
                    ) from error
                if not isinstance(raw, types.CallToolResult):
                    raise OrchestrationError("MCP returned an unsupported Tool result.")
                result = _tool_result_value(raw)
                is_error = raw.is_error is True
                executed.append(ExecutedToolCall(
                    call.id, call.name, deepcopy(call.arguments), result, is_error,
                ))
                executed_keys.add(key)
                messages.append({
                    "role": "tool",
                    "tool_call_id": call.id,
                    "content": json.dumps({"is_error": is_error, "result": result}),
                })

    async def _complete(self, messages, **kwargs):
        """Translate provider failures into a recoverable, credential-safe Host error."""
        try:
            return await self._llm.complete(messages, **kwargs)
        except Exception as error:
            raise OrchestrationError(
                f"LLM completion failed ({type(error).__name__})."
            ) from error


def _tool_result_value(result: types.CallToolResult) -> object:
    """Prefer structured data; represent unsupported non-text blocks explicitly."""
    if result.structured_content is not None:
        return deepcopy(result.structured_content)
    return "\n".join(
        block.text if isinstance(block, types.TextContent)
        else f"[Unsupported MCP content: {block.type}]"
        for block in result.content
    )
