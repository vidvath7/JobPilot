"""Host-owned, single-round coordination between LLM selection and MCP execution.

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
    """Final model content and ordered evidence of the one execution round."""

    answer: str | None
    executed_tool_calls: tuple[ExecutedToolCall, ...]


class JobPilotOrchestrator:
    """Use an already connected MCP client for at most one round per request."""

    def __init__(
        self,
        llm_client: LLMClient,
        mcp_client: JobPilotMCPClient,
        catalog: CapabilityCatalog,
        *,
        allowed_tools: Iterable[str],
    ) -> None:
        self._llm = llm_client
        self._mcp = mcp_client
        self._allowed = frozenset(allowed_tools)
        self._discovered = frozenset(tool.name for tool in catalog.tools)
        self._tools = tuple(
            tool_capability_to_llm_tool(tool)
            for tool in catalog.tools if tool.name in self._allowed
        )

    async def ask(self, user_message: str) -> OrchestrationResult:
        """Select, validate, execute in order, summarize once, then stop."""
        if not user_message.strip():
            raise OrchestrationError("A non-empty user request is required.")
        messages: list[LLMMessage] = [{"role": "user", "content": user_message}]
        first = await self._complete(messages, tools=self._tools, tool_choice="auto")
        if not first.tool_calls:
            return OrchestrationResult(first.content, ())

        # Validate the entire batch before any execution, including correlation
        # IDs needed to pair each continuation result with its model request.
        ids: set[str] = set()
        for call in first.tool_calls:
            if call.name not in self._discovered:
                raise OrchestrationError(f"Model requested unknown Tool: {call.name}")
            if call.name not in self._allowed:
                raise OrchestrationError(f"Tool disallowed by automatic policy: {call.name}")
            if not call.id or call.id in ids:
                raise OrchestrationError("Model returned missing or duplicate Tool-call IDs.")
            ids.add(call.id)

        messages.append({
            "role": "assistant",
            "content": first.content,
            "tool_calls": [
                {"id": call.id, "type": "function", "function": {
                    "name": call.name,
                    "arguments": json.dumps(call.arguments),
                }} for call in first.tool_calls
            ],
        })
        executed = []
        for call in first.tool_calls:
            try:
                raw = await self._mcp.call_tool(call.name, deepcopy(call.arguments))
            except Exception as error:
                # Keep provider/transport diagnostics out of normal user output.
                # Cancellation inherits BaseException and still propagates.
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
            messages.append({
                "role": "tool",
                "tool_call_id": call.id,
                "content": json.dumps({"is_error": is_error, "result": result}),
            })

        # No Tool definitions on this completion: this is a terminal summary,
        # not a recursive agent loop, even if the provider requests more calls.
        final = await self._complete(messages)
        if final.tool_calls:
            raise OrchestrationError("Second model response requested another Tool round.")
        return OrchestrationResult(final.content, tuple(executed))

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
