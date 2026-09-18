"""Host-owned, bounded coordination between LLM selection and MCP execution.

Discovery supplies schemas; explicit Host policy supplies permission. Neither
the provider adapter nor the MCP transport decides which operations may run.
"""

import json
from collections.abc import Awaitable, Callable, Iterable
from copy import deepcopy
from dataclasses import dataclass

from mcp import types

from host.capabilities import CapabilityCatalog
from host.llm import LLMClient, LLMMessage
from host.mcp_client import JobPilotMCPClient
from host.tool_conversion import tool_capability_to_llm_tool
from host.resource_bridge import (
    RESOURCE_BRIDGE_NAME, resource_bridge_definition,
    validate_resource_arguments, resource_result_value,
)


# This is an explicit application policy, not a classification inferred from names.
AUTO_EXECUTABLE_TOOLS = frozenset({"search_jobs", "score_job_match"})
CONFIRMATION_REQUIRED_TOOLS = frozenset({"save_application"})
ApprovalHandler = Callable[[str, dict[str, object]], Awaitable[bool]]


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
        confirmation_required_tools: Iterable[str] = (),
        approval_handler: ApprovalHandler | None = None,
    ) -> None:
        if type(max_tool_rounds) is not int or max_tool_rounds < 1:
            raise ValueError("max_tool_rounds must be a positive integer.")
        self._max_tool_rounds = max_tool_rounds
        self._llm = llm_client
        self._mcp = mcp_client
        self._allowed = frozenset(allowed_tools)
        self._confirmation_required = frozenset(confirmation_required_tools)
        if self._allowed & self._confirmation_required:
            raise ValueError("Automatic and confirmation-required Tool policies must not overlap.")
        self._requestable = self._allowed | self._confirmation_required
        self._approval_handler = approval_handler
        self._discovered = frozenset(tool.name for tool in catalog.tools)
        self._catalog = catalog
        if RESOURCE_BRIDGE_NAME in self._discovered:
            raise OrchestrationError("MCP Tool name conflicts with the Host Resource bridge.")
        self._tools = tuple(
            tool_capability_to_llm_tool(tool)
            for tool in catalog.tools if tool.name in self._requestable
        ) + (resource_bridge_definition(catalog),)

    async def ask(self, user_message: str) -> OrchestrationResult:
        """Keep complete history while bounding execution and rejecting repeats."""
        if not user_message.strip():
            raise OrchestrationError("A non-empty user request is required.")
        return await self.run([{"role": "user", "content": user_message}])

    async def run(self, messages: list[LLMMessage]) -> OrchestrationResult:
        """Start the same bounded engine from Host-converted Prompt or user history.

        Copy caller-owned history so appending execution evidence never changes
        reusable Prompt messages or a caller's conversation snapshot.
        """
        if not messages:
            raise OrchestrationError("A non-empty message history is required.")
        messages = deepcopy(messages)
        ids: set[str] = set()
        executed_keys: set[tuple[str, str]] = set()
        declined_keys: set[tuple[str, str]] = set()
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
                if call.name == RESOURCE_BRIDGE_NAME:
                    try:
                        validate_resource_arguments(call.arguments, self._catalog)
                    except ValueError as error:
                        raise OrchestrationError(str(error)) from error
                else:
                    if call.name not in self._discovered:
                        raise OrchestrationError(f"Model requested unknown Tool: {call.name}")
                    if call.name not in self._requestable:
                        raise OrchestrationError(f"Tool disallowed by automatic policy: {call.name}")
                if not isinstance(call.id, str) or not call.id.strip() or call.id in ids:
                    raise OrchestrationError("Model returned missing or duplicate Tool-call IDs.")
                ids.add(call.id)
                key = (call.name, json.dumps(
                    call.arguments, sort_keys=True, separators=(",", ":"), allow_nan=False,
                ))
                if key in executed_keys:
                    raise OrchestrationError(f"Identical Tool call repeated: {call.name}.")
                if key in declined_keys:
                    raise OrchestrationError(f"Identical declined Tool request repeated: {call.name}.")
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
                # Recheck within the batch too: an earlier call may have just
                # executed or been declined. Approval is never remembered as consent.
                if key in declined_keys:
                    raise OrchestrationError(f"Identical declined Tool request repeated: {call.name}.")
                if key in executed_keys:
                    raise OrchestrationError(f"Identical Tool call repeated: {call.name}.")
                if call.name in self._confirmation_required:
                    if self._approval_handler is None:
                        raise OrchestrationError("No approval handler configured for state-changing action.")
                    try:
                        # Isolate the UI's copy so it cannot change the approved
                        # action's actual arguments before MCP dispatch.
                        approved = await self._approval_handler(call.name, deepcopy(call.arguments))
                    except Exception as error:
                        raise OrchestrationError(f"Approval failed ({type(error).__name__}).") from error
                    if approved is not True:
                        declined_keys.add(key)
                        messages.append({
                            "role": "tool", "tool_call_id": call.id,
                            "content": json.dumps({
                                "approved": False, "executed": False,
                                "reason": "User declined the requested action.",
                            }),
                        })
                        continue
                try:
                    # Host bridge dispatch uses resources/read, never tools/call.
                    if call.name == RESOURCE_BRIDGE_NAME:
                        raw = await self._mcp.read_resource(call.arguments["uri"])
                    else:
                        raw = await self._mcp.call_tool(call.name, deepcopy(call.arguments))
                except Exception as error:
                    # Cancellation still propagates; ordinary protocol/transport
                    # failures become safe Host errors rather than fake success.
                    raise OrchestrationError(
                        f"MCP execution failed for {call.name} ({type(error).__name__})."
                    ) from error
                if call.name == RESOURCE_BRIDGE_NAME:
                    if not isinstance(raw, types.ReadResourceResult):
                        raise OrchestrationError("MCP returned an unsupported Resource result.")
                    try:
                        result = resource_result_value(raw)
                    except ValueError as error:
                        raise OrchestrationError(str(error)) from error
                    # MCP Resource failures arrive as protocol exceptions, not
                    # the is_error flag used by a successfully delivered Tool result.
                    is_error = False
                else:
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
