"""NVIDIA NIM implementation of JobPilot's provider-independent LLM boundary.

The hosted NVIDIA endpoint is OpenAI-compatible, so this adapter contains the
OpenAI SDK dependency and normalizes its responses before they reach Host code.
It never executes Tool calls returned by the model.
"""

import json
import os
from collections.abc import Callable, Sequence
from copy import deepcopy
from typing import Any

from openai import AsyncOpenAI

from host.llm import (
    LLMMessage,
    LLMResponse,
    LLMToolCall,
    LLMToolChoice,
    LLMToolDefinition,
)


NVIDIA_API_KEY_ENVIRONMENT_VARIABLE = "NVIDIA_API_KEY"
DEFAULT_NVIDIA_BASE_URL = "https://integrate.api.nvidia.com/v1"
DEFAULT_NVIDIA_MODEL = "nvidia/nemotron-3.5-lightning-30b-a3b"


class NVIDIAConfigurationError(RuntimeError):
    """Raised when the NVIDIA adapter cannot be configured safely."""


class NVIDIAResponseError(RuntimeError):
    """Raised when a provider response lacks the expected completion structure."""


class MalformedToolArgumentsError(ValueError):
    """Raised when model-requested Tool arguments are not a valid JSON object."""


class NVIDIALLMClient:
    """Send non-streaming NVIDIA chat completions and return Host-owned results."""

    def __init__(
        self,
        *,
        api_key: str | None = None,
        base_url: str = DEFAULT_NVIDIA_BASE_URL,
        model: str = DEFAULT_NVIDIA_MODEL,
        client: Any | None = None,
        client_factory: Callable[..., Any] | None = None,
    ) -> None:
        """Configure NVIDIA defaults while supporting secret-free test injection."""
        self._base_url = base_url
        self._model = model

        if client is not None:
            # Tests may inject a complete SDK-compatible client without requiring
            # credentials. Production construction follows the branch below.
            self._client = client
            return

        resolved_api_key = (
            api_key
            if api_key is not None
            else os.environ.get(NVIDIA_API_KEY_ENVIRONMENT_VARIABLE)
        )
        if not resolved_api_key:
            raise NVIDIAConfigurationError(
                "NVIDIA_API_KEY is required to construct the NVIDIA LLM client."
            )

        factory = client_factory or AsyncOpenAI
        self._client = factory(
            api_key=resolved_api_key,
            base_url=base_url,
            # Do not add an adapter retry policy; failures remain visible to the
            # future Host orchestration layer that will own operational policy.
            max_retries=0,
        )

    @property
    def model(self) -> str:
        """Expose the configured public model identifier without credential data."""
        return self._model

    @property
    def base_url(self) -> str:
        """Expose the configured public endpoint for diagnostics and testing."""
        return self._base_url

    def __repr__(self) -> str:
        """Return safe public configuration while never retaining the API key."""
        return (
            f"NVIDIALLMClient(model={self._model!r}, "
            f"base_url={self._base_url!r})"
        )

    async def complete(
        self,
        messages: Sequence[LLMMessage],
        tools: Sequence[LLMToolDefinition] | None = None,
        tool_choice: LLMToolChoice = "auto",
    ) -> LLMResponse:
        """Request one completion and normalize text or Tool selections."""
        request: dict[str, Any] = {
            "model": self._model,
            "messages": [deepcopy(message) for message in messages],
            "temperature": 0.0,
            "stream": False,
            # NVIDIA documents this chat-template option for disabling extended
            # reasoning when concise structured/tool output is desired.
            "extra_body": {
                "chat_template_kwargs": {"enable_thinking": False}
            },
        }
        if tools:
            request["tools"] = deepcopy(list(tools))
            request["tool_choice"] = deepcopy(tool_choice)

        completion = await self._client.chat.completions.create(**request)
        return _normalize_completion(completion)


def _normalize_completion(completion: Any) -> LLMResponse:
    """Convert one provider completion without leaking its SDK model classes."""
    choices = getattr(completion, "choices", None)
    if not choices:
        raise NVIDIAResponseError("NVIDIA response did not contain a choice.")

    message = getattr(choices[0], "message", None)
    if message is None:
        raise NVIDIAResponseError("NVIDIA response did not contain a message.")

    content = getattr(message, "content", None)
    if content is not None and not isinstance(content, str):
        raise NVIDIAResponseError("NVIDIA response content was not text.")

    normalized_calls = tuple(
        _normalize_tool_call(tool_call)
        for tool_call in (getattr(message, "tool_calls", None) or [])
    )
    return LLMResponse(content=content, tool_calls=normalized_calls)


def _normalize_tool_call(tool_call: Any) -> LLMToolCall:
    """Parse one OpenAI-compatible function call into a Host-owned value."""
    call_id = getattr(tool_call, "id", None)
    function = getattr(tool_call, "function", None)
    name = getattr(function, "name", None)
    raw_arguments = getattr(function, "arguments", None)
    if not isinstance(call_id, str) or not isinstance(name, str):
        raise NVIDIAResponseError(
            "NVIDIA Tool call did not contain a valid ID and function name."
        )
    if not isinstance(raw_arguments, str):
        raise MalformedToolArgumentsError(
            f"Tool call {call_id} arguments were not JSON text."
        )

    try:
        arguments = json.loads(raw_arguments)
    except json.JSONDecodeError as error:
        raise MalformedToolArgumentsError(
            f"Tool call {call_id} arguments were malformed JSON."
        ) from error
    if not isinstance(arguments, dict):
        raise MalformedToolArgumentsError(
            f"Tool call {call_id} arguments must decode to an object."
        )

    return LLMToolCall(id=call_id, name=name, arguments=arguments)
