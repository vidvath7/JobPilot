"""Offline tests for NVIDIA request construction and response normalization.

The injected client mimics only the OpenAI-compatible surface used by the
adapter, preventing automated tests from reading credentials or making network
requests.
"""

import asyncio
from types import SimpleNamespace

import pytest

from host.llm import LLMResponse, LLMToolCall
from host.nvidia_llm import (
    DEFAULT_NVIDIA_BASE_URL,
    DEFAULT_NVIDIA_MODEL,
    MalformedToolArgumentsError,
    NVIDIAConfigurationError,
    NVIDIALLMClient,
)


class FakeCompletions:
    """Capture Chat Completions requests and return one controlled response."""

    def __init__(self, response: object) -> None:
        self.response = response
        self.requests: list[dict[str, object]] = []

    async def create(self, **request: object) -> object:
        self.requests.append(request)
        return self.response


class FakeOpenAIClient:
    """Expose the nested SDK attribute path consumed by NVIDIALLMClient."""

    def __init__(self, response: object) -> None:
        self.completions = FakeCompletions(response)
        self.chat = SimpleNamespace(completions=self.completions)


def test_default_configuration_and_request_are_nvidia_compatible(monkeypatch) -> None:
    """Use environment credentials without retaining or displaying their value."""
    secret = "test-only-secret-value"
    monkeypatch.setenv("NVIDIA_API_KEY", secret)
    captured_configuration: dict[str, object] = {}
    fake_sdk_client = FakeOpenAIClient(_completion(content="Hello"))

    def client_factory(**configuration: object) -> FakeOpenAIClient:
        captured_configuration.update(configuration)
        return fake_sdk_client

    client = NVIDIALLMClient(client_factory=client_factory)
    response = asyncio.run(
        client.complete(
            messages=[{"role": "user", "content": "Find matching jobs."}],
            tools=[_tool_definition()],
        )
    )

    assert captured_configuration == {
        "api_key": secret,
        "base_url": DEFAULT_NVIDIA_BASE_URL,
        "max_retries": 0,
    }
    assert client.model == DEFAULT_NVIDIA_MODEL
    assert client.base_url == DEFAULT_NVIDIA_BASE_URL
    assert secret not in repr(client)
    assert response == LLMResponse(content="Hello", tool_calls=())

    request = fake_sdk_client.completions.requests[0]
    assert request["model"] == DEFAULT_NVIDIA_MODEL
    assert request["messages"] == [
        {"role": "user", "content": "Find matching jobs."}
    ]
    assert request["tools"] == [_tool_definition()]
    assert request["tool_choice"] == "auto"
    assert request["temperature"] == 0.0
    assert request["stream"] is False
    assert request["extra_body"] == {
        "chat_template_kwargs": {"enable_thinking": False}
    }


def test_missing_api_key_raises_safe_configuration_error(monkeypatch) -> None:
    """Fail before SDK construction without exposing any credential material."""
    monkeypatch.delenv("NVIDIA_API_KEY", raising=False)

    with pytest.raises(NVIDIAConfigurationError) as error:
        NVIDIALLMClient()

    assert "NVIDIA_API_KEY" in str(error.value)
    assert "secret" not in str(error.value).casefold()


def test_normal_text_response_has_no_tool_calls() -> None:
    """Normalize assistant text into provider-independent Host output."""
    provider_response = _completion(content="I can help find jobs.")
    fake_sdk_client = FakeOpenAIClient(provider_response)
    client = NVIDIALLMClient(client=fake_sdk_client)

    response = asyncio.run(client.complete([{"role": "user", "content": "Hi"}]))

    assert response == LLMResponse(
        content="I can help find jobs.",
        tool_calls=(),
    )
    assert not isinstance(response, type(provider_response))
    request = fake_sdk_client.completions.requests[0]
    assert "tools" not in request
    assert "tool_choice" not in request


def test_single_tool_call_parses_json_object_arguments() -> None:
    """Preserve call identity and function name while decoding arguments."""
    client = NVIDIALLMClient(
        client=FakeOpenAIClient(
            _completion(
                content=None,
                tool_calls=[
                    _tool_call(
                        "call-1",
                        "search_jobs",
                        '{"role": "AI Engineer"}',
                    )
                ],
            )
        )
    )

    response = asyncio.run(client.complete([{"role": "user", "content": "Find"}]))

    assert response == LLMResponse(
        content=None,
        tool_calls=(
            LLMToolCall(
                id="call-1",
                name="search_jobs",
                arguments={"role": "AI Engineer"},
            ),
        ),
    )


def test_multiple_tool_calls_normalize_in_provider_order() -> None:
    """Represent every returned Tool request without executing any of them."""
    provider_response = _completion(
        content=None,
        tool_calls=[
            _tool_call("call-1", "search_jobs", "{}"),
            _tool_call("call-2", "score_job_match", '{"job_id":"JOB-005"}'),
        ],
    )
    client = NVIDIALLMClient(client=FakeOpenAIClient(provider_response))

    response = asyncio.run(client.complete([{"role": "user", "content": "Find"}]))

    assert [call.id for call in response.tool_calls] == ["call-1", "call-2"]
    assert [call.name for call in response.tool_calls] == [
        "search_jobs",
        "score_job_match",
    ]
    assert response.tool_calls[1].arguments == {"job_id": "JOB-005"}


@pytest.mark.parametrize("arguments", ["{malformed}", "[1, 2]"])
def test_malformed_or_non_object_tool_arguments_raise(arguments: str) -> None:
    """Reject unusable model arguments instead of inventing an empty object."""
    provider_response = _completion(
        content=None,
        tool_calls=[_tool_call("call-1", "search_jobs", arguments)],
    )
    client = NVIDIALLMClient(client=FakeOpenAIClient(provider_response))

    with pytest.raises(MalformedToolArgumentsError):
        asyncio.run(client.complete([{"role": "user", "content": "Find"}]))


def _completion(
    *,
    content: str | None,
    tool_calls: list[object] | None = None,
) -> object:
    """Build the provider response shape used by the adapter, without SDK classes."""
    message = SimpleNamespace(content=content, tool_calls=tool_calls)
    return SimpleNamespace(choices=[SimpleNamespace(message=message)])


def _tool_call(call_id: str, name: str, arguments: str) -> object:
    """Build one OpenAI-compatible Tool-call response object."""
    return SimpleNamespace(
        id=call_id,
        function=SimpleNamespace(name=name, arguments=arguments),
    )


def _tool_definition() -> dict[str, object]:
    """Return a minimal OpenAI-compatible function definition for request tests."""
    return {
        "type": "function",
        "function": {
            "name": "search_jobs",
            "description": "Search jobs.",
            "parameters": {"type": "object", "properties": {}},
        },
    }
