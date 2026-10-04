"""Safe A4 provider diagnostics and exact request-shape regressions, no network."""

import asyncio
import json
from types import SimpleNamespace

import httpx2 as httpx
import pytest
from openai import InternalServerError

from host.external_job import ExternalJobValidationError
from host.external_job_normalizer import ExternalJobNormalizer
from host.nvidia_llm import NVIDIALLMClient, NVIDIAResponseError, safe_provider_error
from tests.test_external_job_normalization import SOURCE, EXTRACTION
from tests.test_host_nvidia_llm import FakeOpenAIClient, _completion


def test_sdk_internal_server_error_is_before_host_parsing_and_safe():
    """A real SDK HTTP 500 error survives unchanged; only vetted metadata is logged."""
    private = "test-only-private-source-and-credential"
    response = httpx.Response(500, request=httpx.Request("POST", "https://example.test/v1"))
    failure = InternalServerError(private, response=response, body={
        "error": {"type": "server_error", "code": "internal_server_error", "message": private},
        "authorization": private,
    })

    async def create(**request):
        raise failure

    sdk = SimpleNamespace(chat=SimpleNamespace(completions=SimpleNamespace(create=create)))
    events = []
    client = NVIDIALLMClient(client=sdk, diagnostic_handler=events.append)
    with pytest.raises(InternalServerError) as caught:
        asyncio.run(client.complete([{"role": "user", "content": private}]))
    assert caught.value is failure
    assert [event["stage"] for event in events] == [
        "nvidia.chat_completions.request", "nvidia.chat_completions.failed"]
    assert events[-1]["http_status"] == 500
    assert events[-1]["exception_class"] == "InternalServerError"
    assert events[-1]["provider_error_type"] == "server_error"
    assert events[-1]["provider_error_code"] == "internal_server_error"
    assert events[-1]["host_response_parsing_started"] is False
    assert private not in json.dumps(events)


def test_arbitrary_provider_metadata_is_redacted():
    """Error code/type fields are untrusted too; they may echo request material."""
    error = RuntimeError("private full payload")
    error.body = {"error": {"type": "private full payload", "code": "Bearer test-secret"}}
    metadata = safe_provider_error(error)
    assert metadata["provider_error_type"] == "[unrecognized metadata redacted]"
    assert metadata["provider_error_code"] == "[unrecognized metadata redacted]"
    assert "private full payload" not in json.dumps(metadata)
    assert "test-secret" not in json.dumps(metadata)


def test_a4_request_is_plain_completion_not_structured_output():
    """Exercise normalizer -> real adapter -> fake SDK, not a guessed request shape."""
    sdk = FakeOpenAIClient(_completion(content=json.dumps(EXTRACTION)))
    events = []
    client = NVIDIALLMClient(client=sdk, diagnostic_handler=events.append)
    job = asyncio.run(ExternalJobNormalizer(client).normalize(
        SOURCE, company_slug="example-labs", job_slug="ml-engineer-123"))
    request = sdk.completions.requests[0]
    assert set(request) == {"model", "messages", "temperature", "stream", "extra_body"}
    assert request["model"] == client.model
    assert [m["role"] for m in request["messages"]] == ["system", "user"]
    assert request["messages"][1]["content"] == SOURCE
    assert request["temperature"] == 0.0 and request["stream"] is False
    assert request["extra_body"] == {"chat_template_kwargs": {"enable_thinking": False}}
    for field in ("tools", "tool_choice", "response_format", "json_schema", "max_tokens"):
        assert field not in request
    assert events[0]["response_format_requested"] is False
    assert events[0]["json_mode_requested"] is False
    assert events[0]["json_schema_requested"] is False
    assert events[-1]["stage"] == "nvidia.response_parsing.completed"
    assert SOURCE not in json.dumps(events)
    assert job.raw_source_text == SOURCE


def test_invalid_sdk_response_is_distinguished_from_http_failure():
    """HTTP completion succeeded but the Host could not normalize its response shape."""
    events = []
    client = NVIDIALLMClient(client=FakeOpenAIClient(SimpleNamespace(choices=[])),
                           diagnostic_handler=events.append)
    with pytest.raises(NVIDIAResponseError):
        asyncio.run(client.complete([{"role": "user", "content": "test"}]))
    assert events[-1]["stage"] == "nvidia.response_parsing.failed"
    assert events[-1]["host_response_parsing_started"] is True
    assert events[-1]["http_status"] is None


@pytest.mark.parametrize("content", ["", "not JSON", None])
def test_bad_extraction_is_after_successful_adapter_parsing(content):
    """Malformed/empty model text remains a strict A4 validation failure, not HTTP 500."""
    events = []
    client = NVIDIALLMClient(client=FakeOpenAIClient(_completion(content=content)),
                           diagnostic_handler=events.append)
    with pytest.raises(ExternalJobValidationError):
        asyncio.run(ExternalJobNormalizer(client).normalize(
            SOURCE, company_slug="example-labs", job_slug="ml-engineer-123"))
    assert events[-1]["stage"] == "nvidia.response_parsing.completed"
