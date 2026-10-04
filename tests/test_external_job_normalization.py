"""A4 boundary tests: controlled source text/model output, no NVIDIA or network."""

import asyncio
import json
from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest
from mcp import types

from host.external_job import ExternalJobValidationError, external_job_id, parse_external_job_id
from host.external_job_normalizer import ExternalJobNormalizer, normalize_experience
from host.external_job_workflow import ExternalJobWorkflow
from host.llm import LLMResponse, LLMToolCall


URL = "https://himalayas.app/companies/example-labs/jobs/ml-engineer-123?utm_source=mcp"
SOURCE = f"""# ML Engineer
Company: Example Labs
Location: Germany
Seniority: Mid-level
Skills: Machine Learning, PyTorch, TensorFlow
## Job Description
Build Machine Learning models with PyTorch and TensorFlow.
## Links
Apply on Himalayas: {URL}
"""
EXTRACTION = {
    "title": "ML Engineer", "company": "Example Labs", "location": "Germany",
    "experience_level": "Mid-level", "required_skills": ["Machine Learning", "PyTorch", "TensorFlow"],
    "description": "Build Machine Learning models with PyTorch and TensorFlow.",
}


class FakeLLM:
    """Record the extraction-only request and return provider-independent values."""

    def __init__(self, data=None, response=None):
        self.response = response or LLMResponse(json.dumps(EXTRACTION if data is None else data), ())
        self.requests = []

    async def complete(self, messages, **kwargs):
        self.requests.append((deepcopy(messages), kwargs))
        return self.response


def normalize(llm, source=SOURCE):
    return asyncio.run(ExternalJobNormalizer(llm).normalize(
        source, company_slug="example-labs", job_slug="ml-engineer-123"))


def test_identity_is_deterministic_and_round_trips():
    """External jobs never consume synthetic JOB-* identifiers."""
    identity = external_job_id("example-labs", "ml-engineer-123")
    assert identity == "himalayas::example-labs::ml-engineer-123"
    assert parse_external_job_id(identity) == ("example-labs", "ml-engineer-123")
    assert external_job_id(*parse_external_job_id(identity)) == identity


@pytest.mark.parametrize("value", ["JOB-001", "other::company::job", "himalayas::a::b::c", "himalayas::a/b::c"])
def test_invalid_external_identity_rejected(value):
    """IDs preserve one selected source and two unambiguous URL path segments."""
    with pytest.raises(ExternalJobValidationError):
        parse_external_job_id(value)


def test_valid_extraction_preserves_source_and_only_explicit_skills():
    """Frozen domain data retains provenance; the model receives no execution Tools."""
    llm = FakeLLM()
    job = normalize(llm)
    assert job.source == "himalayas"
    assert job.external_id == external_job_id("example-labs", "ml-engineer-123")
    assert (job.source_company_slug, job.source_job_slug) == ("example-labs", "ml-engineer-123")
    assert (job.title, job.company, job.location) == ("ML Engineer", "Example Labs", "Germany")
    assert job.experience_level == "Mid-level"
    assert job.required_skills == ("Machine Learning", "PyTorch", "TensorFlow")
    assert "Python" not in job.required_skills
    assert job.description == EXTRACTION["description"]
    assert job.url == URL and job.raw_source_text == SOURCE
    assert llm.requests[0][0][1]["content"] == SOURCE
    assert llm.requests[0][1] == {"tools": None, "tool_choice": "none"}
    with pytest.raises(FrozenInstanceError):
        job.title = "changed"


@pytest.mark.parametrize("raw,expected", [
    ("Mid-level", "Mid-level"), (" MID LEVEL ", "Mid-level"), ("Senior", "Senior"),
    ("Entry-level", "Junior"), ("Junior", "Junior"), (None, None),
    ("Senior, Mid-level", None), ("5 years", None), ("Lead", None), ("", None),
])
def test_seniority_mapping_is_small_and_deterministic(raw, expected):
    """Only known single labels map; years and ambiguous levels do not imply rank."""
    assert normalize_experience(raw) == expected


def test_missing_optional_information_has_explicit_empty_policy():
    """Null location/seniority and an empty skills list are not fabricated evidence."""
    data = {**EXTRACTION, "location": None, "experience_level": None, "required_skills": []}
    job = normalize(FakeLLM(data))
    assert job.location is job.experience_level is None
    assert job.required_skills == ()


@pytest.mark.parametrize("patch", [
    {"title": ""}, {"company": None}, {"description": ""}, {"location": 123},
    {"required_skills": "PyTorch"}, {"required_skills": [4]}, {"required_skills": [""]},
    {"required_skills": ["Python"]}, {"title": "Invented title"},
    {"source": "other"}, {"source_company_slug": "attacker"}, {"source_job_slug": "fake"},
    {"external_id": "JOB-001"}, {"url": "https://evil.test/job"},
])
def test_malformed_unsupported_or_model_provenance_fields_rejected(patch):
    """The model cannot expand qualifications or overwrite Host-owned provenance."""
    with pytest.raises(ExternalJobValidationError):
        normalize(FakeLLM({**EXTRACTION, **patch}))


@pytest.mark.parametrize("raw", ["not JSON", "[]", "{}", "```json\n{}\n```", '{"title":"one","title":"two"}'])
def test_strict_json_and_required_field_validation(raw):
    """No silent JSON repair, duplicate-key override, or partial extraction acceptance."""
    with pytest.raises(ExternalJobValidationError):
        normalize(FakeLLM(response=LLMResponse(raw, ())))


def test_unexpected_tool_request_is_rejected():
    """Extraction is not another orchestration loop."""
    response = LLMResponse(None, (LLMToolCall("1", "search_jobs", {}),))
    with pytest.raises(ExternalJobValidationError, match="without Tool calls"):
        normalize(FakeLLM(response=response))


@pytest.mark.parametrize("url", [
    "https://evil.test/companies/example-labs/jobs/ml-engineer-123",
    "https://himalayas.app/companies/other/jobs/ml-engineer-123",
    "https://himalayas.app/companies/example-labs/jobs/other",
])
def test_wrong_source_url_fails_before_llm(url):
    """Authority/path validation binds the retrieved document to the requested job."""
    llm = FakeLLM()
    with pytest.raises(ExternalJobValidationError, match="lacks a URL"):
        normalize(llm, SOURCE.replace(URL, url))
    assert not llm.requests


class FakeManager:
    """Record qualified dispatch so accidental local Tool usage fails assertions."""

    def __init__(self, result):
        self.result, self.calls = result, []

    async def call_tool(self, name, arguments):
        self.calls.append((name, arguments))
        return self.result


def test_workflow_retrieves_only_himalayas_details_and_returns_in_memory():
    """Real normalization follows one fake manager call, without local service access."""
    result = types.CallToolResult(content=[types.TextContent(type="text", text=SOURCE)])
    before = result.model_dump()
    manager = FakeManager(result)
    workflow = ExternalJobWorkflow(manager, ExternalJobNormalizer(FakeLLM()))
    job = asyncio.run(workflow.get_job("example-labs", "ml-engineer-123"))
    assert manager.calls == [("himalayas::get_job_details", {
        "company_slug": "example-labs", "job_slug": "ml-engineer-123"})]
    assert job.raw_source_text == SOURCE
    assert result.model_dump() == before


@pytest.mark.parametrize("error,contents", [(True, [types.TextContent(type="text", text="failure")]), (False, [])])
def test_failed_or_empty_details_never_reach_extractor(error, contents):
    """An MCP error is not authoritative job data even if it contains readable text."""
    llm = FakeLLM()
    manager = FakeManager(types.CallToolResult(is_error=error, content=contents))
    with pytest.raises(ExternalJobValidationError):
        asyncio.run(ExternalJobWorkflow(manager, ExternalJobNormalizer(llm)).get_job("example-labs", "ml-engineer-123"))
    assert not llm.requests
