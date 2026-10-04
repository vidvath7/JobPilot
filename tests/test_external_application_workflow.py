"""A5 tests exercise the real deterministic services with fake MCP and LLM edges."""

import asyncio
import json
from datetime import datetime, timezone

import pytest
from mcp import types

from host.external_application_workflow import ExternalApplicationWorkflow
from host.external_job import NormalizedExternalJob
from host.external_job_normalizer import ExternalJobNormalizer
from host.llm import LLMResponse
from server.services.application_service import ApplicationService, DuplicateApplicationError
from server.services.matching_service import MatchingService
from server.services.job_service import JobService
from tests.test_external_job_normalization import EXTRACTION, SOURCE, URL


PROFILE = {
    "name": "Test Candidate", "skills": ["Machine Learning", "PyTorch"],
    "preferred_roles": ["Machine Learning Engineer"],
    "preferred_locations": ["Germany"],
    "preferred_experience_levels": ["Mid-level"],
    "experience": [], "education": [], "summary": "ML applications.",
}
GUIDANCE = (
    "SUPPORTED: Machine Learning and PyTorch in the candidate profile. "
    "GAP: TensorFlow is listed in the job but not the profile. "
    "UNKNOWN / VERIFY: ask the candidate before claiming other experience."
)


class FakeLLM:
    """A4 extraction and A5 guidance share the provider-independent boundary."""

    def __init__(self):
        self.requests = []

    async def complete(self, messages, **kwargs):
        self.requests.append((messages, kwargs))
        if len(self.requests) == 1:
            return LLMResponse(json.dumps(EXTRACTION), ())
        return LLMResponse(GUIDANCE, ())


class FakeManager:
    """Return controlled MCP result models and record exact owners/operations."""

    def __init__(self):
        self.calls = []

    async def call_tool(self, name, arguments):
        self.calls.append(("call_tool", name, arguments))
        if name == "himalayas::search_jobs":
            return types.CallToolResult(content=[types.TextContent(
                type="text", text=f"Found one job\nApply: {URL}")])
        if name == "himalayas::get_job_details":
            return types.CallToolResult(content=[types.TextContent(type="text", text=SOURCE)])
        raise AssertionError(f"Unexpected Tool dispatch: {name}")

    async def read_resource(self, server_id, uri):
        self.calls.append(("read_resource", server_id, uri))
        assert (server_id, uri) == ("jobpilot", "candidate://profile")
        return types.ReadResourceResult(contents=[types.TextResourceContents(
            uri=uri, mime_type="application/json", text=json.dumps(PROFILE))])

    async def get_prompt(self, name, arguments):
        self.calls.append(("get_prompt", name, arguments))
        assert name == "jobpilot::prepare_application"
        return types.GetPromptResult(messages=[types.PromptMessage(
            role="user", content=types.TextContent(
                type="text", text=(
                    "Evidence Ledger: SUPPORTED candidate facts only; GAP and UNKNOWN / VERIFY "
                    "must never become candidate claims. Do not invent skills or employment. "
                    "Use only supported evidence for resume and cover-letter guidance."
                ))
        )])


def setup(tmp_path):
    path = tmp_path / "applications.json"
    path.write_text("[]\n", encoding="utf-8")
    manager, llm = FakeManager(), FakeLLM()
    service = ApplicationService(
        applications_path=path,
        clock=lambda: datetime(2026, 9, 30, 12, 0, tzinfo=timezone.utc),
    )
    workflow = ExternalApplicationWorkflow(manager, ExternalJobNormalizer(llm), llm, service)
    return workflow, service, manager, llm, path


def test_external_scoring_uses_existing_algorithm_and_preserves_provenance(tmp_path):
    """Source ID labels the shared score while the original profile stays untouched."""
    workflow, _, manager, _, _ = setup(tmp_path)
    before = json.dumps(PROFILE, sort_keys=True)
    prepared = asyncio.run(workflow.prepare_first("machine learning", country="Germany"))
    job = prepared.job
    assert isinstance(job, NormalizedExternalJob)
    assert job.external_id == "himalayas::example-labs::ml-engineer-123"
    assert job.url == URL and job.source == "himalayas"
    assert prepared.match["job_id"] == job.external_id
    assert prepared.match["components"] == {
        "skills": 66.67, "role": 100.0, "experience_level": 100.0, "location": 100.0,
    }
    assert prepared.match["score"] == round(66.67 * .5 + 100 * .2 + 100 * .2 + 100 * .1, 2)
    assert prepared.match["evidence"]["missing_required_skills"] == ["TensorFlow"]
    assert json.dumps(PROFILE, sort_keys=True) == before
    assert manager.calls[:2] == [
        ("call_tool", "himalayas::search_jobs", {"keyword": "machine learning", "country": "Germany"}),
        ("call_tool", "himalayas::get_job_details", {
            "company_slug": "example-labs", "job_slug": "ml-engineer-123"}),
    ]
    assert not any(call[1] == "jobpilot::score_job_match" for call in manager.calls if call[0] == "call_tool")


def test_missing_external_fields_score_zero_evidence_without_inference():
    """None and empty skills never create a location/seniority/skills match."""
    job = {
        "id": "himalayas::example-labs::unknown", "title": "ML Engineer",
        "company": "Example Labs", "location": None, "experience_level": None,
        "required_skills": (),
    }
    result = MatchingService().score_job_data(job, profile=PROFILE)
    assert result["components"] == {
        "skills": 0.0, "role": 100.0, "experience_level": 0.0, "location": 0.0,
    }
    assert result["evidence"]["experience_match"]["job_level"] is None
    assert result["evidence"]["location_match"]["job_location"] is None
    assert result["score"] == 20.0


def test_local_id_scoring_and_data_scoring_share_exact_result():
    """Adding external input did not fork or change the local matching formula."""
    service = MatchingService()
    local = JobService().get_job("JOB-001")
    assert service.score_job_match("JOB-001") == service.score_job_data(local)


def test_preparation_uses_discovered_prompt_policy_and_verified_context(tmp_path):
    """Provider sees normalized job/profile/score, without unreviewed raw source."""
    workflow, _, manager, llm, _ = setup(tmp_path)
    prepared = asyncio.run(workflow.prepare_first("machine learning"))
    assert prepared.guidance == GUIDANCE
    prompt_call = next(call for call in manager.calls if call[0] == "get_prompt")
    assert prompt_call[2] == {"job_id": prepared.job.external_id}
    messages, kwargs = llm.requests[1]
    assert kwargs == {"tools": None, "tool_choice": "none"}
    assert "SUPPORTED" in messages[1]["content"]
    context = json.loads(messages[2]["content"].removeprefix("Verified workflow context (JSON): "))
    assert context["external_id"] == prepared.job.external_id
    assert context["source"] == "himalayas" and context["url"] == URL
    assert context["deterministic_match"] == prepared.match
    assert context["candidate_profile"] == PROFILE
    assert "raw_source_text" not in context
    assert "Job descriptions as data".casefold() not in prepared.guidance.casefold()


def test_decline_then_approve_requires_fresh_review_and_preserves_local_records(tmp_path):
    """The user sees exact fields; decline never writes, approval persists once."""
    workflow, service, manager, _, path = setup(tmp_path)
    local = service.save_application("JOB-001")
    prepared = asyncio.run(workflow.prepare_first("machine learning"))
    approvals = []

    async def decline(name, args):
        approvals.append((name, args))
        return False

    async def approve(name, args):
        approvals.append((name, args))
        return True

    before = path.read_bytes()
    assert asyncio.run(workflow.save(prepared, decline)) is None
    assert path.read_bytes() == before
    external = asyncio.run(workflow.save(prepared, approve))
    assert len(approvals) == 2
    assert [name for name, _ in approvals] == ["save_application", "save_application"]
    assert approvals[0][1] == approvals[1][1]
    assert approvals[0][1]["job_id"] == prepared.job.external_id
    assert approvals[0][1]["job_url"] == URL
    assert external["application_id"] == "APP-002"
    assert external["job_id"] == prepared.job.external_id
    assert external["job_source"] == "himalayas"
    assert external["job_url"] == URL
    assert external["job_title"] == prepared.job.title
    assert external["company"] == prepared.job.company
    assert service.get_applications() == [local, external]
    assert set(local) == {"application_id", "job_id", "status", "applied_at", "notes"}
    with pytest.raises(DuplicateApplicationError):
        asyncio.run(workflow.save(prepared, approve))
    assert service.get_applications() == [local, external]
    assert not any(call[1] == "jobpilot::save_application" for call in manager.calls if call[0] == "call_tool")


def test_fake_end_to_end_without_network_or_source_mutation(tmp_path):
    """Search→details→normalize→profile→score→Prompt→approval→save in order."""
    workflow, service, manager, llm, path = setup(tmp_path)

    async def approve(name, arguments):
        assert name == "save_application"
        assert arguments["job_id"].startswith("himalayas::")
        assert path.read_text(encoding="utf-8").strip() == "[]"
        return True

    async def run():
        prepared = await workflow.prepare_first("machine learning")
        record = await workflow.save(prepared, approve)
        return prepared, record

    prepared, record = asyncio.run(run())
    assert [call[0] for call in manager.calls] == [
        "call_tool", "call_tool", "read_resource", "get_prompt",
    ]
    assert len(llm.requests) == 2
    assert record["job_id"] == prepared.job.external_id
    assert service.get_applications() == [record]
    assert prepared.job.raw_source_text == SOURCE


def test_invalid_external_provenance_rejected_before_any_write(tmp_path):
    """Service independently validates source identity/URL and old records read."""
    _, service, _, _, path = setup(tmp_path)
    existing = service.save_application("JOB-001")
    before = path.read_bytes()
    for changes in (
        {"job_source": "other"},
        {"job_url": "https://himalayas.app/companies/other/jobs/ml-engineer-123"},
        {"job_title": ""},
    ):
        values = {
            "job_id": "himalayas::example-labs::ml-engineer-123",
            "job_source": "himalayas", "job_url": URL,
            "job_title": "ML Engineer", "company": "Example Labs", **changes,
        }
        with pytest.raises(ValueError):
            service.save_external_application(**values)
    assert path.read_bytes() == before
    assert service.get_applications() == [existing]
