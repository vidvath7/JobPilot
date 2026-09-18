"""Prompt content tests check evidence rules, not model compliance or orchestration.

Direct invocation is appropriate here: the capability only creates instructions;
existing stdio tests independently verify its MCP discovery/retrieval lifecycle.
"""

import pytest

from server.prompts.prepare_application import prepare_application


@pytest.fixture
def guidance():
    """Inspect the actual instruction text without any domain lookup or LLM call."""
    return prepare_application("JOB-005")[0].content.text.casefold()


def test_evidence_ledger_precedes_guidance(guidance):
    """Require distinct evidence categories before any drafting recommendations."""
    for phrase in ("supported", "explicitly present", "gap", "unknown / verify",
                   "never convert gap or unknown", "supporting profile field"):
        assert phrase in guidance
    assert guidance.index("evidence ledger") < guidance.index("6. prepare resume-tailoring guidance")
    assert "using a only for candidate claims" in guidance


@pytest.mark.parametrize("fact", [
    "citizenship", "visa/work authorization", "relocation willingness", "languages",
    "salary expectations", "certifications", "academic coursework/modules",
    "education details", "achievements", "employment responsibilities",
    "stakeholder collaboration", "domain experience", "motivations/preferences",
])
def test_sensitive_or_plausible_candidate_facts_require_evidence(guidance, fact):
    """Each observed fabrication category is explicitly governed by an evidence rule."""
    rule = next(line for line in guidance.splitlines() if "do not infer or assert" in line)
    assert fact in rule
    assert "unless explicitly present in retrieved candidate evidence" in rule
    assert "do not fabricate" in rule


def test_job_responsibilities_are_not_candidate_experience(guidance):
    """Prevent a job requirement from being promoted into employment history."""
    assert "job-description facts describe the job, not the candidate" in guidance
    assert "does not imply" in guidance
    assert "without explicit candidate evidence" in guidance


def test_resume_requires_supported_evidence_and_verification(guidance):
    """Resume suggestions must retain gaps and unknowns rather than polish inventions."""
    resume = guidance.split("resume guidance:")[1].split("cover-letter guidance:")[0]
    for phrase in ("only supported candidate evidence", "never recommend adding an unsupported",
                   "keep absent required skills as gaps", "do not invent related experience",
                   "verify with the candidate before using", "before verification"):
        assert phrase in resume


def test_cover_letter_uses_supported_claims_only(guidance):
    """Unknown motivations or eligibility remain questions, not cover-letter statements."""
    cover = guidance.split("\ncover-letter guidance:\n")[1]
    for phrase in ("only supported candidate qualifications", "do not invent motivations",
                   "relocation willingness", "work authorization", "leadership",
                   "questions to verify", "not statements in the cover letter"):
        assert phrase in cover
