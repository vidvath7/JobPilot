"""Component tests for deterministic matching below the MCP boundary.

Current fixtures verify intended end-to-end scores, while injected service mocks
isolate normalization edge cases without changing production data.
"""

from unittest.mock import Mock

import pytest

from server.services.job_service import JobNotFoundError, JobService
from server.services.matching_service import MatchingService
from server.services.profile_service import ProfileService


def test_strong_match_job_scores_full_marks() -> None:
    """Confirm the intentionally strongest fixture matches every component."""
    result = MatchingService().score_job_match("JOB-001")

    assert result["score"] == 100.0
    assert set(result) == {
        "job_id",
        "job_title",
        "company",
        "score",
        "weights",
        "components",
        "evidence",
    }
    assert result["weights"] == {
        "skills": 0.50,
        "role": 0.20,
        "experience_level": 0.20,
        "location": 0.10,
    }
    assert result["components"] == {
        "skills": 100.0,
        "role": 100.0,
        "experience_level": 100.0,
        "location": 100.0,
    }
    assert set(result["evidence"]) == {
        "matched_required_skills",
        "missing_required_skills",
        "role_match",
        "experience_match",
        "location_match",
    }
    assert result["evidence"]["matched_required_skills"] == [
        "Python",
        "PyTorch",
        "FastAPI",
        "Docker",
        "Git",
    ]
    assert result["evidence"]["missing_required_skills"] == []
    assert result["evidence"]["role_match"] == {
        "job_role": "AI Engineer",
        "matched_preference": "AI Engineer",
        "match_type": "exact",
    }
    assert result["evidence"]["experience_match"] == {
        "job_level": "Mid-level",
        "candidate_preferences": ["Junior", "Mid-level"],
        "matched": True,
    }
    assert result["evidence"]["location_match"] == {
        "job_location": "Berlin, Germany",
        "candidate_preferences": [
            "Berlin, Germany",
            "Hamburg, Germany",
            "Munich, Germany",
            "Remote, EU",
        ],
        "matched_preference": "Berlin, Germany",
        "matched": True,
    }


@pytest.mark.parametrize(
    ("job_id", "expected_score"),
    [("JOB-003", 34.0), ("JOB-010", 0.0)],
)
def test_current_fixtures_include_medium_and_weak_matches(
    job_id: str, expected_score: float
) -> None:
    """Keep representative non-strong fixture scores stable and explainable."""
    assert MatchingService().score_job_match(job_id)["score"] == expected_score


def test_skill_matching_is_case_insensitive() -> None:
    """Treat casing differences as spelling variation, not missing skills."""
    service = _matching_service(candidate_skills=["PYTHON"], required_skills=["python"])

    result = service.score_job_match("TEST-001")

    assert result["components"]["skills"] == 100.0
    assert result["evidence"]["matched_required_skills"] == ["python"]


def test_explicit_skill_aliases_are_equivalent() -> None:
    """Verify only the declared ML, LLM, and JavaScript aliases collapse."""
    service = _matching_service(
        candidate_skills=["Machine Learning", "Large Language Models", "JavaScript"],
        required_skills=["ML", "LLMs", "JS"],
    )

    result = service.score_job_match("TEST-001")

    assert result["components"]["skills"] == 100.0
    assert result["evidence"]["matched_required_skills"] == ["ML", "LLMs", "JS"]


def test_missing_required_skills_preserve_original_job_labels() -> None:
    """Return readable evidence rather than normalized internal tokens."""
    service = _matching_service(
        candidate_skills=["Python"], required_skills=["Python", "Kubernetes"]
    )

    result = service.score_job_match("TEST-001")

    assert result["components"]["skills"] == 50.0
    assert result["evidence"]["matched_required_skills"] == ["Python"]
    assert result["evidence"]["missing_required_skills"] == ["Kubernetes"]


def test_empty_required_skills_have_deterministic_zero_score() -> None:
    """Avoid division by zero and unearned points when requirements are absent."""
    result = _matching_service(required_skills=[]).score_job_match("TEST-001")

    assert result["components"]["skills"] == 0.0
    assert result["evidence"]["matched_required_skills"] == []
    assert result["evidence"]["missing_required_skills"] == []


def test_exact_preferred_role_scores_100() -> None:
    """Protect the highest-confidence role rule using the current fixture."""
    result = MatchingService().score_job_match("JOB-001")

    assert result["components"]["role"] == 100.0
    assert result["evidence"]["role_match"]["match_type"] == "exact"
    assert result["evidence"]["role_match"]["matched_preference"] == "AI Engineer"


def test_explicit_role_family_scores_70() -> None:
    """Recognize Applied AI Engineer without broad semantic inference."""
    result = MatchingService().score_job_match("JOB-003")

    assert result["components"]["role"] == 70.0
    assert result["evidence"]["role_match"]["match_type"] == "family"
    assert result["evidence"]["role_match"]["matched_preference"] == "AI Engineer"


def test_meaningful_role_token_overlap_scores_40() -> None:
    """Award limited alignment for the shared non-generic AI role token."""
    result = MatchingService().score_job_match("JOB-009")

    assert result["components"]["role"] == 40.0
    assert result["evidence"]["role_match"]["match_type"] == "token_overlap"
    assert result["evidence"]["role_match"]["matched_preference"] == "AI Engineer"


def test_weak_match_reports_structured_unmatched_evidence() -> None:
    """Freeze null preferences and false flags for an unaligned fixture job."""
    evidence = MatchingService().score_job_match("JOB-010")["evidence"]

    assert evidence["role_match"] == {
        "job_role": "Cloud Platform Engineer",
        "matched_preference": None,
        "match_type": "none",
    }
    assert evidence["experience_match"] == {
        "job_level": "Senior",
        "candidate_preferences": ["Junior", "Mid-level"],
        "matched": False,
    }
    assert evidence["location_match"] == {
        "job_location": "Dublin, Ireland",
        "candidate_preferences": [
            "Berlin, Germany",
            "Hamburg, Germany",
            "Munich, Germany",
            "Remote, EU",
        ],
        "matched_preference": None,
        "matched": False,
    }


@pytest.mark.parametrize(
    ("job_id", "expected_experience", "expected_location"),
    [("JOB-001", 100.0, 100.0), ("JOB-003", 0.0, 100.0), ("JOB-004", 100.0, 0.0)],
)
def test_experience_and_location_matches_are_binary(
    job_id: str, expected_experience: float, expected_location: float
) -> None:
    """Verify preferred membership and location containment independently."""
    components = MatchingService().score_job_match(job_id)["components"]

    assert components["experience_level"] == expected_experience
    assert components["location"] == expected_location


def test_weighted_score_equals_reported_components() -> None:
    """Ensure consumers can reproduce the final result from returned components."""
    result = MatchingService().score_job_match("JOB-003")
    components = result["components"]
    weights = result["weights"]

    expected_score = round(
        components["skills"] * weights["skills"]
        + components["role"] * weights["role"]
        + components["experience_level"] * weights["experience_level"]
        + components["location"] * weights["location"],
        2,
    )
    assert result["score"] == expected_score


def test_unknown_job_id_propagates_domain_error() -> None:
    """Do not misrepresent an unknown job as a legitimate zero match."""
    with pytest.raises(JobNotFoundError):
        MatchingService().score_job_match("JOB-999")


def _matching_service(
    *,
    candidate_skills: list[str] | None = None,
    required_skills: list[str] | None = None,
) -> MatchingService:
    """Build controlled service dependencies for normalization-focused tests."""
    profile = {
        "skills": candidate_skills if candidate_skills is not None else [],
        "preferred_roles": [],
        "preferred_experience_levels": [],
        "preferred_locations": [],
    }
    job = {
        "id": "TEST-001",
        "title": "Unrelated Role",
        "company": "Synthetic Company",
        "location": "Unlisted Location",
        "experience_level": "Unlisted Level",
        "required_skills": required_skills if required_skills is not None else [],
    }
    profile_service = Mock(spec=ProfileService)
    profile_service.get_profile.return_value = profile
    job_service = Mock(spec=JobService)
    job_service.get_job.return_value = job
    return MatchingService(profile_service, job_service)
