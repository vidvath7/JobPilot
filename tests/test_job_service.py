"""Component tests for ordinary job-search behavior below the MCP boundary.

Direct Python calls are appropriate here because failures should isolate domain
filtering from Tool registration, serialization, and transport concerns.
"""

from server.services.job_service import JobService


SUMMARY_FIELDS = {"id", "title", "company", "location", "experience_level"}


def test_no_filters_returns_all_synthetic_jobs() -> None:
    """Verify that omitting optional filters preserves the complete dataset."""
    results = JobService().search_jobs()

    assert len(results) == 10


def test_role_matching_is_case_insensitive() -> None:
    """Protect the deterministic title-substring contract for role searches."""
    results = JobService().search_jobs(role="mAcHiNe LeArNiNg")

    assert [job["id"] for job in results] == ["JOB-002"]


def test_location_matching_is_case_insensitive() -> None:
    """Isolate location normalization from any MCP argument handling."""
    results = JobService().search_jobs(location="bErLiN")

    assert [job["id"] for job in results] == ["JOB-001"]


def test_experience_level_matching_is_normalized_and_case_insensitive() -> None:
    """Verify trimmed exact matching for the experience-level filter."""
    results = JobService().search_jobs(experience_level="  mID-LeVeL  ")

    assert [job["id"] for job in results] == [
        "JOB-001",
        "JOB-002",
        "JOB-004",
        "JOB-005",
        "JOB-008",
    ]


def test_multiple_filters_use_and_semantics() -> None:
    """Ensure combined filters narrow results instead of matching independently."""
    results = JobService().search_jobs(
        role="engineer",
        location="MUNICH",
        experience_level="mid-level",
    )

    assert [job["id"] for job in results] == ["JOB-002"]


def test_no_matches_returns_empty_list() -> None:
    """Define an empty result as normal search behavior rather than an error."""
    results = JobService().search_jobs(role="Marine Biologist")

    assert results == []


def test_results_contain_only_summary_fields() -> None:
    """Prevent the service boundary from leaking descriptions or skill details."""
    results = JobService().search_jobs()

    assert all(set(job) == SUMMARY_FIELDS for job in results)
