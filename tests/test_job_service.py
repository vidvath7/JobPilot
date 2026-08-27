from server.services.job_service import JobService


SUMMARY_FIELDS = {"id", "title", "company", "location", "experience_level"}


def test_no_filters_returns_all_synthetic_jobs() -> None:
    results = JobService().search_jobs()

    assert len(results) == 10


def test_role_matching_is_case_insensitive() -> None:
    results = JobService().search_jobs(role="mAcHiNe LeArNiNg")

    assert [job["id"] for job in results] == ["JOB-002"]


def test_location_matching_is_case_insensitive() -> None:
    results = JobService().search_jobs(location="bErLiN")

    assert [job["id"] for job in results] == ["JOB-001"]


def test_experience_level_matching_is_normalized_and_case_insensitive() -> None:
    results = JobService().search_jobs(experience_level="  mID-LeVeL  ")

    assert [job["id"] for job in results] == [
        "JOB-001",
        "JOB-002",
        "JOB-004",
        "JOB-005",
        "JOB-008",
    ]


def test_multiple_filters_use_and_semantics() -> None:
    results = JobService().search_jobs(
        role="engineer",
        location="MUNICH",
        experience_level="mid-level",
    )

    assert [job["id"] for job in results] == ["JOB-002"]


def test_no_matches_returns_empty_list() -> None:
    results = JobService().search_jobs(role="Marine Biologist")

    assert results == []


def test_results_contain_only_summary_fields() -> None:
    results = JobService().search_jobs()

    assert all(set(job) == SUMMARY_FIELDS for job in results)
