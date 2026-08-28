"""Smoke checks for the synthetic jobs fixture used by later application layers."""

import json
from pathlib import Path


def test_jobs_data_smoke() -> None:
    """Catch missing or unusable fixture data before service/MCP tests obscure it."""
    # Resolve from the test file so this smoke check works from any shell location.
    jobs_path = Path(__file__).resolve().parents[1] / "data" / "jobs.json"

    with jobs_path.open(encoding="utf-8") as jobs_file:
        jobs = json.load(jobs_file)

    required_fields = {
        "id",
        "title",
        "company",
        "location",
        "experience_level",
        "required_skills",
        "description",
        "url",
    }

    assert isinstance(jobs, list)
    assert jobs
    assert all(required_fields <= job.keys() for job in jobs)
