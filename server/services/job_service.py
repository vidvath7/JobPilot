"""Deterministic job-search logic for JobPilot's application-service layer.

This module deliberately has no MCP dependency. Keeping JSON access, validation,
and filtering here makes the business behavior independently testable and lets
other interfaces reuse it without speaking the MCP protocol.
"""

import json
from pathlib import Path
from typing import Any


# The service projects records onto this allowlist so callers receive stable,
# concise search results rather than descriptions or other unnecessary context.
_SUMMARY_FIELDS = ("id", "title", "company", "location", "experience_level")


class JobService:
    """Load, validate, and deterministically search the local jobs dataset."""

    def __init__(self, jobs_path: str | Path | None = None) -> None:
        """Configure the data source, with injection available for controlled tests.

        The default is anchored to this module instead of the process working
        directory, so CLI clients and test runners resolve the same project data.
        """
        self._jobs_path = (
            Path(jobs_path)
            if jobs_path is not None
            else Path(__file__).resolve().parents[2] / "data" / "jobs.json"
        )

    def search_jobs(
        self,
        role: str | None = None,
        location: str | None = None,
        experience_level: str | None = None,
    ) -> list[dict[str, str]]:
        """Return concise summaries that satisfy every supplied filter.

        Normalized literal comparisons keep results reproducible; semantic search,
        ranking, and model judgment do not belong in this foundation capability.
        """
        normalized_role = self._normalize(role)
        normalized_location = self._normalize(location)
        normalized_experience_level = self._normalize(experience_level)

        summaries = []

        for job in self._load_jobs():
            if (
                normalized_role is not None
                and normalized_role not in self._normalize(job["title"])
            ):
                continue
            if (
                normalized_location is not None
                and normalized_location not in self._normalize(job["location"])
            ):
                continue
            if (
                normalized_experience_level is not None
                and normalized_experience_level
                != self._normalize(job["experience_level"])
            ):
                continue

            # Project into a new dictionary to enforce the public search contract
            # and avoid leaking full source records across higher-level boundaries.
            summaries.append({field: job[field] for field in _SUMMARY_FIELDS})

        return summaries

    def _load_jobs(self) -> list[dict[str, Any]]:
        """Parse JSON and validate only the structure needed for safe searching.

        This intentionally stops short of a comprehensive schema layer: later
        capabilities can add validation when they have a concrete requirement.
        """
        with self._jobs_path.open(encoding="utf-8") as jobs_file:
            jobs = json.load(jobs_file)

        if not isinstance(jobs, list):
            raise ValueError("Jobs data must be a JSON array.")

        for index, job in enumerate(jobs):
            if not isinstance(job, dict):
                raise ValueError(f"Job at index {index} must be a JSON object.")

            missing_fields = [field for field in _SUMMARY_FIELDS if field not in job]
            if missing_fields:
                raise ValueError(
                    f"Job at index {index} is missing fields: "
                    f"{', '.join(missing_fields)}."
                )

            non_string_fields = [
                field for field in _SUMMARY_FIELDS if not isinstance(job[field], str)
            ]
            if non_string_fields:
                raise ValueError(
                    f"Job at index {index} has non-string fields: "
                    f"{', '.join(non_string_fields)}."
                )

        return jobs

    @staticmethod
    def _normalize(value: str | None) -> str | None:
        """Normalize optional filters and fields for deterministic comparison."""
        return value.strip().casefold() if value is not None else None
