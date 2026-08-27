import json
from pathlib import Path
from typing import Any


_SUMMARY_FIELDS = ("id", "title", "company", "location", "experience_level")


class JobService:
    def __init__(self, jobs_path: str | Path | None = None) -> None:
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

            summaries.append({field: job[field] for field in _SUMMARY_FIELDS})

        return summaries

    def _load_jobs(self) -> list[dict[str, Any]]:
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
        return value.strip().casefold() if value is not None else None
