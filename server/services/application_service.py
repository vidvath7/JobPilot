"""Ordinary application tracking and JSON persistence for JobPilot.

This layer owns record validation, ID/timestamp generation, and state changes.
It deliberately has no MCP dependency so persistence behavior can be tested before
being exposed as a protocol Tool or Resource in a later step.
"""

import json
import os
import re
from collections.abc import Callable
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urlsplit

from server.services.job_service import JobService


APPLICATIONS_PATH_ENVIRONMENT_VARIABLE = "JOBPILOT_APPLICATIONS_PATH"
_ALLOWED_STATUSES = frozenset(
    {"applied", "interview", "rejected", "offer", "withdrawn"}
)
_APPLICATION_FIELDS = {
    "application_id",
    "job_id",
    "status",
    "applied_at",
    "notes",
}
_APPLICATION_ID_PATTERN = re.compile(r"APP-(\d+)$")
_HIMALAYAS_ID_PATTERN = re.compile(r"himalayas::([A-Za-z0-9][A-Za-z0-9_-]*)::([A-Za-z0-9][A-Za-z0-9_-]*)$")
_EXTERNAL_RECORD_FIELDS = {"job_source", "job_url", "job_title", "company"}


class InvalidApplicationStatusError(ValueError):
    """Raised when a requested status is outside the approved V1 vocabulary."""


class DuplicateApplicationError(ValueError):
    """Raised when V1 already tracks an application for the requested job."""


class ApplicationService:
    """Persist and retrieve validated application records in the local JSON store."""

    def __init__(
        self,
        applications_path: str | Path | None = None,
        job_service: JobService | None = None,
        clock: Callable[[], datetime] | None = None,
    ) -> None:
        """Configure injectable storage, job validation, and time boundaries."""
        if applications_path is not None:
            resolved_applications_path = Path(applications_path)
        else:
            # Runtime configuration belongs at the shared service-construction
            # boundary so MCP read and write adapters cannot select different
            # stores. Explicit constructor injection still takes precedence.
            configured_path = os.environ.get(
                APPLICATIONS_PATH_ENVIRONMENT_VARIABLE
            )
            resolved_applications_path = (
                Path(configured_path)
                if configured_path
                else Path(__file__).resolve().parents[2]
                / "data"
                / "applications.json"
            )

        self._applications_path = resolved_applications_path
        self._job_service = job_service or JobService()
        self._clock = clock or _utc_now

    def save_application(
        self,
        job_id: str,
        status: str = "applied",
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Validate and append one application without rewriting user notes."""
        normalized_job_id = job_id.strip()
        # Preserve the original validation order for local calls.
        self._normalize_status(status)
        # Reuse JobService as the source of truth for job existence. Its
        # JobNotFoundError intentionally propagates to the caller.
        self._job_service.get_job(normalized_job_id)
        return self._save_record(normalized_job_id, status, notes, {})

    def save_external_application(
        self, job_id: str, *, job_source: str, job_url: str,
        job_title: str, company: str, status: str = "applied",
        notes: str | None = None,
    ) -> dict[str, Any]:
        """Save a Host-normalized external job with independent provenance checks.

        This ordinary service method leaves the existing MCP Tool contract and
        five-field local records intact. The Host must approve every call first.
        """
        normalized_job_id = job_id.strip()
        match = _HIMALAYAS_ID_PATTERN.fullmatch(normalized_job_id)
        if job_source != "himalayas" or match is None:
            raise ValueError("External application requires a Himalayas job identity.")
        try:
            url = urlsplit(job_url)
            valid_url = (
                url.scheme == "https" and url.netloc == "himalayas.app"
                and url.path == f"/companies/{match[1]}/jobs/{match[2]}"
                and not url.fragment
            )
        except (TypeError, ValueError):
            valid_url = False
        if not valid_url or any(not isinstance(value, str) or not value.strip()
                                for value in (job_title, company)):
            raise ValueError("External application provenance is incomplete or invalid.")
        return self._save_record(normalized_job_id, status, notes, {
            "job_source": job_source, "job_url": job_url,
            "job_title": job_title, "company": company,
        })

    def _save_record(self, job_id: str, status: str, notes: str | None,
                     provenance: dict[str, str]) -> dict[str, Any]:
        """Share status, duplicate, timestamp and append rules across job sources."""
        normalized_status = self._normalize_status(status)
        applications = self._load_applications()

        if any(
            existing["job_id"].strip() == job_id
            for existing in applications
        ):
            raise DuplicateApplicationError(
                f"An application already exists for job ID: {job_id}"
            )

        applied_at = self._clock()
        if applied_at.tzinfo is None or applied_at.utcoffset() is None:
            raise ValueError("Application clock must return an aware datetime.")

        record = {
            "application_id": self._next_application_id(applications),
            "job_id": job_id,
            "status": normalized_status,
            "applied_at": applied_at.astimezone(timezone.utc).isoformat(),
            "notes": notes,
            **provenance,
        }

        # Build a new list so validation failures cannot partially mutate the
        # loaded state before the single persistence write.
        self._write_applications([*applications, record])
        return record

    @staticmethod
    def _normalize_status(status: str) -> str:
        """Keep the approved vocabulary identical for local and external saves."""
        normalized = status.strip().casefold()
        if normalized not in _ALLOWED_STATUSES:
            raise InvalidApplicationStatusError(f"Unsupported application status: {status}")
        return normalized

    def get_applications(self) -> list[dict[str, Any]]:
        """Return a newly parsed snapshot of all persisted application records."""
        return self._load_applications()

    def _load_applications(self) -> list[dict[str, Any]]:
        """Load and minimally validate records needed for safe append operations."""
        with self._applications_path.open(encoding="utf-8") as applications_file:
            applications = json.load(applications_file)

        if not isinstance(applications, list):
            raise ValueError("Applications data must be a JSON array.")

        for index, application in enumerate(applications):
            if not isinstance(application, dict):
                raise ValueError(
                    f"Application at index {index} must be a JSON object."
                )

            missing_fields = _APPLICATION_FIELDS - application.keys()
            if missing_fields:
                raise ValueError(
                    f"Application at index {index} is missing fields: "
                    f"{', '.join(sorted(missing_fields))}."
                )

            application_id = application["application_id"]
            if not isinstance(application_id, str) or not _APPLICATION_ID_PATTERN.fullmatch(
                application_id
            ):
                raise ValueError(
                    f"Application at index {index} has an invalid application_id."
                )

            if not isinstance(application["job_id"], str):
                raise ValueError(
                    f"Application at index {index} has an invalid job_id."
                )
            if application["status"] not in _ALLOWED_STATUSES:
                raise ValueError(
                    f"Application at index {index} has an invalid status."
                )
            if not isinstance(application["applied_at"], str):
                raise ValueError(
                    f"Application at index {index} has an invalid applied_at."
                )
            if application["notes"] is not None and not isinstance(
                application["notes"], str
            ):
                raise ValueError(
                    f"Application at index {index} has invalid notes."
                )
            external_fields = _EXTERNAL_RECORD_FIELDS & application.keys()
            if external_fields and external_fields != _EXTERNAL_RECORD_FIELDS:
                raise ValueError(f"Application at index {index} has incomplete provenance.")
            if external_fields:
                match = _HIMALAYAS_ID_PATTERN.fullmatch(application["job_id"])
                try:
                    url = urlsplit(application["job_url"])
                    valid_url = (
                        url.scheme == "https" and url.netloc == "himalayas.app"
                        and match is not None
                        and url.path == f"/companies/{match[1]}/jobs/{match[2]}"
                        and not url.fragment
                    )
                except (TypeError, ValueError):
                    valid_url = False
                if (application["job_source"] != "himalayas" or not valid_url
                        or any(not isinstance(application[name], str)
                               or not application[name].strip()
                               for name in ("job_title", "company"))):
                    raise ValueError(f"Application at index {index} has invalid provenance.")

        return applications

    def _next_application_id(self, applications: list[dict[str, Any]]) -> str:
        """Derive the next sequence from persisted IDs rather than process state."""
        sequence_numbers = [
            int(_APPLICATION_ID_PATTERN.fullmatch(application["application_id"]).group(1))
            for application in applications
        ]
        return f"APP-{max(sequence_numbers, default=0) + 1:03d}"

    def _write_applications(self, applications: list[dict[str, Any]]) -> None:
        """Persist a complete, valid JSON snapshot after all checks succeed."""
        self._applications_path.write_text(
            json.dumps(applications, indent=2) + "\n",
            encoding="utf-8",
        )


def _utc_now() -> datetime:
    """Provide an aware UTC timestamp while allowing deterministic test injection."""
    return datetime.now(timezone.utc)
