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
        normalized_status = status.strip().casefold()

        if normalized_status not in _ALLOWED_STATUSES:
            raise InvalidApplicationStatusError(
                f"Unsupported application status: {status}"
            )

        # Reuse JobService as the source of truth for job existence. Its
        # JobNotFoundError intentionally propagates to the caller.
        self._job_service.get_job(normalized_job_id)
        applications = self._load_applications()

        if any(
            existing["job_id"].strip() == normalized_job_id
            for existing in applications
        ):
            raise DuplicateApplicationError(
                f"An application already exists for job ID: {normalized_job_id}"
            )

        applied_at = self._clock()
        if applied_at.tzinfo is None or applied_at.utcoffset() is None:
            raise ValueError("Application clock must return an aware datetime.")

        record = {
            "application_id": self._next_application_id(applications),
            "job_id": normalized_job_id,
            "status": normalized_status,
            "applied_at": applied_at.astimezone(timezone.utc).isoformat(),
            "notes": notes,
        }

        # Build a new list so validation failures cannot partially mutate the
        # loaded state before the single persistence write.
        self._write_applications([*applications, record])
        return record

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
