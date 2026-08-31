"""Component tests for application tracking below the MCP boundary.

Every test uses an injected temporary store, which protects production application
data while exercising real JSON persistence and fresh-service read-back behavior.
"""

import json
from datetime import datetime, timedelta, timezone
from pathlib import Path

import pytest

from server.services.application_service import (
    ApplicationService,
    DuplicateApplicationError,
    InvalidApplicationStatusError,
)
from server.services.job_service import JobNotFoundError


FIXED_UTC_TIME = datetime(2026, 8, 28, 12, 30, 0, tzinfo=timezone.utc)


@pytest.fixture
def applications_path(tmp_path: Path) -> Path:
    """Create an isolated empty store with the same JSON shape as production."""
    path = tmp_path / "applications.json"
    path.write_text("[]\n", encoding="utf-8")
    return path


def test_empty_store_returns_empty_list(applications_path: Path) -> None:
    """Treat an initialized empty array as a valid application history."""
    assert _service(applications_path).get_applications() == []


def test_successful_save_creates_first_application(applications_path: Path) -> None:
    """Freeze the complete default record, including null notes and UTC time."""
    record = _service(applications_path).save_application("  JOB-005  ")

    assert record == {
        "application_id": "APP-001",
        "job_id": "JOB-005",
        "status": "applied",
        "applied_at": FIXED_UTC_TIME.isoformat(),
        "notes": None,
    }
    parsed_timestamp = datetime.fromisoformat(record["applied_at"])
    assert parsed_timestamp.utcoffset() == timedelta(0)


def test_second_distinct_job_creates_next_persisted_id(
    applications_path: Path,
) -> None:
    """Generate APP-002 from disk state and preserve the first record."""
    _service(applications_path).save_application("JOB-005")
    second_record = _service(applications_path).save_application("JOB-006")

    assert second_record["application_id"] == "APP-002"
    assert [
        application["application_id"]
        for application in _service(applications_path).get_applications()
    ] == ["APP-001", "APP-002"]


def test_non_default_status_is_normalized(applications_path: Path) -> None:
    """Persist approved statuses using their canonical lowercase spelling."""
    record = _service(applications_path).save_application(
        "JOB-005", status="  InTeRvIeW  "
    )

    assert record["status"] == "interview"


def test_invalid_status_does_not_modify_store(applications_path: Path) -> None:
    """Reject unsupported state before any persistence side effect occurs."""
    before = applications_path.read_text(encoding="utf-8")

    with pytest.raises(InvalidApplicationStatusError):
        _service(applications_path).save_application("JOB-005", status="pending")

    assert applications_path.read_text(encoding="utf-8") == before


def test_duplicate_job_does_not_modify_store(applications_path: Path) -> None:
    """Enforce the V1 one-application-per-job rule without appending a record."""
    service = _service(applications_path)
    service.save_application("JOB-005")
    before = applications_path.read_text(encoding="utf-8")

    with pytest.raises(DuplicateApplicationError):
        service.save_application("  JOB-005  ")

    assert applications_path.read_text(encoding="utf-8") == before


def test_unknown_job_propagates_domain_error_without_writing(
    applications_path: Path,
) -> None:
    """Reuse JobService validation instead of persisting an orphan application."""
    before = applications_path.read_text(encoding="utf-8")

    with pytest.raises(JobNotFoundError):
        _service(applications_path).save_application("JOB-999")

    assert applications_path.read_text(encoding="utf-8") == before


def test_notes_are_persisted_without_rewriting(applications_path: Path) -> None:
    """Preserve optional user text exactly rather than interpreting it."""
    notes = "  Applied through the company portal.  "
    record = _service(applications_path).save_application("JOB-005", notes=notes)

    assert record["notes"] == notes
    assert json.loads(applications_path.read_text(encoding="utf-8"))[0][
        "notes"
    ] == notes


def test_fresh_service_reads_persisted_record(applications_path: Path) -> None:
    """Prove state resides in JSON rather than service-process memory."""
    saved_record = _service(applications_path).save_application("JOB-005")

    fresh_service = ApplicationService(applications_path)

    assert fresh_service.get_applications() == [saved_record]


def _service(applications_path: Path) -> ApplicationService:
    """Create a service with deterministic time and production job validation."""
    return ApplicationService(
        applications_path=applications_path,
        clock=lambda: FIXED_UTC_TIME,
    )
