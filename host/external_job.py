"""JobPilot-owned external job values, independent of MCP and provider SDK models.

External identities are not local JOB-* IDs. Nullable fields deliberately prevent
missing source information from masquerading as matching evidence in a later phase.
"""

import re
from dataclasses import dataclass, field
from urllib.parse import urlsplit


class ExternalJobValidationError(ValueError):
    """Source provenance or extracted content cannot form a valid external job."""


def external_job_id(company_slug: str, job_slug: str) -> str:
    """Construct an unambiguous identity without renaming source slug values."""
    for slug in (company_slug, job_slug):
        if not isinstance(slug, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", slug):
            raise ExternalJobValidationError("Expected a nonempty, single-segment Himalayas slug.")
    return f"himalayas::{company_slug}::{job_slug}"


def parse_external_job_id(value: str) -> tuple[str, str]:
    """Resolve only this source's identity format, not MCP capability qualifiers."""
    parts = value.split("::")
    if len(parts) != 3 or parts[0] != "himalayas":
        raise ExternalJobValidationError("Invalid Himalayas external job identity.")
    external_job_id(parts[1], parts[2])
    return parts[1], parts[2]


def validate_source_url(url: str, company_slug: str, job_slug: str) -> None:
    """Require the known HTTPS job-page path; preserve original tracking queries."""
    external_job_id(company_slug, job_slug)
    try:
        parsed = urlsplit(url)
        valid = (parsed.scheme == "https" and parsed.netloc == "himalayas.app"
                 and parsed.path == f"/companies/{company_slug}/jobs/{job_slug}"
                 and not parsed.fragment)
    except (TypeError, ValueError):
        valid = False
    if not valid:
        raise ExternalJobValidationError("URL does not match the selected Himalayas job provenance.")


@dataclass(frozen=True)
class NormalizedExternalJob:
    """Validated in-memory domain value; no persistence or local service adaptation."""

    source_company_slug: str
    source_job_slug: str
    title: str
    company: str
    location: str | None
    experience_level: str | None
    required_skills: tuple[str, ...]
    description: str
    url: str
    raw_source_text: str
    source: str = field(default="himalayas", init=False)
    external_id: str = field(init=False)

    def __post_init__(self) -> None:
        """Reject malformed values even when constructed outside the normalizer."""
        object.__setattr__(self, "external_id", external_job_id(self.source_company_slug, self.source_job_slug))
        for name in ("title", "company", "description", "raw_source_text"):
            value = getattr(self, name)
            if not isinstance(value, str) or not value.strip():
                raise ExternalJobValidationError(f"{name} must be nonempty text.")
        if self.location is not None and (not isinstance(self.location, str) or not self.location.strip()):
            raise ExternalJobValidationError("location must be nonempty text or None.")
        if self.experience_level not in (None, "Junior", "Mid-level", "Senior"):
            raise ExternalJobValidationError("Unsupported normalized experience level.")
        if not isinstance(self.required_skills, tuple) or any(
            not isinstance(skill, str) or not skill.strip() for skill in self.required_skills
        ):
            raise ExternalJobValidationError("required_skills must be a tuple of nonempty strings.")
        validate_source_url(self.url, self.source_company_slug, self.source_job_slug)

