"""Ordinary application service for loading the candidate profile.

Profile storage and validation stay below the MCP boundary so the same structured
profile can be reused by future interfaces without depending on protocol code.
"""

import json
from pathlib import Path
from typing import Any


_EXPECTED_PROFILE_FIELDS = {
    "name",
    "summary",
    "skills",
    "experience",
    "education",
    "preferred_roles",
    "preferred_locations",
    "preferred_experience_levels",
}


class ProfileService:
    """Load and minimally validate JobPilot's structured candidate profile."""

    def __init__(self, profile_path: str | Path | None = None) -> None:
        """Configure an injectable profile path with a repository-relative default."""
        self._profile_path = (
            Path(profile_path)
            if profile_path is not None
            else Path(__file__).resolve().parents[2] / "data" / "profile.json"
        )

    def get_profile(self) -> dict[str, Any]:
        """Return profile data after validating the current top-level contract."""
        with self._profile_path.open(encoding="utf-8") as profile_file:
            profile = json.load(profile_file)

        if not isinstance(profile, dict):
            raise ValueError("Candidate profile data must be a JSON object.")

        missing_fields = sorted(_EXPECTED_PROFILE_FIELDS - profile.keys())
        if missing_fields:
            raise ValueError(
                "Candidate profile is missing fields: "
                f"{', '.join(missing_fields)}."
            )

        return profile
