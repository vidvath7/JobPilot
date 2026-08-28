"""Component tests for profile loading below the MCP protocol boundary.

Direct service calls isolate storage and validation failures from Resource
registration, JSON serialization, and stdio transport behavior.
"""

import json

import pytest

from server.services.profile_service import ProfileService


EXPECTED_PROFILE_FIELDS = {
    "name",
    "summary",
    "skills",
    "experience",
    "education",
    "preferred_roles",
    "preferred_locations",
    "preferred_experience_levels",
}


def test_default_candidate_profile_loads_with_expected_fields() -> None:
    """Verify the repository-relative default resolves the structured fixture."""
    profile = ProfileService().get_profile()

    assert isinstance(profile, dict)
    assert EXPECTED_PROFILE_FIELDS <= profile.keys()


def test_custom_profile_path_is_supported(tmp_path) -> None:
    """Keep storage injectable so tests and future backends need no MCP changes."""
    custom_profile = {field: [] for field in EXPECTED_PROFILE_FIELDS}
    custom_profile["name"] = "Test Candidate"
    custom_profile["summary"] = "Synthetic profile for path injection."
    profile_path = tmp_path / "profile.json"
    profile_path.write_text(json.dumps(custom_profile), encoding="utf-8")

    assert ProfileService(profile_path).get_profile() == custom_profile


def test_profile_root_must_be_an_object(tmp_path) -> None:
    """Reject incompatible storage shapes before they cross the MCP boundary."""
    profile_path = tmp_path / "profile.json"
    profile_path.write_text("[]", encoding="utf-8")

    with pytest.raises(ValueError, match="must be a JSON object"):
        ProfileService(profile_path).get_profile()
