"""MCP Resource adapter for JobPilot's structured candidate profile.

The adapter exposes readable context through MCP while delegating file access and
validation to ``ProfileService``, which remains ordinary application logic.
"""

from typing import Any

if __package__ == "server.resources":
    from ..services.profile_service import ProfileService
else:
    from services.profile_service import ProfileService


CANDIDATE_PROFILE_URI = "candidate://profile"
CANDIDATE_PROFILE_MIME_TYPE = "application/json"
CANDIDATE_PROFILE_DESCRIPTION = (
    "Exposes the candidate profile used for job matching and application context."
)

_profile_service = ProfileService()


def candidate_profile() -> dict[str, Any]:
    """Return structured candidate context for MCP Resource serialization.

    The MCP SDK converts this dictionary to JSON for the declared MIME type; this
    adapter does not duplicate storage or validation behavior.
    """
    return _profile_service.get_profile()
