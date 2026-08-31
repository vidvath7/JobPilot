"""MCP Resource adapter for persisted JobPilot application history.

The Resource is the read side of application tracking. It delegates storage and
validation to ``ApplicationService`` rather than treating the JSON file itself as
an MCP capability.
"""

from typing import Any

if __package__ == "server.resources":
    from ..services.application_service import ApplicationService
else:
    from services.application_service import ApplicationService


APPLICATIONS_URI = "applications://all"
APPLICATIONS_MIME_TYPE = "application/json"
APPLICATIONS_DESCRIPTION = (
    "Exposes the persisted JobPilot application history."
)

# The service resolves the same optional runtime path used by save_application,
# so MCP writes are immediately visible through this read-only Resource.
_application_service = ApplicationService()


def applications() -> list[dict[str, Any]]:
    """Return persisted application records for MCP JSON serialization.

    The adapter performs no file access or validation itself; those ordinary
    application responsibilities remain in ``ApplicationService``.
    """
    return _application_service.get_applications()
