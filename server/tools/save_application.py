"""MCP-facing adapter for JobPilot's application persistence capability.

This module owns only the protocol boundary.
``ApplicationService`` remains responsible for validation, IDs, timestamps, and
JSON persistence so the state-changing behavior stays independently testable.
"""

from typing import Any

if __package__ == "server.tools":
    from ..services.application_service import ApplicationService
else:
    from services.application_service import ApplicationService


# ApplicationService centrally resolves the optional runtime store override. The
# Resource adapter constructs the same way, keeping both MCP capabilities aligned.
_application_service = ApplicationService()


def save_application(
    job_id: str,
    status: str = "applied",
    notes: str | None = None,
) -> dict[str, Any]:
    """Persist an application for a JobPilot job.

    ``job_id`` identifies the known job, ``status`` defaults to ``applied``, and
    ``notes`` are optional. The application ID and UTC timestamp are generated
    internally, and the created application record is returned.
    """
    # Forward the MCP-decoded arguments unchanged. All state and validation rules
    # belong to ApplicationService rather than this protocol adapter.
    return _application_service.save_application(
        job_id=job_id,
        status=status,
        notes=notes,
    )
