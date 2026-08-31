"""MCP-facing adapter for JobPilot's application persistence capability.

This module owns only the protocol boundary and process-level configuration.
``ApplicationService`` remains responsible for validation, IDs, timestamps, and
JSON persistence so the state-changing behavior stays independently testable.
"""

import os
from typing import Any

if __package__ == "server.tools":
    from ..services.application_service import ApplicationService
else:
    from services.application_service import ApplicationService


APPLICATIONS_PATH_ENVIRONMENT_VARIABLE = "JOBPILOT_APPLICATIONS_PATH"

# Resolve the store when the MCP server process starts. Production uses the
# service's repository-relative default; stdio tests can inject an isolated file
# through the subprocess environment without changing domain behavior.
_application_service = ApplicationService(
    applications_path=os.environ.get(APPLICATIONS_PATH_ENVIRONMENT_VARIABLE)
)


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
