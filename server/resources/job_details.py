"""MCP Resource Template adapter for full JobPilot job records.

The URI parameter selects one addressable job. The adapter delegates exact ID
lookup to ``JobService`` so MCP routing never owns JSON or domain behavior.
"""

from typing import Any

if __package__ == "server.resources":
    from ..services.job_service import JobService
else:
    from services.job_service import JobService


JOB_DETAILS_URI_TEMPLATE = "jobs://job/{job_id}"
JOB_DETAILS_MIME_TYPE = "application/json"
JOB_DETAILS_DESCRIPTION = "Returns full details for a known JobPilot job ID."

_job_service = JobService()


def job_details(job_id: str) -> dict[str, Any]:
    """Resolve a template URI's ``job_id`` to its complete job record.

    The parameter name intentionally matches ``{job_id}``; MCP v2 validates that
    correspondence when registering a Resource Template.
    """
    return _job_service.get_job(job_id)
