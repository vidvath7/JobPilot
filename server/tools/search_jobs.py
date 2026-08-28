"""MCP-facing adapter for the ordinary JobPilot job-search capability.

The adapter is intentionally thin: MCP owns discovery and invocation here, while
``JobService`` remains responsible for data access and deterministic filtering.
"""

# As with ``server.main``, this module supports package imports and the MCP CLI's
# direct-file loading context without moving business behavior into the adapter.
if __package__ == "server.tools":
    from ..services.job_service import JobService
else:
    from services.job_service import JobService


# Keeping one service behind the adapter makes the protocol boundary explicit:
# MCP arguments enter here, then ordinary application logic performs the search.
_job_service = JobService()


# MCPServer inspects this callable during registration. Its name, docstring, and
# annotations become Tool metadata and JSON schemas that clients can discover.
def search_jobs(
    role: str | None = None,
    location: str | None = None,
    experience_level: str | None = None,
) -> list[dict[str, str]]:
    """Search available JobPilot jobs using optional role, location, and experience-level filters.

    Returns concise job summaries for matching jobs.
    """
    # Forward arguments unchanged so this layer cannot diverge from the service's
    # independently tested search contract.
    return _job_service.search_jobs(
        role=role,
        location=location,
        experience_level=experience_level,
    )
