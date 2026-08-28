"""Create and run the JobPilot MCP server over the local stdio transport.

This is the protocol composition root: it registers MCP-facing capabilities but
leaves job-search behavior in the ordinary application-service layer.
"""

from mcp.server import MCPServer

# The MCP CLI loads this file directly, while tests import it as ``server.main``.
# Supporting both import contexts keeps one entry point usable for development
# tooling and normal package execution without changing the domain layer.
if __package__:
    from .resources.candidate_profile import (
        CANDIDATE_PROFILE_DESCRIPTION,
        CANDIDATE_PROFILE_MIME_TYPE,
        CANDIDATE_PROFILE_URI,
        candidate_profile,
    )
    from .resources.job_details import (
        JOB_DETAILS_DESCRIPTION,
        JOB_DETAILS_MIME_TYPE,
        JOB_DETAILS_URI_TEMPLATE,
        job_details,
    )
    from .tools.score_job_match import score_job_match
    from .tools.search_jobs import search_jobs
else:
    from resources.candidate_profile import (
        CANDIDATE_PROFILE_DESCRIPTION,
        CANDIDATE_PROFILE_MIME_TYPE,
        CANDIDATE_PROFILE_URI,
        candidate_profile,
    )
    from resources.job_details import (
        JOB_DETAILS_DESCRIPTION,
        JOB_DETAILS_MIME_TYPE,
        JOB_DETAILS_URI_TEMPLATE,
        job_details,
    )
    from tools.score_job_match import score_job_match
    from tools.search_jobs import search_jobs


# MCPServer owns protocol concerns such as capability discovery, schema exposure,
# invocation dispatch, and transport lifecycle. It should not contain job logic.
mcp = MCPServer(name="jobpilot")

# Registering the annotated adapter makes ``search_jobs`` discoverable as an MCP
# Tool. The SDK derives its description and input/output schemas from the
# callable's docstring and type annotations.
mcp.add_tool(search_jobs)

# Matching is an operation over a caller-supplied job ID, so it is exposed as a
# Tool rather than readable Resource context. The adapter preserves the service's
# deterministic result contract across the MCP boundary.
mcp.add_tool(score_job_match)

# A Resource exposes client-readable context rather than an operation. Because
# this URI contains no variables, MCP v2 registers it as one static Resource—not
# a Resource template—and serializes the handler's dictionary as JSON.
mcp.resource(
    CANDIDATE_PROFILE_URI,
    name="candidate_profile",
    description=CANDIDATE_PROFILE_DESCRIPTION,
    mime_type=CANDIDATE_PROFILE_MIME_TYPE,
)(candidate_profile)

# The ``{job_id}`` variable makes this a Resource Template. During a concrete
# read, MCP extracts that URI segment and supplies it to the matching handler
# parameter before the adapter delegates to ordinary application logic.
mcp.resource(
    JOB_DETAILS_URI_TEMPLATE,
    name="job_details",
    description=JOB_DETAILS_DESCRIPTION,
    mime_type=JOB_DETAILS_MIME_TYPE,
)(job_details)


if __name__ == "__main__":
    # Stdio is intentionally the first transport: it provides a real MCP boundary
    # between client and server without introducing networking or deployment.
    mcp.run()
