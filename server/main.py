"""Create and run the JobPilot MCP server over the local stdio transport.

This is the protocol composition root: it registers MCP-facing capabilities but
leaves job-search behavior in the ordinary application-service layer.
"""

from mcp.server import MCPServer

# The MCP CLI loads this file directly, while tests import it as ``server.main``.
# Supporting both import contexts keeps one entry point usable for development
# tooling and normal package execution without changing the domain layer.
if __package__:
    from .tools.search_jobs import search_jobs
else:
    from tools.search_jobs import search_jobs


# MCPServer owns protocol concerns such as capability discovery, schema exposure,
# invocation dispatch, and transport lifecycle. It should not contain job logic.
mcp = MCPServer(name="jobpilot")

# Registering the annotated adapter makes ``search_jobs`` discoverable as an MCP
# Tool. The SDK derives its description and input/output schemas from the
# callable's docstring and type annotations.
mcp.add_tool(search_jobs)


if __name__ == "__main__":
    # Stdio is intentionally the first transport: it provides a real MCP boundary
    # between client and server without introducing networking or deployment.
    mcp.run()
