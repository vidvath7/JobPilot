from mcp.server import MCPServer

if __package__:
    from .tools.search_jobs import search_jobs
else:
    from tools.search_jobs import search_jobs


mcp = MCPServer(name="jobpilot")
mcp.add_tool(search_jobs)


if __name__ == "__main__":
    mcp.run()
