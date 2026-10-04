"""Public A1 discovery configuration; no credentials or business logic."""

from host.remote_mcp_client import RemoteMCPClient

HIMALAYAS_SERVER_ID = "himalayas"
HIMALAYAS_URL = "https://mcp.himalayas.app/mcp"


def create_himalayas_client() -> RemoteMCPClient:
    """Connectable configuration only; constructing the client performs no I/O."""
    return RemoteMCPClient(server_id=HIMALAYAS_SERVER_ID, url=HIMALAYAS_URL)
