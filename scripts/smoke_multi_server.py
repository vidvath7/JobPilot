"""Manual two-transport discovery; never executes Tools or contacts an LLM."""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from host.himalayas import create_himalayas_client
from host.mcp_client import JobPilotMCPClient
from host.mcp_server_manager import MCPServerManager


async def run() -> None:
    """Show real ownership collisions with strict lifecycle cleanup on either failure."""
    clients = (JobPilotMCPClient(), create_himalayas_client())
    manager = MCPServerManager()
    for client in clients:
        manager.register(client)
    async with manager:
        catalog = await manager.discover_capabilities()
        for client in clients:
            counts = {category: sum(item.server_id == client.server_id for item in getattr(catalog, category))
                      for category in ("tools", "resources", "resource_templates", "prompts")}
            print(f"{client.server_id}: {counts}", flush=True)
        print("search_jobs owners:")
        for tool in catalog.find_tools_by_name("search_jobs"):
            print(f"- {tool.server_id}")
        print("Qualified examples:")
        for tool in catalog.tools:
            if tool.capability.name in {"search_jobs", "score_job_match", "get_job_details"}:
                print(f"- {tool.qualified_name}")
    print("Both servers disconnected. No Tools executed.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as error:
        # Provider errors may contain URLs/headers; report only the exception type.
        print(f"Multi-server discovery failed ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(1) from None
