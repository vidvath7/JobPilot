"""Manual remote MCP discovery only: no Tool calls, LLM requests, or data writes."""

import asyncio
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from host.himalayas import create_himalayas_client


async def run() -> None:
    """Observe public remote metadata through the real Streamable HTTP lifecycle."""
    async with create_himalayas_client() as client:
        print(f"Server: {client.server_id}\nURL: {client.url}\nTransport: {client.transport}", flush=True)
        if client.server_info:
            print(f"Advertised identity: {client.server_info.name} {client.server_info.version}")
        catalog = await client.discover_capabilities()
        for label, items, identity in (
            ("Tools", catalog.tools, "name"),
            ("Resources", catalog.resources, "uri"),
            ("Resource Templates", catalog.resource_templates, "uri_template"),
            ("Prompts", catalog.prompts, "name"),
        ):
            print(f"\n{label} ({len(items)}):")
            for item in items:
                print(f"- {getattr(item, identity)}: {item.description or '(no description)'}")
    print("Disconnected cleanly.")


if __name__ == "__main__":
    try:
        asyncio.run(run())
    except Exception as error:
        print(f"Remote discovery failed ({type(error).__name__}).", file=sys.stderr)
        raise SystemExit(1) from None
