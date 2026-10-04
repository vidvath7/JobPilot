"""Offline routing tests isolate Host ownership from transport/business logic."""

import asyncio
import pytest
from host.capabilities import (
    CapabilityCatalog, ToolCapability, ResourceCapability,
    ResourceTemplateCapability, PromptCapability,
)
from host.mcp_server_manager import MCPServerManager
from host.owned_capabilities import QualifiedCapability


class FakeClient:
    """Record original protocol names without contacting any external server."""

    def __init__(self, server_id, events, fail=False):
        self.server_id, self.events, self.fail = server_id, events, fail
        self.is_connected = False
        self.calls = []

    async def connect(self):
        self.events.append((self.server_id, "connect"))
        self.is_connected = True
        if self.fail:
            raise RuntimeError("connection failed")

    async def close(self):
        self.events.append((self.server_id, "close"))
        self.is_connected = False

    async def discover_capabilities(self):
        return CapabilityCatalog(
            (ToolCapability("search_jobs", "Search", {"type": "object"}),),
            (ResourceCapability("test://same", "context", None, "text/plain"),),
            (ResourceTemplateCapability("test://item/{id}", "item", None, None),),
            (PromptCapability("workflow", "Instructions", ()),),
        )

    async def call_tool(self, name, arguments=None):
        self.calls.append(("tool", name, arguments))
        return self.server_id

    async def read_resource(self, uri):
        self.calls.append(("resource", uri))
        return self.server_id

    async def get_prompt(self, name, arguments=None):
        self.calls.append(("prompt", name, arguments))
        return self.server_id


def setup_manager():
    events = []
    clients = [FakeClient(name, events) for name in ("jobpilot", "himalayas")]
    manager = MCPServerManager()
    for client in clients:
        manager.register(client)
    return manager, clients, events


def test_owned_discovery_and_collision_lookup():
    """All categories retain owners, including identical names and Resource URIs."""
    manager, clients, events = setup_manager()

    async def run():
        async with manager:
            catalog = await manager.discover_capabilities()
            for category in (catalog.tools, catalog.resources, catalog.resource_templates, catalog.prompts):
                assert [item.server_id for item in category] == ["jobpilot", "himalayas"]
            assert [t.qualified_name for t in catalog.find_tools_by_name("search_jobs")] == [
                "jobpilot::search_jobs", "himalayas::search_jobs",
            ]
            assert catalog.get_tool("himalayas", "search_jobs") == catalog.tools[1]
            with pytest.raises(KeyError):
                catalog.get_tool("missing", "search_jobs")
        assert not any(client.is_connected for client in clients)
        await manager.close()

    asyncio.run(run())
    assert events[-2:] == [("himalayas", "close"), ("jobpilot", "close")]


def test_qualified_routing_preserves_original_names():
    """Each dispatch reaches only its owner, with Host qualification removed."""
    manager, clients, _ = setup_manager()

    async def run():
        async with manager:
            await manager.discover_capabilities()
            for index, server_id in enumerate(("jobpilot", "himalayas")):
                before = len(clients[1-index].calls)
                assert await manager.call_tool(f"{server_id}::search_jobs", {"role": "AI"}) == server_id
                assert await manager.get_prompt(f"{server_id}::workflow", {"id": "1"}) == server_id
                assert await manager.read_resource(server_id, "test://same") == server_id
                assert len(clients[1-index].calls) == before
                assert clients[index].calls == [
                    ("tool", "search_jobs", {"role": "AI"}),
                    ("prompt", "workflow", {"id": "1"}), ("resource", "test://same"),
                ]

    asyncio.run(run())


@pytest.mark.parametrize("name", ["search_jobs", "missing::search_jobs", "jobpilot::missing", "a::b::c"])
def test_invalid_tool_routes_never_dispatch(name):
    """Ambiguous and unknown identities fail before any MCP request."""
    manager, clients, _ = setup_manager()

    async def run():
        async with manager:
            await manager.discover_capabilities()
            with pytest.raises(ValueError):
                await manager.call_tool(name)
        assert not any(client.calls for client in clients)

    asyncio.run(run())


@pytest.mark.parametrize("server_id", ["", " ", "bad::id", "jobpilot"])
def test_invalid_or_duplicate_registration(server_id):
    """Owner replacement and ambiguous server IDs are forbidden."""
    manager, _, _ = setup_manager()
    with pytest.raises(ValueError):
        manager.register(FakeClient(server_id, []))


def test_strict_partial_failure_unwinds_all_attempts():
    """A failing connection and earlier successes all receive cleanup."""
    manager, clients, events = setup_manager()
    clients[1].fail = True
    with pytest.raises(RuntimeError, match="connection failed"):
        asyncio.run(manager.connect_all())
    assert not any(c.is_connected for c in clients)
    assert events[-2:] == [("himalayas", "close"), ("jobpilot", "close")]


def test_lifecycle_guards_and_body_failure_cleanup():
    """Discovery is required before routing; caller failures unwind connections."""
    manager, clients, _ = setup_manager()

    async def run():
        with pytest.raises(RuntimeError, match="not connected"):
            await manager.discover_capabilities()
        with pytest.raises(ValueError, match="caller"):
            async with manager:
                with pytest.raises(RuntimeError, match="Discover"):
                    await manager.call_tool("jobpilot::search_jobs")
                await manager.discover_capabilities()
                with pytest.raises(ValueError, match="Unknown"):
                    await manager.get_prompt("jobpilot::missing")
                with pytest.raises(ValueError, match="Unknown server"):
                    await manager.read_resource("missing", "test://same")
                raise ValueError("caller")
        assert not any(c.is_connected for c in clients)

    asyncio.run(run())


def test_qualified_identity_round_trip():
    """Central parsing preserves both parts without renaming MCP capabilities."""
    identity = QualifiedCapability.parse("synthetic::original_name")
    assert identity.server_id == "synthetic" and identity.name == "original_name"
    assert str(identity) == "synthetic::original_name"
