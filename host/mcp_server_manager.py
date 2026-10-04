"""Explicit multi-server routing, independent of LLM orchestration and approvals."""

from contextlib import AsyncExitStack
from typing import Protocol

from host.capabilities import CapabilityCatalog
from host.owned_capabilities import (
    MultiServerCapabilityCatalog, OwnedToolCapability, OwnedResourceCapability,
    OwnedResourceTemplateCapability, OwnedPromptCapability, QualifiedCapability,
    validate_server_id,
)


class MCPConnection(Protocol):
    """Structural boundary shared by local and remote clients, not their transports."""

    server_id: str

    @property
    def is_connected(self) -> bool: ...
    async def connect(self) -> object: ...
    async def close(self) -> None: ...
    async def discover_capabilities(self) -> CapabilityCatalog: ...
    async def call_tool(self, name: str, arguments: dict[str, object] | None = None) -> object: ...
    async def read_resource(self, uri: str) -> object: ...
    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None) -> object: ...


class MCPServerManager:
    """Own registered connections with strict all-or-nothing startup.

    Connect, use, and close in the same async task: SDK transport contexts own
    task-local cancellation scopes. Reverse-order cleanup preserves their nesting.
    Routing is explicit low-level execution, not an automatic execution policy.
    """

    def __init__(self) -> None:
        self._clients: dict[str, MCPConnection] = {}
        self._catalogs: dict[str, CapabilityCatalog] = {}
        self._stack: AsyncExitStack | None = None

    def register(self, client: MCPConnection) -> None:
        """Register disconnected clients before startup; never replace an owner."""
        validate_server_id(client.server_id)
        if client.server_id in self._clients:
            raise ValueError(f"Duplicate server ID: {client.server_id}")
        if self._stack is not None or client.is_connected:
            raise RuntimeError("Register disconnected clients before manager startup.")
        self._clients[client.server_id] = client

    async def connect_all(self) -> None:
        """Strict policy only: any failure unwinds every attempted connection."""
        if self._stack is not None:
            raise RuntimeError("Manager is already connected.")
        stack = AsyncExitStack()
        try:
            for client in self._clients.values():
                # Register cleanup first so even a partially opened failing client
                # is closed. ExitStack still runs remaining callbacks on failure.
                stack.push_async_callback(client.close)
                await client.connect()
        except BaseException:
            await stack.aclose()
            raise
        self._stack = stack

    async def close(self) -> None:
        """Invalidate discovery and close all owned connections in reverse order."""
        stack, self._stack = self._stack, None
        self._catalogs = {}
        if stack is not None:
            await stack.aclose()

    async def __aenter__(self) -> "MCPServerManager":
        await self.connect_all()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def discover_capabilities(self) -> MultiServerCapabilityCatalog:
        """Publish only a complete discovery snapshot, retaining every owner."""
        if self._stack is None:
            raise RuntimeError("Manager is not connected.")
        self._catalogs = {}
        catalogs = {}
        for server_id, client in self._clients.items():
            catalogs[server_id] = await client.discover_capabilities()
        self._catalogs = catalogs
        return MultiServerCapabilityCatalog(
            tools=tuple(OwnedToolCapability(s, c) for s, cat in catalogs.items() for c in cat.tools),
            resources=tuple(OwnedResourceCapability(s, c) for s, cat in catalogs.items() for c in cat.resources),
            resource_templates=tuple(OwnedResourceTemplateCapability(s, c) for s, cat in catalogs.items() for c in cat.resource_templates),
            prompts=tuple(OwnedPromptCapability(s, c) for s, cat in catalogs.items() for c in cat.prompts),
        )

    def _owner(self, server_id: str) -> MCPConnection:
        if server_id not in self._clients:
            raise ValueError(f"Unknown server: {server_id}")
        if self._stack is None or not self._clients[server_id].is_connected:
            raise RuntimeError("Manager is not connected.")
        if server_id not in self._catalogs:
            raise RuntimeError("Discover capabilities before routing requests.")
        return self._clients[server_id]

    def _named_owner(self, value: str, category: str) -> tuple[MCPConnection, str]:
        identity = QualifiedCapability.parse(value)
        client = self._owner(identity.server_id)
        if not any(c.name == identity.name for c in getattr(self._catalogs[identity.server_id], category)):
            raise ValueError(f"Unknown {category} capability: {value}")
        return client, identity.name

    async def call_tool(self, qualified_name: str, arguments: dict[str, object] | None = None):
        """Strip Host qualification and send the original Tool name only."""
        client, name = self._named_owner(qualified_name, "tools")
        return await client.call_tool(name, arguments)

    async def get_prompt(self, qualified_name: str, arguments: dict[str, str] | None = None):
        """Route instructions retrieval without starting any Prompt workflow."""
        client, name = self._named_owner(qualified_name, "prompts")
        return await client.get_prompt(name, arguments)

    async def read_resource(self, server_id: str, uri: str):
        """Use explicit ownership; the selected server validates concrete Resource URIs.

        URI-template expansion is deliberately not reimplemented in this registry.
        This generic API does not apply the separate LLM Resource bridge policy.
        """
        return await self._owner(server_id).read_resource(uri)
