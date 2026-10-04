"""Generic Host boundary for an independent Streamable HTTP MCP server.

The SDK owns HTTP framing and protocol negotiation. This wrapper owns lifecycle
and reuses the existing Host catalog normalization; it adds no routing or LLM use.
"""

from contextlib import AsyncExitStack
from types import TracebackType

from mcp import ClientSession, types
from mcp.client.streamable_http import create_mcp_http_client, streamable_http_client

from host.capabilities import CapabilityCatalog, normalize_capabilities


class RemoteMCPConnectionError(RuntimeError):
    """Remote transport setup or MCP initialization failed."""


class RemoteMCPClient:
    """Own one HTTP client, transport, and initialized discovery session."""

    transport = "streamable_http"

    def __init__(self, server_id: str, url: str, *, headers: dict[str, str] | None = None) -> None:
        self.server_id = server_id
        self.url = url
        self._headers = dict(headers) if headers is not None else None
        self._stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None
        self._initialization: types.InitializeResult | None = None

    @property
    def is_connected(self) -> bool:
        """Only a completed initialization makes discovery available."""
        return self._session is not None

    @property
    def server_info(self) -> types.Implementation | None:
        """Expose the server's advertised identity separately from the configured ID."""
        return self._initialization.server_info if self._initialization else None

    async def connect(self) -> "RemoteMCPClient":
        """Open HTTP and initialize exactly once, closing partial setup on failure."""
        if self.is_connected:
            raise RuntimeError("Remote MCP client is already connected.")
        stack = AsyncExitStack()
        try:
            # Supplying a client transfers its lifecycle responsibility to us.
            # LIFO cleanup closes session, transport, then HTTP connection pool.
            http = await stack.enter_async_context(create_mcp_http_client(headers=self._headers))
            read, write = await stack.enter_async_context(streamable_http_client(self.url, http_client=http))
            session = await stack.enter_async_context(ClientSession(read, write, read_timeout_seconds=30.0))
            initialization = await session.initialize()
        except BaseException as error:
            try:
                await stack.aclose()
            finally:
                if isinstance(error, Exception):
                    # Do not print headers or provider response bodies in errors.
                    raise RemoteMCPConnectionError(
                        f"Remote MCP connection/initialization failed ({type(error).__name__})."
                    ) from error
                raise
        self._stack, self._session = stack, session
        self._initialization = initialization
        return self

    async def close(self) -> None:
        """Idempotently close the session and HTTP transport, including session DELETE."""
        stack = self._stack
        self._stack = self._session = self._initialization = None
        if stack is not None:
            await stack.aclose()

    async def __aenter__(self) -> "RemoteMCPClient":
        return await self.connect()

    async def __aexit__(self, exc_type: type[BaseException] | None,
                        exc: BaseException | None, traceback: TracebackType | None) -> None:
        await self.close()

    def _require_session(self) -> ClientSession:
        if self._session is None:
            raise RuntimeError("Remote MCP client is not connected.")
        return self._session

    async def _list(self, capability: str, operation: str, field: str) -> list:
        """Skip unadvertised categories and retain all pages of advertised metadata."""
        session = self._require_session()
        if getattr(self._initialization.capabilities, capability) is None:
            return []
        values, seen = [], set()
        cursor = None
        while True:
            params = types.PaginatedRequestParams(cursor=cursor) if cursor is not None else None
            page = await getattr(session, operation)(params=params)
            values.extend(getattr(page, field))
            cursor = page.next_cursor
            if cursor is None:
                return values
            if cursor in seen:
                raise RuntimeError("Remote MCP discovery repeated a pagination cursor.")
            seen.add(cursor)

    async def list_tools(self) -> list[types.Tool]:
        """Discover server Tools without executing them."""
        return await self._list("tools", "list_tools", "tools")

    async def list_resources(self) -> list[types.Resource]:
        """Discover static Resource metadata without reading content."""
        return await self._list("resources", "list_resources", "resources")

    async def list_resource_templates(self) -> list[types.ResourceTemplate]:
        """Discover URI templates as metadata only."""
        return await self._list("resources", "list_resource_templates", "resource_templates")

    async def list_prompts(self) -> list[types.Prompt]:
        """Discover Prompt metadata without retrieving or executing instructions."""
        return await self._list("prompts", "list_prompts", "prompts")

    async def discover_capabilities(self) -> CapabilityCatalog:
        """Normalize this server's discovery without merging catalogs or adding ownership."""
        return normalize_capabilities(
            tools=await self.list_tools(), resources=await self.list_resources(),
            resource_templates=await self.list_resource_templates(), prompts=await self.list_prompts(),
        )

    async def call_tool(self, name: str, arguments: dict[str, object] | None = None):
        """Delegate an original MCP Tool name; ownership routing belongs to the manager."""
        return await self._require_session().call_tool(name, arguments)

    async def read_resource(self, uri: str):
        """Preserve the SDK Resource result without business interpretation."""
        return await self._require_session().read_resource(uri)

    async def get_prompt(self, name: str, arguments: dict[str, str] | None = None):
        """Retrieve instructions only; this client does not execute Prompt workflows."""
        return await self._require_session().get_prompt(name, arguments)
