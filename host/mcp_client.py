"""Host-side lifecycle wrapper for the JobPilot MCP server.

This module is the first Host layer: it knows how to launch and discover the MCP
server, but it contains no LLM integration, capability selection, or workflow
orchestration. It communicates exclusively through MCP rather than importing
server-side capability implementations.
"""

import sys
from contextlib import AsyncExitStack
from pathlib import Path
from types import TracebackType

from mcp import ClientSession, StdioServerParameters, types
from mcp.client.stdio import stdio_client


class JobPilotMCPClient:
    """Own one initialized MCP session and its local stdio server subprocess."""

    def __init__(
        self,
        project_root: str | Path | None = None,
        python_executable: str | Path | None = None,
    ) -> None:
        """Configure repository-relative server launch details without connecting."""
        self._project_root = (
            Path(project_root)
            if project_root is not None
            else Path(__file__).resolve().parents[1]
        )
        # Using the running Host interpreter keeps server launch inside the same
        # project environment and avoids depending on a global ``mcp`` command.
        self._python_executable = str(python_executable or sys.executable)
        self._exit_stack: AsyncExitStack | None = None
        self._session: ClientSession | None = None

    @property
    def is_connected(self) -> bool:
        """Report whether discovery operations currently have an initialized session."""
        return self._session is not None

    async def connect(self) -> "JobPilotMCPClient":
        """Launch the server, establish one session, and initialize MCP once."""
        if self.is_connected:
            raise RuntimeError("JobPilot MCP client is already connected.")

        exit_stack = AsyncExitStack()
        server_parameters = StdioServerParameters(
            command=self._python_executable,
            args=["-m", "server.main"],
            cwd=self._project_root,
        )

        try:
            # The exit stack closes resources in reverse order: ClientSession
            # first, followed by stdio streams and their owned server subprocess.
            read_stream, write_stream = await exit_stack.enter_async_context(
                stdio_client(server_parameters)
            )
            session = await exit_stack.enter_async_context(
                ClientSession(
                    read_stream,
                    write_stream,
                    read_timeout_seconds=10.0,
                )
            )
            # Initialization negotiates MCP capabilities once. All later Host
            # discovery calls reuse this same session and subprocess.
            await session.initialize()
        except BaseException:
            await exit_stack.aclose()
            raise

        self._exit_stack = exit_stack
        self._session = session
        return self

    async def close(self) -> None:
        """Close the session, stdio streams, and subprocess; repeated calls are safe."""
        exit_stack = self._exit_stack
        # Clear visible connection state before awaiting cleanup so callers cannot
        # start another protocol request while shutdown is in progress.
        self._session = None
        self._exit_stack = None
        if exit_stack is not None:
            await exit_stack.aclose()

    async def __aenter__(self) -> "JobPilotMCPClient":
        """Connect for a bounded Host operation scope."""
        return await self.connect()

    async def __aexit__(
        self,
        exc_type: type[BaseException] | None,
        exc_value: BaseException | None,
        traceback: TracebackType | None,
    ) -> None:
        """Guarantee transport cleanup whether Host work succeeds or fails."""
        await self.close()

    async def list_tools(self) -> list[types.Tool]:
        """Discover Tools through MCP without transforming them for an LLM."""
        return (await self._require_session().list_tools()).tools

    async def list_resources(self) -> list[types.Resource]:
        """Discover static Resources through the initialized MCP session."""
        return (await self._require_session().list_resources()).resources

    async def list_resource_templates(self) -> list[types.ResourceTemplate]:
        """Discover parameterized Resources while preserving their URI metadata."""
        result = await self._require_session().list_resource_templates()
        return result.resource_templates

    async def list_prompts(self) -> list[types.Prompt]:
        """Discover Prompts and arguments for future Host presentation."""
        return (await self._require_session().list_prompts()).prompts

    def _require_session(self) -> ClientSession:
        """Fail clearly when Host code attempts discovery outside its lifecycle."""
        if self._session is None:
            raise RuntimeError("JobPilot MCP client is not connected.")
        return self._session
