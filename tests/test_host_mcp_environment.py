"""Real stdio regressions for parent inheritance and child override precedence.

A launch guard fails before spawning if the regression returns, so tests cannot
accidentally write to production while exercising the actual SDK and server.
"""

import asyncio
import json
import os
from contextlib import asynccontextmanager
from pathlib import Path

import pytest
from mcp import types

import host.mcp_client as client_module
from host.mcp_client import JobPilotMCPClient


PRODUCTION_STORE = Path(__file__).resolve().parents[1] / "data" / "applications.json"
STORE_VARIABLE = "JOBPILOT_APPLICATIONS_PATH"


@pytest.mark.parametrize("explicit_override", [False, True], ids=["parent-inheritance", "override-precedence"])
def test_real_stdio_store_environment(tmp_path, monkeypatch, explicit_override):
    """Production-style construction inherits A; explicit overrides select B."""
    path_a, path_b = tmp_path / "a.json", tmp_path / "b.json"
    for path in (path_a, path_b):
        path.write_text("[]", encoding="utf-8")
    monkeypatch.setenv(STORE_VARIABLE, str(path_a))
    before = PRODUCTION_STORE.read_bytes()
    assert json.loads(before) == []
    target = path_b if explicit_override else path_a
    original_stdio = client_module.stdio_client

    @asynccontextmanager
    async def guarded_stdio(parameters):
        # Observe, never modify, the real client's launch parameters.
        assert parameters.env is not None
        assert parameters.env.get(STORE_VARIABLE) == str(target)
        async with original_stdio(parameters) as streams:
            yield streams

    monkeypatch.setattr(client_module, "stdio_client", guarded_stdio)

    async def run():
        client = (
            JobPilotMCPClient(server_environment={STORE_VARIABLE: str(path_b)})
            if explicit_override else JobPilotMCPClient()
        )
        async with client:
            result = await client.call_tool("save_application", {"job_id": "JOB-005"})
            assert isinstance(result, types.CallToolResult)
            assert result.is_error is False
        assert not client.is_connected

    try:
        asyncio.run(run())
        records = json.loads(target.read_text(encoding="utf-8"))
        assert len(records) == 1
        assert records[0]["application_id"] == "APP-001"
        assert records[0]["job_id"] == "JOB-005"
        untouched = path_a if explicit_override else path_b
        assert json.loads(untouched.read_text(encoding="utf-8")) == []
        assert os.environ[STORE_VARIABLE] == str(path_a)
    finally:
        assert PRODUCTION_STORE.read_bytes() == before


def test_overrides_preserve_runtime_environment_without_mutating_parent(monkeypatch):
    """Check ordinary variables and late configuration without inspecting secrets."""
    monkeypatch.setenv("JOBPILOT_TEST_RUNTIME_SETTING", "parent-value")
    monkeypatch.setenv(STORE_VARIABLE, "parent-store")
    overrides = {STORE_VARIABLE: "child-store"}
    client = JobPilotMCPClient(server_environment=overrides)
    monkeypatch.setenv("JOBPILOT_TEST_LATE_SETTING", "late-value")
    before = dict(os.environ)

    class LaunchInspected(Exception):
        """End this component test after inspecting parameters, before launch."""

    @asynccontextmanager
    async def inspect_stdio(parameters):
        assert parameters.env["JOBPILOT_TEST_RUNTIME_SETTING"] == "parent-value"
        assert parameters.env["JOBPILOT_TEST_LATE_SETTING"] == "late-value"
        for name in ("PATH", "SYSTEMROOT", "TEMP"):
            if name in os.environ:
                assert parameters.env[name] == os.environ[name]
        assert parameters.env[STORE_VARIABLE] == "child-store"
        assert parameters.env is not os.environ
        raise LaunchInspected
        yield  # Async context-manager shape; deliberately never reached.

    monkeypatch.setattr(client_module, "stdio_client", inspect_stdio)
    with pytest.raises(LaunchInspected):
        asyncio.run(client.connect())
    assert dict(os.environ) == before
    assert overrides == {STORE_VARIABLE: "child-store"}
