"""Exercise real SDK HTTP framing and ClientSession against an in-memory endpoint.

Only HTTP I/O is replaced: no live Himalayas service, sockets, or credentials are
needed. This catches transport tuple/API and initialization mistakes that a fake
ClientSession would conceal.
"""

import asyncio
import json

import httpx2
import pytest

import host.remote_mcp_client as remote
from host.himalayas import create_himalayas_client


def endpoint(monkeypatch, *, capabilities=None, fail=False):
    """Install a deterministic Streamable HTTP responder and retain lifecycle evidence."""
    calls, clients = [], []
    advertised = capabilities if capabilities is not None else {"tools": {}, "resources": {}, "prompts": {}}

    async def handle(request):
        if fail:
            return httpx2.Response(503, text="unavailable")
        if request.method != "POST":
            calls.append(request.method)
            return httpx2.Response(204 if request.method == "DELETE" else 405)
        body = json.loads(request.content)
        method = body["method"]
        calls.append(method)
        if "id" not in body:
            return httpx2.Response(202)
        if method == "initialize":
            result = {
                "protocolVersion": body["params"]["protocolVersion"],
                "capabilities": advertised,
                "serverInfo": {"name": "controlled-server", "version": "1.0"},
            }
        elif method == "tools/list":
            second = body.get("params", {}).get("cursor") == "page2"
            result = {"tools": [{"name": "second" if second else "first", "description": "Test Tool",
                                 "inputSchema": {"type": "object", "properties": {}}}]}
            if not second:
                result["nextCursor"] = "page2"
        elif method == "resources/list":
            result = {"resources": [{"uri": "test://context", "name": "context", "mimeType": "application/json"}]}
        elif method == "resources/templates/list":
            result = {"resourceTemplates": [{"uriTemplate": "test://item/{id}", "name": "item"}]}
        elif method == "prompts/list":
            result = {"prompts": [{"name": "workflow", "arguments": [{"name": "id", "required": True}]}]}
        elif method == "tools/call":
            result = {"content": [], "structuredContent": body["params"], "isError": False}
        elif method == "resources/read":
            result = {"contents": [{"uri": body["params"]["uri"], "text": "context"}]}
        elif method == "prompts/get":
            result = {"messages": [{"role": "user", "content": {"type": "text", "text": json.dumps(body["params"])}}]}
        else:
            raise AssertionError(f"Unexpected protocol operation: {method}")
        return httpx2.Response(200, json={"jsonrpc": "2.0", "id": body["id"], "result": result},
                              headers={"mcp-session-id": "test-session"})

    def factory(*, headers):
        http = httpx2.AsyncClient(transport=httpx2.MockTransport(handle), headers=headers)
        clients.append(http)
        return http

    monkeypatch.setattr(remote, "create_mcp_http_client", factory)
    return calls, clients


def test_real_sdk_transport_lifecycle_and_all_discovery(monkeypatch):
    """Real initialization, paginated discovery, identity, headers, and session termination."""
    calls, clients = endpoint(monkeypatch)
    client = remote.RemoteMCPClient("synthetic", "https://test.invalid/mcp", headers={"X-Test": "value"})

    async def run():
        assert not client.is_connected
        async with client:
            assert client.is_connected
            assert client.server_info.name == "controlled-server"
            with pytest.raises(RuntimeError, match="already connected"):
                await client.connect()
            catalog = await client.discover_capabilities()
            assert [item.name for item in catalog.tools] == ["first", "second"]
            assert catalog.tools[0].description == "Test Tool"
            assert catalog.resources[0].uri == "test://context"
            assert catalog.resources[0].mime_type == "application/json"
            assert catalog.resource_templates[0].uri_template == "test://item/{id}"
            assert catalog.prompts[0].arguments[0].required is True
        assert not client.is_connected and client.server_info is None
        await client.close()

    asyncio.run(run())
    assert client.server_id == "synthetic" and client.url == "https://test.invalid/mcp"
    assert calls.count("initialize") == 1
    assert "notifications/initialized" in calls and "DELETE" in calls
    assert clients[0].headers["X-Test"] == "value"
    assert clients[0].is_closed


def test_unadvertised_categories_are_empty(monkeypatch):
    """Tools-only servers need not implement Resource or Prompt listing endpoints."""
    calls, _ = endpoint(monkeypatch, capabilities={"tools": {}})

    async def run():
        async with remote.RemoteMCPClient("test", "https://test.invalid/mcp") as client:
            catalog = await client.discover_capabilities()
            assert catalog.resources == catalog.resource_templates == catalog.prompts == ()

    asyncio.run(run())
    assert "resources/list" not in calls and "prompts/list" not in calls


def test_disconnected_operations_fail():
    """Every entry point enforces lifecycle even before capability checks."""
    client = remote.RemoteMCPClient("test", "https://test.invalid/mcp")

    async def run():
        for method in (client.list_tools, client.list_resources, client.list_resource_templates,
                       client.list_prompts, client.discover_capabilities):
            with pytest.raises(RuntimeError, match="not connected"):
                await method()

    asyncio.run(run())


def test_failure_closes_partial_connection(monkeypatch):
    """HTTP failure during initialization produces a clear Host error and closes the pool."""
    _, clients = endpoint(monkeypatch, fail=True)
    client = remote.RemoteMCPClient("test", "https://test.invalid/mcp")
    with pytest.raises(remote.RemoteMCPConnectionError, match="connection/initialization failed"):
        asyncio.run(client.connect())
    assert not client.is_connected and clients[0].is_closed


def test_context_body_error_still_closes_transport(monkeypatch):
    """Caller failures unwind the MCP session before closing the HTTP client."""
    calls, clients = endpoint(monkeypatch)

    async def run():
        async with remote.RemoteMCPClient("test", "https://test.invalid/mcp"):
            raise ValueError("caller failed")

    with pytest.raises(ValueError, match="caller failed"):
        asyncio.run(run())
    assert "DELETE" in calls and clients[0].is_closed


def test_himalayas_configuration_is_public_streamable_http(monkeypatch):
    """The factory changes only identity and endpoint; it does not configure auth."""
    _, clients = endpoint(monkeypatch)
    client = create_himalayas_client()
    assert client.server_id == "himalayas"
    assert client.url == "https://mcp.himalayas.app/mcp"
    assert client.transport == "streamable_http"
    assert not client.is_connected

    async def run():
        async with client:
            assert "authorization" not in clients[0].headers

    asyncio.run(run())


def test_generic_execution_through_real_sdk_transport(monkeypatch):
    """Exercise execution framing against controlled HTTP, never Himalayas Tools."""
    endpoint(monkeypatch)
    client = remote.RemoteMCPClient("synthetic", "https://test.invalid/mcp")

    async def run():
        for operation, args in ((client.call_tool, ("first",)),
                                (client.read_resource, ("test://context",)),
                                (client.get_prompt, ("workflow",))):
            with pytest.raises(RuntimeError, match="not connected"):
                await operation(*args)
        async with client:
            result = await client.call_tool("first", {"value": 1})
            assert result.structured_content["name"] == "first"
            assert result.structured_content["arguments"] == {"value": 1}
            resource = await client.read_resource("test://context")
            assert resource.contents[0].text == "context"
            prompt = await client.get_prompt("workflow", {"id": "5"})
            params = json.loads(prompt.messages[0].content.text)
            assert params["name"] == "workflow" and params["arguments"] == {"id": "5"}

    asyncio.run(run())
