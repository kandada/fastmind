"""Exercise the example through the MCP transport, registry and FastMind graph."""

import asyncio
import json

import httpx
import pytest

pytest.importorskip("mcp")
from fastmind.examples import parallel_search as example


@pytest.mark.asyncio
@pytest.mark.parametrize("url, expected_tool", [(None, "web_search"),
                                               ("https://example.com", "web_fetch")])
async def test_graph_dispatch_and_transport(monkeypatch, url, expected_tool):
    requests = []
    calls = []
    original_client = httpx.AsyncClient
    schemas = {
        "web_search": {"type": "object", "properties": {"objective": {"type": "string"}},
                       "required": ["objective"]},
        "web_fetch": {"type": "object", "properties": {"urls": {"type": "array", "items": {"type": "string"}}},
                      "required": ["urls"]},
    }

    async def handle(request):
        requests.append(request)
        if request.method != "POST":
            return httpx.Response(405)
        payload = json.loads(request.content)
        if "id" not in payload:
            return httpx.Response(202)
        method = payload["method"]
        if method == "initialize":
            result = {"protocolVersion": "2025-11-25", "capabilities": {"tools": {}},
                      "serverInfo": {"name": "fixture", "version": "1"}}
        elif method == "tools/list":
            # Exercise discovery pagination as well as both schemas.
            name = "web_fetch" if payload.get("params", {}).get("cursor") else "web_search"
            result = {"tools": [{"name": name, "description": name, "inputSchema": schemas[name]}]}
            if name == "web_search":
                result["nextCursor"] = "fetch-page"
        else:
            assert method == "tools/call"
            calls.append(payload["params"])
            result = {"content": [{"type": "text", "text": "Source: https://example.com; useful excerpt"}],
                      "structuredContent": {"url": "https://example.com"}, "isError": False}
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    def client(**kwargs):
        return original_client(transport=httpx.MockTransport(handle), **kwargs)

    original_create = example.create_app

    async def check_schemas(session, conversation_id):
        app = await original_create(session, conversation_id)
        assert set(app.get_tools()) == set(schemas)
        for schema in app.get_tool_schemas():
            function = schema["function"]
            assert function["parameters"] == schemas[function["name"]]
        return app

    monkeypatch.setattr(example.httpx, "AsyncClient", client)
    monkeypatch.setattr(example, "create_app", check_schemas)
    result = json.loads(await example.run("find useful sources", url))
    assert result["structuredContent"]["url"] == "https://example.com"
    assert "useful excerpt" in result["content"][0]["text"]
    assert len(calls) == 1
    assert calls[0]["name"] == expected_tool
    assert len(calls[0]["arguments"]["session_id"]) == 32
    if url:
        assert calls[0]["arguments"]["urls"] == [url]
    else:
        assert calls[0]["arguments"]["search_queries"] == ["find useful sources"]
    assert requests
    for request in requests:
        assert str(request.url) == example.MCP_URL
        assert request.headers["User-Agent"] == example.USER_AGENT
        assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_error_and_cancellation_propagation():
    from mcp.types import CallToolResult, TextContent, Tool

    class Session:
        async def call_tool(self, name, arguments):
            assert arguments["session_id"] == "conversation"
            return CallToolResult(isError=True, content=[TextContent(type="text", text="rate limited")])

    definition = Tool(name="web_search", inputSchema={"type": "object"})
    # Use create_app's real graph and error event path.
    from mcp.types import ListToolsResult

    session = Session()
    async def list_tools(cursor=None):
        return ListToolsResult(tools=[definition, Tool(name="web_fetch", inputSchema={"type": "object"})])
    session.list_tools = list_tools
    app = await example.create_app(session, "conversation")
    with pytest.raises(RuntimeError, match="rate limited"):
        await example.execute(app, "conversation", "web_search", {})

    async def cancelled(name, arguments):
        raise asyncio.CancelledError
    session.call_tool = cancelled
    with pytest.raises(asyncio.CancelledError):
        await app.get_tool("web_search").func(objective="test")
