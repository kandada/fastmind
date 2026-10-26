"""Exercise the deed.page example through the MCP transport, registry and graph."""

import json

import httpx
import pytest

pytest.importorskip("mcp")
from fastmind.examples.deed_page import deed_page as example

PUBLISH_SCHEMA = {"type": "object", "properties": {"files": {"type": "object"},
                                                    "ttl_seconds": {"type": "integer"}},
                  "required": ["files"]}
DELETE_SCHEMA = {"type": "object", "properties": {"slug": {"type": "string"}},
                 "required": ["slug"]}


def mock_server(monkeypatch, tool_result):
    """Serve MCP JSON-RPC from httpx.MockTransport and record requests."""
    requests, calls = [], []
    original_client = httpx.AsyncClient

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
            # deed.page advertises more tools; only the two used here are registered.
            result = {"tools": [
                {"name": "publish_site", "inputSchema": PUBLISH_SCHEMA},
                {"name": "delete_site", "inputSchema": DELETE_SCHEMA},
                {"name": "list_sites", "inputSchema": {"type": "object"}},
            ]}
        else:
            assert method == "tools/call"
            calls.append(payload["params"])
            result = tool_result
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": payload["id"], "result": result})

    def client(**kwargs):
        return original_client(transport=httpx.MockTransport(handle), **kwargs)

    monkeypatch.setattr(example.httpx, "AsyncClient", client)
    return requests, calls


def text(payload, is_error=False):
    return {"content": [{"type": "text", "text": json.dumps(payload)}], "isError": is_error}


@pytest.mark.asyncio
async def test_publish_through_graph_without_token(monkeypatch):
    published = {"ok": True, "slug": "calm-mesa-1a2b",
                 "url": "https://calm-mesa-1a2b.deed.page/", "token": "new-owner-token"}
    requests, calls = mock_server(monkeypatch, text(published))
    arguments = {"files": {"index.html": example.SAMPLE_PAGE}, "ttl_seconds": 3600}
    result = await example.run("publish_site", arguments)
    assert result == published
    assert calls == [{"name": "publish_site", "arguments": arguments}]
    for request in requests:
        assert str(request.url) == example.MCP_URL
        assert request.headers["User-Agent"] == example.USER_AGENT
        assert "Authorization" not in request.headers


@pytest.mark.asyncio
async def test_delete_sends_owner_token_as_bearer(monkeypatch):
    requests, calls = mock_server(monkeypatch, text({"ok": True, "deleted": "calm-mesa-1a2b"}))
    result = await example.run("delete_site", {"slug": "calm-mesa-1a2b"}, token="owner-token")
    assert result["deleted"] == "calm-mesa-1a2b"
    assert calls == [{"name": "delete_site", "arguments": {"slug": "calm-mesa-1a2b"}}]
    assert all(r.headers["Authorization"] == "Bearer owner-token" for r in requests)


@pytest.mark.asyncio
async def test_tool_error_is_raised_plainly(monkeypatch):
    error = text({"code": "not_found", "message": "No site at gone-site."}, is_error=True)
    mock_server(monkeypatch, error)
    with pytest.raises(RuntimeError, match="not_found"):
        await example.run("delete_site", {"slug": "gone-site"}, token="owner-token")


def test_token_is_saved_private_and_env_wins(monkeypatch, tmp_path):
    monkeypatch.setattr(example, "TOKEN_FILE", tmp_path / "deed-page" / "token")
    monkeypatch.delenv("DEEDPAGE_TOKEN", raising=False)
    assert example.load_token() is None
    example.save_token("saved-token")
    assert example.TOKEN_FILE.stat().st_mode & 0o777 == 0o600
    assert example.TOKEN_FILE.parent.stat().st_mode & 0o777 == 0o700
    assert example.load_token() == "saved-token"
    monkeypatch.setenv("DEEDPAGE_TOKEN", "env-token")
    assert example.load_token() == "env-token"
