"""Publish and delete deed.page sites through FastMind's tool registry and graph.

See examples/deed_page/deed_page.md for installation and usage. This
deterministic example dispatches MCP tools without an LLM and reads no .env.
"""

import argparse
import asyncio
import json
import os
import uuid
from datetime import timedelta
from pathlib import Path

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from fastmind import Event, FastMind, Graph, Tool, ToolNode, __version__
from fastmind.contrib import FastMindAPI

MCP_URL = "https://deed.page/mcp"
USER_AGENT = f"FastMind/{__version__} DeedPageExample"
TOOLS = ("publish_site", "delete_site")
TOKEN_FILE = Path.home() / ".config" / "deed-page" / "token"
SAMPLE_PAGE = (
    "<!doctype html><html><head><meta charset=\"utf-8\"><title>FastMind</title>"
    "</head><body><h1>Published from a FastMind graph</h1></body></html>"
)


def load_token():
    token = os.environ.get("DEEDPAGE_TOKEN", "").strip()
    if not token and TOKEN_FILE.exists():
        token = TOKEN_FILE.read_text().strip()
    return token or None


def save_token(token):
    TOKEN_FILE.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    TOKEN_FILE.touch(mode=0o600, exist_ok=True)
    TOKEN_FILE.write_text(token + "\n")


def make_tool(session, definition):
    """Keep the server's input schema and return its JSON text result."""
    async def call(**arguments):
        result = await session.call_tool(definition.name, arguments=arguments)
        text = "".join(c.text for c in result.content if getattr(c, "text", None))
        if result.isError:
            raise RuntimeError(text or result.model_dump_json(by_alias=True))
        return text

    return Tool(
        name=definition.name,
        description=definition.description or "",
        func=call,
        schema={
            "type": "function",
            "function": {
                "name": definition.name,
                "description": definition.description or "",
                "parameters": definition.inputSchema,
            },
        },
    )


async def create_app(session):
    """Discover publish_site and delete_site and register them via public APIs."""
    app = FastMind()
    cursor = None
    while True:
        page = await session.list_tools(cursor=cursor)
        for definition in page.tools:
            if definition.name in TOOLS:
                app.register_tool(definition.name, make_tool(session, definition))
        cursor = page.nextCursor
        if not cursor:
            break
    if set(app.get_tools()) != set(TOOLS):
        raise RuntimeError("deed.page MCP did not advertise publish_site and delete_site")

    async def request(state, event):
        state["tool_calls"] = [{
            "id": "deed_page_call",
            "function": {
                "name": event.payload["tool"],
                "arguments": json.dumps(event.payload["arguments"]),
            },
        }]
        return state

    async def respond(state, event):
        result = state["tool_results"][0]["result"]
        if result.startswith("Error executing tool:"):
            return state, [Event("error", {"error": result}, event.session_id)]
        return state, [Event("stream.chunk", {"delta": result}, event.session_id),
                       Event("stream.end", {}, event.session_id)]

    graph = Graph()
    graph.add_node("request", request)
    graph.add_node("tools", ToolNode(app.get_tools()))
    graph.add_node("respond", respond)
    graph.add_edge("request", "tools")
    graph.add_edge("tools", "respond")
    graph.set_entry_point("request")
    app.register_graph("main", graph)
    return app


async def execute(app, conversation_id, tool, arguments):
    """Push a real FastMind event and collect the graph's tool output."""
    api = FastMindAPI(app)
    await api.start()
    try:
        await api.push_event(
            conversation_id,
            Event("user.message", {"tool": tool, "arguments": arguments}, conversation_id),
        )
        async for event in api.stream_events(conversation_id):
            if event.type == "error":
                raise RuntimeError(event.payload["error"])
            if event.type == "stream.chunk":
                return event.payload["delta"]
        raise RuntimeError("The graph ended without a tool result")
    finally:
        await api.stop()


async def run(tool, arguments, token=None):
    """Call one deed.page tool through the graph and return its parsed JSON."""
    headers = {"User-Agent": USER_AGENT}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    async with httpx.AsyncClient(headers=headers, timeout=30.0) as client:
        async with streamable_http_client(MCP_URL, http_client=client) as streams:
            async with ClientSession(
                streams[0], streams[1], read_timeout_seconds=timedelta(seconds=60),
            ) as session:
                await session.initialize()
                app = await create_app(session)
                try:
                    result = await execute(app, uuid.uuid4().hex, tool, arguments)
                except RuntimeError as exc:
                    # Raise after the transport closes, so the caller gets the
                    # tool error itself rather than an anyio ExceptionGroup.
                    error = exc
                else:
                    return json.loads(result)
    raise error


def main():
    parser = argparse.ArgumentParser(description="Publish or delete a deed.page site.")
    sub = parser.add_subparsers(dest="command", required=True)
    publish = sub.add_parser("publish", help="Publish an HTML file as index.html")
    publish.add_argument("file", nargs="?", help="HTML file (default: a sample page)")
    publish.add_argument("--slug", help="Subdomain to publish to (optional)")
    publish.add_argument("--ttl", type=int, default=3600,
                         help="Seconds until the page expires; 0 keeps it")
    delete = sub.add_parser("delete", help="Delete a page you own")
    delete.add_argument("slug")
    args = parser.parse_args()

    token = load_token()
    if args.command == "delete":
        if not token:
            parser.error("no owner token; set DEEDPAGE_TOKEN or publish first")
        tool, arguments = "delete_site", {"slug": args.slug}
    else:
        html = Path(args.file).read_text() if args.file else SAMPLE_PAGE
        arguments = {"files": {"index.html": html}}
        if args.slug:
            arguments["slug"] = args.slug
        if args.ttl:
            arguments["ttl_seconds"] = args.ttl
        tool = "publish_site"

    try:
        result = asyncio.run(asyncio.wait_for(run(tool, arguments, token), timeout=90))
    except RuntimeError as exc:
        parser.exit(1, f"{exc}\n")
    if tool == "delete_site":
        print(f"Deleted {result['deleted']}")
        return
    if result.get("token"):
        save_token(result["token"])
        print(f"Owner token saved to {TOKEN_FILE}")
    print(result["url"])
    if result.get("expires_at"):
        print(f"Expires {result['expires_at']}")


if __name__ == "__main__":
    main()
