"""Keyless Parallel Search MCP through FastMind's tool registry and graph.

See examples/parallel_search.md for installation and usage. This deterministic
example dispatches tools without an LLM; it does not load credentials or .env.
"""

import argparse
import asyncio
from datetime import timedelta
import json
import uuid

import httpx
from mcp import ClientSession
from mcp.client.streamable_http import streamable_http_client

from fastmind import Event, FastMind, Graph, Tool, ToolNode, __version__
from fastmind.contrib import FastMindAPI


MCP_URL = "https://search.parallel.ai/mcp"
USER_AGENT = f"FastMind/{__version__} ParallelSearchExample"


def make_tool(session, definition, conversation_id):
    """Keep the server's input schema and structured/text result intact."""
    async def call(**arguments):
        arguments["session_id"] = conversation_id
        result = await session.call_tool(definition.name, arguments=arguments)
        if result.isError:
            raise RuntimeError(result.model_dump_json(by_alias=True))
        return result.model_dump_json(by_alias=True, exclude_none=True)

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


async def create_app(session, conversation_id):
    """Discover the two remote tools and register them through public APIs."""
    app = FastMind()
    cursor = None
    while True:
        page = await session.list_tools(cursor=cursor)
        for definition in page.tools:
            if definition.name in ("web_search", "web_fetch"):
                app.register_tool(
                    definition.name, make_tool(session, definition, conversation_id)
                )
        cursor = page.nextCursor
        if not cursor:
            break
    if set(app.get_tools()) != {"web_search", "web_fetch"}:
        raise RuntimeError("Parallel MCP did not advertise web_search and web_fetch")

    async def request(state, event):
        state["tool_calls"] = [{
            "id": "parallel_call",
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


async def run(query, url=None):
    conversation_id = uuid.uuid4().hex
    # This fresh client sends no Authorization header and reads no saved settings.
    async with httpx.AsyncClient(
        headers={"User-Agent": USER_AGENT}, timeout=30.0, follow_redirects=True,
    ) as client:
        async with streamable_http_client(MCP_URL, http_client=client) as streams:
            async with ClientSession(
                streams[0], streams[1], read_timeout_seconds=timedelta(seconds=60),
            ) as session:
                await session.initialize()
                app = await create_app(session, conversation_id)
                tool = "web_fetch" if url else "web_search"
                arguments = {"urls": [url], "objective": query} if url else {
                    "objective": query, "search_queries": [query],
                }
                return await execute(app, conversation_id, tool, arguments)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("query", help="Search query or fetch objective")
    parser.add_argument("--url", help="Fetch this URL instead of searching")
    args = parser.parse_args()

    async def bounded_run():
        return await asyncio.wait_for(run(args.query, args.url), timeout=90)

    print(asyncio.run(bounded_run()))


if __name__ == "__main__":
    main()
