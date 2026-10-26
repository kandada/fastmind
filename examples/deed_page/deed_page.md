# Optional third-party deed.page MCP example

This optional third-party example uses a remote service operated by deed.page,
with separate dependencies and tests. It was contributed by the deed.page team
(Avatar 8 LLC); the same server is also packaged as an MIT agent skill at
[f-o-x11/deed-page-skill](https://github.com/f-o-x11/deed-page-skill).

[deed.page](https://deed.page/docs) publishes static HTML to a public
`https://<slug>.deed.page/` URL without an account or API key. The example
connects to `https://deed.page/mcp` using Streamable HTTP, discovers
`publish_site` and `delete_site`, keeps their input schemas, and registers them
with `FastMind.register_tool`. A FastMind event runs a request → `ToolNode` →
response graph and prints the result. Routing is deterministic; no LLM is used.

From a clean checkout, install the optional dependencies in a virtual environment
(Python 3.10 or later). Run the module from the directory containing the
`fastmind` checkout, as with the other FastMind examples:

```bash
git clone https://github.com/kandada/fastmind.git
python -m venv .venv
source .venv/bin/activate
pip install -r fastmind/examples/deed_page/requirements-mcp.txt
python -m fastmind.examples.deed_page.deed_page publish              # sample page, expires in 1 hour
python -m fastmind.examples.deed_page.deed_page publish report.html --slug my-report-1234 --ttl 0
python -m fastmind.examples.deed_page.deed_page delete my-report-1234
```

`--ttl` sets `ttl_seconds` (default 3600); `--ttl 0` keeps the page. No API key,
LLM key, or `.env` file is needed.

The first publish returns an owner token, which is needed to update or delete
the page. The example saves it to `~/.config/deed-page/token` (directory 700,
file 600) and never prints it; set `DEEDPAGE_TOKEN` to use a token from
elsewhere instead. Later runs send it as an `Authorization: Bearer` header, so
updates and deletes stay with the same owner. Keep the token private.

Published pages are public: do not publish secrets or private data. The example
sends the HTML you publish, the slug, the `FastMind/<version> DeedPageExample`
User-Agent and MCP client metadata to deed.page, which also receives the
connection's source IP address. Anonymous pages show a small deed.page badge;
see the [docs](https://deed.page/docs) for limits.

Calls use a 30-second HTTP timeout, a 60-second MCP response timeout, and a
90-second overall CLI timeout. Tool errors (for example deleting a page that no
longer exists) print the server's message and exit with status 1.

The isolated tests use `httpx.MockTransport` and need no network or credentials.
They live under `examples/tests/` and are not part of the default core `tests/`
collection. Run them explicitly from the directory containing the checkout:

```bash
pip install pytest pytest-asyncio
python -m pytest fastmind/examples/tests/test_deed_page.py -q
```

To use the tools in your own graph, keep the MCP session open for the lifetime
of the `ToolNode`. `create_app` shows registration before the node is built, and
`app.get_tool_schemas()` exposes the live OpenAI-compatible schemas, ready for an
LLM-driven graph.
