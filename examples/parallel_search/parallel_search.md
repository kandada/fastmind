# Optional third-party Parallel Search MCP example

This optional third-party example uses a remote service operated by Parallel,
with separate dependencies and tests. It connects to the anonymous [Parallel Search MCP](https://docs.parallel.ai/integrations/mcp/search-mcp)
at `https://search.parallel.ai/mcp` using Streamable HTTP. It discovers `web_search` and `web_fetch`, preserves their
input schemas, and registers them with `FastMind.register_tool`. A FastMind event
runs a request → `ToolNode` → response graph and prints the MCP result, including
source URLs and excerpts. The routing is deterministic; no LLM is used.

From a clean checkout, install the optional dependencies in a virtual environment
(Python 3.10 or later). Run the module from the directory containing the `fastmind`
checkout, as with the other FastMind examples:

```bash
git clone https://github.com/kandada/fastmind.git
python -m venv .venv
source .venv/bin/activate
pip install -r fastmind/examples/requirements-mcp.txt
python -m fastmind.examples.parallel_search "Python asyncio task cancellation"
python -m fastmind.examples.parallel_search "Explain task cancellation" --url https://docs.python.org/3/library/asyncio-task.html
```

No Parallel API key, LLM key, or `.env` file is needed. Anonymous access is free
for exploration and light use, with lower server-enforced rate limits than
account-authenticated access. The [Search MCP documentation](https://docs.parallel.ai/integrations/mcp/search-mcp)
does not specify a fixed anonymous request quota. The random `session_id` is used
for free-tier rate limiting and log correlation; do not change it to bypass limits.
Rate-limit errors cause a nonzero exit and are not retried. Model inference would
be a separate dependency and cost if you adapt this to an LLM agent.

The example transmits your query as both `objective` and `search_queries` for
search, or your requested URL and objective for fetch, to Parallel. It also sends
one random conversation `session_id` per run, the
`FastMind/<version> ParallelSearchExample` User-Agent, and MCP protocol/client
metadata for initialization and discovery. Parallel receives the connection's
source IP address. The example does not load credentials or `.env`, send a model
name, or upload local files or the graph's full state. Avoid sensitive queries or
URLs, including URLs containing secrets; anonymous access does not imply zero
data retention. See [Parallel's privacy policy](https://parallel.ai/privacy-policy).

Calls use a 30-second HTTP timeout, a 60-second MCP response timeout, and a
90-second overall CLI timeout; errors cause a nonzero exit.

The isolated example tests use a local mock transport and need no network or
credentials. They live under `examples/tests/`; the repository's default pytest
collection targets only core `tests/`, even if MCP is installed. Run them explicitly
from the directory containing the checkout after installing the optional dependencies:

```bash
pip install pytest pytest-asyncio
python -m pytest fastmind/examples/tests/test_parallel_search.py -q
```

To adapt the discovered tools to your own graph, keep the MCP session open for
the entire lifetime of the `ToolNode`. `create_app` demonstrates registration
before constructing the node, and `app.get_tool_schemas()` exposes the live
OpenAI-compatible schemas. `make_tool` supplies the stable conversation ID on
each call and preserves the full MCP response as JSON in `tool_results`. Existing
tools and examples are unaffected because these dependencies and registration
are opt-in.
