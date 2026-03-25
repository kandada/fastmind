# API Reference

## FastMind

Main framework class with decorator-style API.

```python
from fastmind import FastMind

app = FastMind()
```

### @app.tool

Decorator: Register a tool

```python
@app.tool(name="get_weather", description="Get weather for a city")
async def get_weather(city: str) -> str:
    return f"{city} is sunny"
```

### @app.agent

Decorator: Register an Agent

```python
@app.agent(name="chat", tools=["get_weather"], stream=False)
async def chat(state, event):
    return state
```

### @app.perception

Decorator: Register a perception loop

```python
@app.perception(interval=5.0, name="sensor")
async def sensor(app):
    while True:
        yield Event("data", {}, "system")
        await asyncio.sleep(5.0)
```

### app.get_tool_schemas()

Get OpenAI schemas for all tools

```python
schemas = app.get_tool_schemas()
```

## Graph

State graph class

```python
from fastmind import Graph

graph = Graph(name="my_graph")
```

### graph.add_node(name, node)

Add a node

```python
graph.add_node("start", start_func)
graph.add_node("subgraph", sub_graph)  # Supports subgraphs
```

### graph.add_edge(source, target, condition=None)

Add a regular edge

```python
graph.add_edge("start", "middle")
graph.add_edge("middle", "end", condition=lambda s: s.get("ready"))
```

### graph.add_conditional_edges(source, router, path_map)

Add conditional edges

```python
def router(state, event):
    return "tools" if state.get("tool_calls") else None

graph.add_conditional_edges("agent", router, {
    "tools": "tools",
    None: "__end__"
})
```

### graph.add_interrupt(name, prompt, resume_node, cancel_node=None)

Add an interrupt node

```python
graph.add_interrupt("confirm", "Confirm execution?", "yes", "no")
```

### graph.set_entry_point(name)

Set entry point

```python
graph.set_entry_point("start")
```

### graph.detect_cycles()

Detect cycles in the graph

```python
cycles = graph.detect_cycles()
if cycles:
    print(f"Cycle detected: {cycles}")
```

### graph.validate()

Validate graph configuration and emit warnings

```python
graph.validate()
# Outputs configuration warnings (if any)
```

### graph.iteration_limit

Set maximum iterations to prevent infinite loops (default 999)

```python
graph.iteration_limit = 100
```

## Session

Session class

```python
session = api.get_session("user1")
```

### session.state

Get session state

```python
state = session.state
```

### session.session_state

Get session state constant (STATE_CREATED, STATE_RUNNING, STATE_IDLE, STATE_INTERRUPTED, STATE_STOPPED)

```python
if session.session_state == STATE_RUNNING:
    print("Session running")
```

### session.is_running

Check if session is running

```python
if session.is_running:
    print("running")
```

### session.is_alive

Check if session is alive (not stopped)

```python
if session.is_alive:
    print("alive")
```

### session.wait_for_output(timeout=None)

Wait for output event

```python
await session.wait_for_output(timeout=5.0)
```

## State API

State convenience methods

```python
state = session.state
```

### state.add_message(role, content)

Add message (if message doesn't exist)

```python
state.add_message("user", "Hello")
```

### state.add_message_if_new(role, content)

Add message only if it doesn't exist

```python
state.add_message_if_new("user", "Hello")
```

### state.get_last_message(role=None)

Get last message

```python
last = state.get_last_message()
last_user = state.get_last_message("user")
```

### state.pop_messages(role=None)

Pop all messages

```python
all = state.pop_messages()
user_msgs = state.pop_messages("user")
```

### state.get_message_count(role=None)

Get message count

```python
count = state.get_message_count()
user_count = state.get_message_count("user")
```

## FastMindAPI

External API interface

```python
from fastmind.contrib import FastMindAPI

api = FastMindAPI(app)
```

### api.start()

Start engine and perception loops

```python
await api.start()
```

### api.stop()

Stop engine and perception loops

```python
await api.stop()
```

### api.push_event(session_id, event, graph_name="main")

Push external event

```python
await api.push_event("user1", Event("message", {"text": "hi"}, "user1"))
```

### api.get_state(session_id)

Get state snapshot

```python
state = api.get_state("user1")
```

### api.stream_events(session_id, event_types=None)

Stream session events (no polling)

```python
async for ev in api.stream_events("user1"):
    if ev.type == "stream.chunk":
        print(ev.payload.get("delta", ""), end="", flush=True)
    elif ev.type == "stream.end":
        print()
```

### api.run_streaming(session_id, user_input, on_chunk=None, on_end=None)

Convenience method: Run streaming conversation

```python
full_text = await api.run_streaming(
    "user1",
    "Hello!",
    on_chunk=lambda delta: print(delta, end="", flush=True),
)
```

### api.resume_session(session_id, user_input)

Resume interrupted session

```python
await api.resume_session("user1", "confirm")
```

### api.list_sessions()

List all sessions

```python
sessions = api.list_sessions()
```

### api.delete_session(session_id)

Delete session

```python
await api.delete_session("user1")
```
