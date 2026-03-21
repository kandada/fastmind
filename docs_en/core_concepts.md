# Core Concepts

## State

State is the global data snapshot of a session, essentially a dict. All nodes share and update this state.

```python
state = {}

@app.agent(name="chat")
async def chat(state, event):
    # Freely define keys
    state.setdefault("messages", [])
    state["messages"].append(event.payload["text"])
    return state
```

Common conventions (optional):
- `messages`: Conversation history
- `tool_calls`: Pending tool calls
- `tool_results`: Tool execution results
- `metadata`: Metadata

## Event

Event is the trigger that drives graph execution.

```python
from fastmind import Event

event = Event(
    type="user.message",      # Event type
    payload={"text": "Hello"}, # Event data
    session_id="user_001"    # Session ID
)
```

Common event types:
- `user.message`: User message
- `stream.chunk`: Streaming output chunk
- `stream.end`: Streaming output end
- `interrupt`: Interrupt event
- `sensor.data`: Sensor data

## Node

Node is an async function that receives (state, event) and returns the updated state.

```python
async def my_node(state: dict, event: Event) -> dict:
    state["processed"] = True
    return state

# Supports returning output_events
async def my_streaming_node(state, event):
    output_events = [Event("chunk", {"delta": "..."}, event.session_id)]
    return state, output_events
```

## Edge

Edge defines the execution order between nodes.

```python
graph.add_edge("start", "middle")      # Unconditional jump
graph.add_edge("middle", "end", condition=lambda s: s.get("ready"))
```

## ConditionalEdge

ConditionalEdge determines the next node based on state content.

```python
def router(state, event):
    if state.get("need_tool"):
        return "tools"
    return None  # End

graph.add_conditional_edges("agent", router, {
    "tools": "tools",
    None: "__end__"
})
```

## Graph

Graph is a collection of nodes and edges, representing a complete workflow.

```python
graph = Graph()
graph.add_node("start", start_node)
graph.add_node("end", end_node)
graph.add_edge("start", "end")
graph.set_entry_point("start")
app.register_graph("main", graph)
```

## Session

Each session_id has its own State instance, event queue, and execution context.

```python
api = FastMindAPI(app)
await api.push_event("user1", Event("msg", {}, "user1"))
state = api.get_state("user1")
```
