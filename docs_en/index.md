# FastMind Framework

A lightweight multi-agent framework for embodied AI systems.

## Core Features

- **Lightweight**: Core code under 8000 lines, no LangChain/LangGraph dependencies
- **Event-Driven**: Asyncio-based architecture, supports thousands of concurrent sessions
- **Embodied Native**: Built-in perception loops for sensors, timers, and external triggers
- **Human-in-the-loop**: Supports human confirmation and interruption
- **Developer-Friendly**: FastAPI-like decorator pattern

## Quick Start

```python
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.agent(name="chat")
async def chat(state, event):
    state["reply"] = f"Received: {event.payload.get('text')}"
    return state

graph = Graph()
graph.add_node("chat", chat)
graph.set_entry_point("chat")
app.register_graph("main", graph)

async def main():
    api = FastMindAPI(app)
    await api.start()
    await api.push_event("user1", Event("msg", {"text": "Hello"}, "user1"))
    # ...
    await api.stop()
```

## Directory Structure

```
fastmind/
├── core/           # Core modules
│   ├── app.py      # FastMind main class
│   ├── graph.py    # Graph class
│   ├── event.py    # Event class
│   ├── state.py    # State class
│   ├── tool.py     # Tool system
│   ├── node.py     # Agent nodes
│   ├── engine.py   # Execution engine
│   └── perception.py  # Perception loops
├── contrib/        # Extension modules
│   └── api.py      # FastMindAPI
├── examples/       # Example code
└── tests/         # Test code
```

## Examples

```bash
cd fastmind
PYTHONPATH=. python3 examples/simple_chat.py
```

## Documentation

- [Quick Start](quickstart.md)
- [Core Concepts](core_concepts.md)
- [API Reference](api_reference.md)
- [Examples](examples.md)
- Advanced Guides
  - [Streaming](advanced/streaming.md)
  - [Human-in-the-Loop](advanced/human_in_loop.md)
  - [Perception Loop](advanced/perception_loop.md)
  - [Subgraph](advanced/subgraph.md)
- Best Practices
  - [Tool Design](best_practices/tool_design.md)
  - [State Design](best_practices/state_design.md)
