# FastMind 🧠

A lightweight, event-driven multi-agent framework for embodied AI systems.

[![PyPI version](https://badge.fury.io/py/fastmind.svg)](https://badge.fury.io/py/fastmind)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Features

- **FastAPI-like Decorators**: Familiar `@app.agent`, `@app.tool`, `@app.perception` syntax, easy to learn
- **State Graph Architecture**: Build agent workflows like flowcharts, not nested loops
- **Event-Driven**: Asyncio-based, zero polling, high-performance async execution
- **Built-in Streaming**: Real-time streaming output with backpressure control
- **Human-in-the-Loop**: Interrupt and resume sessions for human approval
- **Perception Loops**: Native support for sensors, timers, and external triggers
- **Tool Calling**: ReAct-style agent-tool-agent loops
- **Session Isolation**: Multi-user support with isolated session state
- **Lightweight**: ~8000 lines, no big dependencies

## Installation

### From PyPI (Recommended)

```bash
pip install fastmind
```

### From GitHub

```bash
pip install git+https://github.com/kandada/fastmind.git
```

With examples:

```bash
pip install git+https://github.com/kandada/fastmind.git#egg=fastmind[examples]
```

For development:

```bash
git clone https://github.com/kandada/fastmind.git
cd fastmind
pip install -e ".[all]"
```

## Quick Start

```python
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.agent(name="chat_agent")
async def chat_agent(state: dict, event: Event) -> dict:
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": event.payload.get("text", "")})
    # Your LLM call here
    state["messages"].append({"role": "assistant", "content": "Hello!"})
    return state

graph = Graph()
graph.add_node("chat", chat_agent)
graph.set_entry_point("chat")
app.register_graph("main", graph)

async def main():
    api = FastMindAPI(app)
    await api.start()
    await api.push_event("user_001", Event("user.message", {"text": "Hello!"}, "user_001"))
    await api.stop()

import asyncio
asyncio.run(main())
```

## Core Concepts

### State

A dict-like container for session data shared across nodes:

```python
state["messages"].append({"role": "user", "content": "Hello"})
```

### Node

An async function that processes events and returns updated state:

```python
async def my_node(state: dict, event: Event) -> dict:
    state["processed"] = True
    return state
```

### Graph

A collection of nodes and edges defining your workflow:

```python
graph = Graph()
graph.add_node("agent", chat_agent)
graph.add_edge("agent", "tool_node")
graph.set_entry_point("agent")
```

### Event

External or internal triggers that drive graph execution:

```python
event = Event(type="user.message", payload={"text": "Hello"}, session_id="user_001")
```

## Streaming Output

Real-time streaming with zero polling:

```python
@app.agent(name="chat_agent", stream=True)
async def chat_agent(state: dict, event: Event) -> dict:
    output_queue = state["_output_queue"]
    session_id = state["_session_id"]
    
    async def stream_llm():
        for chunk in llm_stream():
            for char in chunk:
                output_queue.put_nowait(Event(
                    type="stream.chunk",
                    payload={"delta": char},
                    session_id=session_id
                ))
                await asyncio.sleep(0.03)
        output_queue.put_nowait(Event(type="stream.end", payload={}, session_id=session_id))
    
    asyncio.create_task(stream_llm())
    return state
```

## Human-in-the-Loop

Interrupt and resume for human approval:

```python
@app.agent(name="order_agent")
async def order_agent(state: dict, event: Event) -> dict:
    state.setdefault("orders", [])
    amount = event.payload.get("amount", 0)
    state["orders"].append({"amount": amount, "status": "pending"})
    if amount > 1000:
        state["need_approval"] = True
    return state

async def approve_node(state: dict, event: Event) -> tuple[dict, list[Event]]:
    return state, [Event(
        type="interrupt",
        payload={"prompt": "Approve this transaction?", "resume_node": "confirm"},
        session_id=event.session_id
    )]

async def confirm_node(state: dict, event: Event) -> dict:
    if state.get("orders"):
        state["orders"][-1]["status"] = "confirmed"
    return state

async def reject_node(state: dict, event: Event) -> dict:
    if state.get("orders"):
        state["orders"][-1]["status"] = "rejected"
    return state

graph = Graph()
graph.add_node("order", order_agent)
graph.add_node("approve", approve_node)
graph.add_node("confirm", confirm_node)
graph.add_node("reject", reject_node)

graph.add_edge("order", "approve", condition=lambda s: s.get("need_approval"))
graph.add_edge("approve", "confirm")
graph.add_edge("approve", "reject")
graph.set_entry_point("order")

app.register_graph("main", graph)
```

Handle the interrupt in your application:

```python
async def main():
    api = FastMindAPI(app)
    await api.start()
    
    event = Event("user.message", {"amount": 2000}, "user_001")
    await api.push_event("user_001", event)
    
    async for ev in api.stream_events("user_001"):
        if ev.type == "interrupt":
            print(f"Interrupt: {ev.payload['prompt']}")
            await api.resume_session("user_001", "confirm")  # or "reject"
```

## Perception Loop

React to sensors, timers, and external events:

```python
@app.perception(interval=5.0, name="sensor_monitor")
async def sensor_monitor(app: FastMind):
    while True:
        data = await read_sensor()
        yield Event(type="sensor.data", payload=data, session_id="system")
        await asyncio.sleep(5.0)
```

## Tool Calling (ReAct)

```python
@app.tool(name="get_weather", description="Get weather")
async def get_weather(city: str) -> str:
    return f"{city} is sunny"

from fastmind import ToolNode

tool_node = ToolNode(app.get_tools())

def has_tool_calls(state: dict, event: Event) -> str:
    return "tools" if state.get("tool_calls") else None

graph.add_conditional_edges("agent", has_tool_calls, {None: "__end__"})
graph.add_edge("tools", "agent")
```

## Examples

| Example | Description |
|---------|-------------|
| [simple_chat.py](examples/simple_chat.py) | Basic chat agent |
| [simple_chat_with_tool.py](examples/simple_chat_with_tool.py) | Agent with tool calling (ReAct) |
| [streaming_chat.py](examples/streaming_chat.py) | Real-time streaming output |
| [human_in_loop.py](examples/human_in_loop.py) | Human approval workflow |
| [perception_loop.py](examples/perception_loop.py) | Sensor data processing |
| [drone.py](examples/drone.py) | Timer-based perception |
| [companion_bot.py](examples/companion_bot.py) | Multi-agent conversation |
| [humanoid_robot.py](examples/humanoid_robot.py) | Multi-tool collaboration |
| [sleep_assessment.py](examples/sleep_assessment.py) | Multi-state HITL flow |
| [comprehensive_assistant.py](examples/comprehensive_assistant.py) | Full-featured assistant |

Run an example:

```bash
python -m fastmind.examples.simple_chat
```

## API Reference

### FastMindAPI

```python
api = FastMindAPI(app)

await api.start()                    # Start engine and perception loops
await api.push_event(session_id, event)  # Push event to session
async for ev in api.stream_events(session_id):  # Stream output events
    print(ev)
await api.stop()                     # Stop engine
```

### Session

```python
session = api.get_session(session_id)
state = session.state                 # Get session state
await session.wait_for_output(timeout=5.0)  # Wait for output event
```

## Architecture

```
┌─────────────────────────────────────────────────────────┐
│                      FastMindAPI                         │
│  ┌─────────────────┐    ┌────────────────────────────┐  │
│  │ PerceptionLoop   │───▶│        Engine              │  │
│  │ Scheduler       │    │  ┌──────────────────────┐  │  │
│  └─────────────────┘    │  │ Session (per user)   │  │  │
│                         │  │  ├─ State           │  │  │
│                         │  │  ├─ Event Queue     │  │  │
│                         │  │  └─ Output Queue    │  │  │
│                         │  └──────────────────────┘  │  │
│                         └────────────────────────────┘  │
└─────────────────────────────────────────────────────────┘
```

## Testing

```bash
pip install fastmind[dev]
pytest tests/ -v
```

## Changelog

### v0.1.3
- **Bug Fix**: Fixed `stream_events` timeout issue when agent returns no output events
- **Improvement**: Enhanced debug logging in engine for better observability
- **Improvement**: Added `_has_conditional_edges()` helper method to Graph class
- **Tests**: Added comprehensive test suite for ReAct loops and node execution protection

### v0.1.2
- Initial release

## License

GPL-3.0 License - see [LICENSE](LICENSE) for details.

## Acknowledgments

Inspired by [LangGraph](https://github.com/langchain-ai/langgraph) for the state graph architecture design.

## Author

[xiefujin](https://github.com/kandada)

## Links

- [Documentation](https://fastmind.ai/docs)
- [GitHub](https://github.com/kandada/fastmind)
- [PyPI](https://pypi.org/project/fastmind/)
