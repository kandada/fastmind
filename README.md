# FastMind 🧠

A lightweight, event-driven framework for building embodied AI agents with dual-loop architecture (LLM + VLA).

[![PyPI version](https://badge.fury.io/py/fastmind.svg)](https://badge.fury.io/py/fastmind)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## Features

- **Dual-Loop Architecture**: LLM slow loop (planning, reasoning) + VLA fast loop (real-time control) run concurrently in one session
- **Signal & Event**: Two parallel communication primitives — `Event` for discrete messages, `Signal` for continuous high-frequency data
- **FastAPI-like Decorators**: Familiar `@app.agent`, `@app.tool`, `@app.vla`, `@app.vla_action`, `@app.signal` syntax
- **State Graph**: Build agent workflows like flowcharts with nodes, edges, and conditional routing
- **Event-Driven**: Asyncio-based, zero polling, high-performance async execution
- **Human-in-the-Loop**: Interrupt and resume sessions for human approval
- **Perception Loops**: Native support for sensors, timers, and external triggers
- **Action Channel Routing**: VLA outputs map to multiple action executors via channel names (N:M)
- **Session Isolation**: Multi-user support with isolated state per session
- **Lightweight**: ~8000 lines core, no big dependencies

## Installation

```bash
pip install fastmind
```

## Quick Start

### LLM Agent

```python
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.agent(name="chat_agent")
async def chat_agent(state: dict, event: Event) -> dict:
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": event.payload.get("text", "")})
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

### LLM Agent with Tool Calling (ReAct)

```python
from fastmind import FastMind, Graph, Event, ToolNode, Tool
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.tool(name="get_weather", description="Get weather for a city")
async def get_weather(city: str) -> str:
    return f"{city} is sunny, 20°C"

async def chat_agent(state: dict, event: Event) -> dict:
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": event.payload.get("text", "")})
    # Simulate LLM deciding to call a tool
    if "weather" in event.payload.get("text", "").lower():
        state["tool_calls"] = [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}}
        ]
    else:
        state["messages"].append({"role": "assistant", "content": "I can check weather for you!"})
    return state

tool_node = ToolNode(app.get_tools())

def has_tool_calls(state: dict, event: Event) -> str:
    return "tools" if state.get("tool_calls") else None

graph = Graph()
graph.add_node("agent", chat_agent)
graph.add_node("tools", tool_node)
graph.add_conditional_edges("agent", has_tool_calls, {"tools": "tools", None: "__end__"})
graph.add_edge("tools", "agent")
graph.set_entry_point("agent")
app.register_graph("main", graph)
```

### VLA Agent (NPC Control)

```python
from fastmind import FastMind, Graph, Event, ActionSpace
from fastmind.contrib import FastMindAPI

app = FastMind()

# High-frequency sensor signal (30fps, zero-copy)
@app.signal(name="vision", interval=1/30)
async def npc_vision():
    return {"frame_id": 1, "objects": []}

# VLA fast loop (30Hz, time-driven, bypasses graph)
@app.vla(name="navigation", frequency=30.0)
async def navigation_vla(state, signal_bus):
    vision = signal_bus.read("vision")
    goal = state.get("llm", {}).get("goal", "idle")
    action = [0.5, 0.0, 0.0]  # mock: move forward
    return {"body": action}

# Action executor receives routed action vector
@app.vla_action(name="body", action_space=ActionSpace(3))
async def body_executor(action):
    await game_engine.move(action[0], action[1], action[2])

# LLM slow loop (event-driven)
@app.agent(name="npc_brain")
async def npc_brain(state, event):
    if event.type == "user.message":
        state.setdefault("llm", {})["goal"] = "go_to_castle"
    return state

graph = Graph()
graph.add_node("brain", npc_brain)
graph.set_entry_point("brain")
app.register_graph("main", graph)
```

## Core Concepts

| Concept | Description |
|---------|-------------|
| **State** | Per-session dict shared across all loops |
| **Event** | Discrete messages (user input, LLM response) — queued, push-based |
| **Signal** | Continuous data (camera frames, joint angles) — last-value cache, pull-based |
| **Graph** | LLM workflow topology (nodes + edges) |
| **@app.agent** | LLM reasoning node, event-driven |
| **@app.vla** | VLA inference node, time-driven, runs on its own scheduler |
| **@app.vla_action** | Action executor, receives VLA output via channel name |
| **@app.signal** | High-frequency sensor source, writes to SignalBus |
| **@app.perception** | Low-frequency sensor source, yields Events (existing) |
| **Action Channel** | Named bus that routes VLA output to executors (N:M mapping) |

## Architecture

```
Session
├── SignalBus                     ← high-frequency data (zero-copy)
├── LLM Task (_run)               ← slow loop, event-driven
│   └── Graph: Agent → Tool → ...
├── VLA Task (_vla_scheduler)     ← fast loop, time-driven
│   ├── @app.vla inference
│   ├── Action Channel routing
│   └── @app.vla_action execution
└── State (Blackboard)
    ├── llm/: goal, plan, messages
    └── vla/: actions, status, memory
```

## Examples

| Example | Description |
|---------|-------------|
| [simple_chat.py](examples/simple_chat.py) | Basic chat |
| [simple_chat_with_tool.py](examples/simple_chat_with_tool.py) | Tool calling (ReAct) |
| [streaming_chat.py](examples/streaming_chat.py) | Real-time streaming |
| [human_in_loop.py](examples/human_in_loop.py) | Human approval workflow |
| [perception_loop.py](examples/perception_loop.py) | Sensor processing |
| [drone.py](examples/drone.py) | Timer-based perception |
| [companion_bot.py](examples/companion_bot.py) | Multi-agent conversation |
| [humanoid_robot.py](examples/humanoid_robot.py) | Multi-tool robot control |
| [sleep_assessment.py](examples/sleep_assessment.py) | Multi-state HITL flow |
| [comprehensive_assistant.py](examples/comprehensive_assistant.py) | Full-featured assistant |
| [npc_vla.py](examples/npc_vla.py) | **VLA dual-loop NPC (new)** |

```bash
python -m fastmind.examples.npc_vla
```

## API Reference

### FastMindAPI

```python
api = FastMindAPI(app)

await api.start()
await api.push_event(session_id, event)
async for ev in api.stream_events(session_id): ...

# New VLA/Signal API methods:
frame = api.read_signal(session_id, "vision")      # read signal
api.write_signal(session_id, "gps", data)           # write signal
signals = api.list_signals(session_id)              # list signals
api.pause_vla(session_id)                            # pause VLA loop
api.resume_vla(session_id)                           # resume VLA loop

await api.stop()
```

## Changelog

### v0.2.0
- **Major**: New VLA dual-loop architecture — `@app.vla` for high-frequency inference (time-driven), `@app.vla_action` for action execution via channel routing, `@app.signal` for zero-copy sensor data (parallel to Event)
- **Major**: Dual-loop Session — VLA fast loop runs concurrently with LLM slow loop, communicates via shared State (Blackboard pattern), N:M action channel mapping
- **New Feature**: `FastMindAPI.read_signal()` / `write_signal()` / `list_signals()` / `pause_vla()` / `resume_vla()`
- **Bug Fix**: Fixed `_save_checkpoint` crash on unpicklable state objects (now uses `_safe_deepcopy` with graceful fallback)
- **Bug Fix**: Fixed `human_in_loop` checkpoint pickle error
- **Bug Fix**: Fixed VLA action executor error isolation (one executor crash no longer blocks other actions in same tick)
- **Reliability**: Added 20 VLA stress/reliability tests (long-running, error recovery, concurrent access, multi-session, pause/resume cycles, override cycles)

## License

GPL-3.0 License — see [LICENSE](LICENSE) for details.

## Author

[xiefujin](https://github.com/kandada) email:490021684@qq.com
