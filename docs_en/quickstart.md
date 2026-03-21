# 5-Minute Quick Start

## Installation

```bash
pip install fastmind
# Or install from source
cd fastmind
pip install -e .
```

## Your First Application

```python
import asyncio
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.agent(name="hello")
async def hello_agent(state, event):
    state.setdefault("messages", [])
    state["messages"].append({
        "role": "user",
        "content": event.payload.get("text", "")
    })
    state["messages"].append({
        "role": "assistant",
        "content": f"You said: {event.payload.get('text', '')}"
    })
    return state

graph = Graph()
graph.add_node("hello", hello_agent)
graph.set_entry_point("hello")
app.register_graph("main", graph)

async def main():
    api = FastMindAPI(app)
    await api.start()
    
    await api.push_event("user1", Event(
        "message", {"text": "Hello"}, "user1"
    ))
    
    await asyncio.sleep(0.5)
    state = api.get_state("user1")
    print(state["messages"])
    
    await api.stop()

asyncio.run(main())
```

## Adding Tools

```python
app = FastMind()

@app.tool(name="get_weather")
async def get_weather(city: str):
    return f"{city} is sunny"

@app.agent(name="weather")
async def weather_agent(state, event):
    # ...
    if "weather" in event.payload.get("text", ""):
        state["tool_calls"] = [{
            "id": "1",
            "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}
        }]
    return state
```

## Run Examples

```bash
cd fastmind
PYTHONPATH=. python3 examples/simple_chat.py
```
