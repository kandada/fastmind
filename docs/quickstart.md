# 5 分钟快速入门

## 安装

```bash
pip install fastmind
# 或从源码安装
cd fastmind
pip install -e .
```

## 第一个应用

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
        "content": f"你说了: {event.payload.get('text', '')}"
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
        "message", {"text": "你好"}, "user1"
    ))
    
    await asyncio.sleep(0.5)
    state = api.get_state("user1")
    print(state["messages"])
    
    await api.stop()

asyncio.run(main())
```

## 添加工具

```python
app = FastMind()

@app.tool(name="get_weather")
async def get_weather(city: str):
    return f"{city} 天气晴朗"

@app.agent(name="weather")
async def weather_agent(state, event):
    # ...
    if "天气" in event.payload.get("text", ""):
        state["tool_calls"] = [{
            "id": "1",
            "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}
        }]
    return state
```

## 运行示例

```bash
cd fastmind
PYTHONPATH=. python3 examples/simple_chat.py
```
