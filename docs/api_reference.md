# API 参考

## FastMind

框架主类，提供装饰器风格的 API。

```python
from fastmind import FastMind

app = FastMind()
```

### @app.tool

装饰器：注册工具

```python
@app.tool(name="get_weather", description="获取城市天气")
async def get_weather(city: str) -> str:
    return f"{city} 天气晴朗"
```

### @app.agent

装饰器：注册 Agent

```python
@app.agent(name="chat", tools=["get_weather"], stream=False)
async def chat(state, event):
    return state
```

### @app.perception

装饰器：注册感知循环

```python
@app.perception(interval=5.0, name="sensor")
async def sensor(app):
    while True:
        yield Event("data", {}, "system")
        await asyncio.sleep(5.0)
```

### app.get_tool_schemas()

获取所有工具的 OpenAI schema

```python
schemas = app.get_tool_schemas()
```

## Graph

状态图类

```python
from fastmind import Graph

graph = Graph(name="my_graph")
```

### graph.add_node(name, node)

添加节点

```python
graph.add_node("start", start_func)
graph.add_node("subgraph", sub_graph)  # 支持子图
```

### graph.add_edge(source, target, condition=None)

添加普通边

```python
graph.add_edge("start", "middle")
graph.add_edge("middle", "end", condition=lambda s: s.get("ready"))
```

### graph.add_conditional_edges(source, router, path_map)

添加条件边

```python
def router(state, event):
    return "tools" if state.get("tool_calls") else None

graph.add_conditional_edges("agent", router, {
    "tools": "tools",
    None: "__end__"
})
```

### graph.add_interrupt(name, prompt, resume_node, cancel_node=None)

添加中断节点

```python
graph.add_interrupt("confirm", "确认执行？", "yes", "no")
```

### graph.set_entry_point(name)

设置入口点

```python
graph.set_entry_point("start")
```

## FastMindAPI

对外 API 接口

```python
from fastmind.contrib import FastMindAPI

api = FastMindAPI(app)
```

### api.start()

启动引擎和感知循环

```python
await api.start()
```

### api.stop()

停止引擎和感知循环

```python
await api.stop()
```

### api.push_event(session_id, event, graph_name="main")

推送外部事件

```python
await api.push_event("user1", Event("message", {"text": "hi"}, "user1"))
```

### api.get_state(session_id)

获取状态快照

```python
state = api.get_state("user1")
```

### api.stream_events(session_id, event_types=None)

流式获取会话事件（无轮询）

```python
async for ev in api.stream_events("user1"):
    if ev.type == "stream.chunk":
        print(ev.payload.get("delta", ""), end="", flush=True)
    elif ev.type == "stream.end":
        print()
```

### api.run_streaming(session_id, user_input, on_chunk=None, on_end=None)

便捷方法：运行流式对话

```python
full_text = await api.run_streaming(
    "user1",
    "Hello!",
    on_chunk=lambda delta: print(delta, end="", flush=True),
)
```

### api.resume_session(session_id, user_input)

恢复中断会话

```python
await api.resume_session("user1", "confirm")
```

### api.list_sessions()

列出所有会话

```python
sessions = api.list_sessions()
```

### api.delete_session(session_id)

删除会话

```python
await api.delete_session("user1")
```
