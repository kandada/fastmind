# 核心概念

## State

State 是会话的全局数据快照，本质是一个 dict。所有节点共享和更新这个状态。

```python
state = {}

@app.agent(name="chat")
async def chat(state, event):
    # 自由定义 key
    state.setdefault("messages", [])
    state["messages"].append(event.payload["text"])
    return state
```

常用约定（可选）：
- `messages`: 对话历史
- `tool_calls`: 待执行的工具调用
- `tool_results`: 工具执行结果
- `metadata`: 元数据

## Event

Event 是驱动图执行的触发器。

```python
from fastmind import Event

event = Event(
    type="user.message",      # 事件类型
    payload={"text": "你好"}, # 事件数据
    session_id="user_001"    # 会话 ID
)
```

常用事件类型：
- `user.message`: 用户消息
- `stream.chunk`: 流式输出片段
- `stream.end`: 流式输出结束
- `interrupt`: 中断事件
- `sensor.data`: 传感器数据

## Node

Node 是一个异步函数，接收 (state, event)，返回更新后的 state。

```python
async def my_node(state: dict, event: Event) -> dict:
    state["processed"] = True
    return state

# 支持返回 output_events
async def my_streaming_node(state, event):
    output_events = [Event("chunk", {"delta": "..."}, event.session_id)]
    return state, output_events
```

## Edge

Edge 定义节点之间的执行顺序。

```python
graph.add_edge("start", "middle")      # 无条件跳转
graph.add_edge("middle", "end", condition=lambda s: s.get("ready"))
```

## ConditionalEdge

ConditionalEdge 根据 state 内容决定下一个节点。

```python
def router(state, event):
    if state.get("need_tool"):
        return "tools"
    return None  # 结束

graph.add_conditional_edges("agent", router, {
    "tools": "tools",
    None: "__end__"
})
```

## Graph

Graph 是节点和边的集合，代表一个完整的工作流。

```python
graph = Graph()
graph.add_node("start", start_node)
graph.add_node("end", end_node)
graph.add_edge("start", "end")
graph.set_entry_point("start")
app.register_graph("main", graph)
```

## Session

每个 session_id 拥有独立的 State 实例、事件队列和执行上下文。

```python
api = FastMindAPI(app)
await api.push_event("user1", Event("msg", {}, "user1"))
state = api.get_state("user1")
```
