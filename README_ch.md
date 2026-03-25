# FastMind 🧠

轻量级事件驱动的具身智能多 Agent 系统框架。

[![PyPI 版本](https://badge.fury.io/py/fastmind.svg)](https://badge.fury.io/py/fastmind)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![许可证: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## 核心特性

- **类 FastAPI 装饰器**：熟悉的 `@app.agent`、`@app.tool`、`@app.perception` 语法，极易上手
- **状态图架构**：像画流程图一样构建智能体工作流，无需编写复杂嵌套循环
- **事件驱动**：基于 asyncio，无需轮询，高性能异步执行
- **内置流式输出**：实时流式输出，支持背压控制
- **Human-in-the-Loop**：支持中断和恢复，可进行人工审批
- **感知循环**：原生支持传感器、定时器和外部触发器
- **工具调用**：ReAct 风格的 Agent → Tool → Agent 循环
- **会话隔离**：多用户支持，每个会话状态独立
- **轻量级**：约 8000 行代码，不依赖其他框架

## 安装

### 从 PyPI 安装（推荐）

```bash
pip install fastmind
```

### 从 GitHub 安装

```bash
pip install git+https://github.com/kandada/fastmind.git
```

安装包含示例的版本：

```bash
pip install git+https://github.com/kandada/fastmind.git#egg=fastmind[examples]
```

本地开发安装：

```bash
git clone https://github.com/kandada/fastmind.git
cd fastmind
pip install -e ".[all]"
```

## 快速开始

```python
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.agent(name="chat_agent")
async def chat_agent(state: dict, event: Event) -> dict:
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": event.payload.get("text", "")})
    # 在这里调用你的 LLM
    state["messages"].append({"role": "assistant", "content": "你好！"})
    return state

graph = Graph()
graph.add_node("chat", chat_agent)
graph.set_entry_point("chat")
app.register_graph("main", graph)

async def main():
    api = FastMindAPI(app)
    await api.start()
    await api.push_event("user_001", Event("user.message", {"text": "你好！"}, "user_001"))
    await api.stop()

import asyncio
asyncio.run(main())
```

## 核心概念

### State（状态）

会话数据的容器，类似 dict，节点间共享：

```python
state["messages"].append({"role": "user", "content": "你好"})
```

### Node（节点）

处理事件并返回更新状态的异步函数：

```python
async def my_node(state: dict, event: Event) -> dict:
    state["processed"] = True
    return state
```

### Graph（图）

定义工作流的节点和边的集合：

```python
graph = Graph()
graph.add_node("agent", chat_agent)
graph.add_edge("agent", "tool_node")
graph.set_entry_point("agent")
```

### Event（事件）

驱动图执行的外部或内部触发器：

```python
event = Event(type="user.message", payload={"text": "你好"}, session_id="user_001")
```

## 流式输出

零轮询的实时流式输出：

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

## Human-in-the-Loop（人工介入）

中断和恢复以进行人工审批：

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
        payload={"prompt": "是否批准此交易？", "resume_node": "confirm"},
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

在应用层处理中断：

```python
async def main():
    api = FastMindAPI(app)
    await api.start()
    
    event = Event("user.message", {"amount": 2000}, "user_001")
    await api.push_event("user_001", event)
    
    async for ev in api.stream_events("user_001"):
        if ev.type == "interrupt":
            print(f"中断: {ev.payload['prompt']}")
            await api.resume_session("user_001", "confirm")  # 或 "reject"
```

## 感知循环

响应传感器、定时器和外部事件：

```python
@app.perception(interval=5.0, name="sensor_monitor")
async def sensor_monitor(app: FastMind):
    while True:
        data = await read_sensor()
        yield Event(type="sensor.data", payload=data, session_id="system")
        await asyncio.sleep(5.0)
```

## 工具调用（ReAct）

```python
@app.tool(name="get_weather", description="获取天气")
async def get_weather(city: str) -> str:
    return f"{city} 天气晴朗"

from fastmind import ToolNode

tool_node = ToolNode(app.get_tools())

def has_tool_calls(state: dict, event: Event) -> str:
    return "tools" if state.get("tool_calls") else None

graph.add_conditional_edges("agent", has_tool_calls, {None: "__end__"})
graph.add_edge("tools", "agent")
```

## 示例

| 示例 | 描述 |
|------|------|
| [simple_chat.py](examples/simple_chat.py) | 基础聊天 |
| [simple_chat_with_tool.py](examples/simple_chat_with_tool.py) | 工具调用（ReAct）|
| [streaming_chat.py](examples/streaming_chat.py) | 实时流式输出 |
| [human_in_loop.py](examples/human_in_loop.py) | 人工审批工作流 |
| [perception_loop.py](examples/perception_loop.py) | 传感器数据处理 |
| [drone.py](examples/drone.py) | 定时感知 |
| [companion_bot.py](examples/companion_bot.py) | 多智能体对话 |
| [humanoid_robot.py](examples/humanoid_robot.py) | 多工具协作 |
| [sleep_assessment.py](examples/sleep_assessment.py) | 多状态 HITL 流程 |
| [comprehensive_assistant.py](examples/comprehensive_assistant.py) | 全功能助手 |

运行示例：

```bash
python -m fastmind.examples.simple_chat
```

## API 参考

### FastMindAPI

```python
api = FastMindAPI(app)

await api.start()                    # 启动引擎和感知循环
await api.push_event(session_id, event)  # 推送事件到会话
async for ev in api.stream_events(session_id):  # 流式获取输出事件
    print(ev)
await api.stop()                     # 停止引擎
```

### Session

```python
session = api.get_session(session_id)
state = session.state                 # 获取会话状态
await session.wait_for_output(timeout=5.0)  # 等待输出事件
```

## 架构图

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

## 测试

```bash
pip install fastmind[dev]
pytest tests/ -v
```

## 许可证

GPL-3.0 许可证 - 详见 [LICENSE](LICENSE)。

## 致谢

状态图架构设计灵感来源于 [LangGraph](https://github.com/langchain-ai/langgraph)。

## 作者

[xiefujin](https://github.com/kandada)

## 链接

- [文档](https://fastmind.ai/docs)
- [GitHub](https://github.com/kandada/fastmind)
- [PyPI](https://pypi.org/project/fastmind/)
