# FastMind 🧠

轻量级事件驱动的具身智能 Agent 框架，支持 LLM + VLA 双循环架构。

[![PyPI 版本](https://badge.fury.io/py/fastmind.svg)](https://badge.fury.io/py/fastmind)
[![Python](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![许可证: GPL-3.0](https://img.shields.io/badge/License-GPL--3.0-blue.svg)](https://www.gnu.org/licenses/gpl-3.0.en.html)

## 核心特性

- **双循环架构**：LLM 慢循环（规划/推理）与 VLA 快循环（实时控制）在同一个 Session 中并发运行
- **Signal + Event 双通信元语**：Event 处理离散消息，Signal 处理高频连续数据（last-value cache，零拷贝）
- **类 FastAPI 装饰器**：`@app.agent`、`@app.tool`、`@app.vla`、`@app.vla_action`、`@app.signal` 统一风格
- **状态图架构**：像画流程图一样构建工作流
- **事件驱动**：基于 asyncio，无轮询
- **Human-in-the-Loop**：支持中断恢复、人工审批
- **感知循环**：原生支持传感器、定时器
- **动作通道路由**：VLA 输出通过通道名映射到多个执行器（N:M）
- **会话隔离**：多用户、独立状态
- **轻量级**：核心约 8000 行

## 安装

```bash
pip install fastmind
```

## 快速开始

### LLM Agent

```python
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.agent(name="chat_agent")
async def chat_agent(state: dict, event: Event) -> dict:
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": event.payload.get("text", "")})
    state["messages"].append({"role": "assistant", "content": "你好！"})
    return state

graph = Graph()
graph.add_node("chat", chat_agent)
graph.set_entry_point("chat")
app.register_graph("main", graph)

async def main():
    api = FastMindAPI(app)
    await api.start()
    await api.push_event("user_001", Event("user.message", {"text": "你好"}, "user_001"))
    await api.stop()

import asyncio
asyncio.run(main())
```

### LLM Agent + 工具调用（ReAct）

```python
from fastmind import FastMind, Graph, Event, ToolNode, Tool
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.tool(name="get_weather", description="获取城市天气")
async def get_weather(city: str) -> str:
    return f"{city} 晴朗，20°C"

async def chat_agent(state: dict, event: Event) -> dict:
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": event.payload.get("text", "")})
    if "天气" in event.payload.get("text", ""):
        state["tool_calls"] = [
            {"id": "call_1", "function": {"name": "get_weather", "arguments": '{"city": "北京"}'}}
        ]
    else:
        state["messages"].append({"role": "assistant", "content": "我可以帮你查天气！"})
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

### VLA Agent（NPC 控制）

```python
from fastmind import FastMind, Graph, Event, ActionSpace
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.signal(name="vision", interval=1/30)
async def npc_vision():
    return {"frame_id": 1, "objects": []}

@app.vla(name="navigation", frequency=30.0)
async def navigation_vla(state, signal_bus):
    vision = signal_bus.read("vision")
    goal = state.get("llm", {}).get("goal", "idle")
    return {"body": [0.5, 0.0, 0.0]}

@app.vla_action(name="body", action_space=ActionSpace(3))
async def body_executor(action):
    await game_engine.move(action[0], action[1], action[2])

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

## 核心概念

| 概念 | 说明 |
|------|------|
| **State** | 会话级共享字典，所有循环共用 |
| **Event** | 离散消息（用户输入、LLM 回复）— 队列式、push |
| **Signal** | 连续数据（摄像头帧、关节角）— 最新值缓存、pull |
| **Graph** | LLM 工作流拓扑（节点 + 边） |
| **@app.agent** | LLM 推理节点，event-driven |
| **@app.vla** | VLA 推理节点，time-driven，独立调度 |
| **@app.vla_action** | 动作执行器，通过通道名接收 VLA 输出 |
| **@app.signal** | 高频信号源，直写 SignalBus |
| **@app.perception** | 低频感知，yield Event（已有） |
| **Action Channel** | 通道路由，N:M 映射 VLA → 执行器 |

## 架构

```
Session
├── SignalBus                     ← 高频数据通道（零拷贝）
├── LLM Task (_run)               ← 慢循环，event-driven
│   └── Graph: Agent → Tool → ...
├── VLA Task (_vla_scheduler)     ← 快循环，time-driven
│   ├── @app.vla 推理
│   ├── 动作通道路由
│   └── @app.vla_action 执行
└── State（黑板模式）
    ├── llm/: goal, plan, messages
    └── vla/: actions, status, memory
```

## 示例

| 示例 | 说明 |
|------|------|
| [simple_chat.py](examples/simple_chat.py) | 基础聊天 |
| [simple_chat_with_tool.py](examples/simple_chat_with_tool.py) | 工具调用（ReAct）|
| [streaming_chat.py](examples/streaming_chat.py) | 实时流式输出 |
| [human_in_loop.py](examples/human_in_loop.py) | 人工审批工作流 |
| [perception_loop.py](examples/perception_loop.py) | 传感器处理 |
| [drone.py](examples/drone.py) | 定时感知 |
| [companion_bot.py](examples/companion_bot.py) | 多 Agent 对话 |
| [humanoid_robot.py](examples/humanoid_robot.py) | 多工具机器人控制 |
| [sleep_assessment.py](examples/sleep_assessment.py) | 多状态 HITL 流程 |
| [comprehensive_assistant.py](examples/comprehensive_assistant.py) | 全功能助手 |
| [npc_vla.py](examples/npc_vla.py) | **VLA 双循环 NPC（新）** |

```bash
python -m fastmind.examples.npc_vla
```

## API 参考

### FastMindAPI

```python
api = FastMindAPI(app)

await api.start()
await api.push_event(session_id, event)
async for ev in api.stream_events(session_id): ...

# VLA/Signal 新增方法:
frame = api.read_signal(session_id, "vision")      # 读信号
api.write_signal(session_id, "gps", data)           # 写信号
signals = api.list_signals(session_id)              # 列出信号
api.pause_vla(session_id)                            # 暂停 VLA
api.resume_vla(session_id)                           # 恢复 VLA

await api.stop()
```

## 更新日志

### v0.2.1
- **重构**：`_merge_state` 从全量替换改为 `update`，防止节点返回部分 state 时丢失未涉及的 key
- **重构**：输出队列从 `asyncio.Queue` 替换为 `EventBuffer`（只追加环形缓冲区 + 游标读取），`stream_events` 支持多消费者并行独立消费

### v0.2.0
- **重大更新**：新增 VLA 双循环架构 — `@app.vla`（高频推理，time-driven）、`@app.vla_action`（动作通道路由）、`@app.signal`（零拷贝信号通道，与 Event 平行）
- **重大更新**：Session 双循环 — VLA 快循环与 LLM 慢循环并发运行，通过 State（黑板模式）通信，支持 N:M 动作通道映射
- **新功能**：`FastMindAPI.read_signal()` / `write_signal()` / `list_signals()` / `pause_vla()` / `resume_vla()`
- **Bug 修复**：修复 `_save_checkpoint` 在 state 含不可序列化对象时崩溃（改用 `_safe_deepcopy` 优雅回退）
- **Bug 修复**：修复 `human_in_loop` checkpoint pickle 错误
- **Bug 修复**：修复 VLA action executor 异常隔离（一个 executor 崩溃不再阻塞同 tick 的其他 action）
- **可靠性**：新增 20 项 VLA 压力/可靠性测试（长时间运行、异常恢复、并发访问、多会话、暂停/恢复周期、override 周期）

## 引用

如果使用 FastMind 进行研究，请引用：

```bibtex
@misc{xie2026fastmind,
  title  = {FastMind: A Framework-Centric Architecture for Dual-Loop Embodied Intelligence},
  author = {Xie, Fujin},
  year   = {2026},
  doi    = {10.6084/m9.figshare.32692677},
  url    = {https://doi.org/10.6084/m9.figshare.32692677}
}
```

> **预印本**: https://doi.org/10.6084/m9.figshare.32692677

## 许可证

GPL-3.0 许可证 — 详见 [LICENSE](LICENSE)。

## 相关项目

- [fastclaw](https://github.com/kandada/fastclaw) - 基于 FastMind 构建的通用 Agent 框架，提供可复用的工具、技能库和多智能体编排模式。
- [fastbot](https://github.com/kandada/fastbot) - 基于 FastMind 构建的双循环具身智能机器人仿真 Demo，展示 VLA + LLM 协控在 3D 机器人环境中的应用。

## 作者

[xiefujin](https://github.com/kandada) email:490021684@qq.com.

Copyright (c) 2024-2026 xiefujin <490021684@qq.com>. Licensed under GNU GPLv3.
