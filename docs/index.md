# FastMind 框架

具身智能多 Agent 系统框架

## 核心特性

- **轻量高效**: 核心代码不超过 8000 行，不依赖 LangChain、LangGraph 等框架
- **事件驱动**: 基于 asyncio 的事件驱动架构，支持千级并发会话
- **具身原生**: 内置感知循环，支持传感器、定时器等外部驱动
- **Human-in-the-loop**: 支持人工确认和中断
- **开发者友好**: 类似 FastAPI 的装饰器模式

## 快速开始

```python
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI

app = FastMind()

@app.agent(name="chat")
async def chat(state, event):
    state["reply"] = f"收到: {event.payload.get('text')}"
    return state

graph = Graph()
graph.add_node("chat", chat)
graph.set_entry_point("chat")
app.register_graph("main", graph)

async def main():
    api = FastMindAPI(app)
    await api.start()
    await api.push_event("user1", Event("msg", {"text": "你好"}, "user1"))
    # ...
    await api.stop()
```

## 目录结构

```
fastmind/
├── core/           # 核心模块
│   ├── app.py      # FastMind 主类
│   ├── graph.py    # Graph 类
│   ├── event.py    # Event 类
│   ├── state.py    # State 类
│   ├── tool.py     # 工具系统
│   ├── node.py     # Agent 节点
│   ├── engine.py   # 执行引擎
│   └── perception.py  # 感知循环
├── contrib/        # 扩展模块
│   └── api.py      # FastMindAPI
├── examples/       # 示例代码
└── tests/         # 测试代码
```

## 示例

```bash
cd fastmind
PYTHONPATH=. python3 examples/simple_chat.py
```

## 文档

- [快速入门](quickstart.md)
- [核心概念](core_concepts.md)
- [API 参考](api_reference.md)
- [示例](examples.md)
- 进阶指南
  - [流式输出](advanced/streaming.md)
  - [人工介入](advanced/human_in_loop.md)
  - [感知循环](advanced/perception_loop.md)
  - [子图](advanced/subgraph.md)
- 最佳实践
  - [工具设计](best_practices/tool_design.md)
  - [状态设计](best_practices/state_design.md)
