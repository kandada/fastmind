# 示例目录

本目录包含 FastMind 框架的各种示例，可以直接运行。

## 基础示例

### simple_chat.py

简单的对话示例，展示最基本的 Agent 使用方式。

```bash
PYTHONPATH=. python3 examples/simple_chat.py
```

### simple_chat_with_tool.py

带工具调用的对话示例，展示 Agent-Tool-Agent 的 ReAct 循环。

```bash
PYTHONPATH=. python3 examples/simple_chat_with_tool.py
```

### streaming_chat.py

流式输出示例，展示如何处理 `stream.chunk` 和 `stream.end` 事件。

```bash
PYTHONPATH=. python3 examples/streaming_chat.py
```

## 进阶示例

### human_in_loop.py

Human-in-the-loop 示例，展示中断和恢复机制。

```bash
PYTHONPATH=. python3 examples/human_in_loop.py
```

### perception_loop.py

感知循环示例，展示定时器感知和事件路由。

```bash
PYTHONPATH=. python3 examples/perception_loop.py
```

## 应用示例

### humanoid_robot.py

人形机器人示例，展示多工具协作和语音交互。

```bash
PYTHONPATH=. python3 examples/humanoid_robot.py
```

### companion_bot.py

陪伴机器人示例，展示情感识别和复杂对话。

```bash
PYTHONPATH=. python3 examples/companion_bot.py
```

### sleep_assessment.py

睡眠评估示例，展示多状态管理和 HITL 流程。

```bash
PYTHONPATH=. python3 examples/sleep_assessment.py
```

### drone.py

无人机示例，展示定时感知和飞行控制。

```bash
PYTHONPATH=. python3 examples/drone.py
```

### comprehensive_assistant.py

全功能助手示例，展示集成了工具调用、流式输出、人机交互的完整助手应用。

```bash
PYTHONPATH=. python3 examples/comprehensive_assistant.py
```
