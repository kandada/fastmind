# 事件监听：stream_events 与 wait_for_output

FastMind 提供了两种优雅的方式监听会话输出事件，替代机械的 `asyncio.sleep()` 轮询。

## 核心概念

| 方法 | 模式 | 适用场景 |
|------|------|----------|
| `stream_events()` | 异步迭代器 | 持续监听、实时处理事件 |
| `wait_for_output()` | 单次等待 | 等待特定结果、超时控制 |
| `output_queue.get()` | 直接队列 | 低层控制、需要更多灵活性 |

## 1. stream_events() - 异步迭代器（推荐）

`stream_events()` 返回一个异步迭代器，自动阻塞等待事件，直到会话结束。

### 基本用法

```python
async def main():
    api = FastMindAPI(app)
    await api.start()

    await api.push_event("user_001", Event("user.message", {"text": "Hello"}, "user_001"))

    async for ev in api.stream_events("user_001"):
        if ev.type == "stream.chunk":
            print(ev.payload.get("delta", ""), end="", flush=True)
        elif ev.type == "stream.end":
            print("\n--- Streaming complete ---")
        elif ev.type == "interrupt":
            print(f"\nInterrupt: {ev.payload['prompt']}")
            await api.resume_session("user_001", "confirm")

    await api.stop()
```

### 处理所有事件类型

```python
async for ev in api.stream_events("user_001"):
    if ev.type == "error":
        print(f"Error: {ev.payload['error']}")
        break
    elif ev.type == "interrupt":
        print(f"Need approval: {ev.payload['prompt']}")
        await api.resume_session(ev.session_id, "yes")
    elif ev.type == "stream.chunk":
        # 处理流式输出
        print(ev.payload.get("delta", ""), end="", flush=True)
    elif ev.type == "stream.end":
        print("\nDone")
        break
```

### 过滤特定事件类型

```python
# 只监听 error 和 interrupt 事件
async for ev in api.stream_events("user_001", event_types=["error", "interrupt"]):
    print(f"Event: {ev.type}", ev.payload)
```

## 2. wait_for_output() - 单次等待

`wait_for_output()` 等待一个输出事件，支持超时控制。

### 基础用法

```python
session = api.get_session("user_001")

# 等待最多 10 秒
event = await session.wait_for_output(timeout=10.0)

if event:
    print(f"Got event: {event.type}", event.payload)
else:
    print("Timeout, no event received")
```

### 配合循环使用

```python
session = api.get_session("user_001")

while True:
    event = await session.wait_for_output(timeout=30.0)

    if event is None:
        print("Timeout waiting for response")
        break

    if event.type == "stream.end":
        print("Done")
        break

    if event.type == "error":
        print(f"Error: {event.payload['error']}")
        break

    # 处理事件
    print(f"Event: {event.type}", event.payload)
```

### 轮询状态（非推荐，但有时有用）

```python
session = api.get_session("user_001")

while True:
    state = api.get_state("user_001")

    if state.get("done"):
        print("Task completed!")
        break

    # 每秒检查一次
    await asyncio.sleep(1)
```

## 3. output_queue 直接操作

对于需要更底层控制的场景，可以直接操作 `output_queue`。

### 获取下一个事件（阻塞）

```python
session = api.get_session("user_001")
event = await session.output_queue.get()  # 永久阻塞
```

### 非阻塞获取

```python
event = await session.get_output()  # 立即返回，None 表示队列空
```

### 组合使用

```python
session = api.get_session("user_001")

while True:
    try:
        # 最多等 5 秒
        event = await asyncio.wait_for(
            session.output_queue.get(),
            timeout=5.0
        )
    except asyncio.TimeoutError:
        print("No event for 5 seconds, continuing...")
        continue

    if event.type == "stream.end":
        break

    process_event(event)
```

## 4. 完整示例

### 流式对话机器人

```python
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI
import asyncio

app = FastMind()

@app.agent(name="chat")
async def chat_agent(state: dict, event: Event) -> dict:
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": event.payload.get("text", "")})

    # 模拟 LLM 流式输出
    response = f"Echo: {event.payload.get('text', '')}"
    output_queue = state["_output_queue"]
    session_id = state["_session_id"]

    async def stream_response():
        for char in response:
            await output_queue.put(Event(
                type="stream.chunk",
                payload={"delta": char},
                session_id=session_id
            ))
            await asyncio.sleep(0.02)
        await output_queue.put(Event(
            type="stream.end",
            payload={},
            session_id=session_id
        ))

    asyncio.create_task(stream_response())
    return state

graph = Graph()
graph.add_node("chat", chat_agent)
graph.set_entry_point("chat")
app.register_graph("main", graph)

async def main():
    api = FastMindAPI(app)
    await api.start()

    await api.push_event("user_001", Event("user.message", {"text": "Hello!"}, "user_001"))

    print("Response: ", end="", flush=True)
    async for ev in api.stream_events("user_001"):
        if ev.type == "stream.chunk":
            print(ev.payload.get("delta", ""), end="", flush=True)
        elif ev.type == "stream.end":
            print("\nConversation ended")

    await api.stop()

asyncio.run(main())
```

### 带超时的任务处理

```python
async def process_with_timeout(api, session_id, timeout=30.0):
    session = api.get_session(session_id)

    async def listen():
        results = []
        async for ev in api.stream_events(session_id):
            if ev.type == "stream.chunk":
                results.append(ev.payload.get("delta", ""))
            elif ev.type == "stream.end":
                return "".join(results)
            elif ev.type == "error":
                return f"Error: {ev.payload['error']}"
        return "".join(results)

    try:
        return await asyncio.wait_for(listen(), timeout=timeout)
    except asyncio.TimeoutError:
        return f"Timeout after {timeout} seconds"
```

## 5. 对比总结

```
┌─────────────────────────────────────────────────────────────┐
│                    事件监听方式对比                          │
├──────────────┬──────────────┬───────────────────────────────┤
│ stream_events│ 异步迭代器    │ 持续监听，自动阻塞，最推荐     │
│ wait_for_out │ 单次等待     │ 需要超时控制时使用             │
│ output_queue │ 队列直接操作 │ 需要底层控制时使用             │
│ asyncio.sleep│ 轮询 ❌      │ 不推荐，浪费资源              │
└──────────────┴──────────────┴───────────────────────────────┘
```

## 6. 最佳实践

1. **优先使用 `stream_events()`**：最符合事件驱动理念
2. **需要超时控制用 `wait_for_output()`**：配合循环实现复杂逻辑
3. **避免 `asyncio.sleep()` 轮询**：除非有特殊需求（如轮询外部 API）
4. **记得处理 `error` 事件**：确保异常能被正确捕获
5. **使用 `stream.end` 判断结束**：不要依赖外部超时
