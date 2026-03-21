# 流式输出

Agent 节点可以返回 `tuple[dict, list[Event]]` 来实现流式输出。

## 返回格式

```python
async def streaming_agent(state, event):
    output_events = []
    full_content = ""
    
    for chunk in stream_response():
        output_events.append(Event(
            type="stream.chunk",
            payload={"delta": chunk},
            session_id=event.session_id
        ))
        full_content += chunk
    
    output_events.append(Event(
        type="stream.end",
        payload={"content": full_content},
        session_id=event.session_id
    ))
    
    state["response"] = full_content
    return state, output_events
```

## 消费流式事件

```python
api = FastMindAPI(app)
await api.start()

await api.push_event("user1", Event("msg", {"text": "hi"}, "user1"))

# 方式1: stream_events (推荐，无轮询)
async for ev in api.stream_events("user1"):
    if ev.type == "stream.chunk":
        print(ev.payload["delta"], end="", flush=True)
    elif ev.type == "stream.end":
        print()

# 方式2: 直接使用队列
session = api.get_session("user1")
while True:
    ev = await session.output_queue.get()  # 阻塞等待，无轮询
    print(ev)
```

## run_streaming 便捷方法

```python
full_text = await api.run_streaming(
    "user1",
    "Hello!",
    on_chunk=lambda delta: print(delta, end="", flush=True),
)
```
