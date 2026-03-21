# Streaming Output

Agent nodes can return `tuple[dict, list[Event]]` to implement streaming output.

## Return Format

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

## Consuming Streaming Events

```python
api = FastMindAPI(app)
await api.start()

await api.push_event("user1", Event("msg", {"text": "hi"}, "user1"))

# Method 1: stream_events (recommended, no polling)
async for ev in api.stream_events("user1"):
    if ev.type == "stream.chunk":
        print(ev.payload["delta"], end="", flush=True)
    elif ev.type == "stream.end":
        print()

# Method 2: Direct queue usage
session = api.get_session("user1")
while True:
    ev = await session.output_queue.get()  # Blocking wait, no polling
    print(ev)
```

## run_streaming Convenience Method

```python
full_text = await api.run_streaming(
    "user1",
    "Hello!",
    on_chunk=lambda delta: print(delta, end="", flush=True),
)
```
