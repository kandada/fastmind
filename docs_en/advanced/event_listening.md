# Event Listening: stream_events and wait_for_output

FastMind provides two elegant ways to listen to session output events, replacing mechanical `asyncio.sleep()` polling.

## Core Concepts

| Method | Mode | Use Case |
|--------|------|----------|
| `stream_events()` | Async iterator | Continuous listening, real-time event processing |
| `wait_for_output()` | Single wait | Waiting for specific result, timeout control |
| `output_queue.get()` | Direct queue | Lower-level control, more flexibility |

## 1. stream_events() - Async Iterator (Recommended)

`stream_events()` returns an async iterator that automatically blocks and waits for events until the session ends.

### Basic Usage

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

### Handle All Event Types

```python
async for ev in api.stream_events("user_001"):
    if ev.type == "error":
        print(f"Error: {ev.payload['error']}")
        break
    elif ev.type == "interrupt":
        print(f"Need approval: {ev.payload['prompt']}")
        await api.resume_session(ev.session_id, "yes")
    elif ev.type == "stream.chunk":
        # Handle streaming output
        print(ev.payload.get("delta", ""), end="", flush=True)
    elif ev.type == "stream.end":
        print("\nDone")
        break
```

### Filter Specific Event Types

```python
# Only listen to error and interrupt events
async for ev in api.stream_events("user_001", event_types=["error", "interrupt"]):
    print(f"Event: {ev.type}", ev.payload)
```

## 2. wait_for_output() - Single Wait

`wait_for_output()` waits for one output event with timeout control.

### Basic Usage

```python
session = api.get_session("user_001")

# Wait up to 10 seconds
event = await session.wait_for_output(timeout=10.0)

if event:
    print(f"Got event: {event.type}", event.payload)
else:
    print("Timeout, no event received")
```

### Use with Loop

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

    # Handle event
    print(f"Event: {event.type}", event.payload)
```

### Polling State (Not Recommended, but Sometimes Useful)

```python
session = api.get_session("user_001")

while True:
    state = api.get_state("user_001")

    if state.get("done"):
        print("Task completed!")
        break

    # Check every second
    await asyncio.sleep(1)
```

## 3. Direct output_queue Operations

For scenarios requiring more low-level control, you can directly manipulate `output_queue`.

### Get Next Event (Blocking)

```python
session = api.get_session("user_001")
event = await session.output_queue.get()  # Permanent block
```

### Non-blocking Get

```python
event = await session.get_output()  # Return immediately, None if queue is empty
```

### Combined Usage

```python
session = api.get_session("user_001")

while True:
    try:
        # Wait up to 5 seconds
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

## 4. Complete Examples

### Streaming Chat Bot

```python
from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI
import asyncio

app = FastMind()

@app.agent(name="chat")
async def chat_agent(state: dict, event: Event) -> dict:
    state.setdefault("messages", [])
    state["messages"].append({"role": "user", "content": event.payload.get("text", "")})

    # Simulate LLM streaming output
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

### Task Processing with Timeout

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

## 5. Comparison Summary

```
┌─────────────────────────────────────────────────────────────┐
│                  Event Listening Comparison                  │
├──────────────┬──────────────┬───────────────────────────────┤
│ stream_events│ Async iter   │ Continuous listening, best     │
│ wait_for_out │ Single wait  │ Use when timeout control needed│
│ output_queue │ Queue ops    │ Use when low-level control needed│
│ asyncio.sleep│ Polling ❌   │ Not recommended, wastes resources│
└──────────────┴──────────────┴───────────────────────────────┘
```

## 6. Best Practices

1. **Prefer `stream_events()`**: Most aligned with event-driven philosophy
2. **Use `wait_for_output()` when timeout control needed**: Combine with loop for complex logic
3. **Avoid `asyncio.sleep()` polling**: Unless you have specific needs (like polling external API)
4. **Remember to handle `error` events**: Ensure exceptions are caught properly
5. **Use `stream.end` to determine completion**: Don't rely on external timeout
