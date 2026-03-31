# Perception Loop

Perception loops allow generating events periodically for sensor data collection, scheduled tasks, external triggers, etc.

## Define a Perception Loop

Use the `@app.perception` decorator to register a perception loop:

```python
@app.perception(interval=5.0, name="sensor_monitor")
async def sensor_monitor(app: FastMind):
    while True:
        data = await read_sensor()
        yield Event(
            type="sensor.data",
            payload=data,
            session_id="system"
        )
        await asyncio.sleep(5.0)
```

## Supported Event Types

The perception system supports all event types, not just `sensor.data`:

```python
@app.perception(interval=1.0, name="multi_sensor")
async def multi_sensor(app: FastMind):
    # Sensor data
    yield Event(type="sensor.data", payload={"temp": 25}, session_id="user_001")
    # Timer event
    yield Event(type="timer.tick", payload={"count": 1}, session_id="user_001")
    # User message
    yield Event(type="user.message", payload={"text": "triggered"}, session_id="user_001")
    # Custom event
    yield Event(type="custom.event", payload={"data": "..."}, session_id="user_001")
```

**Auto-routing**: All perception events with non-`system` session are automatically routed to the corresponding Session.

**Auto-creation**: If the Session doesn't exist, it will be created automatically.

## Sync vs Async Generators

### Async Generators (Recommended)

Async generators re-execute every loop:

```python
@app.perception(interval=1.0, name="async_sensor")
async def async_sensor(app: FastMind):
    """Re-executes every loop"""
    while True:
        data = await read_sensor()  # Called every loop
        yield Event(type="sensor.data", payload=data, session_id="user_001")
        await asyncio.sleep(1.0)
```

### Sync Generators (State Preservation)

Sync generators initialize only once, maintaining state across loops:

```python
@app.perception(interval=1.0, name="sync_sensor")
def sync_sensor(app: FastMind):
    """Initializes once, preserves state"""
    counter = 0
    while True:
        counter += 1
        yield Event(type="counter", payload={"count": counter}, session_id="user_001")
        time.sleep(1.0)
```

## Handle Perception Events

### In an Agent

```python
@app.agent(name="sensor_processor")
async def processor(state, event):
    if event.type == "sensor.data":
        state["latest_data"] = event.payload
    elif event.type == "timer.tick":
        state["tick_count"] = event.payload.get("count")
    elif event.type == "user.message":
        state["messages"] = state.get("messages", [])
        state["messages"].append(event.payload.get("text"))
    return state
```

### Event Auto-routing

Perception events are automatically routed from `PerceptionScheduler` to the corresponding Session:

```
Perception function yields event
       │
       ▼
PerceptionScheduler._handle_event()
       │
       ▼
FastMindAPI._handle_perception_event()
       │  (Type restriction removed, supports all event types)
       ▼
Engine.get_or_create_session()  ── Auto-creates non-existent Sessions
       │
       ▼
Session.push_event()
       │
       ▼
Event enters Session processing queue
```

## Auto Start

All perception loops start automatically when `FastMindAPI.start()` is called.

## Exception Handling

Exceptions in perception loops are logged and don't interrupt other handlers:

```python
# fastmind/core/perception.py
async def _handle_event(self, event: Event) -> None:
    for handler in self._event_handlers:
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in perception event handler: {e}")
```

## Examples

See [perception_loop.py](../../examples/perception_loop.py) for a complete example.
