# Perception Loop

Perception loops allow generating events periodically for sensor data collection, scheduled tasks, etc.

## Define a Perception Loop

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

## Handle Perception Events

```python
@app.agent(name="sensor_processor")
async def processor(state, event):
    if event.type == "sensor.data":
        state["latest_data"] = event.payload
    return state
```

## Auto Start

All perception loops start automatically when `FastMindAPI.start()` is called.
