# 感知循环

感知循环允许定期生成事件，用于传感器数据采集、定时任务等场景。

## 定义感知循环

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

## 处理感知事件

```python
@app.agent(name="sensor_processor")
async def processor(state, event):
    if event.type == "sensor.data":
        state["latest_data"] = event.payload
    return state
```

## 自动启动

`FastMindAPI.start()` 时所有感知循环自动启动。
