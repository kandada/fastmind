# 感知循环

感知循环允许定期生成事件，用于传感器数据采集、定时任务、外部触发等场景。

## 定义感知循环

使用 `@app.perception` 装饰器注册感知循环：

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

## 支持的事件类型

感知系统支持所有类型的事件，不限于 `sensor.data`：

```python
@app.perception(interval=1.0, name="multi_sensor")
async def multi_sensor(app: FastMind):
    # 传感器数据
    yield Event(type="sensor.data", payload={"temp": 25}, session_id="user_001")
    # 定时器事件
    yield Event(type="timer.tick", payload={"count": 1}, session_id="user_001")
    # 用户消息
    yield Event(type="user.message", payload={"text": "triggered"}, session_id="user_001")
    # 自定义事件
    yield Event(type="custom.event", payload={"data": "..."}, session_id="user_001")
```

**自动路由**：所有非 `system` session 的感知事件都会自动路由到对应的 Session。

**自动创建**：如果 Session 不存在，会自动创建。

## 同步 vs 异步生成器

### 异步生成器（推荐）

异步生成器每次循环都会重新执行：

```python
@app.perception(interval=1.0, name="async_sensor")
async def async_sensor(app: FastMind):
    """每次循环都会重新执行"""
    while True:
        data = await read_sensor()  # 每次循环都调用
        yield Event(type="sensor.data", payload=data, session_id="user_001")
        await asyncio.sleep(1.0)
```

### 同步生成器（状态保持）

同步生成器只初始化一次，状态在多次循环间保持：

```python
@app.perception(interval=1.0, name="sync_sensor")
def sync_sensor(app: FastMind):
    """只初始化一次，状态保持"""
    counter = 0
    while True:
        counter += 1
        yield Event(type="counter", payload={"count": counter}, session_id="user_001")
        time.sleep(1.0)
```

## 处理感知事件

### 在 Agent 中处理

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

### 事件自动路由

感知事件自动从 `PerceptionScheduler` 路由到对应的 Session：

```
感知函数 yield 事件
       │
       ▼
PerceptionScheduler._handle_event()
       │
       ▼
FastMindAPI._handle_perception_event()
       │  （移除类型限制，支持所有事件类型）
       ▼
Engine.get_or_create_session()  ── 自动创建不存在的 Session
       │
       ▼
Session.push_event()
       │
       ▼
事件进入 Session 处理队列
```

## 自动启动

`FastMindAPI.start()` 时所有感知循环自动启动。

## 异常处理

感知循环中的异常会被记录到日志，不会中断其他处理：

```python
# fastmind/core/perception.py
async def _handle_event(self, event: Event) -> None:
    for handler in self._event_handlers:
        try:
            await handler(event)
        except Exception as e:
            logger.error(f"Error in perception event handler: {e}")
```

## 示例

参考 [perception_loop.py](../../examples/perception_loop.py) 了解完整示例。
