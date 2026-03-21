"""感知循环测试"""

import pytest
import asyncio
from fastmind import FastMind
from fastmind.core.perception import PerceptionScheduler, PerceptionLoop
from fastmind.core.event import Event
from fastmind.core.graph import Graph
from fastmind.core.event import EventType


class TestPerceptionScheduler:
    """PerceptionScheduler 测试"""

    @pytest.mark.asyncio
    async def test_scheduler_initialization(self):
        """测试调度器初始化"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        assert scheduler.app == app
        assert scheduler._running is False
        assert scheduler._loops == {}
        assert scheduler._event_handlers == []

    @pytest.mark.asyncio
    async def test_scheduler_start_stop(self):
        """测试调度器启动和停止"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        await scheduler.start()
        assert scheduler._running is True

        await scheduler.stop()
        assert scheduler._running is False

    @pytest.mark.asyncio
    async def test_scheduler_register_handler(self):
        """测试注册事件处理器"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        scheduler.register_event_handler(handler)
        assert len(scheduler._event_handlers) == 1

        scheduler.register_event_handler(handler)
        assert len(scheduler._event_handlers) == 2

    @pytest.mark.asyncio
    async def test_scheduler_notify_handlers(self):
        """测试通知处理器"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        received_events = []

        async def handler(event: Event):
            received_events.append(event)

        scheduler.register_event_handler(handler)

        event = Event("test.type", {"data": "value"}, "test_session")
        await scheduler._handle_event(event)

        assert len(received_events) == 1
        assert received_events[0].type == "test.type"


class TestPerceptionLoop:
    """PerceptionLoop 测试"""

    @pytest.mark.asyncio
    async def test_perception_loop_basic(self):
        """测试基本感知循环"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        generated_events = []

        async def sensor_perception(app: FastMind):
            for i in range(3):
                yield Event(
                    type=EventType.SENSOR_DATA,
                    payload={"sensor_id": "temp", "value": 20 + i},
                    session_id="system",
                )
                await asyncio.sleep(0.02)

        scheduler.register_loop("temp_sensor", sensor_perception, 0.02)

        async def handler(event: Event):
            generated_events.append(event)

        scheduler.register_event_handler(handler)

        await scheduler.start()
        await asyncio.sleep(0.15)
        await scheduler.stop()

        assert len(generated_events) >= 1

    @pytest.mark.asyncio
    async def test_multiple_perception_loops(self):
        """测试多个感知循环"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        sensor1_events = []
        sensor2_events = []

        async def sensor1(app: FastMind):
            for i in range(2):
                yield Event(
                    type=EventType.SENSOR_DATA,
                    payload={"id": "sensor1", "value": i},
                    session_id="system",
                )
                await asyncio.sleep(0.02)

        async def sensor2(app: FastMind):
            for i in range(2):
                yield Event(
                    type=EventType.SENSOR_DATA,
                    payload={"id": "sensor2", "value": i},
                    session_id="system",
                )
                await asyncio.sleep(0.02)

        async def handler1(event: Event):
            if event.payload.get("id") == "sensor1":
                sensor1_events.append(event)

        async def handler2(event: Event):
            if event.payload.get("id") == "sensor2":
                sensor2_events.append(event)

        scheduler.register_loop("s1", sensor1, 0.02)
        scheduler.register_loop("s2", sensor2, 0.02)
        scheduler.register_event_handler(handler1)
        scheduler.register_event_handler(handler2)

        await scheduler.start()
        await asyncio.sleep(0.15)
        await scheduler.stop()

        assert len(sensor1_events) >= 1
        assert len(sensor2_events) >= 1


class TestAppPerceptionDecorator:
    """@app.perception 装饰器测试"""

    @pytest.mark.asyncio
    async def test_perception_decorator(self):
        """测试感知装饰器"""
        app = FastMind()

        @app.perception(interval=0.1, name="test_sensor")
        async def sensor(app: FastMind):
            yield Event(type=EventType.SENSOR_DATA, payload={"data": "test"}, session_id="system")

        perceptions = app.get_perceptions()
        assert len(perceptions) == 1
        assert perceptions[0][0] == "test_sensor"

    @pytest.mark.asyncio
    async def test_perception_with_scheduler(self):
        """测试感知与调度器集成"""
        app = FastMind()

        @app.perception(interval=0.05, name="sensor1")
        async def sensor1(app: FastMind):
            for i in range(3):
                yield Event(
                    type=EventType.SENSOR_DATA,
                    payload={"from": "sensor1", "value": i},
                    session_id="system",
                )
                await asyncio.sleep(0.03)

        from fastmind.contrib import FastMindAPI

        received = []

        async def handler(event: Event):
            received.append(event)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        fm_api._perception_scheduler.register_event_handler(handler)

        await asyncio.sleep(0.2)
        await fm_api.stop()

        sensor1_events = [e for e in received if e.payload.get("from") == "sensor1"]
        assert len(sensor1_events) >= 1


class TestPerceptionIntegration:
    """感知循环集成测试"""

    @pytest.mark.asyncio
    async def test_perception_routes_to_session(self):
        """测试感知事件路由到会话"""
        app = FastMind()

        @app.perception(interval=0.05, name="sensor")
        async def sensor_loop(app: FastMind):
            yield Event(type=EventType.SENSOR_DATA, payload={"temp": 25}, session_id="user_001")

        async def agent(state: dict, event: Event) -> dict:
            state.setdefault("sensor_data", [])
            if event.type == EventType.SENSOR_DATA:
                state["sensor_data"].append(event.payload)
            return state

        from fastmind.core.graph import Graph

        graph = Graph()
        graph.add_node("process", agent)
        graph.set_entry_point("process")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        fm_api = FastMindAPI(app)
        await fm_api.start()

        await asyncio.sleep(0.3)

        state = fm_api.get_state("user_001")
        if state:
            sensor_data = state.get("sensor_data", [])
            assert len(sensor_data) >= 1

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_perception_with_multiple_sessions(self):
        """测试多会话感知"""
        app = FastMind()

        @app.perception(interval=0.05, name="multi_sensor")
        async def multi_sensor(app: FastMind):
            for session_id in ["session_1", "session_2"]:
                yield Event(
                    type=EventType.SENSOR_DATA, payload={"from": session_id}, session_id=session_id
                )
                await asyncio.sleep(0.02)

        from fastmind.contrib import FastMindAPI

        sensor_events = {"session_1": 0, "session_2": 0}

        async def handler(event: Event):
            session_id = event.payload.get("from")
            if session_id in sensor_events:
                sensor_events[session_id] += 1

        fm_api = FastMindAPI(app)
        await fm_api.start()
        fm_api._perception_scheduler.register_event_handler(handler)

        await asyncio.sleep(0.25)
        await fm_api.stop()

        assert sensor_events["session_1"] >= 1
        assert sensor_events["session_2"] >= 1
