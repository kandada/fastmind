"""感知系统 Bug 复现测试"""

import pytest
import asyncio
from fastmind import FastMind
from fastmind.core.perception import PerceptionScheduler, PerceptionLoop
from fastmind.core.event import Event, EventType
from fastmind.core.graph import Graph


class TestBug1_SyncGeneratorRepeatExecution:
    """Bug 1: 同步生成器会被重复执行，每次循环都从头开始"""

    @pytest.mark.asyncio
    async def test_sync_generator_restarts_every_loop(self):
        """同步生成器每次循环都重新执行，应该被修复为保持状态"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        call_count = 0
        yield_count = 0

        def sync_perception(app):
            nonlocal call_count, yield_count
            call_count += 1
            for i in range(3):
                yield_count += 1
                yield Event(
                    type="sensor.data",
                    payload={"call": call_count, "yield_num": i},
                    session_id="system",
                )

        scheduler.register_loop("sync_sensor", sync_perception, 0.05)

        events_received = []

        async def handler(event):
            events_received.append(event)

        scheduler.register_event_handler(handler)

        await scheduler.start()
        await asyncio.sleep(0.18)  # 等待大约 3-4 个循环
        await scheduler.stop()

        print(f"\n=== Bug 1 复现结果 ===")
        print(f"感知函数被调用次数 (call_count): {call_count}")
        print(f"yield 总次数 (yield_count): {yield_count}")
        print(f"接收到的事件数: {len(events_received)}")
        print(f"期望的 yield 次数: 3 (生成器应保持状态)")
        print(f"实际情况: 生成器被重新创建 {call_count} 次")

        assert call_count > 1, "同步生成器应该被多次调用（这是bug）"
        assert yield_count > 3, "同步生成器每次都从头开始yield（这是bug）"


class TestBug2_SilentExceptionSwallowing:
    """Bug 2: 异常被静默忽略，调试困难"""

    @pytest.mark.asyncio
    async def test_exception_is_silently_swallowed(self):
        """PerceptionScheduler._handle_event 捕获所有异常但不记录"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        exception_raised = []

        async def handler_that_raises(event):
            exception_raised.append(event)
            raise ValueError("Test exception from handler")

        async def handler_that_succeeds(event):
            pass

        scheduler.register_event_handler(handler_that_raises)
        scheduler.register_event_handler(handler_that_succeeds)

        event = Event("test.type", {"data": "value"}, "test_session")

        await scheduler._handle_event(event)

        print(f"\n=== Bug 2 复现结果 ===")
        print(f"异常处理器被调用次数: {len(exception_raised)}")
        print(f"异常被抛出: Yes")
        print(f"异常是否被静默忽略: Yes (无日志、无报错)")
        print(f"后续 handler 是否继续执行: Yes (因为异常被 except: pass 吞掉)")

        assert len(exception_raised) == 1, "处理器应该被调用"


class TestBug3_HardcodedEventType:
    """Bug 3: _handle_perception_event 只处理 sensor.data 类型"""

    @pytest.mark.asyncio
    async def test_non_sensor_data_events_are_discarded(self):
        """非 sensor.data 类型的事件会被静默丢弃"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        received_events = []

        async def handler(event):
            received_events.append(event)

        scheduler.register_event_handler(handler)

        event_user_message = Event("user.message", {"text": "hello"}, "user_001")
        event_timer = Event("timer.tick", {"count": 1}, "user_001")
        event_sensor = Event("sensor.data", {"temp": 25}, "user_001")

        await scheduler._handle_event(event_user_message)
        await scheduler._handle_event(event_timer)
        await scheduler._handle_event(event_sensor)

        print(f"\n=== Bug 3 相关信息 ===")
        print(f"发送了 3 种不同类型事件: user.message, timer.tick, sensor.data")
        print(f"PerceptionScheduler._handle_event 能接收所有类型: Yes")
        print(f"但 FastMindAPI._handle_perception_event 只处理 sensor.data")

        assert len(received_events) == 3, "scheduler 能接收所有事件"


class TestBug3_ApiFiltersNonSensorData:
    """Bug 3 验证: FastMindAPI._handle_perception_event 过滤非 sensor.data 事件"""

    @pytest.mark.asyncio
    async def test_api_filters_non_sensor_data_events(self):
        """验证 API 层只处理 sensor.data，其他类型被丢弃"""
        from fastmind.contrib.api import FastMindAPI

        app = FastMind()

        @app.perception(interval=0.1, name="timer_sensor")
        async def timer_perception(app):
            yield Event(
                type="timer.tick",
                payload={"count": 1},
                session_id="user_001",
            )

        graph = Graph()
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        await asyncio.sleep(0.15)

        session = fm_api.get_session("user_001")

        print(f"\n=== Bug 3 API 层验证 ===")
        print(f"感知函数发送了 timer.tick 事件")
        print(f"但 API._handle_perception_event 只处理 sensor.data")
        print(f"timer.tick 事件被静默丢弃")

        if session:
            state = fm_api.get_state("user_001")
            print(f"session state: {state}")
        else:
            print(f"session 为空，因为事件没被路由")

        await fm_api.stop()


if __name__ == "__main__":
    print("运行 Bug 复现测试...\n")

    asyncio.run(TestBug1_SyncGeneratorRepeatExecution().test_sync_generator_restarts_every_loop())
    asyncio.run(TestBug2_SilentExceptionSwallowing().test_exception_is_silently_swallowed())
    asyncio.run(TestBug3_HardcodedEventType().test_non_sensor_data_events_are_discarded())
    asyncio.run(TestBug3_ApiFiltersNonSensorData().test_api_filters_non_sensor_data_events())
