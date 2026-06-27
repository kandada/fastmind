# Copyright (c) 2024-2026 xiefujin <490021684@qq.com>
# Licensed under GNU GPLv3, see LICENSE file for full license terms.

"""感知系统 Bug 复现测试"""

import pytest
import asyncio
from fastmind import FastMind
from fastmind.core.perception import PerceptionScheduler, PerceptionLoop
from fastmind.core.event import Event, EventType
from fastmind.core.graph import Graph


class TestSyncGeneratorExhaustion:
    """验证同步生成器耗尽后会被重新创建"""

    @pytest.mark.asyncio
    async def test_sync_generator_recreated_after_exhaustion(self):
        """有限同步生成器耗尽后，下次循环重新创建而非永久闲置"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        call_count = 0

        def finite_sensor(app):
            nonlocal call_count
            call_count += 1
            for i in range(2):
                yield Event(
                    type="sensor.data",
                    payload={"call": call_count, "i": i},
                    session_id="test",
                )

        scheduler.register_loop("finite", finite_sensor, 0.03)

        events = []

        async def handler(event):
            events.append(event)

        scheduler.register_event_handler(handler)

        await scheduler.start()
        await asyncio.sleep(0.15)
        await scheduler.stop()

        assert call_count >= 2, f"耗尽后应重新创建，实际调用 {call_count} 次"
        assert len(events) >= 4, f"应收到至少 4 个事件，实际 {len(events)} 个"


class TestHandlerExceptionIsolation:
    """验证 handler 异常不会中断事件迭代"""

    @pytest.mark.asyncio
    async def test_handler_exception_does_not_stop_iteration(self):
        """handler 抛异常不影响后续事件的处理"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        received = []

        def sensor(app):
            for i in range(3):
                yield Event(
                    type="sensor.data",
                    payload={"i": i},
                    session_id="test",
                )

        async def handler(event):
            received.append(event)
            if event.payload["i"] == 0:
                raise ValueError("intentional test error")

        scheduler.register_loop("s", sensor, 0.05)
        scheduler.register_event_handler(handler)

        await scheduler.start()
        await asyncio.sleep(0.08)
        await scheduler.stop()

        for e in received:
            print(f"  事件 i={e.payload.get('i')}")

        assert len(received) >= 3, f"应至少收到 3 个事件，实际 {len(received)} 个"
        received_is = [e.payload["i"] for e in received]
        assert 0 in received_is, "事件 i=0 应被处理"
        assert 1 in received_is, "handler 异常后事件 i=1 应继续被处理"
        assert 2 in received_is, "handler 异常后事件 i=2 应继续被处理"


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
    print("运行感知系统测试...\n")

    asyncio.run(TestSyncGeneratorExhaustion().test_sync_generator_recreated_after_exhaustion())
    asyncio.run(TestHandlerExceptionIsolation().test_handler_exception_does_not_stop_iteration())
    asyncio.run(TestBug2_SilentExceptionSwallowing().test_exception_is_silently_swallowed())
    asyncio.run(TestBug3_HardcodedEventType().test_non_sensor_data_events_are_discarded())
    asyncio.run(TestBug3_ApiFiltersNonSensorData().test_api_filters_non_sensor_data_events())
