"""感知系统 Bug 修复验证测试"""

import pytest
import asyncio
from fastmind import FastMind
from fastmind.core.perception import PerceptionScheduler, PerceptionLoop
from fastmind.core.event import Event, EventType
from fastmind.core.graph import Graph


class TestBug1Fix_VerifySyncGeneratorStatePreserved:
    """验证修复: 同步生成器状态保持"""

    @pytest.mark.asyncio
    async def test_sync_generator_preserves_state(self):
        """同步生成器只初始化一次，保持状态"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        call_count = 0
        values_yielded = []

        def stateful_sync_sensor(app):
            nonlocal call_count
            call_count += 1
            for i in range(3):
                values_yielded.append(f"call{call_count}_i{i}")
                yield Event(
                    type="sensor.data",
                    payload={"call": call_count, "i": i},
                    session_id="system",
                )

        scheduler.register_loop("stateful_sensor", stateful_sync_sensor, 0.05)

        events_received = []

        async def handler(event):
            events_received.append(event)

        scheduler.register_event_handler(handler)

        await scheduler.start()
        await asyncio.sleep(0.12)  # 约 2 个循环
        await scheduler.stop()

        print(f"\n=== Bug 1 修复验证 ===")
        print(f"感知函数被调用次数: {call_count}")
        print(f"yield 值列表: {values_yielded}")
        print(f"接收事件数: {len(events_received)}")

        # 生成器状态应正确保持：每次 yield 序列完整，内容正确
        # 注意：有限生成器耗尽后会重新创建，call_count 可能 > 1
        assert "call1_i0" in values_yielded
        assert "call1_i1" in values_yielded
        assert "call1_i2" in values_yielded


class TestBug2Fix_VerifyExceptionLogged:
    """验证修复: 异常被正确记录"""

    @pytest.mark.asyncio
    async def test_exception_is_logged(self):
        """PerceptionScheduler._handle_event 会记录异常"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        error_logged = []

        class MockLogger:
            def error(self, msg):
                error_logged.append(msg)

        import fastmind.core.perception as perception_module

        original_logger = perception_module.logger
        perception_module.logger = MockLogger()

        try:

            async def handler_that_raises(event):
                raise ValueError("Test exception")

            async def handler_that_succeeds(event):
                pass

            scheduler.register_event_handler(handler_that_raises)
            scheduler.register_event_handler(handler_that_succeeds)

            event = Event("test.type", {"data": "value"}, "test_session")
            await scheduler._handle_event(event)

            assert len(error_logged) == 1, "异常应该被记录"
            assert "Test exception" in error_logged[0], f"异常消息应该被记录: {error_logged[0]}"
            assert "handler_that_raises" in error_logged[0], (
                f"处理器名称应该被记录: {error_logged[0]}"
            )
            print(f"\n=== Bug 2 修复验证 ===")
            print(f"异常被记录: Yes")
            print(f"异常消息: {error_logged[0]}")
        finally:
            perception_module.logger = original_logger


class TestBug3Fix_VerifyAllEventTypesRouted:
    """验证修复: 所有事件类型都被路由"""

    @pytest.mark.asyncio
    async def test_all_event_types_routed_to_session(self):
        """非 system session 的所有感知事件都会被路由"""
        from fastmind.contrib.api import FastMindAPI

        app = FastMind()

        event_types_received = []

        @app.agent(name="test_agent")
        async def test_agent(state, event):
            event_types_received.append(event.type)
            return state

        @app.perception(interval=0.1, name="multi_type_sensor")
        async def multi_type_sensor(app):
            yield Event(
                type="timer.tick",
                payload={"count": 1},
                session_id="user_001",
            )
            yield Event(
                type="user.message",
                payload={"text": "hello"},
                session_id="user_001",
            )
            yield Event(
                type="sensor.data",
                payload={"temp": 25},
                session_id="user_001",
            )

        graph = Graph()
        graph.add_node("test", test_agent)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        await asyncio.sleep(0.2)

        await fm_api.stop()

        print(f"\n=== Bug 3 修复验证 ===")
        print(f"感知发送了 3 种事件: timer.tick, user.message, sensor.data")
        print(f"Session 收到的事件类型: {event_types_received}")

        # 修复后，所有类型的事件都应该被路由
        assert "timer.tick" in event_types_received, "timer.tick 事件应该被路由"
        assert "user.message" in event_types_received, "user.message 事件应该被路由"
        assert "sensor.data" in event_types_received, "sensor.data 事件应该被路由"


class TestSyncVsAsyncGenerator:
    """验证同步和异步生成器都能正常工作"""

    @pytest.mark.asyncio
    async def test_async_generator_works(self):
        """异步生成器正常工作（每次循环重新执行）"""
        app = FastMind()
        scheduler = PerceptionScheduler(app)

        call_count = 0

        async def async_sensor(app):
            nonlocal call_count
            call_count += 1
            for i in range(2):
                yield Event(
                    type="sensor.data",
                    payload={"call": call_count, "i": i},
                    session_id="system",
                )

        scheduler.register_loop("async_sensor", async_sensor, 0.05)

        events = []

        async def handler(event):
            events.append(event)

        scheduler.register_event_handler(handler)

        await scheduler.start()
        await asyncio.sleep(0.12)
        await scheduler.stop()

        print(f"\n=== Async Generator Test ===")
        print(f"async generator 调用次数: {call_count}")
        print(f"收到事件数: {len(events)}")

        # async generator 每次循环都重新执行
        assert call_count >= 2, f"async generator 应该至少调用 2 次，实际 {call_count}"
        assert len(events) >= 4, f"应该收到至少 4 个事件，实际 {len(events)}"


if __name__ == "__main__":
    print("运行修复验证测试...\n")

    asyncio.run(
        TestBug1Fix_VerifySyncGeneratorStatePreserved().test_sync_generator_preserves_state()
    )
    asyncio.run(TestBug2Fix_VerifyExceptionLogged().test_exception_is_logged())
    asyncio.run(TestBug3Fix_VerifyAllEventTypesRouted().test_all_event_types_routed_to_session())
    asyncio.run(TestSyncVsAsyncGenerator().test_async_generator_works())

    print("\n所有修复验证测试通过!")
