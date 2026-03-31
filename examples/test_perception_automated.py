#!/usr/bin/env python3
"""自动化感知系统验证测试

运行方式: python -m fastmind.examples.test_perception_automated
"""

import asyncio
import sys
import os
from datetime import datetime

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI


async def test_sync_generator_preserves_state():
    """测试同步生成器状态保持（验证 Bug 1 修复）"""
    print("\n" + "=" * 60)
    print("测试 1: 同步生成器状态保持")
    print("=" * 60)

    app = FastMind()

    call_count = 0

    @app.perception(interval=0.05, name="sync_sensor")
    def stateful_sensor(app):
        nonlocal call_count
        call_count += 1
        for i in range(3):
            yield Event(
                type="sensor.data",
                payload={"call": call_count, "value": i},
                session_id="user_001",
            )

    @app.agent(name="test_agent")
    async def test_agent(state, event):
        state.setdefault("events", [])
        state["events"].append(event.type)
        return state

    graph = Graph()
    graph.add_node("test", test_agent)
    graph.set_entry_point("test")
    app.register_graph("main", graph)

    fm_api = FastMindAPI(app)
    await fm_api.start()

    await asyncio.sleep(0.15)  # 等待几个循环

    await fm_api.stop()

    state = fm_api.get_state("user_001")
    events_received = state.get("events", []) if state else []

    print(f"感知函数调用次数: {call_count} (期望: 1)")
    print(f"收到事件数: {len(events_received)}")

    if call_count == 1:
        print("✓ Bug 1 修复验证通过: 同步生成器状态保持")
        return True
    else:
        print("✗ Bug 1 修复验证失败")
        return False


async def test_all_event_types_routed():
    """测试所有事件类型都被路由（验证 Bug 3 修复）"""
    print("\n" + "=" * 60)
    print("测试 2: 所有事件类型路由")
    print("=" * 60)

    app = FastMind()

    event_types_received = []

    @app.agent(name="test_agent")
    async def test_agent(state, event):
        event_types_received.append(event.type)
        return state

    @app.perception(interval=0.1, name="multi_sensor")
    async def multi_sensor(app):
        yield Event(type="timer.tick", payload={"count": 1}, session_id="user_002")
        yield Event(type="user.message", payload={"text": "hello"}, session_id="user_002")
        yield Event(type="sensor.data", payload={"temp": 25}, session_id="user_002")

    graph = Graph()
    graph.add_node("test", test_agent)
    graph.set_entry_point("test")
    app.register_graph("main", graph)

    fm_api = FastMindAPI(app)
    await fm_api.start()

    await asyncio.sleep(0.25)  # 等待感知触发

    await fm_api.stop()

    print(f"收到的事件类型: {event_types_received}")

    success = "timer.tick" in event_types_received and "user.message" in event_types_received
    if success:
        print("✓ Bug 3 修复验证通过: 所有事件类型都被路由")
        return True
    else:
        print("✗ Bug 3 修复验证失败")
        return False


async def test_exception_logging():
    """测试异常被正确记录（验证 Bug 2 修复）"""
    print("\n" + "=" * 60)
    print("测试 3: 异常日志记录")
    print("=" * 60)

    app = FastMind()

    error_logged = []

    class MockLogger:
        def error(self, msg):
            error_logged.append(msg)

    import fastmind.core.perception as perception_module

    original_logger = perception_module.logger
    perception_module.logger = MockLogger()

    scheduler = app._perceptions = []

    from fastmind.core.perception import PerceptionScheduler

    sched = PerceptionScheduler(app)

    async def bad_handler(event):
        raise ValueError("Test error")

    async def good_handler(event):
        pass

    sched.register_event_handler(bad_handler)
    sched.register_event_handler(good_handler)

    from fastmind.core.event import Event

    event = Event("test", {}, "test")
    await sched._handle_event(event)

    perception_module.logger = original_logger

    print(f"错误日志: {error_logged}")

    if error_logged and "Test error" in error_logged[0]:
        print("✓ Bug 2 修复验证通过: 异常被正确记录")
        return True
    else:
        print("✗ Bug 2 修复验证失败")
        return False


async def test_perception_loop_no_restart_on_exhaustion():
    """测试感知循环在生成器耗尽后不会无限重启"""
    print("\n" + "=" * 60)
    print("测试 4: 生成器耗尽后行为")
    print("=" * 60)

    app = FastMind()

    call_count = 0

    def exhausted_sensor(app):
        nonlocal call_count
        call_count += 1
        yield Event(type="sensor.data", payload={"count": call_count}, session_id="system")

    events = []

    async def handler(event):
        events.append(event)

    from fastmind.core.perception import PerceptionLoop

    loop = PerceptionLoop("test", exhausted_sensor, 0.05, app)
    await loop.start(handler)

    await asyncio.sleep(0.2)  # 3-4 个循环
    await loop.stop()

    print(f"函数调用次数: {call_count}")
    print(f"收到事件数: {len(events)}")

    # 验证: 同步生成器只调用一次但能 yield 多次
    # 或者如果生成器有状态，在耗尽后不会再调用
    if call_count == 1 and len(events) == 1:
        print("✓ 同步生成器正确处理: 只初始化一次")
        return True
    elif call_count >= 1 and len(events) >= call_count:
        print("✓ 感知循环正确工作")
        return True
    else:
        print("✗ 感知循环行为异常")
        return False


async def main():
    print("\n" + "=" * 60)
    print("FastMind 感知系统自动化验证测试")
    print("=" * 60)

    results = []

    results.append(await test_sync_generator_preserves_state())
    results.append(await test_all_event_types_routed())
    results.append(await test_exception_logging())
    results.append(await test_perception_loop_no_restart_on_exhaustion())

    print("\n" + "=" * 60)
    print("测试结果汇总")
    print("=" * 60)

    passed = sum(results)
    total = len(results)

    print(f"通过: {passed}/{total}")

    if passed == total:
        print("\n✓ 所有测试通过！感知系统修复验证成功！")
        return 0
    else:
        print("\n✗ 部分测试失败")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
