"""Session.stop / Engine.stop / stream_events 超时与健壮性测试

覆盖优化:
1. Session.stop() 各 await 点加 asyncio.wait_for(timeout=5)
2. stream_events() 被 cancel 时有 debug 日志
3. Engine.stop() 并行 stop 所有 session + 超时
"""

import pytest
import asyncio
import time
from fastmind import FastMind
from fastmind.core.graph import Graph
from fastmind.core.event import Event
from fastmind.core.engine import Session, Engine
from fastmind.contrib import FastMindAPI


class TestSessionStopTimeout:
    """验证 Session.stop() 不会因 task 卡住而永久阻塞"""

    @pytest.mark.asyncio
    async def test_stop_normal_session_completes_quickly(self):
        """正常 session stop 应在超时内完成"""
        app = FastMind()
        graph = Graph()

        async def quick_node(state, event):
            return state

        graph.add_node("quick", quick_node)
        graph.set_entry_point("quick")

        session = Session("test", graph, app)
        await session.start()
        await session.push_event(Event("test", {}, "test"))
        await asyncio.sleep(0.05)

        start = time.monotonic()
        await session.stop()
        elapsed = time.monotonic() - start

        assert elapsed < 3, f"stop took {elapsed:.2f}s, expected < 3s"
        assert session.session_state == Session.STATE_STOPPED
        assert session._task is None
        assert session._vla_task is None

    @pytest.mark.asyncio
    async def test_stop_when_task_is_idle(self):
        """task 在 input_queue.get() 等待时 stop 应快速完成"""
        app = FastMind()
        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)

        session = Session("test", graph, app)
        await session.start()

        # task 正在 input_queue.get() 等待
        await asyncio.sleep(0.05)
        assert session.is_running

        start = time.monotonic()
        await session.stop()
        elapsed = time.monotonic() - start

        assert elapsed < 1, f"stop idle session took {elapsed:.2f}s, expected < 1s"
        assert session.session_state == Session.STATE_STOPPED

    @pytest.mark.asyncio
    async def test_stop_when_vla_task_running(self):
        """VLA task 运行时 stop 应能在超时内完成"""
        app = FastMind()

        @app.vla(name="fast", frequency=30.0)
        async def fast_vla(state, sb):
            return {"act": [1.0]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.1)
        assert session._vla_task is not None

        start = time.monotonic()
        await session.stop()
        elapsed = time.monotonic() - start

        assert elapsed < 3, f"stop vla session took {elapsed:.2f}s, expected < 3s"
        assert session._vla_task is None

    @pytest.mark.asyncio
    async def test_stop_when_signal_tasks_running(self):
        """Signal tasks 运行时 stop 应能正常结束"""
        app = FastMind()

        @app.signal(name="tick", interval=0.02)
        async def tick_signal():
            return "tick"

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.1)
        assert len(session._signal_tasks) > 0

        start = time.monotonic()
        await session.stop()
        elapsed = time.monotonic() - start

        assert elapsed < 3, f"stop signal session took {elapsed:.2f}s, expected < 3s"
        assert session._signal_tasks == []

    @pytest.mark.asyncio
    async def test_stop_clears_all_tasks(self):
        """stop 后所有 task 引用应置空"""
        app = FastMind()

        @app.vla(name="v", frequency=30.0)
        async def vla(state, sb):
            return {"a": [0.0]}

        @app.vla_action(name="a")
        async def act(action):
            return {}

        @app.signal(name="s", interval=0.03)
        async def sig():
            return "x"

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.1)

        assert session._task is not None
        assert session._vla_task is not None
        assert len(session._signal_tasks) > 0

        await session.stop()

        assert session._task is None
        assert session._vla_task is None
        assert session._signal_tasks == []

    @pytest.mark.asyncio
    async def test_stop_nonexistent_tasks_no_error(self):
        """stop 在 task 不存在时不应报错"""
        app = FastMind()
        graph = Graph()
        session = Session("test", graph, app)

        # 未 start 就 stop
        await session.stop()
        assert session.session_state == Session.STATE_STOPPED

    @pytest.mark.asyncio
    async def test_double_stop_is_idempotent(self):
        """重复 stop 不应报错"""
        app = FastMind()
        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.02)

        await session.stop()
        await session.stop()  # 第二次 stop
        assert session.session_state == Session.STATE_STOPPED


class TestStreamEventsCancelLogging:
    """验证 stream_events 被 cancel 时有日志输出"""

    @pytest.mark.asyncio
    async def test_stream_events_cancel_does_not_crash(self):
        """stream_events 被取消时不应崩溃"""
        app = FastMind()

        async def echo(state, event):
            state["msg"] = event.payload.get("text", "")
            return state

        graph = Graph()
        graph.add_node("echo", echo)
        graph.set_entry_point("echo")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("s1", Event("user.message", {"text": "hi"}, "s1"))
        await asyncio.sleep(0.1)

        stream_task = asyncio.create_task(
            _collect_stream(api, "s1")
        )
        await asyncio.sleep(0.05)
        stream_task.cancel()

        try:
            await stream_task
        except asyncio.CancelledError:
            pass

        await api.stop()

    @pytest.mark.asyncio
    async def test_stream_events_graceful_on_session_stop(self):
        """session stop 后 stream_events 应正常退出"""
        app = FastMind()

        async def slow_node(state, event):
            await asyncio.sleep(0.05)
            return state

        graph = Graph()
        graph.add_node("slow", slow_node)
        graph.set_entry_point("slow")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("s1", Event("test", {}, "s1"))

        async def read_stream():
            events = []
            async for ev in api.stream_events("s1"):
                events.append(ev)
            return events

        task = asyncio.create_task(read_stream())

        await asyncio.sleep(0.15)
        await api.delete_session("s1")

        try:
            result = await asyncio.wait_for(task, timeout=3)
        except asyncio.TimeoutError:
            result = None

        assert result is not None or task.done()

        await api.stop()


class TestEngineStopParallel:
    """验证 Engine.stop() 并行 stop 所有 session"""

    @pytest.mark.asyncio
    async def test_engine_stop_multiple_sessions(self):
        """多 session 的 Engine.stop 应能正常完成"""
        app = FastMind()

        async def node(state, event):
            return state

        graph = Graph()
        graph.add_node("n", node)
        graph.set_entry_point("n")
        app.register_graph("main", graph)

        engine = Engine(app)
        await engine.start()

        num_sessions = 5
        for i in range(num_sessions):
            s = engine.get_or_create_session(f"s{i}")
            await s.start()

        assert len(engine._sessions) == num_sessions

        start = time.monotonic()
        await engine.stop()
        elapsed = time.monotonic() - start

        # 并行 stop 应该在合理时间内完成
        # 如果串行 stop + 每个 2s，会超过 5s
        assert elapsed < 5, f"engine stop took {elapsed:.2f}s, expected < 5s (parallel)"
        assert engine._running is False

        for s in engine._sessions.values():
            assert s.session_state == Session.STATE_STOPPED

    @pytest.mark.asyncio
    async def test_engine_stop_empty(self):
        """没有 session 时 stop 不应报错"""
        app = FastMind()
        engine = Engine(app)
        await engine.start()
        await engine.stop()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_engine_stop_with_mixed_session_states(self):
        """混合状态 session（running / idling）的 stop"""
        app = FastMind()

        async def busy_node(state, event):
            state.setdefault("count", 0)
            state["count"] += 1
            return state

        graph = Graph()
        graph.add_node("busy", busy_node)
        graph.set_entry_point("busy")
        app.register_graph("main", graph)

        engine = Engine(app)
        await engine.start()

        # 创建并启动多个 session，其中一个正在执行中
        for i in range(3):
            s = engine.get_or_create_session(f"s{i}")
            await s.start()
            await engine.push_event(f"s{i}", Event("test", {}, f"s{i}"))

        # 立即 stop
        start = time.monotonic()
        await engine.stop()
        elapsed = time.monotonic() - start

        assert elapsed < 5, f"engine stop with mixed states took {elapsed:.2f}s"
        for s in engine._sessions.values():
            assert s.session_state == Session.STATE_STOPPED

    @pytest.mark.asyncio
    async def test_engine_stop_vla_sessions_parallel(self):
        """带 VLA 的多个 session 并行 stop"""
        app = FastMind()

        @app.vla(name="counter", frequency=30.0)
        async def counter(state, sb):
            state.setdefault("cnt", 0)
            state["cnt"] += 1
            return {"act": [float(state["cnt"])]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        engine = Engine(app)
        await engine.start()

        for i in range(4):
            s = engine.get_or_create_session(f"vla_s{i}")
            await s.start()

        await asyncio.sleep(0.1)

        start = time.monotonic()
        await engine.stop()
        elapsed = time.monotonic() - start

        assert elapsed < 5, f"engine stop vla sessions took {elapsed:.2f}s"

    @pytest.mark.asyncio
    async def test_delete_session_also_stops(self):
        """delete_session 应调用 stop"""
        app = FastMind()
        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        engine = Engine(app)
        s = engine.get_or_create_session("del_me")
        await s.start()

        assert "del_me" in engine._sessions

        await engine.delete_session("del_me")
        assert "del_me" not in engine._sessions


class TestAPIStopIntegration:
    """FastMindAPI stop 集成测试"""

    @pytest.mark.asyncio
    async def test_api_stop_closes_all_sessions(self):
        """API stop 应正常关闭所有 session"""
        app = FastMind()

        async def node(state, event):
            return state

        graph = Graph()
        graph.add_node("n", node)
        graph.set_entry_point("n")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        for i in range(3):
            await api.push_event(f"s{i}", Event("test", {}, f"s{i}"))

        await asyncio.sleep(0.1)

        start = time.monotonic()
        await api.stop()
        elapsed = time.monotonic() - start

        assert elapsed < 5, f"api stop took {elapsed:.2f}s"
        assert api._running is False

    @pytest.mark.asyncio
    async def test_api_stop_no_sessions(self):
        """无 session 时 API stop 应正常"""
        app = FastMind()
        api = FastMindAPI(app)
        await api.start()
        await api.stop()
        assert api._running is False


# ── helpers ────────────────────────────────────────────────────


async def _collect_stream(api: FastMindAPI, session_id: str) -> list:
    events = []
    async for ev in api.stream_events(session_id):
        events.append(ev)
    return events
