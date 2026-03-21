"""Engine 和 Session 单元测试"""

import pytest
import asyncio
from fastmind import FastMind
from fastmind.core.graph import Graph
from fastmind.core.event import Event
from fastmind.core.engine import Session, Engine


class TestSession:
    """Session 测试"""

    @pytest.mark.asyncio
    async def test_session_initialization(self):
        """测试 Session 初始化"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        assert session.session_id == "test_session"
        assert session.graph == graph
        assert session.app == app
        assert isinstance(session.state, dict)
        assert isinstance(session.input_queue, asyncio.Queue)
        assert isinstance(session.output_queue, asyncio.Queue)

    @pytest.mark.asyncio
    async def test_session_state_contains_queues(self):
        """测试 Session 初始化时 state 包含队列引用"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        assert "_output_queue" in session.state
        assert "_session_id" in session.state
        assert session.state["_output_queue"] is session.output_queue
        assert session.state["_session_id"] == "test_session"

    @pytest.mark.asyncio
    async def test_session_start_stop(self):
        """测试 Session 启动和停止"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            state["executed"] = True
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")

        session = Session("test_session", graph, app)
        await session.start()
        assert session._running is True
        assert session._task is not None

        await session.stop()
        assert session._running is False

    @pytest.mark.asyncio
    async def test_session_push_event(self):
        """测试推送事件"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        event = Event("test.type", {"data": "value"}, "test_session")
        await session.push_event(event)

        pushed_event = await session.input_queue.get()
        assert pushed_event.type == "test.type"
        assert pushed_event.payload["data"] == "value"

    @pytest.mark.asyncio
    async def test_session_output_queue(self):
        """测试输出队列"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        event = Event("output.type", {"result": "test"}, "test_session")
        await session.output_queue.put(event)

        result = await asyncio.wait_for(session.output_queue.get(), timeout=0.1)
        assert result.type == "output.type"
        assert result.payload["result"] == "test"

    @pytest.mark.asyncio
    async def test_session_get_output_timeout(self):
        """测试获取输出超时"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        result = await session.get_output()
        assert result is None

    @pytest.mark.asyncio
    async def test_session_wait_for_output(self):
        """测试等待输出事件（事件驱动）"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        async def put_event_after_delay():
            await asyncio.sleep(0.1)
            await session._put_output(Event("test.type", {"data": "value"}, "test_session"))

        task = asyncio.create_task(put_event_after_delay())

        result = await session.wait_for_output(timeout=1.0)
        assert result is not None
        assert result.type == "test.type"
        assert result.payload["data"] == "value"

        await task

    @pytest.mark.asyncio
    async def test_session_wait_for_output_immediate(self):
        """测试队列已有事件时立即返回"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        await session._put_output(Event("immediate.type", {}, "test_session"))

        result = await session.wait_for_output(timeout=1.0)
        assert result is not None
        assert result.type == "immediate.type"

    @pytest.mark.asyncio
    async def test_session_wait_for_output_timeout(self):
        """测试等待输出超时"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        result = await session.wait_for_output(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_session_checkpoint_save_restore(self):
        """测试检查点保存和恢复"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        session.state["data"] = "original"
        session.state["count"] = 10
        session._current_node = "test_node"

        session._save_checkpoint("test_node")

        assert session._checkpoint is not None
        assert session._checkpoint["state"]["data"] == "original"
        assert session._checkpoint["state"]["count"] == 10
        assert session._checkpoint["current_node"] == "test_node"

        session.state["data"] = "modified"
        session.state["count"] = 99

        session._restore_from_checkpoint()

        assert session.state["data"] == "original"
        assert session.state["count"] == 10
        assert session._current_node == "test_node"

    @pytest.mark.asyncio
    async def test_session_interrupt_flag(self):
        """测试中断标志"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        assert session._interrupted is False

        session._save_checkpoint("test_node")
        session._interrupted = True

        assert session._interrupted is True


class TestEngine:
    """Engine 测试"""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """测试 Engine 初始化"""
        app = FastMind()
        engine = Engine(app)

        assert engine.app == app
        assert engine._sessions == {}
        assert engine._running is False
        assert engine._tasks == []

    @pytest.mark.asyncio
    async def test_engine_start_stop(self):
        """测试 Engine 启动和停止"""
        app = FastMind()
        engine = Engine(app)

        await engine.start()
        assert engine._running is True

        await engine.stop()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_engine_get_or_create_session(self):
        """测试获取或创建会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)

        session1 = engine.get_or_create_session("session_1")
        assert session1 is not None
        assert session1.session_id == "session_1"

        session2 = engine.get_or_create_session("session_1")
        assert session2 is session1

        session3 = engine.get_or_create_session("session_2", "main")
        assert session3.session_id == "session_2"
        assert session3 is not session1

    @pytest.mark.asyncio
    async def test_engine_get_or_create_session_invalid_graph(self):
        """测试获取不存在图时会话"""
        app = FastMind()
        engine = Engine(app)

        with pytest.raises(ValueError, match="not found"):
            engine.get_or_create_session("session_1", "nonexistent")

    @pytest.mark.asyncio
    async def test_engine_push_event_creates_session(self):
        """测试推送事件自动创建会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)

        event = Event("test", {}, "new_session")
        session = await engine.push_event("new_session", event)

        assert session is not None
        assert "new_session" in engine._sessions

    @pytest.mark.asyncio
    async def test_engine_get_session(self):
        """测试获取会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)

        session = engine.get_or_create_session("test_session")
        result = engine.get_session("test_session")
        assert result is session

        result_none = engine.get_session("nonexistent")
        assert result_none is None

    @pytest.mark.asyncio
    async def test_engine_get_session_state(self):
        """测试获取会话状态"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            state["value"] = 42
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)
        session = engine.get_or_create_session("test_session")
        await engine.push_event("test_session", Event("test", {}, "test_session"))

        await asyncio.sleep(0.2)

        state = engine.get_session_state("test_session")
        assert state is not None
        assert state["value"] == 42

        state_none = engine.get_session_state("nonexistent")
        assert state_none is None

    @pytest.mark.asyncio
    async def test_engine_list_sessions(self):
        """测试列出所有会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)
        engine.get_or_create_session("session_1")
        engine.get_or_create_session("session_2")

        sessions = engine.list_sessions()
        assert "session_1" in sessions
        assert "session_2" in sessions
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_engine_delete_session(self):
        """测试删除会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)
        engine.get_or_create_session("session_1")

        assert "session_1" in engine._sessions

        await engine.delete_session("session_1")

        assert "session_1" not in engine._sessions

    @pytest.mark.asyncio
    async def test_engine_multiple_sessions_isolated(self):
        """测试多会话状态隔离"""
        app = FastMind()

        async def counter_agent(state: dict, event: Event) -> dict:
            state.setdefault("count", 0)
            state["count"] += 1
            state["session_id"] = state["_session_id"]
            return state

        graph = Graph()
        graph.add_node("counter", counter_agent)
        graph.set_entry_point("counter")
        app.register_graph("main", graph)

        engine = Engine(app)
        session1 = engine.get_or_create_session("session_1")
        session2 = engine.get_or_create_session("session_2")

        await engine.start()
        await engine.push_event("session_1", Event("test", {}, "session_1"))
        await engine.push_event("session_2", Event("test", {}, "session_2"))

        await asyncio.sleep(0.3)

        state1 = engine.get_session_state("session_1")
        state2 = engine.get_session_state("session_2")

        assert state1["count"] == 1
        assert state2["count"] == 1
        assert state1["session_id"] == "session_1"
        assert state2["session_id"] == "session_2"

        await engine.push_event("session_1", Event("test", {}, "session_1"))
        await asyncio.sleep(0.2)

        state1_new = engine.get_session_state("session_1")
        assert state1_new["count"] == 2
        assert state2["count"] == 1

        await engine.stop()
