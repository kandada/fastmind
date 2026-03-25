"""FastMindAPI 单元测试"""

import pytest
import asyncio
from fastmind import FastMind
from fastmind.core.graph import Graph
from fastmind.core.event import Event
from fastmind.contrib import FastMindAPI


class TestFastMindAPI:
    """FastMindAPI 测试"""

    @pytest.mark.asyncio
    async def test_api_initialization(self):
        """测试 API 初始化"""
        app = FastMind()
        api = FastMindAPI(app)

        assert api.app == app
        assert api._running is False

    @pytest.mark.asyncio
    async def test_api_start_stop(self):
        """测试 API 启动和停止"""
        app = FastMind()
        api = FastMindAPI(app)

        await api.start()
        assert api._running is True

        await api.stop()
        assert api._running is False

    @pytest.mark.asyncio
    async def test_api_push_event(self):
        """测试推送事件"""
        app = FastMind()

        async def echo_agent(state: dict, event: Event) -> dict:
            state["received"] = event.payload.get("text", "")
            return state

        graph = Graph()
        graph.add_node("echo", echo_agent)
        graph.set_entry_point("echo")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        event = Event("user.message", {"text": "Hello"}, "test_session")
        session = await api.push_event("test_session", event)

        assert session is not None
        assert session.session_id == "test_session"

        await asyncio.sleep(0.2)

        state = api.get_state("test_session")
        assert state["received"] == "Hello"

        await api.stop()

    @pytest.mark.asyncio
    async def test_api_get_session(self):
        """测试获取会话"""
        app = FastMind()

        async def simple_agent(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("simple", simple_agent)
        graph.set_entry_point("simple")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("session_1", Event("test", {}, "session_1"))

        session = api.get_session("session_1")
        assert session is not None
        assert session.session_id == "session_1"

        session_none = api.get_session("nonexistent")
        assert session_none is None

        await api.stop()

    @pytest.mark.asyncio
    async def test_api_get_state(self):
        """测试获取状态"""
        app = FastMind()

        async def agent(state: dict, event: Event) -> dict:
            state["value"] = 42
            state["name"] = event.payload.get("name", "unknown")
            return state

        graph = Graph()
        graph.add_node("agent", agent)
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("session_1", Event("test", {"name": "test"}, "session_1"))

        await asyncio.sleep(0.2)

        state = api.get_state("session_1")
        assert state is not None
        assert state["value"] == 42
        assert state["name"] == "test"

        state_none = api.get_state("nonexistent")
        assert state_none is None

        await api.stop()

    @pytest.mark.asyncio
    async def test_api_list_sessions(self):
        """测试列出会话"""
        app = FastMind()

        async def simple_agent(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("simple", simple_agent)
        graph.set_entry_point("simple")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("s1", Event("test", {}, "s1"))
        await api.push_event("s2", Event("test", {}, "s2"))

        await asyncio.sleep(0.2)

        sessions = api.list_sessions()
        assert "s1" in sessions
        assert "s2" in sessions

        await api.stop()

    @pytest.mark.asyncio
    async def test_api_delete_session(self):
        """测试删除会话"""
        app = FastMind()

        async def simple_agent(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("simple", simple_agent)
        graph.set_entry_point("simple")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("temp", Event("test", {}, "temp"))

        await asyncio.sleep(0.1)

        assert api.get_session("temp") is not None

        await api.delete_session("temp")

        assert api.get_session("temp") is None

        await api.stop()

    @pytest.mark.asyncio
    async def test_api_get_tool_schemas(self):
        """测试获取工具 schema"""
        app = FastMind()

        @app.tool(name="get_weather", description="获取天气")
        async def get_weather(city: str) -> str:
            return f"{city} 天气晴朗"

        schemas = app.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "get_weather"


class TestStreamEvents:
    """stream_events 测试"""

    @pytest.mark.asyncio
    async def test_stream_events_basic(self):
        """测试基本流式事件"""
        app = FastMind()

        @app.agent(name="streaming_agent")
        async def streaming_agent(state: dict, event: Event) -> dict:
            output_queue = state["_output_queue"]
            session_id = state["_session_id"]

            for i in range(3):
                output_queue.put_nowait(
                    Event(type="stream.chunk", payload={"delta": str(i)}, session_id=session_id)
                )
                await asyncio.sleep(0.05)

            output_queue.put_nowait(Event(type="stream.end", payload={}, session_id=session_id))
            return state

        graph = Graph()
        graph.add_node("agent", streaming_agent)
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("start", {}, "test"))

        chunks = []
        async for ev in api.stream_events("test"):
            if ev.type == "stream.chunk":
                chunks.append(ev.payload.get("delta", ""))
            elif ev.type == "stream.end":
                break

        assert len(chunks) == 3
        assert "".join(chunks) == "012"

        await api.stop()

    @pytest.mark.asyncio
    async def test_stream_events_with_filter(self):
        """测试流式事件过滤"""
        app = FastMind()

        @app.agent(name="filter_agent")
        async def filter_agent(state: dict, event: Event) -> dict:
            output_queue = state["_output_queue"]
            session_id = state["_session_id"]

            output_queue.put_nowait(
                Event(type="custom.event", payload={"data": "1"}, session_id=session_id)
            )
            output_queue.put_nowait(
                Event(type="stream.chunk", payload={"delta": "A"}, session_id=session_id)
            )
            output_queue.put_nowait(
                Event(type="custom.event", payload={"data": "2"}, session_id=session_id)
            )
            output_queue.put_nowait(
                Event(type="stream.chunk", payload={"delta": "B"}, session_id=session_id)
            )
            output_queue.put_nowait(Event(type="stream.end", payload={}, session_id=session_id))
            return state

        graph = Graph()
        graph.add_node("agent", filter_agent)
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("start", {}, "test"))

        chunks = []
        async for ev in api.stream_events("test", event_types=["stream.chunk"]):
            if ev.type == "stream.chunk":
                chunks.append(ev.payload.get("delta", ""))

        assert len(chunks) == 2
        assert "".join(chunks) == "AB"

        await api.stop()

    @pytest.mark.asyncio
    async def test_stream_events_auto_stop_on_end(self):
        """测试流式事件自动停止"""
        app = FastMind()

        @app.agent(name="end_agent")
        async def end_agent(state: dict, event: Event) -> dict:
            output_queue = state["_output_queue"]
            session_id = state["_session_id"]

            output_queue.put_nowait(
                Event(type="stream.chunk", payload={"delta": "X"}, session_id=session_id)
            )
            output_queue.put_nowait(Event(type="stream.end", payload={}, session_id=session_id))
            output_queue.put_nowait(
                Event(type="stream.chunk", payload={"delta": "Y"}, session_id=session_id)
            )
            return state

        graph = Graph()
        graph.add_node("agent", end_agent)
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("start", {}, "test"))

        chunks = []
        count = 0
        async for ev in api.stream_events("test"):
            count += 1
            if ev.type == "stream.chunk":
                chunks.append(ev.payload.get("delta", ""))
            elif ev.type == "stream.end":
                break

        assert len(chunks) == 1
        assert chunks[0] == "X"
        assert count == 2

        await api.stop()


class TestRunStreaming:
    """run_streaming 测试"""

    @pytest.mark.asyncio
    async def test_run_streaming_basic(self):
        """测试基本流式对话"""
        app = FastMind()

        @app.agent(name="streaming_chat")
        async def streaming_chat(state: dict, event: Event) -> dict:
            output_queue = state["_output_queue"]
            session_id = state["_session_id"]

            text = "Hello, World!"
            for char in text:
                output_queue.put_nowait(
                    Event(type="stream.chunk", payload={"delta": char}, session_id=session_id)
                )
                await asyncio.sleep(0.02)

            output_queue.put_nowait(Event(type="stream.end", payload={}, session_id=session_id))
            return state

        graph = Graph()
        graph.add_node("agent", streaming_chat)
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        received_chunks = []

        full_text = await api.run_streaming(
            "test", "Hi", on_chunk=lambda delta: received_chunks.append(delta)
        )

        assert "".join(received_chunks) == "Hello, World!"
        assert full_text == "Hello, World!"

        await api.stop()

    @pytest.mark.asyncio
    async def test_run_streaming_with_on_end(self):
        """测试流式对话 with on_end 回调"""
        app = FastMind()

        @app.agent(name="end_chat")
        async def end_chat(state: dict, event: Event) -> dict:
            output_queue = state["_output_queue"]
            session_id = state["_session_id"]

            for char in "Test":
                output_queue.put_nowait(
                    Event(type="stream.chunk", payload={"delta": char}, session_id=session_id)
                )
                await asyncio.sleep(0.02)

            output_queue.put_nowait(Event(type="stream.end", payload={}, session_id=session_id))
            return state

        graph = Graph()
        graph.add_node("agent", end_chat)
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        end_callback_received = []

        full_text = await api.run_streaming(
            "test",
            "Hi",
            on_chunk=lambda delta: None,
            on_end=lambda text: end_callback_received.append(text),
        )

        assert len(end_callback_received) == 1
        assert end_callback_received[0] == "Test"
        assert full_text == "Test"

        await api.stop()


class TestFastMindAPIEdgeCases:
    """FastMindAPI 边界情况测试"""

    @pytest.mark.asyncio
    async def test_api_without_graph(self):
        """测试没有注册图的情况"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("simple", simple_node)
        graph.set_entry_point("simple")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        session = await api.push_event("test", Event("test", {}, "test"))

        assert session is not None

        await api.stop()

    @pytest.mark.asyncio
    async def test_multiple_push_events(self):
        """测试连续推送多个事件"""
        app = FastMind()

        async def counter_agent(state: dict, event: Event) -> dict:
            state.setdefault("count", 0)
            state["count"] += 1
            return state

        graph = Graph()
        graph.add_node("counter", counter_agent)
        graph.set_entry_point("counter")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        for i in range(5):
            await api.push_event("test", Event("inc", {}, "test"))

        await asyncio.sleep(0.3)

        state = api.get_state("test")
        assert state["count"] == 5

        await api.stop()

    @pytest.mark.asyncio
    async def test_stream_events_nonexistent_session(self):
        """测试获取不存在会话的事件流应抛出异常"""
        app = FastMind()
        api = FastMindAPI(app)
        await api.start()

        with pytest.raises(RuntimeError, match="does not exist"):
            async for _ in api.stream_events("nonexistent"):
                pass

        await api.stop()

    @pytest.mark.asyncio
    async def test_push_event_none_raises(self):
        """测试 push_event 接收 None 应抛出异常"""
        app = FastMind()
        api = FastMindAPI(app)
        await api.start()

        with pytest.raises(ValueError, match="event cannot be None"):
            await api.push_event("s1", None)

        await api.stop()

    @pytest.mark.asyncio
    async def test_resume_nonexistent_session_raises(self):
        """测试恢复不存在的会话应抛出异常"""
        app = FastMind()
        api = FastMindAPI(app)
        await api.start()

        with pytest.raises(ValueError, match="does not exist"):
            await api.resume_session("nonexistent", "input")

        await api.stop()

    @pytest.mark.asyncio
    async def test_resume_non_interrupted_session_raises(self):
        """测试恢复非中断状态的会话应抛出异常"""
        from fastmind import Graph

        app = FastMind()

        async def agent(state, event):
            return state

        graph = Graph()
        graph.add_node("start", agent)
        graph.set_entry_point("start")
        app.register_graph("main", graph)

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("s1", Event("test", {}, "s1"))
        await asyncio.sleep(0.2)

        with pytest.raises(RuntimeError, match="is not interrupted"):
            await api.resume_session("s1", "input")

        await api.stop()
