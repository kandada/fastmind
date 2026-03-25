"""集成测试"""

import pytest
import asyncio
from fastmind import FastMind
from fastmind.core.graph import Graph
from fastmind.core.event import Event
from fastmind.core.tool import ToolNode
from fastmind.contrib import FastMindAPI


class TestIntegration:
    """集成测试"""

    @pytest.mark.asyncio
    async def test_simple_chat_flow(self):
        """测试简单对话流程"""
        app = FastMind()

        async def echo_agent(state: dict, event: Event) -> dict:
            state.setdefault("messages", [])
            state["messages"].append({"role": "user", "content": event.payload.get("text", "")})
            state["messages"].append(
                {"role": "assistant", "content": f"Echo: {event.payload.get('text', '')}"}
            )
            return state

        graph = Graph()
        graph.add_node("chat", echo_agent)
        graph.set_entry_point("chat")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        event = Event("user.message", {"text": "Hello"}, "user_001")
        await fm_api.push_event("user_001", event)

        await asyncio.sleep(0.5)

        state = fm_api.get_state("user_001")
        assert state is not None
        assert len(state["messages"]) == 2
        assert state["messages"][0]["content"] == "Hello"
        assert state["messages"][1]["content"] == "Echo: Hello"

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_agent_tool_agent_flow(self):
        """测试 Agent-Tool-Agent 流程"""
        app = FastMind()

        @app.tool(name="get_weather", description="获取天气")
        async def get_weather(city: str) -> str:
            return f"{city} 天气晴朗"

        async def agent(state: dict, event: Event) -> dict:
            state.setdefault("messages", [])
            state["messages"].append({"role": "user", "content": event.payload.get("text", "")})

            if state.get("tool_results"):
                last_result = state["tool_results"][-1]["result"]
                state["messages"].append(
                    {"role": "assistant", "content": f"工具返回: {last_result}"}
                )
                state.pop("tool_results", None)
            elif state.get("tool_calls"):
                pass
            else:
                state["messages"].append({"role": "assistant", "content": "需要查询天气"})
                state["tool_calls"] = [
                    {
                        "id": "call_1",
                        "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
                    }
                ]

            return state

        tool_node = ToolNode(app.get_tools())

        def has_tool_calls(state: dict, event: Event) -> str:
            return "tools" if state.get("tool_calls") else None

        graph = Graph()
        graph.add_node("agent", agent)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges("agent", has_tool_calls, {"tools": "tools", None: "__end__"})
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        event = Event("user.message", {"text": "北京天气如何？"}, "user_001")
        await fm_api.push_event("user_001", event)

        await asyncio.sleep(1)

        state = fm_api.get_state("user_001")
        assert state is not None
        assert "tool_results" in state or "北京 天气晴朗" in str(state.get("messages", []))

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_multi_session(self):
        """测试多会话"""
        app = FastMind()

        async def counter_agent(state: dict, event: Event) -> dict:
            state.setdefault("count", 0)
            state["count"] += 1
            return state

        graph = Graph()
        graph.add_node("counter", counter_agent)
        graph.set_entry_point("counter")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        await fm_api.push_event("session_1", Event("test", {}, "session_1"))
        await fm_api.push_event("session_2", Event("test", {}, "session_2"))
        await fm_api.push_event("session_1", Event("test", {}, "session_1"))

        await asyncio.sleep(0.5)

        state1 = fm_api.get_state("session_1")
        state2 = fm_api.get_state("session_2")

        assert state1 is not None
        assert state2 is not None
        assert state1["count"] == 2
        assert state2["count"] == 1

        sessions = fm_api.list_sessions()
        assert "session_1" in sessions
        assert "session_2" in sessions

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_delete_session(self):
        """测试删除会话"""
        app = FastMind()

        async def simple_agent(state: dict, event: Event) -> dict:
            state["executed"] = True
            return state

        graph = Graph()
        graph.add_node("simple", simple_agent)
        graph.set_entry_point("simple")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        await fm_api.push_event("temp_session", Event("test", {}, "temp_session"))
        await asyncio.sleep(0.1)

        state = fm_api.get_state("temp_session")
        assert state is not None

        await fm_api.delete_session("temp_session")

        state = fm_api.get_state("temp_session")
        assert state is None

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_node_error_handling(self):
        """测试节点错误处理 - 错误不应影响整个会话"""
        app = FastMind()

        async def error_agent(state: dict, event: Event) -> dict:
            if event.payload.get("cause_error"):
                raise ValueError("Intentional error for testing")
            state["executed"] = True
            return state

        async def normal_agent(state: dict, event: Event) -> dict:
            state["after_error"] = "success"
            return state

        graph = Graph()
        graph.add_node("error", error_agent)
        graph.add_node("normal", normal_agent)
        graph.add_edge("error", "normal")
        graph.set_entry_point("error")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        event = Event("user.message", {"text": "test", "cause_error": True}, "error_session")
        await fm_api.push_event("error_session", event)
        await asyncio.sleep(0.5)

        session = fm_api.get_session("error_session")
        assert session is not None

        error_event = None
        while True:
            ev = await session.get_output()
            if ev is None:
                break
            if ev.type == "error":
                error_event = ev
                break

        assert error_event is not None, "Error event should be captured"
        assert "Intentional error" in error_event.payload["error"]

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_logging_module(self):
        """测试日志模块"""
        from fastmind.utils.logging import get_logger

        logger1 = get_logger("test1")
        logger2 = get_logger("test1")

        assert logger1 is logger2, "Same name should return same logger"

        logger = get_logger("test_unique")
        assert logger.name == "test_unique"
        assert logger.level is not None


class TestRobustness:
    """健壮性测试"""

    @pytest.mark.asyncio
    async def test_idempotency(self):
        """测试幂等性 - 相同事件多次发送只处理一次"""
        app = FastMind()

        call_count = 0

        async def counting_agent(state: dict, event: Event) -> dict:
            nonlocal call_count
            call_count += 1
            state["call_count"] = call_count
            return state

        graph = Graph()
        graph.add_node("count", counting_agent)
        graph.set_entry_point("count")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        event = Event("test", {"data": "same"}, "session_id")

        await fm_api.push_event("session_id", event)
        await asyncio.sleep(0.2)

        session = fm_api.get_session("session_id")
        assert session._is_event_processed(event) is True

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_concurrent_sessions(self):
        """测试并发会话 - 多个会话同时运行互不干扰"""
        app = FastMind()

        async def session_agent(state: dict, event: Event) -> dict:
            state["session_id"] = event.session_id
            state["value"] = state.get("value", 0) + 1
            return state

        graph = Graph()
        graph.add_node("main", session_agent)
        graph.set_entry_point("main")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        tasks = []
        for i in range(5):
            task = asyncio.create_task(
                fm_api.push_event(f"concurrent_{i}", Event("test", {}, f"concurrent_{i}"))
            )
            tasks.append(task)

        await asyncio.gather(*tasks)
        await asyncio.sleep(0.5)

        for i in range(5):
            state = fm_api.get_state(f"concurrent_{i}")
            assert state is not None
            assert state.get("session_id") == f"concurrent_{i}"

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_session_recovery_after_error(self):
        """测试错误后会话恢复 - 节点出错后仍能处理新事件"""
        app = FastMind()

        error_count = 0

        async def fragile_agent(state: dict, event: Event) -> dict:
            nonlocal error_count
            if event.payload.get("fail"):
                error_count += 1
                raise RuntimeError("Simulated failure")
            state["success"] = True
            return state

        graph = Graph()
        graph.add_node("fragile", fragile_agent)
        graph.set_entry_point("fragile")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        event_fail = Event("test", {"fail": True}, "recovery_session")
        await fm_api.push_event("recovery_session", event_fail)
        await asyncio.sleep(0.3)

        event_success = Event("test", {"fail": False}, "recovery_session")
        await fm_api.push_event("recovery_session", event_success)
        await asyncio.sleep(0.3)

        state = fm_api.get_state("recovery_session")
        assert state is not None
        assert state.get("success") is True

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_multiple_events_same_session(self):
        """测试同一会话多个事件 - 事件按顺序处理"""
        app = FastMind()

        results = []

        async def collector_agent(state: dict, event: Event) -> dict:
            results.append(event.payload.get("seq"))
            return state

        graph = Graph()
        graph.add_node("collector", collector_agent)
        graph.set_entry_point("collector")
        app.register_graph("main", graph)

        fm_api = FastMindAPI(app)
        await fm_api.start()

        for i in range(5):
            await fm_api.push_event("seq_session", Event("test", {"seq": i}, "seq_session"))
            await asyncio.sleep(0.1)

        await asyncio.sleep(0.5)

        assert len(results) == 5
        assert results == [0, 1, 2, 3, 4]

        await fm_api.stop()

    @pytest.mark.asyncio
    async def test_graph_validate_with_cycles(self):
        """测试循环图验证"""
        from fastmind.core.graph import GraphValidationError

        g = Graph()
        g.add_node("a", lambda s, e: s)
        g.add_node("b", lambda s, e: s)
        g.add_node("c", lambda s, e: s)
        g.add_edge("a", "b")
        g.add_edge("b", "c")
        g.add_edge("c", "a")
        g.set_entry_point("a")

        warnings = g.validate()
        assert any("cycle" in w.lower() for w in warnings)

    @pytest.mark.asyncio
    async def test_tool_node_with_empty_calls(self):
        """测试 ToolNode 处理空 tool_calls"""
        from fastmind.core.tool import ToolNode

        app = FastMind()
        tool_node = ToolNode(app.get_tools())

        state = {"tool_calls": []}
        new_state, events = await tool_node.execute(state, Event("test", {}, "s1"))

        assert new_state == state
        assert events == []

    @pytest.mark.asyncio
    async def test_state_message_helper_methods(self):
        """测试 State 消息辅助方法"""
        from fastmind.core.state import State

        state = State()

        state.add_message("msgs", "user", "hello")
        state.add_message("msgs", "assistant", "hi")

        assert state.get_message_count("msgs") == 2

        last = state.get_last_message("msgs")
        assert last["content"] == "hi"

        last_user = state.get_last_message("msgs", role="user")
        assert last_user["content"] == "hello"

        popped = state.pop_messages("msgs", 1)
        assert len(popped) == 1
        assert state.get_message_count("msgs") == 1

    @pytest.mark.asyncio
    async def test_reserved_node_names_blocked(self):
        """测试保留节点名称被阻止"""
        from fastmind.core.graph import GraphValidationError

        g = Graph()

        with pytest.raises(GraphValidationError):
            g.add_node("__start__", lambda s, e: s)

        with pytest.raises(GraphValidationError):
            g.add_node("__end__", lambda s, e: s)
