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
