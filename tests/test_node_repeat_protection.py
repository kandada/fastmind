"""节点防重执行和 ReAct 循环测试"""

import pytest
import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastmind import FastMind, Graph, Event, ToolNode
from fastmind.contrib import FastMindAPI
from fastmind.core.engine import Session


class TestNodeExecution:
    """节点执行测试"""

    @pytest.mark.asyncio
    async def test_simple_node_execution(self):
        """测试简单节点执行"""
        app = FastMind()
        execution_count = {"count": 0}

        async def counting_node(state: dict, event: Event) -> tuple[dict, list]:
            execution_count["count"] += 1
            state["executions"] = execution_count["count"]
            return state, []

        graph = Graph()
        graph.add_node("start", counting_node)
        graph.add_edge("start", "end")
        graph.add_node("end", lambda s, e: (s, []))
        graph.set_entry_point("start")

        session = Session("test_simple", graph, app)
        await session.start()
        await session.push_event(Event("user.message", {"text": "test"}, "test_simple"))
        await asyncio.sleep(0.3)
        await session.stop()

        assert execution_count["count"] == 1, (
            f"Expected 1 execution, got {execution_count['count']}"
        )

    @pytest.mark.asyncio
    async def test_max_iterations_protection(self):
        """测试 max_iterations 保护机制"""
        app = FastMind()
        execution_count = {"agent": 0}

        async def looping_agent(state: dict, event: Event) -> tuple[dict, list]:
            execution_count["agent"] += 1
            return state, []

        graph = Graph(max_iterations=5)
        graph.add_node("agent", looping_agent)
        graph.add_edge("agent", "agent")
        graph.set_entry_point("agent")

        session = Session("test_max_iter", graph, app)
        await session.start()
        await session.push_event(Event("user.message", {"text": "test"}, "test_max_iter"))

        output_events = []
        timeout = 2.0
        start = asyncio.get_event_loop().time()
        while session.is_alive and (asyncio.get_event_loop().time() - start) < timeout:
            ev = await session.get_output()
            if ev:
                output_events.append(ev)
                if ev.type == "error":
                    break
            await asyncio.sleep(0.05)

        await session.stop()

        assert any(
            e.type == "error" and "max iterations" in e.payload.get("error", "")
            for e in output_events
        ), "Should hit max iterations error"


class TestReActLoop:
    """ReAct 循环测试"""

    @pytest.mark.asyncio
    async def test_react_loop_basic(self):
        """测试基本 ReAct 循环"""
        app = FastMind()
        execution_count = {"agent": 0, "tools": 0}

        @app.tool(name="test_tool", description="测试工具")
        async def test_tool(arg: str) -> str:
            execution_count["tools"] += 1
            return f"result: {arg}"

        async def agent_node(state: dict, event: Event) -> tuple[dict, list]:
            execution_count["agent"] += 1
            state.setdefault("messages", [])

            if state.get("tool_results"):
                for result in state["tool_results"]:
                    state["messages"].append(
                        {
                            "role": "tool",
                            "tool_call_id": result["tool_call_id"],
                            "content": str(result["result"]),
                        }
                    )
                del state["tool_results"]

            if execution_count["agent"] == 1:
                state["tool_calls"] = [
                    {"id": "1", "function": {"name": "test_tool", "arguments": '{"arg": "hello"}'}}
                ]
            else:
                state["_end"] = True

            return state, []

        tool_node = ToolNode(app.get_tools())

        def route(state: dict, event: Event) -> str:
            if state.get("tool_calls"):
                return "tools"
            elif state.get("_end"):
                return "__end__"
            return "agent"

        graph = Graph(max_iterations=10)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges(
            "agent", route, {"tools": "tools", None: "__end__", "__end__": "__end__"}
        )
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")

        session = Session("test_react", graph, app)
        await session.start()
        await session.push_event(Event("user.message", {"text": "test"}, "test_react"))

        timeout = 5.0
        start = asyncio.get_event_loop().time()
        while session.is_alive and (asyncio.get_event_loop().time() - start) < timeout:
            ev = await session.get_output()
            if ev and ev.type == "error" and "max iterations" in ev.payload.get("error", ""):
                break
            await asyncio.sleep(0.05)

        await session.stop()

        assert execution_count["agent"] == 2, (
            f"Expected 2 agent executions, got {execution_count['agent']}"
        )
        assert execution_count["tools"] == 1, (
            f"Expected 1 tools execution, got {execution_count['tools']}"
        )


class TestStateConsistency:
    """状态一致性测试"""

    @pytest.mark.asyncio
    async def test_state_snapshots(self):
        """测试状态在执行过程中的变化"""
        app = FastMind()
        snapshots = []

        async def agent_node(state: dict, event: Event) -> tuple[dict, list]:
            state["tool_calls"] = [{"id": "1"}]
            snapshots.append(("agent", dict(state)))
            return state, []

        async def tools_node(state: dict, event: Event) -> tuple[dict, list]:
            if "tool_calls" in state:
                del state["tool_calls"]
            state["tool_results"] = [{"id": "result"}]
            snapshots.append(("tools", dict(state)))
            return state, []

        def route(state: dict, event: Event) -> str:
            if state.get("tool_calls"):
                return "tools"
            return "end"

        graph = Graph()
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tools_node)
        graph.add_conditional_edges("agent", route, {"tools": "tools", "end": "end"})
        graph.set_entry_point("agent")

        session = Session("test_snapshots", graph, app)
        await session.start()
        await session.push_event(Event("user.message", {"text": "test"}, "test_snapshots"))
        await asyncio.sleep(0.3)
        await session.stop()

        assert len(snapshots) >= 2, "Should capture at least agent and tools snapshots"
        agent_snapshot = next(s for s in snapshots if s[0] == "agent")
        tools_snapshot = next(s for s in snapshots if s[0] == "tools")
        assert "tool_calls" in agent_snapshot[1], "Agent should set tool_calls"
        assert "tool_results" in tools_snapshot[1], "Tools should set tool_results"


class TestEdgeCases:
    """边界情况测试"""

    @pytest.mark.asyncio
    async def test_empty_tool_calls_list(self):
        """测试空 tool_calls 列表的处理"""
        app = FastMind()

        async def agent_node(state: dict, event: Event) -> tuple[dict, list]:
            state["tool_calls"] = []
            return state, []

        async def tools_node(state: dict, event: Event) -> tuple[dict, list]:
            return state, []

        def route(state: dict, event: Event) -> str:
            return "tools" if state.get("tool_calls") else "end"

        graph = Graph()
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tools_node)
        graph.add_conditional_edges("agent", route, {"tools": "tools", None: "end"})
        graph.set_entry_point("agent")

        session = Session("test_empty_tool_calls", graph, app)
        await session.start()
        await session.push_event(Event("user.message", {"text": "test"}, "test_empty_tool_calls"))
        await asyncio.sleep(0.2)
        await session.stop()

        assert session.state.get("tool_calls") == []

    @pytest.mark.asyncio
    async def test_multiple_tools(self):
        """测试多个工具调用"""
        app = FastMind()
        tool_calls_received = []

        @app.tool(name="tool_a", description="工具A")
        async def tool_a() -> str:
            tool_calls_received.append("a")
            return "A"

        @app.tool(name="tool_b", description="工具B")
        async def tool_b() -> str:
            tool_calls_received.append("b")
            return "B"

        async def agent_node(state: dict, event: Event) -> tuple[dict, list]:
            state.setdefault("messages", [])
            if not state.get("tool_results") and not state.get("tool_calls"):
                state["tool_calls"] = [
                    {"id": "1", "function": {"name": "tool_a", "arguments": "{}"}},
                    {"id": "2", "function": {"name": "tool_b", "arguments": "{}"}},
                ]
            return state, []

        tool_node = ToolNode(app.get_tools())

        def route(state: dict, event: Event) -> str:
            if state.get("tool_calls"):
                return "tools"
            elif state.get("_end"):
                return "__end__"
            return "agent"

        graph = Graph(max_iterations=10)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges(
            "agent", route, {"tools": "tools", None: "__end__", "__end__": "__end__"}
        )
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")

        session = Session("test_multiple_tools", graph, app)
        await session.start()
        await session.push_event(Event("user.message", {"text": "test"}, "test_multiple_tools"))

        timeout = 5.0
        start = asyncio.get_event_loop().time()
        while session.is_alive and (asyncio.get_event_loop().time() - start) < timeout:
            ev = await session.get_output()
            if ev and ev.type == "error" and "max iterations" in ev.payload.get("error", ""):
                break
            await asyncio.sleep(0.05)

        await session.stop()

        assert "a" in tool_calls_received, "tool_a should be called"
        assert "b" in tool_calls_received, "tool_b should be called"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
