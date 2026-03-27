"""性能测试"""

import pytest
import asyncio
import time
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastmind import FastMind, Graph, Event, ToolNode
from fastmind.core.engine import Session


class TestPerformance:
    """性能测试"""

    @pytest.mark.asyncio
    async def test_session_creation_performance(self):
        """测试会话创建性能"""
        app = FastMind()
        graph = Graph()
        graph.add_node("start", lambda s, e: (s, []))
        graph.set_entry_point("start")

        start = time.time()
        for i in range(100):
            session = Session(f"perf_session_{i}", graph, app)
        elapsed = time.time() - start

        assert elapsed < 1.0, f"Creating 100 sessions took {elapsed:.2f}s, should be < 1s"

    @pytest.mark.asyncio
    async def test_event_processing_performance(self):
        """测试事件处理性能"""
        app = FastMind()
        execution_count = {"count": 0}

        async def fast_node(state: dict, event: Event) -> tuple[dict, list]:
            execution_count["count"] += 1
            return state, []

        graph = Graph()
        graph.add_node("start", fast_node)
        graph.add_edge("start", "end")
        graph.add_node("end", lambda s, e: (s, []))
        graph.set_entry_point("start")

        session = Session("perf_test", graph, app)
        await session.start()

        start = time.time()
        for i in range(50):
            await session.push_event(Event("user.message", {"text": f"test_{i}"}, "perf_test"))
        elapsed = time.time() - start

        await asyncio.sleep(1.0)
        await session.stop()

        print(f"\nProcessed 50 events in {elapsed:.2f}s ({50 / elapsed:.1f} events/sec)")

    @pytest.mark.asyncio
    async def test_state_update_performance(self):
        """测试状态更新性能"""
        app = FastMind()

        async def state_update_node(state: dict, event: Event) -> tuple[dict, list]:
            for i in range(100):
                state[f"key_{i}"] = i
            return state, []

        graph = Graph()
        graph.add_node("start", state_update_node)
        graph.add_edge("start", "end")
        graph.add_node("end", lambda s, e: (s, []))
        graph.set_entry_point("start")

        session = Session("state_perf", graph, app)
        await session.start()

        iterations = 100
        start = time.time()
        for i in range(iterations):
            await session.push_event(Event("user.message", {"text": "test"}, "state_perf"))
        elapsed = time.time() - start

        await asyncio.sleep(2.0)
        await session.stop()

        print(f"\n{iterations} iterations with 100 state updates each completed in {elapsed:.2f}s")

    @pytest.mark.asyncio
    async def test_react_loop_performance(self):
        """测试 ReAct 循环性能"""
        app = FastMind()
        execution_count = {"agent": 0, "tools": 0}

        @app.tool(name="fast_tool", description="快速工具")
        async def fast_tool(value: str) -> str:
            execution_count["tools"] += 1
            return f"processed: {value}"

        async def agent_node(state: dict, event: Event) -> tuple[dict, list]:
            execution_count["agent"] += 1
            state.setdefault("iterations", 0)
            state["iterations"] += 1

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

            if state["iterations"] < 5 and not state.get("tool_calls"):
                state["tool_calls"] = [
                    {
                        "id": f"call_{state['iterations']}",
                        "function": {"name": "fast_tool", "arguments": '{"value": "test"}'},
                    }
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

        graph = Graph(max_iterations=20)
        graph.add_node("agent", agent_node)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges(
            "agent", route, {"tools": "tools", None: "__end__", "__end__": "__end__"}
        )
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")

        session = Session("react_perf", graph, app)
        await session.start()

        start = time.time()
        await session.push_event(Event("user.message", {"text": "test"}, "react_perf"))

        timeout = 10.0
        while session.is_alive and time.time() - start < timeout:
            ev = await session.get_output()
            if ev and ev.type == "error":
                break
            await asyncio.sleep(0.01)

        await session.stop()
        elapsed = time.time() - start

        print(
            f"\nReAct loop ({execution_count['agent']} agent, {execution_count['tools']} tools) completed in {elapsed:.2f}s"
        )

    @pytest.mark.asyncio
    async def test_memory_usage(self):
        """测试内存使用"""
        app = FastMind()

        async def memory_node(state: dict, event: Event) -> tuple[dict, list]:
            state["data"] = "x" * 1000
            return state, []

        graph = Graph()
        graph.add_node("start", memory_node)
        graph.add_edge("start", "end")
        graph.add_node("end", lambda s, e: (s, []))
        graph.set_entry_point("start")

        sessions = []
        for i in range(10):
            session = Session(f"memory_test_{i}", graph, app)
            sessions.append(session)

        for session in sessions:
            await session.start()
            await session.push_event(Event("user.message", {"text": "test"}, session.session_id))

        await asyncio.sleep(0.5)

        for session in sessions:
            await session.stop()

        import gc

        gc.collect()

        print(f"\nCreated and destroyed 10 sessions without memory issues")

    @pytest.mark.asyncio
    async def test_concurrent_sessions_performance(self):
        """测试并发会话性能"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> tuple[dict, list]:
            state["processed"] = True
            return state, []

        graph = Graph()
        graph.add_node("start", simple_node)
        graph.set_entry_point("start")

        num_sessions = 20
        sessions = []

        start = time.time()
        for i in range(num_sessions):
            session = Session(f"concurrent_{i}", graph, app)
            sessions.append(session)
            await session.start()
            await session.push_event(Event("user.message", {"text": "test"}, session.session_id))

        await asyncio.sleep(1.0)

        for session in sessions:
            await session.stop()

        elapsed = time.time() - start
        print(
            f"\n{num_sessions} concurrent sessions processed in {elapsed:.2f}s ({num_sessions / elapsed:.1f} sessions/sec)"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
