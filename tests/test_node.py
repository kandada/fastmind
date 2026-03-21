"""Node 装饰器和 Graph 执行测试"""

import pytest
import asyncio
from fastmind import FastMind
from fastmind.core.graph import Graph
from fastmind.core.event import Event


class TestAgentDecorator:
    """@app.agent 装饰器测试"""

    @pytest.mark.asyncio
    async def test_agent_decorator_basic(self):
        """测试装饰器基本功能"""
        app = FastMind()

        @app.agent(name="test_agent")
        async def test_agent(state: dict, event: Event) -> dict:
            state["executed"] = True
            return state

        assert "test_agent" in app.get_agents()

    @pytest.mark.asyncio
    async def test_agent_with_tools(self):
        """测试带工具的 Agent"""
        app = FastMind()

        @app.tool(name="test_tool")
        async def test_tool(arg: str) -> str:
            return f"Result: {arg}"

        @app.agent(name="agent_with_tools", tools=["test_tool"])
        async def agent(state: dict, event: Event) -> dict:
            return state

        agent_obj = app.get_agent("agent_with_tools")
        assert agent_obj is not None

    @pytest.mark.asyncio
    async def test_agent_execution(self):
        """测试 Agent 执行"""
        app = FastMind()

        @app.agent(name="exec_agent")
        async def exec_agent(state: dict, event: Event) -> dict:
            state["executed"] = True
            state["event_type"] = event.type
            return state

        graph = Graph()
        graph.add_node("exec_agent", exec_agent)
        graph.set_entry_point("exec_agent")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.2)

        state = api.get_state("test")
        assert state["executed"] is True
        assert state["event_type"] == "test"

        await api.stop()


class TestNodeReturnTypes:
    """节点返回类型测试"""

    @pytest.mark.asyncio
    async def test_node_return_dict(self):
        """测试节点返回 dict"""
        app = FastMind()

        async def simple_agent(state: dict, event: Event) -> dict:
            state["result"] = "dict_return"
            return state

        graph = Graph()
        graph.add_node("agent", simple_agent)
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.2)

        state = api.get_state("test")
        assert state["result"] == "dict_return"

        await api.stop()

    @pytest.mark.asyncio
    async def test_node_return_tuple(self):
        """测试节点返回 tuple (state, events)"""
        app = FastMind()

        async def event_agent(state: dict, event: Event) -> tuple[dict, list[Event]]:
            state["result"] = "tuple_return"

            events = [
                Event(type="custom.event", payload={"data": "test"}, session_id=event.session_id)
            ]

            return state, events

        graph = Graph()
        graph.add_node("agent", event_agent)
        graph.set_entry_point("agent")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.2)

        state = api.get_state("test")
        assert state["result"] == "tuple_return"

        session = api.get_session("test")
        received_events = []

        while True:
            try:
                ev = await asyncio.wait_for(session.output_queue.get(), timeout=0.2)
                received_events.append(ev)
            except asyncio.TimeoutError:
                break

        custom_events = [e for e in received_events if e.type == "custom.event"]
        assert len(custom_events) == 1
        assert custom_events[0].payload["data"] == "test"

        await api.stop()

    @pytest.mark.asyncio
    async def test_sync_node(self):
        """测试同步节点"""
        app = FastMind()

        def sync_agent(state: dict, event: Event) -> dict:
            state["sync"] = True
            return state

        graph = Graph()
        graph.add_node("sync", sync_agent)
        graph.set_entry_point("sync")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.2)

        state = api.get_state("test")
        assert state["sync"] is True

        await api.stop()


class TestGraphEdgeExecution:
    """Graph 边执行测试"""

    @pytest.mark.asyncio
    async def test_simple_edge_execution(self):
        """测试简单边执行"""
        app = FastMind()
        execution_order = []

        async def node_a(state: dict, event: Event) -> dict:
            execution_order.append("a")
            return state

        async def node_b(state: dict, event: Event) -> dict:
            execution_order.append("b")
            return state

        graph = Graph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_edge("a", "b")
        graph.set_entry_point("a")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.3)

        assert "a" in execution_order
        assert "b" in execution_order
        assert execution_order.index("a") < execution_order.index("b")

        await api.stop()

    @pytest.mark.asyncio
    async def test_conditional_edge(self):
        """测试条件边"""
        app = FastMind()
        executed_nodes = []

        async def start_node(state: dict, event: Event) -> dict:
            state["go_to_b"] = True
            executed_nodes.append("start")
            return state

        async def node_b(state: dict, event: Event) -> dict:
            executed_nodes.append("b")
            return state

        async def node_c(state: dict, event: Event) -> dict:
            executed_nodes.append("c")
            return state

        def router(state: dict, event: Event) -> str:
            return "b" if state.get("go_to_b") else "c"

        graph = Graph()
        graph.add_node("start", start_node)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_conditional_edges("start", router, {"b": "b", "c": "c"})
        graph.set_entry_point("start")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.3)

        assert "start" in executed_nodes
        assert "b" in executed_nodes
        assert "c" not in executed_nodes

        await api.stop()

    @pytest.mark.asyncio
    async def test_multiple_edges(self):
        """测试多条边（只执行第一条匹配的边）"""
        app = FastMind()
        execution_order = []

        async def node_a(state: dict, event: Event) -> dict:
            execution_order.append("a")
            return state

        async def node_b(state: dict, event: Event) -> dict:
            execution_order.append("b")
            return state

        async def node_c(state: dict, event: Event) -> dict:
            execution_order.append("c")
            return state

        graph = Graph()
        graph.add_node("a", node_a)
        graph.add_node("b", node_b)
        graph.add_node("c", node_c)
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")
        graph.set_entry_point("a")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.3)

        assert "a" in execution_order
        assert "b" in execution_order

        await api.stop()


class TestSubgraph:
    """子图测试"""

    @pytest.mark.asyncio
    async def test_subgraph_execution(self):
        """测试子图执行"""
        app = FastMind()
        executed = []

        async def sub_task(state: dict, event: Event) -> dict:
            executed.append("sub_task")
            state["sub_result"] = "done"
            return state

        child_graph = Graph()
        child_graph.add_node("sub_task", sub_task)
        child_graph.set_entry_point("sub_task")

        async def parent_node(state: dict, event: Event) -> dict:
            executed.append("parent")
            return state

        graph = Graph()
        graph.add_node("parent", parent_node)
        graph.add_node("child", child_graph)
        graph.add_edge("parent", "child")
        graph.add_edge("child", "__end__")
        graph.set_entry_point("parent")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.3)

        assert "parent" in executed
        assert "sub_task" in executed

        state = api.get_state("test")
        assert state["sub_result"] == "done"

        await api.stop()


class TestErrorHandling:
    """错误处理测试"""

    @pytest.mark.asyncio
    async def test_node_exception(self):
        """测试节点异常"""
        app = FastMind()

        async def error_node(state: dict, event: Event) -> dict:
            raise ValueError("Test error")

        graph = Graph()
        graph.add_node("error", error_node)
        graph.set_entry_point("error")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.3)

        session = api.get_session("test")
        received_events = []

        while True:
            try:
                ev = await asyncio.wait_for(session.output_queue.get(), timeout=0.2)
                received_events.append(ev)
            except asyncio.TimeoutError:
                break

        error_events = [e for e in received_events if e.type == "error"]
        assert len(error_events) >= 1

        await api.stop()

    @pytest.mark.asyncio
    async def test_nonexistent_node(self):
        """测试不存在的节点"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("simple", simple_node)
        graph.set_entry_point("nonexistent")
        app.register_graph("main", graph)

        from fastmind.contrib import FastMindAPI

        api = FastMindAPI(app)
        await api.start()

        await api.push_event("test", Event("test", {}, "test"))

        await asyncio.sleep(0.2)

        state = api.get_state("test")
        assert state is not None

        await api.stop()
