"""Graph 类的单元测试"""

import pytest
from fastmind.core.graph import Graph, GraphValidationError
from fastmind.core.event import Event


class TestGraph:
    """Graph 测试"""

    @pytest.fixture
    def sample_node(self):
        """示例节点"""

        async def node(state: dict, event: Event) -> dict:
            state["executed"] = True
            return state

        return node

    def test_create_graph(self):
        """测试创建图"""
        graph = Graph()
        assert graph.name is not None
        assert len(graph.nodes) == 0

    def test_add_node(self, sample_node):
        """测试添加节点"""
        graph = Graph()
        graph.add_node("start", sample_node)

        assert "start" in graph.nodes
        assert graph.get_node("start") == sample_node

    def test_add_edge(self, sample_node):
        """测试添加边"""
        graph = Graph()
        graph.add_node("start", sample_node)
        graph.add_node("end", sample_node)
        graph.add_edge("start", "end")

        assert len(graph.edges["start"]) == 1
        assert graph.edges["start"][0]["target"] == "end"

    def test_add_conditional_edges(self, sample_node):
        """测试添加条件边"""
        graph = Graph()
        graph.add_node("agent", sample_node)
        graph.add_node("tools", sample_node)

        def router(state: dict, event: Event) -> str:
            return "tools" if state.get("tool_calls") else None

        graph.add_conditional_edges("agent", router, {"tools": "tools", None: "__end__"})

        assert "agent" in graph.conditional_edges

    def test_set_entry_point(self, sample_node):
        """测试设置入口点"""
        graph = Graph()
        graph.add_node("start", sample_node)
        graph.set_entry_point("start")

        assert graph.entry_point == "start"

    def test_get_next_node_simple_edge(self, sample_node):
        """测试普通边的下一个节点"""
        graph = Graph()
        graph.add_node("start", sample_node)
        graph.add_node("end", sample_node)
        graph.add_edge("start", "end")
        graph.set_entry_point("start")

        state = {}
        event = Event("test", {}, "session_001")

        next_node = graph.get_next_node("start", state, event)
        assert next_node == "end"

    def test_get_next_node_conditional_edge(self, sample_node):
        """测试条件边的下一个节点"""
        graph = Graph()
        graph.add_node("agent", sample_node)
        graph.add_node("tools", sample_node)

        def router(state: dict, event: Event) -> str:
            return "tools" if state.get("tool_calls") else None

        graph.add_conditional_edges("agent", router, {"tools": "tools", None: "__end__"})

        state_with_tool = {"tool_calls": [{"id": "1"}]}
        state_without_tool = {}
        event = Event("test", {}, "session_001")

        next_with = graph.get_next_node("agent", state_with_tool, event)
        assert next_with == "tools"

        next_without = graph.get_next_node("agent", state_without_tool, event)
        assert next_without is None

    def test_chain_methods(self, sample_node):
        """测试链式调用"""
        graph = Graph()
        graph.add_node("start", sample_node).add_node("end", sample_node).add_edge(
            "start", "end"
        ).set_entry_point("start")

        assert graph.has_node("start")
        assert graph.has_node("end")
        assert graph.entry_point == "start"

    def test_graph_repr(self, sample_node):
        """测试图的字符串表示"""
        graph = Graph(name="test_graph")
        graph.add_node("start", sample_node)

        repr_str = repr(graph)
        assert "test_graph" in repr_str
        assert "start" in repr_str

    def test_reserved_node_name(self, sample_node):
        """测试保留名称检测"""
        graph = Graph()

        with pytest.raises(GraphValidationError):
            graph.add_node("__start__", sample_node)

        with pytest.raises(GraphValidationError):
            graph.add_node("__end__", sample_node)

    def test_detect_cycles(self, sample_node):
        """测试循环检测"""
        graph = Graph()
        graph.add_node("a", sample_node)
        graph.add_node("b", sample_node)
        graph.add_node("c", sample_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", "c")
        graph.add_edge("c", "a")
        graph.set_entry_point("a")

        cycles = graph.detect_cycles()
        assert len(cycles) > 0

    def test_no_cycles(self, sample_node):
        """测试无循环"""
        graph = Graph()
        graph.add_node("a", sample_node)
        graph.add_node("b", sample_node)
        graph.add_edge("a", "b")
        graph.set_entry_point("a")

        cycles = graph.detect_cycles()
        assert len(cycles) == 0

    def test_validate_warnings(self, sample_node):
        """测试验证警告"""
        graph = Graph()
        graph.add_node("a", sample_node)
        graph.add_edge("a", "nonexistent")
        graph.set_entry_point("a")

        warnings = graph.validate()
        assert len(warnings) > 0
        assert any("nonexistent" in w for w in warnings)

    def test_validate_cycle_warning(self, sample_node):
        """测试循环警告"""
        graph = Graph()
        graph.add_node("a", sample_node)
        graph.add_node("b", sample_node)
        graph.add_edge("a", "b")
        graph.add_edge("b", "a")
        graph.set_entry_point("a")

        warnings = graph.validate()
        assert any("cycle" in w.lower() for w in warnings)

    def test_get_all_edges(self, sample_node):
        """测试获取所有边"""
        graph = Graph()
        graph.add_node("a", sample_node)
        graph.add_node("b", sample_node)
        graph.add_node("c", sample_node)
        graph.add_edge("a", "b")
        graph.add_edge("a", "c")

        edges = graph.get_all_edges("a")
        assert len(edges) == 2

    def test_conditional_edges_with_fallback(self, sample_node):
        """测试带 fallback 的条件边"""
        graph = Graph()
        graph.add_node("agent", sample_node)
        graph.add_node("tools", sample_node)
        graph.add_node("fallback", sample_node)

        def router(state: dict, event: Event) -> str:
            return "unknown_route"

        graph.add_conditional_edges(
            "agent",
            router,
            {"tools": "tools"},
            fallback_edges=[{"target": "fallback", "condition": None}],
        )

        state = {}
        event = Event("test", {}, "session_001")
        next_node = graph.get_next_node("agent", state, event)
        assert next_node == "fallback"

    def test_max_iterations(self, sample_node):
        """测试最大迭代次数配置"""
        graph = Graph(max_iterations=50)
        assert graph.max_iterations == 50

    def test_default_max_iterations(self):
        """测试默认最大迭代次数"""
        graph = Graph()
        assert graph.max_iterations == 999


class TestGraphInterrupt:
    """中断节点测试"""

    def test_add_interrupt(self):
        """测试添加中断节点"""
        graph = Graph()
        graph.add_interrupt(
            name="ask_confirm", prompt="确认执行？", resume_node="confirm", cancel_node="cancel"
        )

        assert "ask_confirm" in graph.nodes

        assert "ask_confirm" in graph._interrupt_nodes
        assert graph._interrupt_nodes["ask_confirm"]["prompt"] == "确认执行？"
