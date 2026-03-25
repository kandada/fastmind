"""Graph 类定义 - 状态图"""

from __future__ import annotations

from typing import Callable, Any, Optional, Union, TYPE_CHECKING
from collections import defaultdict
import asyncio

if TYPE_CHECKING:
    from .event import Event


NodeFunc = Callable[..., Union[dict, tuple[dict, list["Event"]]]]
RouterFunc = Callable[..., Optional[str]]
EdgeType = dict[str, Any]


class GraphValidationError(Exception):
    """Graph 验证错误"""

    pass


class GraphCycleError(Exception):
    """图循环错误"""

    pass


class Graph:
    """状态图类，包含节点和边的集合

    用于定义智能体工作流，支持：
    - 添加普通节点和条件边
    - 设置入口点和结束点
    - 支持子图嵌套

    图执行语义:
    1. 从 entry_point 开始执行
    2. 节点执行后，通过 get_next_node 确定下一个节点
    3. 如果有条件边（add_conditional_edges），使用 router 函数决定路由
    4. 如果是普通边（add_edge），按顺序检查 condition 函数
    5. 返回 None 则结束，或到达 __end__ 节点

    注意: 条件边和普通边可以同时存在，条件边优先

    用法示例:
        graph = Graph()
        graph.add_node("start", start_node)
        graph.add_node("end", end_node)
        graph.add_edge("start", "end")
        graph.set_entry_point("start")
    """

    START_NODE = "__start__"
    END_NODE = "__end__"
    DEFAULT_MAX_ITERATIONS = 999

    def __init__(self, name: Optional[str] = None, max_iterations: Optional[int] = None):
        """初始化 Graph

        Args:
            name: 图的名称
            max_iterations: 最大迭代次数，防止无限循环。None 表示禁用检测
        """
        self.name = name or f"graph_{id(self)}"
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, list[EdgeType]] = defaultdict(list)
        self.conditional_edges: dict[str, dict[str, Any]] = {}
        self.entry_point: Optional[str] = None
        self._interrupt_nodes: dict[str, dict[str, Any]] = {}
        self.max_iterations = max_iterations or self.DEFAULT_MAX_ITERATIONS

    def add_node(self, name: str, node: Union[NodeFunc, "Graph"]) -> "Graph":
        """添加节点

        Args:
            name: 节点名称
            node: 节点函数或子图

        Returns:
            self，支持链式调用
        """
        if name in (self.START_NODE, self.END_NODE):
            raise GraphValidationError(f"Cannot use reserved node name: {name}")
        self.nodes[name] = node
        return self

    def add_edge(self, source: str, target: str, condition: Optional[Callable] = None) -> "Graph":
        """添加普通边（无条件跳转）

        Args:
            source: 源节点名称
            target: 目标节点名称，None 或 "__end__" 表示结束
            condition: 可选的条件函数，返回 True 时才跳转

        Returns:
            self，支持链式调用
        """
        self.edges[source].append({"target": target, "condition": condition})
        return self

    def add_conditional_edges(
        self,
        source: str,
        router: RouterFunc,
        path_map: dict[Optional[str], str],
        fallback_edges: Optional[list[EdgeType]] = None,
    ) -> "Graph":
        """添加条件边

        条件边根据 router 函数的返回值决定下一个节点。
        如果 router 返回的值在 path_map 中有对应条目，则跳转到该节点。
        如果没有对应条目（如返回 None 但 path_map 中没有 None 键），则使用 fallback_edges。

        Args:
            source: 源节点名称
            router: 路由函数，接收 (state, event)，返回目标节点标识或 None
            path_map: 路由返回值到目标节点的映射，如 {"tools": "tools", None: "__end__"}
            fallback_edges: 可选的备用边列表，当 router 返回值不在 path_map 中时使用

        Returns:
            self，支持链式调用

        用法示例:
            def has_tool_calls(state, event):
                return "tools" if state.get("tool_calls") else None

            graph.add_conditional_edges(
                "agent",
                has_tool_calls,
                {"tools": "tools", None: "__end__"}
            )
        """
        if source not in self.nodes:
            raise GraphValidationError(
                f"Source node '{source}' must be added before adding conditional edges"
            )
        self.conditional_edges[source] = {
            "router": router,
            "path_map": path_map,
            "fallback_edges": fallback_edges or [],
        }
        return self

    def add_interrupt(
        self,
        name: str,
        prompt: str,
        resume_node: str = "resume",
        cancel_node: Optional[str] = None,
    ) -> "Graph":
        """添加中断节点（用于 Human-in-the-loop）"""

        async def interrupt_node(state: dict, event: Event) -> tuple[dict, list[Event]]:
            return state, [
                Event(
                    type="interrupt",
                    payload={
                        "prompt": prompt,
                        "resume_node": resume_node,
                        "cancel_node": cancel_node,
                    },
                    session_id=event.session_id,
                )
            ]

        self.nodes[name] = interrupt_node
        self._interrupt_nodes[name] = {
            "prompt": prompt,
            "resume_node": resume_node,
            "cancel_node": cancel_node,
        }
        return self

    def set_entry_point(self, name: str) -> "Graph":
        """设置入口节点"""
        self.entry_point = name
        return self

    def get_next_node(self, current_node: str, state: dict, event: Event) -> Optional[str]:
        """获取下一个节点名称

        执行流程:
        1. 如果有条件边，使用 router 决定路由
        2. 如果 router 返回值在 path_map 中有对应节点，使用该节点
        3. 如果 router 返回值不在 path_map 中，尝试 fallback_edges
        4. 如果没有条件边或 fallback，检查普通边

        Args:
            current_node: 当前节点名称
            state: 当前状态
            event: 当前事件

        Returns:
            下一个节点名称，如果结束则返回 None
        """
        if current_node in self.conditional_edges:
            router_info = self.conditional_edges[current_node]
            router = router_info["router"]
            path_map = router_info["path_map"]
            fallback_edges = router_info.get("fallback_edges", [])

            route_key = router(state, event)

            if route_key in path_map:
                next_node = path_map[route_key]
                if next_node == self.END_NODE:
                    return None
                return next_node

            for edge in fallback_edges:
                condition = edge.get("condition")
                if condition is None or condition(state):
                    target = edge["target"]
                    if target == self.END_NODE:
                        return None
                    return target

            return None

        edges = self.edges.get(current_node, [])
        for edge in edges:
            condition = edge.get("condition")
            if condition is None or condition(state):
                target = edge["target"]
                if target == self.END_NODE:
                    return None
                return target

        return None

    def get_node(self, name: str) -> Optional[Union[NodeFunc, "Graph"]]:
        """获取节点"""
        return self.nodes.get(name)

    def has_node(self, name: str) -> bool:
        """检查节点是否存在"""
        return name in self.nodes

    def get_all_edges(self, node_name: str) -> list[EdgeType]:
        """获取节点的所有边（条件边转为普通边）"""
        edges = list(self.edges.get(node_name, []))

        if node_name in self.conditional_edges:
            info = self.conditional_edges[node_name]
            for route_key, target in info["path_map"].items():
                edges.append(
                    {
                        "target": target,
                        "condition": lambda s, e, key=route_key: True,
                        "route_key": route_key,
                    }
                )
            edges.extend(info.get("fallback_edges", []))

        return edges

    def detect_cycles(self) -> list[list[str]]:
        """检测图中的循环路径

        Returns:
            循环路径列表，每个循环是一个节点名称列表
        """
        cycles = []
        visited = set()
        path = []

        def dfs(node: str) -> bool:
            if node in path:
                cycle_start = path.index(node)
                cycle = path[cycle_start:] + [node]
                cycles.append(cycle)
                return True

            if node in visited:
                return False

            visited.add(node)
            path.append(node)

            for edge in self.edges.get(node, []):
                target = edge.get("target")
                if target and target != self.END_NODE:
                    dfs(target)

            path.pop()
            return False

        if self.entry_point:
            dfs(self.entry_point)

        return cycles

    def validate(self) -> list[str]:
        """验证图配置，返回警告列表

        Returns:
            警告信息列表，空列表表示验证通过
        """
        warnings = []

        if not self.entry_point:
            warnings.append("No entry_point set")

        if not self.nodes:
            warnings.append("No nodes defined")

        if self.entry_point and self.entry_point not in self.nodes:
            warnings.append(f"Entry point '{self.entry_point}' is not a defined node")

        for source, edges in self.edges.items():
            if source not in self.nodes and source != self.START_NODE:
                warnings.append(f"Edge source '{source}' is not a defined node")

            for edge in edges:
                target = edge.get("target")
                if target and target != self.END_NODE and target not in self.nodes:
                    warnings.append(f"Edge target '{target}' is not a defined node")

        for source in self.conditional_edges:
            if source not in self.nodes:
                warnings.append(f"Conditional edge source '{source}' is not a defined node")

            path_map = self.conditional_edges[source]["path_map"]
            for key, target in path_map.items():
                if target != self.END_NODE and target not in self.nodes:
                    warnings.append(
                        f"Conditional edge path_map target '{target}' is not a defined node"
                    )

        cycles = self.detect_cycles()
        for cycle in cycles:
            warnings.append(f"Potential infinite cycle detected: {' -> '.join(cycle)}")

        return warnings

    def __repr__(self) -> str:
        return f"Graph(name={self.name}, nodes={list(self.nodes.keys())})"
