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


class Graph:
    """状态图类，包含节点和边的集合

    用于定义智能体工作流，支持：
    - 添加普通节点和条件边
    - 设置入口点和结束点
    - 支持子图嵌套

    用法示例:
        graph = Graph()
        graph.add_node("start", start_node)
        graph.add_node("end", end_node)
        graph.add_edge("start", "end")
        graph.set_entry_point("start")
    """

    START_NODE = "__start__"
    END_NODE = "__end__"

    def __init__(self, name: Optional[str] = None):
        """初始化 Graph

        Args:
            name: 图的名称
        """
        self.name = name or f"graph_{id(self)}"
        self.nodes: dict[str, Any] = {}
        self.edges: dict[str, list[EdgeType]] = defaultdict(list)
        self.conditional_edges: dict[str, dict[str, Any]] = {}
        self.entry_point: Optional[str] = None
        self._interrupt_nodes: dict[str, dict[str, Any]] = {}

    def add_node(self, name: str, node: Union[NodeFunc, "Graph"]) -> "Graph":
        """添加节点

        Args:
            name: 节点名称
            node: 节点函数或子图

        Returns:
            self，支持链式调用
        """
        self.nodes[name] = node
        return self

    def add_edge(
        self, source: str, target: str, condition: Optional[Callable] = None
    ) -> "Graph":
        """添加普通边（无条件跳转）

        Args:
            source: 源节点名称
            target: 目标节点名称
            condition: 可选的条件函数，返回 True 时才跳转

        Returns:
            self，支持链式调用
        """
        self.edges[source].append(
            {
                "target": target,
                "condition": condition,
            }
        )
        return self

    def add_conditional_edges(
        self, source: str, router: RouterFunc, path_map: dict[Optional[str], str]
    ) -> "Graph":
        """添加条件边

        条件边根据 router 函数的返回值决定下一个节点。

        Args:
            source: 源节点名称
            router: 路由函数，接收 (state, event)，返回目标节点名或 None
            path_map: 路由返回值到目标节点的映射

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
        self.conditional_edges[source] = {
            "router": router,
            "path_map": path_map,
        }
        return self

    def add_interrupt(
        self,
        name: str,
        prompt: str,
        resume_node: str = "resume",
        cancel_node: Optional[str] = None,
    ) -> "Graph":
        """添加中断节点（用于 Human-in-the-loop）

        Args:
            name: 节点名称
            prompt: 提示信息
            resume_node: 恢复后继续执行的节点
            cancel_node: 取消后跳转的节点

        Returns:
            self，支持链式调用
        """

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
        """设置入口节点

        Args:
            name: 节点名称

        Returns:
            self，支持链式调用
        """
        self.entry_point = name
        return self

    def get_next_node(
        self, current_node: str, state: dict, event: Event
    ) -> Optional[str]:
        """获取下一个节点名称

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

            if asyncio.iscoroutinefunction(router):
                result = router(state, event)
                if asyncio.iscoroutine(result):
                    return None
                route_key = result
            else:
                route_key = router(state, event)

            return path_map.get(route_key)

        edges = self.edges.get(current_node, [])
        for edge in edges:
            condition = edge.get("condition")
            if condition is None or condition(state):
                return edge["target"]

        return None

    def get_node(self, name: str) -> Optional[Union[NodeFunc, "Graph"]]:
        """获取节点

        Args:
            name: 节点名称

        Returns:
            节点函数或子图
        """
        return self.nodes.get(name)

    def has_node(self, name: str) -> bool:
        """检查节点是否存在"""
        return name in self.nodes

    def __repr__(self) -> str:
        return f"Graph(name={self.name}, nodes={list(self.nodes.keys())})"
