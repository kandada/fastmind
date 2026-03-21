"""FastMind 主类"""

from typing import Callable, Any, Optional
from collections.abc import AsyncGenerator
import asyncio

from .graph import Graph
from .event import Event
from .tool import ToolRegistry, Tool
from .node import AgentRegistry, Agent


class FastMind:
    """FastMind 框架主类

    提供装饰器风格的 API，用于注册 agent、tool、graph 和感知循环。

    用法示例:
        app = FastMind()

        @app.tool(name="get_weather", description="获取城市天气")
        async def get_weather(city: str) -> str:
            return f"{city} 天气晴朗"

        @app.agent(name="chat_agent", tools=["get_weather"])
        async def chat_agent(state: dict, event: Event) -> dict:
            ...

        @app.graph(name="main")
        def create_graph():
            graph = Graph()
            ...
            return graph
    """

    def __init__(self):
        self._tool_registry = ToolRegistry()
        self._agent_registry = AgentRegistry()
        self._graphs: dict[str, Graph] = {}
        self._perceptions: list[tuple[str, Callable, float]] = []

    def tool(
        self,
        name: Optional[str] = None,
        description: str = "",
    ) -> Callable:
        """装饰器：注册工具

        Args:
            name: 工具名称，默认使用函数名
            description: 工具描述

        Returns:
            装饰器函数

        用法示例:
            @app.tool(name="get_weather", description="获取城市天气")
            async def get_weather(city: str) -> str:
                return f"{city} 天气晴朗"
        """
        return self._tool_registry.register(name=name, description=description)

    def agent(
        self,
        name: Optional[str] = None,
        tools: Optional[list[str]] = None,
        stream: bool = False,
    ) -> Callable:
        """装饰器：注册 Agent

        Args:
            name: Agent 名称，默认使用函数名
            tools: 工具名称列表
            stream: 是否支持流式输出

        Returns:
            装饰器函数

        用法示例:
            @app.agent(name="chat_agent", tools=["get_weather"])
            async def chat_agent(state: dict, event: Event) -> dict:
                state.setdefault("messages", [])
                state["messages"].append({"role": "user", "content": event.payload["text"]})
                return state
        """
        return self._agent_registry.register(name=name, tools=tools, stream=stream)

    def graph(self, name: Optional[str] = None) -> Callable:
        """装饰器：注册图

        Args:
            name: 图名称

        Returns:
            装饰器函数

        用法示例:
            @app.graph(name="main")
            def create_graph():
                graph = Graph()
                ...
                return graph
        """

        def decorator(func: Callable) -> Callable:
            graph_name = name or func.__name__
            graph = func()
            self._graphs[graph_name] = graph
            return func

        return decorator

    def register_graph(self, name: str, graph: Graph) -> None:
        """手动注册图

        Args:
            name: 图名称
            graph: Graph 实例
        """
        self._graphs[name] = graph

    def register_tool(self, name: str, tool: Tool) -> None:
        """手动注册工具

        Args:
            name: 工具名称
            tool: Tool 实例
        """
        self._tool_registry._tools[name] = tool

    def register_agent(self, name: str, agent: Agent) -> None:
        """手动注册 Agent

        Args:
            name: Agent 名称
            agent: Agent 实例
        """
        self._agent_registry._agents[name] = agent

    def perception(
        self,
        interval: float = 1.0,
        name: Optional[str] = None,
    ) -> Callable:
        """装饰器：注册感知循环

        Args:
            interval: 触发间隔（秒）
            name: 感知名称

        Returns:
            装饰器函数

        用法示例:
            @app.perception(interval=5.0, name="sensor_monitor")
            async def sensor_monitor(app: FastMind):
                while True:
                    data = await read_sensor()
                    yield Event(type="sensor.data", payload=data, session_id="system")
                    await asyncio.sleep(5.0)
        """

        def decorator(func: Callable) -> Callable:
            self._perceptions.append((name or func.__name__, func, interval))
            return func

        return decorator

    def get_graph(self, name: str) -> Optional[Graph]:
        """获取图"""
        return self._graphs.get(name)

    def get_graphs(self) -> dict[str, Graph]:
        """获取所有图"""
        return self._graphs.copy()

    def get_tools(self) -> dict[str, Tool]:
        """获取所有工具"""
        return self._tool_registry.get_all()

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tool_registry.get(name)

    def get_tool_schemas(self) -> list[dict]:
        """获取所有工具的 schema（用于 LLM 调用）"""
        return self._tool_registry.get_schemas()

    def get_agents(self) -> dict[str, Agent]:
        """获取所有 Agent"""
        return self._agent_registry.get_all()

    def get_agent(self, name: str) -> Optional[Agent]:
        """获取 Agent"""
        return self._agent_registry.get(name)

    def get_perceptions(self) -> list[tuple[str, Callable, float]]:
        """获取所有感知循环"""
        return self._perceptions.copy()

    def __repr__(self) -> str:
        return (
            f"FastMind("
            f"graphs={list(self._graphs.keys())}, "
            f"tools={list(self._tool_registry._tools.keys())}, "
            f"agents={list(self._agent_registry._agents.keys())}"
            f")"
        )
