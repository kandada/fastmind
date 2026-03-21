"""Agent 节点装饰器"""

from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from collections.abc import AsyncGenerator

from .event import Event


@dataclass
class Agent:
    """Agent 定义

    Attributes:
        name: Agent 名称
        func: Agent 函数
        tools: 工具名称列表
        stream: 是否支持流式输出
    """

    name: str
    func: Callable
    tools: list[str] = field(default_factory=list)
    stream: bool = False


class AgentRegistry:
    """Agent 注册表"""

    def __init__(self):
        self._agents: dict[str, Agent] = {}

    def register(
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
        """

        def decorator(func: Callable) -> Callable:
            agent_name = name or func.__name__
            agent = Agent(
                name=agent_name,
                func=func,
                tools=tools or [],
                stream=stream,
            )
            self._agents[agent_name] = agent
            return func

        return decorator

    def get(self, name: str) -> Optional[Agent]:
        """获取 Agent"""
        return self._agents.get(name)

    def get_all(self) -> dict[str, Agent]:
        """获取所有 Agent"""
        return self._agents.copy()

    def __contains__(self, name: str) -> bool:
        return name in self._agents


class AgentNode:
    """Agent 节点封装

    用于在 Graph 中调用 Agent。
    """

    def __init__(self, agent: Agent, app: Any = None):
        """初始化 AgentNode

        Args:
            agent: Agent 实例
            app: FastMind 应用实例（用于获取工具 schema）
        """
        self.agent = agent
        self.app = app

    async def execute(self, state: dict, event: Event) -> tuple[dict, list[Event]]:
        """执行 Agent

        Args:
            state: 当前状态
            event: 当前事件

        Returns:
            (更新后的状态, 输出事件列表)
        """
        func = self.agent.func

        if asyncio.iscoroutinefunction(func):
            result = await func(state, event)
        else:
            result = func(state, event)

        if isinstance(result, tuple):
            return result
        else:
            return result, []

    def get_tools(self) -> list[str]:
        """获取 Agent 使用的工具列表"""
        return self.agent.tools


import asyncio
