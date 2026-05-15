"""FastMind 主类"""

from typing import Callable, Optional

from .graph import Graph
from .tool import ToolRegistry, Tool
from .node import AgentRegistry, Agent
from .signal import Signal
from .vla import VLAConfig, VLARegistry, VLActionRegistry, VLAActionNode, ActionSpace


class FastMind:
    """FastMind 框架主类

    提供装饰器风格的 API，用于注册 agent、tool、graph、感知循环、信号和 VLA。

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
        self._signals: dict[str, Signal] = {}
        self._vla_registry = VLARegistry()
        self._vla_action_registry = VLActionRegistry()

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

        Raises:
            TypeError: 如果装饰的函数没有返回 Graph 实例

        用法示例:
            @app.graph(name="main")
            def create_graph():
                graph = Graph()
                ...
                return graph
        """

        def decorator(func: Callable) -> Callable:
            graph_name = name or func.__name__
            result = func()
            if not isinstance(result, Graph):
                raise TypeError(
                    f"@app.graph decorated function '{func.__name__}' "
                    f"must return a Graph instance, got {type(result).__name__}"
                )
            self._graphs[graph_name] = result
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
        self._tool_registry.add(name, tool)

    def register_agent(self, name: str, agent: Agent) -> None:
        """手动注册 Agent

        Args:
            name: Agent 名称
            agent: Agent 实例
        """
        self._agent_registry.add(name, agent)

    def perception(
        self,
        interval: float = 1.0,
        name: Optional[str] = None,
    ) -> Callable:
        """装饰器：注册感知循环

        Args:
            interval: 触发间隔（秒），必须大于 0
            name: 感知名称

        Returns:
            装饰器函数

        Raises:
            ValueError: 如果 interval <= 0

        用法示例:
            @app.perception(interval=5.0, name="sensor_monitor")
            async def sensor_monitor(app: FastMind):
                while True:
                    data = await read_sensor()
                    yield Event(type="sensor.data", payload=data, session_id="system")
                    await asyncio.sleep(5.0)
        """
        if interval <= 0:
            raise ValueError(f"perception interval must be positive, got {interval}")

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

    def get_tools(self, tools: Optional[list[str]] = None) -> dict[str, Tool]:
        """获取工具

        Args:
            tools: 工具名称列表，只返回这些工具。None 表示返回全部工具。

        Returns:
            工具字典 {name: Tool}
        """
        if tools is None:
            return self._tool_registry.get_all()
        return {
            name: self._tool_registry.get(name)
            for name in tools
            if name in self._tool_registry
        }

    def get_tool(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tool_registry.get(name)

    def get_tool_schemas(self, tools: Optional[list[str]] = None) -> list[dict]:
        """获取工具的 OpenAI schema（用于 LLM 调用）

        Args:
            tools: 工具名称列表，只返回这些工具的 schema。None 表示返回全部。

        Returns:
            schema 列表
        """
        if tools is None:
            return self._tool_registry.get_schemas()
        return [
            self._tool_registry.get(name).to_openai_schema()
            for name in tools
            if name in self._tool_registry
        ]

    def get_agents(self) -> dict[str, Agent]:
        """获取所有 Agent"""
        return self._agent_registry.get_all()

    def get_agent(self, name: str) -> Optional[Agent]:
        """获取 Agent"""
        return self._agent_registry.get(name)

    def signal(
        self,
        name: Optional[str] = None,
        interval: float = 1.0,
    ) -> Callable:
        """装饰器：注册高频信号

        信号与 Event 平行，是高频连续数据的载体。
        信号数据直写 SignalBus，不经过 event 队列。

        Args:
            name: 信号名称，默认使用函数名
            interval: 更新间隔（秒）

        Returns:
            装饰器函数

        用法示例:
            @app.signal(name="vision", interval=1/30)
            async def npc_vision():
                return game_engine.render_npc_view()
        """
        if interval <= 0:
            raise ValueError(f"signal interval must be positive, got {interval}")

        def decorator(func: Callable) -> Callable:
            signal_name = name or func.__name__
            self._signals[signal_name] = Signal(
                name=signal_name,
                interval=interval,
                func=func,
            )
            return func

        return decorator

    def vla(
        self,
        name: Optional[str] = None,
        frequency: float = 10.0,
        input_signals: Optional[list[str]] = None,
    ) -> Callable:
        """装饰器：注册 VLA 推理节点

        VLA 节点由框架按 frequency 自动调度，不经过 graph。
        签名: async (state, signal_bus) -> dict[str, list[float]]
        返回: 动作通道名 → 动作向量

        Args:
            name: VLA 名称，默认使用函数名
            frequency: 目标控制频率 (Hz)
            input_signals: 输入信号名称列表

        Returns:
            装饰器函数

        用法示例:
            @app.vla(name="navigation", frequency=30.0)
            async def navigation_vla(state, signal_bus):
                vision = signal_bus.read("vision")
                return {"body": [0.5, 0, 0, 0.1]}
        """
        return self._vla_registry.register(
            name=name,
            frequency=frequency,
            input_signals=input_signals,
        )

    def vla_action(
        self,
        name: Optional[str] = None,
        action_space: Optional[ActionSpace] = None,
    ) -> Callable:
        """装饰器：注册 VLA 动作执行器

        VLA 的动作输出通过通道名路由到对应的动作执行器。

        Args:
            name: 动作名称（通道名），默认使用函数名
            action_space: 动作空间定义

        Returns:
            装饰器函数

        用法示例:
            @app.vla_action(name="body", action_space=ActionSpace(4))
            async def body_executor(action):
                await game_engine.move(action[0], action[1], action[2], action[3])
        """
        return self._vla_action_registry.register(
            name=name,
            action_space=action_space,
        )

    def register_signal(self, name: str, signal: Signal) -> None:
        """手动注册信号"""
        self._signals[name] = signal

    def register_vla(self, name: str, cfg: VLAConfig) -> None:
        """手动注册 VLA 配置"""
        self._vla_registry.add(name, cfg)

    def register_vla_action(self, name: str, node: VLAActionNode) -> None:
        """手动注册 VLA 动作执行器"""
        self._vla_action_registry.add(name, node)

    def get_signals(self) -> dict[str, Signal]:
        """获取所有信号"""
        return self._signals.copy()

    def get_signal(self, name: str) -> Optional[Signal]:
        """获取信号"""
        return self._signals.get(name)

    def get_vlas(self) -> dict[str, VLAConfig]:
        """获取所有 VLA 配置"""
        return self._vla_registry.get_all()

    def get_vla(self, name: str) -> Optional[VLAConfig]:
        """获取 VLA 配置"""
        return self._vla_registry.get(name)

    def get_vla_actions(self) -> dict[str, VLAActionNode]:
        """获取所有 VLA 动作执行器"""
        return self._vla_action_registry.get_all()

    def get_vla_action(self, name: str) -> Optional[VLAActionNode]:
        """获取 VLA 动作执行器"""
        return self._vla_action_registry.get(name)

    def has_vla(self) -> bool:
        """检查是否注册了任何 VLA"""
        return len(self._vla_registry) > 0

    def get_perceptions(self) -> list[tuple[str, Callable, float]]:
        """获取所有感知循环"""
        return self._perceptions.copy()

    def __repr__(self) -> str:
        return (
            f"FastMind("
            f"graphs={list(self._graphs.keys())}, "
            f"tools={list(self._tool_registry.get_all().keys())}, "
            f"agents={list(self._agent_registry.get_all().keys())}"
            f")"
        )
