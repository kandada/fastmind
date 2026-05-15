"""VLA 模块 - VLA 推理节点与动作执行器"""

from typing import Callable, Optional
from dataclasses import dataclass, field
import asyncio


@dataclass
class ActionSpace:
    """动作空间定义

    Attributes:
        dim: 动作向量维度
        low: 各维度的下界（可选）
        high: 各维度的上界（可选）
    """

    dim: int
    low: Optional[list[float]] = None
    high: Optional[list[float]] = None


@dataclass
class VLAConfig:
    """VLA 配置

    Attributes:
        name: VLA 名称
        func: VLA 推理函数，签名: async (state, signal_bus) -> dict[str, list[float]]
        frequency: 目标控制频率 (Hz)
        input_signals: 输入信号名称列表
    """

    name: str
    func: Callable
    frequency: float
    input_signals: list[str] = field(default_factory=list)


class VLARegistry:
    """VLA 注册表"""

    def __init__(self):
        self._vlas: dict[str, VLAConfig] = {}

    def register(
        self,
        name: Optional[str] = None,
        frequency: float = 10.0,
        input_signals: Optional[list[str]] = None,
    ) -> Callable:
        """装饰器：注册 VLA

        Args:
            name: VLA 名称，默认使用函数名
            frequency: 目标控制频率 (Hz)
            input_signals: 输入信号名称列表

        Returns:
            装饰器函数
        """

        def decorator(func: Callable) -> Callable:
            vla_name = name or func.__name__
            cfg = VLAConfig(
                name=vla_name,
                func=func,
                frequency=frequency,
                input_signals=input_signals or [],
            )
            self._vlas[vla_name] = cfg
            return func

        return decorator

    def get(self, name: str) -> Optional[VLAConfig]:
        """获取 VLA 配置"""
        return self._vlas.get(name)

    def get_all(self) -> dict[str, VLAConfig]:
        """获取所有 VLA 配置"""
        return self._vlas.copy()

    def add(self, name: str, cfg: VLAConfig) -> None:
        """直接添加 VLA 配置"""
        self._vlas[name] = cfg

    def __contains__(self, name: str) -> bool:
        return name in self._vlas

    def __len__(self) -> int:
        return len(self._vlas)


class VLActionRegistry:
    """VLA 动作执行器注册表"""

    def __init__(self):
        self._actions: dict[str, "VLAActionNode"] = {}

    def register(
        self,
        name: Optional[str] = None,
        action_space: Optional[ActionSpace] = None,
    ) -> Callable:
        """装饰器：注册 VLA 动作执行器

        Args:
            name: 动作名称（通道名），默认使用函数名
            action_space: 动作空间定义

        Returns:
            装饰器函数
        """

        def decorator(func: Callable) -> Callable:
            action_name = name or func.__name__
            node = VLAActionNode(
                name=action_name,
                func=func,
                action_space=action_space,
            )
            self._actions[action_name] = node
            return func

        return decorator

    def get(self, name: str) -> Optional["VLAActionNode"]:
        """获取动作执行器"""
        return self._actions.get(name)

    def get_all(self) -> dict[str, "VLAActionNode"]:
        """获取所有动作执行器"""
        return self._actions.copy()

    def add(self, name: str, node: "VLAActionNode") -> None:
        """直接添加动作执行器"""
        self._actions[name] = node

    def __contains__(self, name: str) -> bool:
        return name in self._actions


class VLAActionNode:
    """VLA 动作执行节点

    对标 ToolNode，但处理连续动作向量而非 JSON tool_calls。
    VLA 模型的输出动作向量通过通道名路由到对应的 VLAActionNode。

    Attributes:
        name: 动作名称（通道名）
        func: 动作执行函数，async (action_vector) -> dict
        action_space: 动作空间定义
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        action_space: Optional[ActionSpace] = None,
    ):
        self.name = name
        self.func = func
        self.action_space = action_space

    async def execute(self, action: list[float]) -> dict:
        """执行动作

        Args:
            action: 动作向量

        Returns:
            执行结果 dict
        """
        if asyncio.iscoroutinefunction(self.func):
            return await self.func(action)
        return self.func(action)
