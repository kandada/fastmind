"""工具注册与执行"""

from typing import Callable, Any, Optional
from dataclasses import dataclass, field
from collections.abc import AsyncGenerator
import asyncio
import inspect

from .event import Event


@dataclass
class Tool:
    """工具定义

    Attributes:
        name: 工具名称
        description: 工具描述
        func: 工具函数
        schema: OpenAI 格式的工具 schema
    """

    name: str
    description: str
    func: Callable
    schema: dict = field(default_factory=dict)

    def to_openai_schema(self) -> dict:
        """转换为 OpenAI 格式的工具 schema"""
        if self.schema:
            return self.schema

        sig = inspect.signature(self.func)
        params = sig.parameters

        properties = {}
        required = []
        for param_name, param in params.items():
            if param_name in ("self", "state", "event"):
                continue

            param_type = "string"
            if param.annotation == int:
                param_type = "integer"
            elif param.annotation == float:
                param_type = "number"
            elif param.annotation == bool:
                param_type = "boolean"
            elif param.annotation == list:
                param_type = "array"
            elif param.annotation == dict:
                param_type = "object"

            properties[param_name] = {
                "type": param_type,
                "description": f"Parameter {param_name}",
            }

            if param.default == inspect.Parameter.empty:
                required.append(param_name)

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": self.description,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


class ToolRegistry:
    """工具注册表"""

    def __init__(self):
        self._tools: dict[str, Tool] = {}

    def register(
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
        """

        def decorator(func: Callable) -> Callable:
            tool_name = name or func.__name__
            tool = Tool(
                name=tool_name,
                description=description or func.__doc__ or "",
                func=func,
            )
            self._tools[tool_name] = tool
            return func

        return decorator

    def get(self, name: str) -> Optional[Tool]:
        """获取工具"""
        return self._tools.get(name)

    def get_all(self) -> dict[str, Tool]:
        """获取所有工具"""
        return self._tools.copy()

    def get_schemas(self) -> list[dict]:
        """获取所有工具的 schema"""
        return [tool.to_openai_schema() for tool in self._tools.values()]

    def __contains__(self, name: str) -> bool:
        return name in self._tools


class ToolNode:
    """工具执行节点

    用于执行 agent 产生的 tool_calls，并将结果写入 state。
    """

    def __init__(self, tools: dict[str, Tool]):
        """初始化 ToolNode

        Args:
            tools: 工具注册表或工具字典
        """
        if isinstance(tools, ToolRegistry):
            self.tools = tools.get_all()
        else:
            self.tools = tools

    async def execute(self, state: dict, event: Event) -> tuple[dict, list[Event]]:
        """执行工具调用

        Args:
            state: 当前状态，包含 tool_calls
            event: 当前事件

        Returns:
            (更新后的状态, 输出事件列表)
        """
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return state, []

        results = []
        output_events = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            arguments = tool_call.get("function", {}).get("arguments", "{}")

            if isinstance(arguments, str):
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            tool = self.tools.get(tool_name)
            if not tool:
                result = f"Tool '{tool_name}' not found"
            else:
                try:
                    if asyncio.iscoroutinefunction(tool.func):
                        result = await tool.func(**arguments)
                    else:
                        result = tool.func(**arguments)
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"

            results.append(
                {
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": tool_name,
                    "result": result,
                }
            )

        state["tool_results"] = results
        if "tool_calls" in state:
            del state["tool_calls"]

        return state, output_events


class StreamingToolNode(ToolNode):
    """支持流式输出的工具执行节点"""

    async def execute(self, state: dict, event: Event) -> tuple[dict, list[Event]]:
        """执行工具调用（流式版本）"""
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return state, []

        results = []
        output_events = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            arguments = tool_call.get("function", {}).get("arguments", "{}")

            if isinstance(arguments, str):
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            output_events.append(
                Event(
                    type="stream.chunk",
                    payload={"delta": f"[Calling tool: {tool_name}]"},
                    session_id=event.session_id,
                )
            )

            tool = self.tools.get(tool_name)
            if not tool:
                result = f"Tool '{tool_name}' not found"
            else:
                try:
                    if asyncio.iscoroutinefunction(tool.func):
                        result = await tool.func(**arguments)
                    else:
                        result = tool.func(**arguments)
                except Exception as e:
                    result = f"Error executing tool: {str(e)}"

            output_events.append(
                Event(
                    type="stream.chunk",
                    payload={"delta": f"[Tool result: {result}]\n"},
                    session_id=event.session_id,
                )
            )

            results.append(
                {
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": tool_name,
                    "result": result,
                }
            )

        state["tool_results"] = results
        if "tool_calls" in state:
            del state["tool_calls"]

        return state, output_events
