"""工具注册与执行的单元测试"""

import pytest
import asyncio
from fastmind.core.tool import Tool, ToolRegistry, ToolNode
from fastmind.core.event import Event


class TestTool:
    """Tool 测试"""

    def test_create_tool(self):
        """测试创建工具"""

        async def get_weather(city: str) -> str:
            return f"{city} 天气晴朗"

        tool = Tool(
            name="get_weather",
            description="获取城市天气",
            func=get_weather,
        )

        assert tool.name == "get_weather"
        assert tool.description == "获取城市天气"
        assert tool.func == get_weather

    def test_tool_openai_schema(self):
        """测试生成 OpenAI schema"""

        async def get_weather(city: str) -> str:
            return f"{city} 天气晴朗"

        tool = Tool(
            name="get_weather",
            description="获取城市天气",
            func=get_weather,
        )

        schema = tool.to_openai_schema()

        assert schema["type"] == "function"
        assert schema["function"]["name"] == "get_weather"
        assert "parameters" in schema["function"]


class TestToolRegistry:
    """工具注册表测试"""

    def test_register_decorator(self):
        """测试装饰器注册"""
        registry = ToolRegistry()

        @registry.register(name="calc", description="计算器")
        async def calc(a: int, b: int) -> int:
            return a + b

        assert "calc" in registry._tools
        tool = registry.get("calc")
        assert tool.name == "calc"
        assert tool.description == "计算器"

    def test_get_all_tools(self):
        """测试获取所有工具"""
        registry = ToolRegistry()

        @registry.register(name="t1")
        async def tool1():
            pass

        @registry.register(name="t2")
        async def tool2():
            pass

        tools = registry.get_all()
        assert len(tools) == 2
        assert "t1" in tools
        assert "t2" in tools

    def test_get_schemas(self):
        """测试获取所有 schema"""
        registry = ToolRegistry()

        @registry.register(name="weather", description="天气")
        async def weather(city: str) -> str:
            return "sunny"

        schemas = registry.get_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "weather"


class TestToolNode:
    """工具节点测试"""

    @pytest.mark.asyncio
    async def test_execute_no_tool_calls(self):
        """测试无工具调用时"""

        async def dummy_tool():
            pass

        tool = Tool("dummy", "", dummy_tool)
        node = ToolNode({"dummy": tool})

        state = {"messages": []}
        event = Event("test", {}, "session_001")

        new_state, output = await node.execute(state, event)

        assert new_state == state
        assert output == []

    @pytest.mark.asyncio
    async def test_execute_with_tool_call(self):
        """测试执行工具调用"""

        async def get_weather(city: str) -> str:
            return f"{city} 天气晴朗"

        tool = Tool("get_weather", "获取天气", get_weather)
        node = ToolNode({"get_weather": tool})

        state = {
            "tool_calls": [
                {
                    "id": "call_1",
                    "function": {"name": "get_weather", "arguments": '{"city": "北京"}'},
                }
            ]
        }
        event = Event("test", {}, "session_001")

        new_state, output = await node.execute(state, event)

        assert "tool_results" in new_state
        assert len(new_state["tool_results"]) == 1
        assert new_state["tool_results"][0]["tool_name"] == "get_weather"
        assert new_state["tool_results"][0]["result"] == "北京 天气晴朗"
        assert "tool_calls" not in new_state

    @pytest.mark.asyncio
    async def test_execute_unknown_tool(self):
        """测试调用未知工具"""
        node = ToolNode({})

        state = {
            "tool_calls": [
                {"id": "call_1", "function": {"name": "unknown_tool", "arguments": "{}"}}
            ]
        }
        event = Event("test", {}, "session_001")

        new_state, output = await node.execute(state, event)

        assert "tool_results" in new_state
        assert "not found" in new_state["tool_results"][0]["result"]
