"""FastMind 主类的单元测试"""

import pytest
from fastmind import FastMind
from fastmind.core.graph import Graph
from fastmind.core.event import Event


class TestFastMind:
    """FastMind 测试"""

    @pytest.fixture
    def app(self):
        """创建 FastMind 实例"""
        return FastMind()

    def test_create_app(self, app):
        """测试创建应用"""
        assert app is not None
        assert len(app._graphs) == 0
        assert len(app._tool_registry._tools) == 0
        assert len(app._agent_registry._agents) == 0

    def test_tool_decorator(self, app):
        """测试工具装饰器"""

        @app.tool(name="test_tool", description="测试工具")
        async def test_tool(arg1: str) -> str:
            return f"result: {arg1}"

        assert "test_tool" in app._tool_registry._tools
        tool = app.get_tool("test_tool")
        assert tool.name == "test_tool"
        assert tool.description == "测试工具"

    def test_agent_decorator(self, app):
        """测试 Agent 装饰器"""

        @app.agent(name="test_agent", tools=["tool1"])
        async def test_agent(state: dict, event: Event) -> dict:
            return state

        assert "test_agent" in app._agent_registry._agents
        agent = app.get_agent("test_agent")
        assert agent.name == "test_agent"
        assert agent.tools == ["tool1"]

    def test_register_graph(self, app):
        """测试注册图"""
        graph = Graph()
        graph.add_node("start", lambda s, e: s)

        app.register_graph("test_graph", graph)

        assert "test_graph" in app._graphs
        assert app.get_graph("test_graph") == graph

    def test_get_tool_schemas(self, app):
        """测试获取工具 schema"""

        @app.tool(name="weather", description="获取天气")
        async def get_weather(city: str) -> str:
            return "sunny"

        schemas = app.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "weather"

    def test_perception_decorator(self, app):
        """测试感知装饰器"""

        @app.perception(interval=5.0, name="test_sensor")
        async def sensor(app: FastMind):
            while True:
                yield Event("sensor.data", {}, "system")
                await asyncio.sleep(5.0)

        import asyncio

        perceptions = app.get_perceptions()
        assert len(perceptions) == 1
        assert perceptions[0][0] == "test_sensor"
        assert perceptions[0][2] == 5.0

    def test_app_repr(self, app):
        """测试应用字符串表示"""

        @app.tool(name="t1")
        async def tool1():
            pass

        @app.agent(name="a1")
        async def agent1(s, e):
            pass

        repr_str = repr(app)
        assert "FastMind" in repr_str
        assert "t1" in repr_str
        assert "a1" in repr_str


import asyncio
