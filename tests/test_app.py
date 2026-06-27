# Copyright (c) 2024-2026 xiefujin <490021684@qq.com>
# Licensed under GNU GPLv3, see LICENSE file for full license terms.

"""FastMind 主类的单元测试"""

import pytest
import asyncio
from fastmind import FastMind, ActionSpace, VLAActionNode
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
        """测试获取工具 schema（全量）"""

        @app.tool(name="weather", description="获取天气")
        async def get_weather(city: str) -> str:
            return "sunny"

        schemas = app.get_tool_schemas()
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "weather"

    def test_get_tools_filtered(self, app):
        """测试过滤获取工具"""

        @app.tool(name="tool_a")
        async def tool_a():
            pass

        @app.tool(name="tool_b")
        async def tool_b():
            pass

        tools = app.get_tools(tools=["tool_a"])
        assert len(tools) == 1
        assert "tool_a" in tools
        assert "tool_b" not in tools

    def test_get_tool_schemas_filtered(self, app):
        """测试过滤获取工具 schema"""

        @app.tool(name="weather", description="获取天气")
        async def get_weather(city: str) -> str:
            return "sunny"

        @app.tool(name="calculate")
        async def calculate(expr: str) -> str:
            return str(eval(expr))

        schemas = app.get_tool_schemas(tools=["weather"])
        assert len(schemas) == 1
        assert schemas[0]["function"]["name"] == "weather"

class TestSignalDecorator:
    """@app.signal 装饰器测试"""

    @pytest.fixture
    def app(self):
        return FastMind()

    def test_signal_decorator(self, app):
        """测试注册信号"""

        @app.signal(name="vision", interval=1/30)
        async def npc_vision():
            return "frame_data"

        signals = app.get_signals()
        assert "vision" in signals
        sig = app.get_signal("vision")
        assert sig.name == "vision"
        assert sig.interval == pytest.approx(0.0333, rel=0.01)
        assert sig.func is npc_vision

    def test_signal_uses_function_name(self, app):
        """测试未指定名称时使用函数名"""

        @app.signal(interval=0.1)
        async def custom_signal():
            return 42

        assert "custom_signal" in app.get_signals()

    def test_signal_interval_validation(self, app):
        """测试 interval 必须为正数"""
        with pytest.raises(ValueError, match="must be positive"):

            @app.signal(interval=0)
            async def bad():
                pass

        with pytest.raises(ValueError, match="must be positive"):

            @app.signal(interval=-1)
            async def also_bad():
                pass

    def test_register_signal_manually(self, app):
        """测试手动注册信号"""
        from fastmind import Signal

        async def src():
            return "data"

        sig = Signal(name="manual", interval=0.5, func=src)
        app.register_signal("manual", sig)
        assert "manual" in app.get_signals()
        assert app.get_signal("manual").interval == 0.5

    def test_multiple_signals(self, app):
        """测试注册多个信号"""

        @app.signal(name="vision", interval=1/30)
        async def vision():
            return "frame"

        @app.signal(name="hearing", interval=1/10)
        async def hearing():
            return "sound"

        signals = app.get_signals()
        assert len(signals) == 2
        assert "vision" in signals
        assert "hearing" in signals


class TestVLADecorator:
    """@app.vla 装饰器测试"""

    @pytest.fixture
    def app(self):
        return FastMind()

    def test_vla_decorator(self, app):
        """测试注册 VLA"""

        @app.vla(name="navigation", frequency=30.0)
        async def navigation_vla(state, signal_bus):
            return {"body": [0.5, 0, 0]}

        vlas = app.get_vlas()
        assert "navigation" in vlas
        cfg = app.get_vla("navigation")
        assert cfg.frequency == 30.0
        assert cfg.func is navigation_vla

    def test_vla_default_name(self, app):
        """测试未指定名称"""

        @app.vla(frequency=20.0)
        async def my_vla(state, sb):
            return {}

        assert "my_vla" in app.get_vlas()

    def test_vla_default_frequency(self, app):
        """测试默认频率"""

        @app.vla(name="test")
        async def test_vla(state, sb):
            return {}

        cfg = app.get_vla("test")
        assert cfg.frequency == 10.0

    def test_vla_with_input_signals(self, app):
        """测试指定输入信号"""

        @app.vla(name="nav", frequency=30.0, input_signals=["vision", "proprioception"])
        async def nav_vla(state, sb):
            return {}

        cfg = app.get_vla("nav")
        assert cfg.input_signals == ["vision", "proprioception"]

    def test_vla_empty_input_signals(self, app):
        """测试默认输入信号为空"""

        @app.vla(name="test")
        async def test_vla(state, sb):
            return {}

        assert app.get_vla("test").input_signals == []

    def test_register_vla_manually(self, app):
        """测试手动注册 VLA"""
        from fastmind import VLAConfig

        async def fn(state, sb):
            return {"body": [0.0]}

        cfg = VLAConfig(name="manual", func=fn, frequency=5.0)
        app.register_vla("manual", cfg)
        assert "manual" in app.get_vlas()
        assert app.get_vla("manual").frequency == 5.0

    def test_has_vla_false(self, app):
        """测试没有 VLA 时返回 False"""
        assert app.has_vla() is False

    def test_has_vla_true(self, app):
        """测试有 VLA 时返回 True"""

        @app.vla(name="test")
        async def test_vla(state, sb):
            return {}

        assert app.has_vla() is True

    def test_multiple_vlas(self, app):
        """测试注册多个 VLA"""

        @app.vla(name="nav", frequency=30.0)
        async def nav(state, sb):
            return {}

        @app.vla(name="face", frequency=10.0)
        async def face(state, sb):
            return {}

        assert len(app.get_vlas()) == 2


class TestVLAActionDecorator:
    """@app.vla_action 装饰器测试"""

    @pytest.fixture
    def app(self):
        return FastMind()

    def test_vla_action_decorator(self, app):
        """测试注册动作执行器"""
        space = ActionSpace(dim=7)

        @app.vla_action(name="body", action_space=space)
        async def body_executor(action):
            return {"done": True}

        actions = app.get_vla_actions()
        assert "body" in actions
        node = app.get_vla_action("body")
        assert node.name == "body"
        assert node.action_space.dim == 7

    def test_vla_action_default_name(self, app):
        """测试未指定名称"""

        @app.vla_action()
        async def my_action(action):
            return {}

        assert "my_action" in app.get_vla_actions()

    def test_vla_action_without_action_space(self, app):
        """测试不指定动作空间"""

        @app.vla_action(name="test")
        async def test_action(action):
            return {}

        node = app.get_vla_action("test")
        assert node.action_space is None

    def test_register_vla_action_manually(self, app):
        """测试手动注册动作执行器"""

        async def fn(action):
            return {}

        node = VLAActionNode(name="manual", func=fn)
        app.register_vla_action("manual", node)
        assert "manual" in app.get_vla_actions()

    def test_multiple_actions(self, app):
        """测试注册多个动作执行器"""

        @app.vla_action(name="body")
        async def body(action):
            return {}

        @app.vla_action(name="face")
        async def face(action):
            return {}

        @app.vla_action(name="speech")
        async def speech(action):
            return {}

        assert len(app.get_vla_actions()) == 3


class TestPerceptionDecorator:
    """@app.perception 装饰器测试（从 TestFastMind 移出）"""

    @pytest.fixture
    def app(self):
        return FastMind()

    def test_perception_decorator(self, app):
        """测试感知装饰器"""

        @app.perception(interval=5.0, name="test_sensor")
        async def sensor(app: FastMind):
            while True:
                yield Event("sensor.data", {}, "system")
                await asyncio.sleep(5.0)

        perceptions = app.get_perceptions()
        assert len(perceptions) == 1
        assert perceptions[0][0] == "test_sensor"
        assert perceptions[0][2] == 5.0



