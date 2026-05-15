"""VLA 模块的单元测试"""

import pytest

from fastmind.core.vla import (
    ActionSpace,
    VLAConfig,
    VLARegistry,
    VLActionRegistry,
    VLAActionNode,
)


class TestActionSpace:
    """ActionSpace 测试"""

    def test_create_with_dim_only(self):
        """测试仅指定维度"""
        space = ActionSpace(dim=7)
        assert space.dim == 7
        assert space.low is None
        assert space.high is None

    def test_create_with_bounds(self):
        """测试指定上下界"""
        space = ActionSpace(
            dim=3,
            low=[-1.0, -1.0, -3.14],
            high=[1.0, 1.0, 3.14],
        )
        assert space.dim == 3
        assert space.low == [-1.0, -1.0, -3.14]
        assert space.high == [1.0, 1.0, 3.14]

    def test_default_dim(self):
        """测试默认值"""
        space = ActionSpace(dim=2)
        assert space.dim == 2


class TestVLAConfig:
    """VLAConfig 测试"""

    def test_create_config(self):
        """测试创建配置"""
        async def dummy_vla(state, signal_bus):
            return {"body": [0.0]}

        cfg = VLAConfig(
            name="navigation",
            func=dummy_vla,
            frequency=30.0,
            input_signals=["vision", "proprioception"],
        )
        assert cfg.name == "navigation"
        assert cfg.func is dummy_vla
        assert cfg.frequency == 30.0
        assert cfg.input_signals == ["vision", "proprioception"]

    def test_default_input_signals(self):
        """测试默认输入信号为空"""
        async def dummy(state, sb):
            return {}

        cfg = VLAConfig(name="test", func=dummy, frequency=10.0)
        assert cfg.input_signals == []

    def test_frequency_default(self):
        """测试默认频率"""
        async def dummy(state, sb):
            return {}

        cfg = VLAConfig(name="test", func=dummy, frequency=10.0)
        assert cfg.frequency == 10.0


class TestVLAActionNode:
    """VLAActionNode 测试"""

    @pytest.mark.asyncio
    async def test_execute_async_function(self):
        """测试异步执行函数"""
        async def move_executor(action):
            return {"executed": True, "action": action}

        node = VLAActionNode(name="body", func=move_executor)
        result = await node.execute([1.0, 2.0, 3.0])
        assert result["executed"] is True
        assert result["action"] == [1.0, 2.0, 3.0]

    @pytest.mark.asyncio
    async def test_execute_sync_function(self):
        """测试同步执行函数"""
        def move_executor(action):
            return {"moved": action[0]}

        node = VLAActionNode(name="body", func=move_executor)
        result = await node.execute([5.0])
        assert result["moved"] == 5.0

    def test_with_action_space(self):
        """测试带动作空间"""
        space = ActionSpace(dim=7, low=[-1]*7, high=[1]*7)

        async def executor(action):
            return {}

        node = VLAActionNode(name="full_body", func=executor, action_space=space)
        assert node.action_space.dim == 7
        assert node.action_space.low == [-1]*7
        assert node.action_space.high == [1]*7

    def test_name(self):
        """测试节点名称"""
        async def fn(action):
            return {}

        node = VLAActionNode(name="speech", func=fn)
        assert node.name == "speech"

    @pytest.mark.asyncio
    async def test_execute_returns_empty_dict(self):
        """测试返回空字典"""
        async def fn(action):
            return {}

        node = VLAActionNode(name="test", func=fn)
        result = await node.execute([0.0])
        assert result == {}


class TestVLARegistry:
    """VLARegistry 测试"""

    @pytest.fixture
    def registry(self):
        return VLARegistry()

    def test_register_decorator(self, registry):
        """测试装饰器注册"""
        @registry.register(name="nav", frequency=30.0)
        async def navigation_vla(state, signal_bus):
            return {"body": [0.0]}

        assert "nav" in registry
        cfg = registry.get("nav")
        assert cfg.frequency == 30.0
        assert cfg.func is navigation_vla

    def test_register_uses_function_name(self, registry):
        """测试未指定名称时使用函数名"""
        @registry.register(frequency=20.0)
        async def my_vla(state, sb):
            return {}

        assert "my_vla" in registry

    def test_get_nonexistent(self, registry):
        """测试获取不存在的 VLA"""
        assert registry.get("nonexistent") is None

    def test_get_all(self, registry):
        """测试获取所有 VLA"""
        @registry.register(name="vla_a")
        async def a(state, sb):
            return {}

        @registry.register(name="vla_b")
        async def b(state, sb):
            return {}

        all_vlas = registry.get_all()
        assert len(all_vlas) == 2
        assert "vla_a" in all_vlas
        assert "vla_b" in all_vlas

    def test_get_all_isolation(self, registry):
        """测试 get_all 返回拷贝"""
        @registry.register(name="vla_a")
        async def a(state, sb):
            return {}

        all_vlas = registry.get_all()
        all_vlas["fake"] = "value"
        assert "fake" not in registry.get_all()

    def test_add_direct(self, registry):
        """测试直接添加"""
        async def fn(state, sb):
            return {}

        cfg = VLAConfig(name="manual", func=fn, frequency=5.0)
        registry.add("manual", cfg)
        assert "manual" in registry
        assert registry.get("manual").frequency == 5.0

    def test_contains(self, registry):
        """测试包含检查"""
        @registry.register(name="exists")
        async def fn(state, sb):
            return {}

        assert "exists" in registry
        assert "missing" not in registry

    def test_len(self, registry):
        """测试长度"""
        @registry.register(name="a")
        async def fn_a(state, sb):
            return {}

        @registry.register(name="b")
        async def fn_b(state, sb):
            return {}

        assert len(registry) == 2


class TestVLActionRegistry:
    """VLActionRegistry 测试"""

    @pytest.fixture
    def registry(self):
        return VLActionRegistry()

    def test_register_decorator(self, registry):
        """测试装饰器注册"""
        space = ActionSpace(dim=4)

        @registry.register(name="body", action_space=space)
        async def body_executor(action):
            return {"done": True}

        node = registry.get("body")
        assert node is not None
        assert node.name == "body"
        assert node.action_space.dim == 4

    def test_register_uses_function_name(self, registry):
        """测试未指定名称时使用函数名"""
        @registry.register(action_space=ActionSpace(1))
        async def hand_executor(action):
            return {}

        assert "hand_executor" in registry

    def test_get_nonexistent(self, registry):
        """测试获取不存在"""
        assert registry.get("missing") is None

    def test_get_all(self, registry):
        """测试获取所有"""
        @registry.register(name="act_a")
        async def a(action):
            return {}

        @registry.register(name="act_b")
        async def b(action):
            return {}

        assert len(registry.get_all()) == 2

    def test_add_direct(self, registry):
        """测试直接添加"""
        async def fn(action):
            return {}

        node = VLAActionNode(name="manual", func=fn)
        registry.add("manual", node)
        assert "manual" in registry

    def test_contains(self, registry):
        """测试包含检查"""
        @registry.register(name="exists")
        async def fn(action):
            return {}

        assert "exists" in registry
        assert "no" not in registry

    @pytest.mark.asyncio
    async def test_registered_node_executes(self, registry):
        """测试注册后的节点可执行"""
        @registry.register(name="test")
        async def test_executor(action):
            return {"sum": sum(action)}

        node = registry.get("test")
        result = await node.execute([1.0, 2.0, 3.0])
        assert result["sum"] == 6.0
