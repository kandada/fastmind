"""State 类的单元测试"""

import pytest
from fastmind.core.state import State, StateKey


class TestState:
    """State 测试"""

    def test_create_empty_state(self):
        """测试创建空状态"""
        state = State()
        assert len(state) == 0

    def test_create_with_initial(self):
        """测试使用初始数据创建"""
        state = State({"name": "test", "value": 123})
        assert state["name"] == "test"
        assert state["value"] == 123

    def test_create_with_kwargs(self):
        """测试使用关键字参数创建"""
        state = State(name="test", value=123)
        assert state["name"] == "test"
        assert state["value"] == 123

    def test_copy(self):
        """测试状态复制"""
        state = State({"name": "test", "items": [1, 2, 3]})
        copied = state.copy()

        assert copied["name"] == state["name"]
        assert copied["items"] == state["items"]

    def test_get_nested(self):
        """测试获取嵌套值"""
        state = State({"user": {"profile": {"name": "Alice"}}})

        assert state.get_nested("user.profile.name") == "Alice"
        assert state.get_nested("user.profile.age") is None
        assert state.get_nested("user.profile.age", 25) == 25

    def test_set_nested(self):
        """测试设置嵌套值"""
        state = State()
        state.set_nested("user.profile.name", "Bob")

        assert state["user"]["profile"]["name"] == "Bob"


class TestStateKey:
    """StateKey 常量测试"""

    def test_state_key_constants(self):
        """测试状态键常量"""
        assert StateKey.MESSAGES == "messages"
        assert StateKey.TOOL_CALLS == "tool_calls"
        assert StateKey.TOOL_RESULTS == "tool_results"
        assert StateKey.METADATA == "metadata"
