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

    def test_add_message(self):
        """测试 add_message 方法"""
        state = State()
        state.add_message("messages", "user", "hello")
        state.add_message("messages", "assistant", "hi")

        assert len(state["messages"]) == 2
        assert state["messages"][0] == {"role": "user", "content": "hello"}
        assert state["messages"][1] == {"role": "assistant", "content": "hi"}

    def test_add_message_with_extra_fields(self):
        """测试 add_message 带额外字段"""
        state = State()
        state.add_message("tool_results", role="system", content="result", tool_name="test")

        assert state["tool_results"][0]["tool_name"] == "test"

    def test_add_message_chain(self):
        """测试 add_message 链式调用"""
        state = State()
        result = state.add_message("messages", "user", "hello").add_message(
            "messages", "assistant", "hi"
        )

        assert result is state
        assert len(state["messages"]) == 2

    def test_add_message_if_new(self):
        """测试 add_message_if_new 方法"""
        state = State()
        state.add_message("messages", "user", "hello")

        result1 = state.add_message_if_new("messages", "user", "hello")
        assert result1 is False
        assert len(state["messages"]) == 1

        result2 = state.add_message_if_new("messages", "user", "world")
        assert result2 is True
        assert len(state["messages"]) == 2
        assert state["messages"][1]["content"] == "world"

    def test_get_last_message(self):
        """测试 get_last_message 方法"""
        state = State()
        state.add_message("messages", "user", "hello")
        state.add_message("messages", "assistant", "hi")
        state.add_message("messages", "user", "third")

        last = state.get_last_message("messages")
        assert last["content"] == "third"

        last_user = state.get_last_message("messages", role="user")
        assert last_user["content"] == "third"

        last_assistant = state.get_last_message("messages", role="assistant")
        assert last_assistant["content"] == "hi"

    def test_get_last_message_empty(self):
        """测试空消息列表"""
        state = State()
        assert state.get_last_message("messages") is None

    def test_pop_messages(self):
        """测试 pop_messages 方法"""
        state = State()
        state.add_message("messages", "user", "hello")
        state.add_message("messages", "assistant", "hi")
        state.add_message("messages", "user", "world")

        popped = state.pop_messages("messages", 2)
        assert len(popped) == 2
        assert len(state["messages"]) == 1
        assert state["messages"][0]["content"] == "hello"

    def test_get_message_count(self):
        """测试 get_message_count 方法"""
        state = State()
        assert state.get_message_count("messages") == 0

        state.add_message("messages", "user", "hello")
        assert state.get_message_count("messages") == 1

        state.add_message("messages", "assistant", "hi")
        assert state.get_message_count("messages") == 2


class TestStateKey:
    """StateKey 常量测试"""

    def test_state_key_constants(self):
        """测试状态键常量"""
        assert StateKey.MESSAGES == "messages"
        assert StateKey.TOOL_CALLS == "tool_calls"
        assert StateKey.TOOL_RESULTS == "tool_results"
        assert StateKey.METADATA == "metadata"
