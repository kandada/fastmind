"""State 类定义 - 字典别名"""

from typing import Any, Optional, Dict


class State(dict):
    """状态类，继承自 dict。

    State 是会话的全局数据快照，所有节点共享和更新这个状态。
    开发者自由定义 key，如 messages、tool_calls、tool_results 等。

    用法示例:
        state = State()
        state["messages"] = []
        state["messages"].append({"role": "user", "content": "hello"})
        state.setdefault("count", 0)
        state["count"] += 1

    也推荐使用高层 API 来管理消息历史:
        state.add_message("messages", "user", "hello")
    """

    def __init__(self, initial: Optional[Dict[str, Any]] = None, **kwargs):
        """初始化 State

        Args:
            initial: 初始字典数据
            **kwargs: 关键字参数，会被合并到状态中
        """
        super().__init__()
        if initial:
            self.update(initial)
        self.update(kwargs)

    def copy(self) -> "State":
        """返回状态的浅拷贝"""
        return State(dict(self))

    def deep_copy(self) -> "State":
        """返回状态的深拷贝"""
        import copy

        return State(copy.deepcopy(dict(self)))

    def get_nested(self, key_path: str, default: Any = None) -> Any:
        """获取嵌套的值，使用点号分隔的路径

        Args:
            key_path: 点号分隔的路径，如 "user.profile.name"
            default: 默认值

        Returns:
            查找到的值或默认值
        """
        keys = key_path.split(".")
        value = self
        for key in keys:
            if isinstance(value, dict) and key in value:
                value = value[key]
            else:
                return default
        return value

    def set_nested(self, key_path: str, value: Any) -> None:
        """设置嵌套的值，使用点号分隔的路径

        Args:
            key_path: 点号分隔的路径，如 "user.profile.name"
            value: 要设置的值
        """
        keys = key_path.split(".")
        target = self
        for key in keys[:-1]:
            if key not in target:
                target[key] = {}
            target = target[key]
        target[keys[-1]] = value

    def add_message(
        self,
        key: str = "messages",
        role: str = "user",
        content: str = "",
        **extra_fields,
    ) -> "State":
        """添加消息到列表，自动初始化

        自动避免重复添加相同的 user 消息（在 ReAct 循环中常见问题）。

        Args:
            key: 消息列表的 key，默认 "messages"
            role: 消息角色，如 "user", "assistant", "system"
            content: 消息内容
            **extra_fields: 其他字段，如 "name", "tool_call_id" 等

        Returns:
            self，支持链式调用

        用法示例:
            state.add_message("messages", "user", "hello")
            state.add_message("messages", "assistant", "hi!")
            state.add_message("tool_results", role="system", content="[tool] result")
        """
        self.setdefault(key, [])
        message = {"role": role, "content": content, **extra_fields}
        self[key].append(message)
        return self

    def add_message_if_new(
        self,
        key: str = "messages",
        role: str = "user",
        content: str = "",
        **extra_fields,
    ) -> bool:
        """添加消息，仅当与最后一条不完全相同时才添加

        适用于避免在 ReAct 循环中重复添加相同的 user 消息。

        Args:
            key: 消息列表的 key，默认 "messages"
            role: 消息角色
            content: 消息内容
            **extra_fields: 其他字段

        Returns:
            True 如果添加了新消息，False 如果跳过了
        """
        self.setdefault(key, [])
        message = {"role": role, "content": content, **extra_fields}

        last = self[key][-1] if self[key] else None
        if last and last.get("role") == role and last.get("content") == content:
            return False

        self[key].append(message)
        return True

    def get_last_message(self, key: str = "messages", role: Optional[str] = None) -> Optional[Dict]:
        """获取最后一条消息

        Args:
            key: 消息列表的 key，默认 "messages"
            role: 可选，按角色过滤

        Returns:
            最后一条消息或 None
        """
        messages = self.get(key, [])
        if not messages:
            return None
        if role is None:
            return messages[-1]
        for msg in reversed(messages):
            if msg.get("role") == role:
                return msg
        return None

    def pop_messages(self, key: str = "messages", count: int = 1) -> list:
        """弹出最后 count 条消息

        用于需要撤销操作的场景。

        Args:
            key: 消息列表的 key，默认 "messages"
            count: 弹出的消息数量

        Returns:
            被弹出的消息列表
        """
        messages = self.get(key, [])
        popped = messages[-count:] if count > 0 else []
        self[key] = messages[:-count] if count > 0 else messages
        return popped

    def get_message_count(self, key: str = "messages") -> int:
        """获取消息数量"""
        return len(self.get(key, []))


class StateKey:
    """State 常用 key 约定（可选使用）

    MESSAGES = "messages"           # 对话历史
    TOOL_CALLS = "tool_calls"       # 待执行的工具调用
    TOOL_RESULTS = "tool_results"   # 工具执行结果
    METADATA = "metadata"           # 元数据
    """

    MESSAGES = "messages"
    TOOL_CALLS = "tool_calls"
    TOOL_RESULTS = "tool_results"
    METADATA = "metadata"
