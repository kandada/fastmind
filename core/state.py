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
