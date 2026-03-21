"""Event 类定义"""

from dataclasses import dataclass, field
from typing import Any, Optional
import uuid


@dataclass
class Event:
    """事件类，用于在框架中传递消息和触发动作。

    Attributes:
        type: 事件类型，如 "user.message", "stream.chunk", "interrupt"
        payload: 事件数据，可以是任意结构
        session_id: 会话 ID，用于隔离不同会话
        event_id: 事件唯一标识符
    """

    type: str
    payload: dict[str, Any]
    session_id: str
    event_id: str = field(default_factory=lambda: str(uuid.uuid4()))

    def to_dict(self) -> dict[str, Any]:
        """将事件转换为字典"""
        return {
            "type": self.type,
            "payload": self.payload,
            "session_id": self.session_id,
            "event_id": self.event_id,
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> "Event":
        """从字典创建事件"""
        return cls(
            type=data["type"],
            payload=data["payload"],
            session_id=data["session_id"],
            event_id=data.get("event_id", str(uuid.uuid4())),
        )

    def is_type(self, type_prefix: str) -> bool:
        """检查事件类型是否以指定前缀开头"""
        return self.type.startswith(type_prefix)

    def copy(self) -> "Event":
        """创建事件的副本"""
        return Event(
            type=self.type,
            payload=self.payload.copy(),
            session_id=self.session_id,
            event_id=str(uuid.uuid4()),
        )


class EventType:
    """常用事件类型常量"""

    USER_MESSAGE = "user.message"
    STREAM_CHUNK = "stream.chunk"
    STREAM_END = "stream.end"
    INTERRUPT = "interrupt"
    SENSOR_DATA = "sensor.data"
    RESUME = "resume"
    TIMER = "timer"
