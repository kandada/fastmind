"""Event 类的单元测试"""

import pytest
from fastmind.core.event import Event, EventType


class TestEvent:
    """Event 测试"""

    def test_create_event(self):
        """测试创建事件"""
        event = Event(type="user.message", payload={"text": "hello"}, session_id="session_001")

        assert event.type == "user.message"
        assert event.payload == {"text": "hello"}
        assert event.session_id == "session_001"
        assert event.event_id is not None

    def test_event_to_dict(self):
        """测试事件转字典"""
        event = Event(type="user.message", payload={"text": "hello"}, session_id="session_001")

        d = event.to_dict()
        assert d["type"] == "user.message"
        assert d["payload"] == {"text": "hello"}
        assert d["session_id"] == "session_001"
        assert "event_id" in d

    def test_event_from_dict(self):
        """测试从字典创建事件"""
        data = {
            "type": "user.message",
            "payload": {"text": "hello"},
            "session_id": "session_001",
            "event_id": "test_id",
        }

        event = Event.from_dict(data)
        assert event.type == "user.message"
        assert event.payload == {"text": "hello"}
        assert event.session_id == "session_001"
        assert event.event_id == "test_id"

    def test_event_is_type(self):
        """测试事件类型检查"""
        event = Event(type="user.message", payload={}, session_id="session_001")

        assert event.is_type("user") is True
        assert event.is_type("user.") is True
        assert event.is_type("admin") is False

    def test_event_copy(self):
        """测试事件复制"""
        event = Event(type="user.message", payload={"text": "hello"}, session_id="session_001")

        copied = event.copy()
        assert copied.type == event.type
        assert copied.payload == event.payload
        assert copied.session_id == event.session_id
        assert copied.event_id != event.event_id


class TestEventType:
    """EventType 常量测试"""

    def test_event_type_constants(self):
        """测试事件类型常量"""
        assert EventType.USER_MESSAGE == "user.message"
        assert EventType.STREAM_CHUNK == "stream.chunk"
        assert EventType.STREAM_END == "stream.end"
        assert EventType.INTERRUPT == "interrupt"
        assert EventType.SENSOR_DATA == "sensor.data"
