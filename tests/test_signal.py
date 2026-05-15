"""Signal 和 SignalBus 的单元测试"""

import pytest
import threading
import time

from fastmind.core.signal import Signal, SignalBus


class TestSignal:
    """Signal 数据类测试"""

    def test_create_signal(self):
        """测试创建信号"""
        async def dummy():
            pass

        sig = Signal(name="vision", interval=1/30, func=dummy)
        assert sig.name == "vision"
        assert sig.interval == pytest.approx(0.0333, rel=0.01)
        assert sig.func is dummy


class TestSignalBus:
    """SignalBus 测试"""

    @pytest.fixture
    def bus(self):
        return SignalBus()

    def test_write_and_read(self, bus):
        """测试写入和读取"""
        bus.write("vision", "frame_data_123")
        assert bus.read("vision") == "frame_data_123"

    def test_read_nonexistent(self, bus):
        """测试读取不存在的信号"""
        assert bus.read("nonexistent") is None

    def test_has(self, bus):
        """测试信号存在检查"""
        assert bus.has("vision") is False
        bus.write("vision", "data")
        assert bus.has("vision") is True

    def test_overwrite(self, bus):
        """测试覆盖旧值"""
        bus.write("sensor", "value1")
        bus.write("sensor", "value2")
        assert bus.read("sensor") == "value2"

    def test_all(self, bus):
        """测试获取所有信号快照"""
        bus.write("a", 1)
        bus.write("b", 2)
        snapshot = bus.all()
        assert snapshot == {"a": 1, "b": 2}

    def test_all_isolation(self, bus):
        """测试 all() 返回的字典是拷贝"""
        bus.write("x", 10)
        snapshot = bus.all()
        snapshot["x"] = 99
        assert bus.read("x") == 10

    def test_multiple_signals_independent(self, bus):
        """测试多个信号互不干扰"""
        bus.write("vision", "frame")
        bus.write("hearing", "sound")
        bus.write("proprioception", "state")
        assert bus.read("vision") == "frame"
        assert bus.read("hearing") == "sound"
        assert bus.read("proprioception") == "state"

    def test_write_none(self, bus):
        """测试写入 None"""
        bus.write("empty", None)
        assert bus.read("empty") is None
        assert bus.has("empty") is True

    def test_concurrent_write_and_read(self, bus):
        """测试并发读写（线程安全）"""
        errors = []
        def writer():
            for i in range(100):
                bus.write("counter", i)
                time.sleep(0.0001)

        def reader():
            for _ in range(50):
                val = bus.read("counter")
                if val is None:
                    continue
                if not isinstance(val, int):
                    errors.append(f"Unexpected type: {type(val)}")
                time.sleep(0.0002)

        threads = [
            threading.Thread(target=writer),
            threading.Thread(target=reader),
            threading.Thread(target=reader),
        ]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        assert len(errors) == 0

    def test_concurrent_all_safe(self, bus):
        """测试并发 all() 不抛异常"""
        def writer():
            for i in range(50):
                bus.write(f"key_{i % 5}", i)
                time.sleep(0.0001)

        threads = [threading.Thread(target=writer) for _ in range(4)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()

        snapshot = bus.all()
        assert len(snapshot) == 5

    def test_signal_dataclass_defaults(self):
        """测试信号数据类字段"""
        async def src():
            return 42

        sig = Signal(name="test", interval=0.1, func=src)
        assert sig.name == "test"
        assert sig.interval == 0.1
        assert sig.func is src
