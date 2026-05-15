"""Signal - 与 Event 平行的高频数据通道"""

from typing import Any, Optional, Callable
from dataclasses import dataclass
import threading


@dataclass
class Signal:
    """信号定义

    Signal 是高频连续数据的载体，与 Event 平行：
    - Event：离散、队列、push、适合低频/重要信息
    - Signal：连续、最新值缓存、pull、适合高频/传感器数据

    Attributes:
        name: 信号名称
        interval: 更新间隔（秒）
        func: 信号源函数，async () -> Any
    """

    name: str
    interval: float
    func: Callable


class SignalBus:
    """信号总线

    每个 Session 持有一个 SignalBus 实例。
    写操作 O(1) 覆盖旧值，读操作 O(1) 返回最新值。
    线程安全，支持并发读写。

    用法:
        signal_bus.write("vision", camera_frame)
        frame = signal_bus.read("vision")
    """

    def __init__(self):
        self._data: dict[str, Any] = {}
        self._lock = threading.Lock()

    def write(self, name: str, data: Any) -> None:
        """写信号

        Args:
            name: 信号名称
            data: 信号数据（覆盖旧值）
        """
        with self._lock:
            self._data[name] = data

    def read(self, name: str) -> Optional[Any]:
        """读信号

        Args:
            name: 信号名称

        Returns:
            最新值，如果信号不存在返回 None
        """
        with self._lock:
            return self._data.get(name)

    def has(self, name: str) -> bool:
        """检查信号是否存在"""
        with self._lock:
            return name in self._data

    def all(self) -> dict[str, Any]:
        """获取所有信号的快照"""
        with self._lock:
            return self._data.copy()
