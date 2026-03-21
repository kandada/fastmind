"""感知循环调度器"""

from typing import Callable, Any, Optional
import asyncio
from collections import defaultdict

from .event import Event
from .app import FastMind


class PerceptionLoop:
    """感知循环

    定期生成事件，用于传感器数据、定时器等外部驱动。
    """

    def __init__(
        self,
        name: str,
        func: Callable,
        interval: float,
        app: Any = None,
    ):
        self.name = name
        self.func = func
        self.interval = interval
        self.app = app
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self, event_handler: Callable) -> None:
        """启动感知循环

        Args:
            event_handler: 事件处理器，接收 Event
        """
        self._running = True
        self._task = asyncio.create_task(self._run(event_handler))

    async def stop(self) -> None:
        """停止感知循环"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self, event_handler: Callable) -> None:
        """感知循环主函数"""
        while self._running:
            try:
                gen = self.func(self.app)

                if asyncio.iscoroutine(gen):
                    events = await gen
                else:
                    events = gen

                if hasattr(events, "__anext__"):
                    async for event in events:
                        if not self._running:
                            break
                        await event_handler(event)
                else:
                    for event in events:
                        if not self._running:
                            break
                        await event_handler(event)

                await asyncio.sleep(self.interval)

            except asyncio.CancelledError:
                break
            except Exception as e:
                await asyncio.sleep(self.interval)


class PerceptionScheduler:
    """感知循环调度器

    管理所有感知循环，支持跨会话感知。
    """

    def __init__(self, app: FastMind):
        self.app = app
        self._loops: dict[str, PerceptionLoop] = {}
        self._running = False
        self._event_handlers: list[Callable] = []

    def register_loop(
        self,
        name: str,
        func: Callable,
        interval: float,
    ) -> None:
        """注册感知循环

        Args:
            name: 感知名称
            func: 感知函数（生成器）
            interval: 触发间隔
        """
        loop = PerceptionLoop(name, func, interval, self.app)
        self._loops[name] = loop

    def register_event_handler(self, handler: Callable) -> None:
        """注册事件处理器

        Args:
            handler: 事件处理器函数
        """
        self._event_handlers.append(handler)

    async def start(self) -> None:
        """启动所有感知循环"""
        self._running = True

        for name, func, interval in self.app.get_perceptions():
            self.register_loop(name, func, interval)

        for loop in self._loops.values():
            await loop.start(self._handle_event)

    async def stop(self) -> None:
        """停止所有感知循环"""
        self._running = False

        for loop in self._loops.values():
            await loop.stop()

        self._loops.clear()

    async def _handle_event(self, event: Event) -> None:
        """处理感知事件"""
        for handler in self._event_handlers:
            try:
                if asyncio.iscoroutinefunction(handler):
                    await handler(event)
                else:
                    handler(event)
            except Exception:
                pass

    def get_loop(self, name: str) -> Optional[PerceptionLoop]:
        """获取感知循环"""
        return self._loops.get(name)

    def list_loops(self) -> list[str]:
        """列出所有感知循环"""
        return list(self._loops.keys())


class Timer:
    """定时器

    简单的定时器实现。
    """

    def __init__(
        self,
        interval: float,
        callback: Callable,
        session_id: Optional[str] = None,
    ):
        self.interval = interval
        self.callback = callback
        self.session_id = session_id
        self._task: Optional[asyncio.Task] = None
        self._running = False

    async def start(self) -> None:
        """启动定时器"""
        self._running = True
        self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """停止定时器"""
        self._running = False
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass

    async def _run(self) -> None:
        """定时器主循环"""
        while self._running:
            try:
                await asyncio.sleep(self.interval)

                if asyncio.iscoroutinefunction(self.callback):
                    await self.callback()
                else:
                    self.callback()

            except asyncio.CancelledError:
                break
            except Exception:
                pass


class SensorManager:
    """传感器管理器

    管理多个传感器，定期收集数据。
    """

    def __init__(self):
        self._sensors: dict[str, Callable] = {}
        self._intervals: dict[str, float] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []
        self._event_handlers: list[Callable] = []

    def register(
        self,
        name: str,
        sensor_func: Callable,
        interval: float = 1.0,
    ) -> None:
        """注册传感器

        Args:
            name: 传感器名称
            sensor_func: 传感器读取函数
            interval: 采样间隔
        """
        self._sensors[name] = sensor_func
        self._intervals[name] = interval

    def register_event_handler(self, handler: Callable) -> None:
        """注册事件处理器"""
        self._event_handlers.append(handler)

    async def start(self) -> None:
        """启动所有传感器"""
        self._running = True

        for name, sensor_func in self._sensors.items():
            interval = self._intervals.get(name, 1.0)
            task = asyncio.create_task(self._run_sensor(name, sensor_func, interval))
            self._tasks.append(task)

    async def stop(self) -> None:
        """停止所有传感器"""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        self._tasks.clear()

    async def _run_sensor(self, name: str, sensor_func: Callable, interval: float) -> None:
        """运行单个传感器"""
        while self._running:
            try:
                if asyncio.iscoroutinefunction(sensor_func):
                    data = await sensor_func()
                else:
                    data = sensor_func()

                event = Event(
                    type="sensor.data",
                    payload={
                        "sensor": name,
                        "data": data,
                    },
                    session_id="system",
                )

                for handler in self._event_handlers:
                    if asyncio.iscoroutinefunction(handler):
                        await handler(event)
                    else:
                        handler(event)

                await asyncio.sleep(interval)

            except asyncio.CancelledError:
                break
            except Exception:
                await asyncio.sleep(interval)
