"""FastMindAPI - 对外接口"""

from typing import Any, Optional, Callable
import asyncio
from collections.abc import AsyncIterator

from ..core.app import FastMind
from ..core.engine import Engine, Session
from ..core.event import Event
from ..core.perception import PerceptionScheduler
from ..utils.logging import get_logger

logger = get_logger("fastmind.api")


class FastMindAPI:
    """FastMind API

    提供外部调用接口，包括事件推送、状态查询、会话管理等功能。

    核心方法:
        - push_event: 推送外部事件
        - stream_events: 流式获取事件（无轮询）
        - run_streaming: 便捷流式对话
        - get_state: 获取状态快照
        - resume_session: 恢复中断会话
        - start/stop: 启动/停止引擎
        - list_sessions: 列出所有会话
        - delete_session: 删除会话

    用法示例:
        fm_api = FastMindAPI(app)
        await fm_api.start()

        # 方式1: stream_events (推荐，无轮询)
        async for ev in fm_api.stream_events("user_123"):
            print(ev)

        # 方式2: run_streaming 便捷方法
        full_text = await fm_api.run_streaming("user_123", "Hello!")

        # 方式3: 直接使用队列
        session = fm_api.get_session("user_123")
        ev = await session.output_queue.get()  # 阻塞等待
    """

    def __init__(self, app: FastMind):
        """初始化 FastMindAPI

        Args:
            app: FastMind 应用实例
        """
        self.app = app
        self._engine = Engine(app)
        self._perception_scheduler = PerceptionScheduler(app)
        self._running = False

    async def start(self) -> None:
        """启动引擎和感知循环"""
        logger.info("Starting FastMindAPI")
        self._running = True
        await self._engine.start()
        await self._perception_scheduler.start()

        self._perception_scheduler.register_event_handler(self._handle_perception_event)

    async def stop(self) -> None:
        """停止引擎和感知循环"""
        logger.info("Stopping FastMindAPI")
        self._running = False
        await self._perception_scheduler.stop()
        await self._engine.stop()

    async def _handle_perception_event(self, event: Event) -> None:
        """处理感知事件"""
        if event.type == "sensor.data":
            session_id = event.session_id or "system"
            if session_id != "system":
                session = self._engine.get_session(session_id)
                if session:
                    await session.push_event(event)

    async def push_event(
        self,
        session_id: str,
        event: Event,
        graph_name: str = "main",
    ) -> Session:
        """推送外部事件

        自动创建不存在的会话。

        Args:
            session_id: 会话 ID
            event: 事件
            graph_name: 图名称

        Returns:
            Session 实例
        """
        logger.debug(f"Push event {event.type} to session {session_id}")
        return await self._engine.push_event(session_id, event, graph_name)

    async def stream_events(
        self,
        session_id: str,
        event_types: Optional[list[str]] = None,
    ) -> AsyncIterator[Event]:
        """流式获取会话事件（无轮询）

        使用 asyncio.Queue.get() 阻塞等待，零CPU浪费。

        Args:
            session_id: 会话 ID
            event_types: 只接收指定类型的事件，None 表示全部接收

        Yields:
            Event: 输出事件

        用法:
            async for ev in fm_api.stream_events("user_123"):
                if ev.type == "stream.chunk":
                    print(ev.payload.get("delta", ""), end="", flush=True)
                elif ev.type == "stream.end":
                    break
        """
        session = self._engine.get_session(session_id)
        if not session:
            return

        while self._running:
            try:
                event = await session.output_queue.get()
                if event_types is None or event.type in event_types:
                    yield event
                if event.type in ("stream.end", "error", "interrupt"):
                    break
            except asyncio.CancelledError:
                break

    def get_state(self, session_id: str) -> Optional[dict]:
        """获取状态快照

        Args:
            session_id: 会话 ID

        Returns:
            状态字典
        """
        return self._engine.get_session_state(session_id)

    async def resume_session(
        self,
        session_id: str,
        user_input: Any = None,
    ) -> None:
        """恢复中断会话

        Args:
            session_id: 会话 ID
            user_input: 用户输入
        """
        await self._engine.resume_session(session_id, user_input)

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话

        Args:
            session_id: 会话 ID

        Returns:
            Session 实例
        """
        return self._engine.get_session(session_id)

    def list_sessions(self) -> list[str]:
        """列出所有会话 ID

        Returns:
            会话 ID 列表
        """
        return self._engine.list_sessions()

    async def delete_session(self, session_id: str) -> None:
        """删除会话及状态

        Args:
            session_id: 会话 ID
        """
        await self._engine.delete_session(session_id)

    def get_tool_schemas(self) -> list[dict]:
        """获取所有工具的 schema

        Returns:
            工具 schema 列表
        """
        return self.app.get_tool_schemas()

    def get_graph(self, name: str = "main"):
        """获取图

        Args:
            name: 图名称
        """
        return self.app.get_graph(name)

    async def get_output_event(self, session_id: str) -> Optional[Event]:
        """获取单个输出事件（非阻塞）

        Args:
            session_id: 会话 ID

        Returns:
            Event 或 None
        """
        return await self._engine.get_session_output(session_id)

    async def wait_for_output_event(
        self,
        session_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Event]:
        """等待单个输出事件（阻塞直到有输出或超时）

        Args:
            session_id: 会话 ID
            timeout: 超时时间（秒），None 表示一直等待

        Returns:
            Event 或 None
        """
        return await self._engine.wait_for_session_output(session_id, timeout)

    async def run_streaming(
        self,
        session_id: str,
        user_input: str,
        on_chunk: Optional[Callable[[str], None]] = None,
        on_end: Optional[Callable[[str], None]] = None,
    ) -> str:
        """便捷方法：运行流式对话

        Args:
            session_id: 会话 ID
            user_input: 用户输入
            on_chunk: 收到 chunk 时的回调，签名: on_chunk(delta: str)
            on_end: 流结束时回调，签名: on_end(full_text: str)

        Returns:
            完整的回复文本
        """
        full_text = []
        event = Event("user.message", {"text": user_input}, session_id)
        await self.push_event(session_id, event)

        async for ev in self.stream_events(session_id):
            if ev.type == "stream.chunk":
                delta = ev.payload.get("delta", "")
                full_text.append(delta)
                if on_chunk:
                    on_chunk(delta)
            elif ev.type == "stream.end":
                if on_end:
                    on_end("".join(full_text))
                break

        return "".join(full_text)


class SessionQueue:
    """会话事件队列封装

    提供更方便的队列消费接口。
    """

    def __init__(self, api: FastMindAPI, session_id: str):
        self.api = api
        self.session_id = session_id

    def __aiter__(self) -> AsyncIterator[Event]:
        return self.api.stream_events(self.session_id)

    async def get(self) -> Optional[Event]:
        """获取单个事件（阻塞等待）"""
        session = self.api.get_session(self.session_id)
        if not session:
            return None
        return await session.output_queue.get()
