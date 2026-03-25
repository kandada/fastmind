"""执行引擎 - 事件驱动的图执行"""

from typing import Any, Optional
import asyncio
import copy
from collections import deque
import time

from .event import Event
from .graph import Graph
from .app import FastMind
from .state import State
from ..utils.logging import get_logger

logger = get_logger("fastmind.engine")


class Session:
    """会话实例

    每个 session_id 拥有独立的状态、事件队列和执行上下文。

    Session 生命周期:
    1. CREATED - 刚创建，未启动
    2. RUNNING - 正在处理事件
    3. IDLE - 等待新事件
    4. INTERRUPTED - 被中断，等待恢复
    5. STOPPED - 已停止

    状态转换:
    - CREATED -> RUNNING: 调用 start()
    - RUNNING -> IDLE: 处理完一个事件，等待下一个
    - RUNNING -> INTERRUPTED: 节点返回 interrupt 事件
    - INTERRUPTED -> RUNNING: 收到 resume 事件
    - ANY -> STOPPED: 调用 stop()
    """

    STATE_CREATED = "created"
    STATE_RUNNING = "running"
    STATE_IDLE = "idle"
    STATE_INTERRUPTED = "interrupted"
    STATE_STOPPED = "stopped"

    def __init__(
        self,
        session_id: str,
        graph: Graph,
        app: FastMind,
    ):
        self.session_id = session_id
        self.graph = graph
        self.app = app
        self.state: dict = State()
        self.input_queue: asyncio.Queue[Event] = asyncio.Queue()
        self.output_queue: asyncio.Queue[Event] = asyncio.Queue()
        self._output_event: asyncio.Event = asyncio.Event()
        self._task: Optional[asyncio.Task] = None
        self._state = self.STATE_CREATED
        self._checkpoint: Optional[dict] = None
        self._interrupted = False
        self._current_node: Optional[str] = None
        self._pending_input_events: deque[Event] = deque()
        self._event_history: list[str] = []
        self._max_history = 100
        self._last_event_time: float = time.time()

        self.state["_output_queue"] = self.output_queue
        self.state["_session_id"] = self.session_id
        self._output_event.set()

    @property
    def session_state(self) -> str:
        """获取会话当前状态"""
        return self._state

    @property
    def is_running(self) -> bool:
        """检查会话是否正在运行"""
        return self._state in (self.STATE_RUNNING, self.STATE_IDLE)

    @property
    def is_alive(self) -> bool:
        """检查会话是否活跃（可以处理事件）"""
        return self._state not in (self.STATE_STOPPED,)

    async def start(self) -> None:
        """启动会话处理循环"""
        if self._state == self.STATE_STOPPED:
            return
        self._state = self.STATE_RUNNING
        if self._task is None or self._task.done():
            self._task = asyncio.create_task(self._run())

    async def stop(self) -> None:
        """停止会话"""
        self._state = self.STATE_STOPPED
        if self._task:
            self._task.cancel()
            try:
                await self._task
            except asyncio.CancelledError:
                pass
            self._task = None

    async def push_event(self, event: Event) -> None:
        """推送事件到输入队列"""
        self._pending_input_events.append(event)
        await self.input_queue.put(event)

    async def _put_output(self, event: Event) -> None:
        """内部方法：推送输出事件并触发信号"""
        await self.output_queue.put(event)
        self._output_event.set()

    async def get_output(self) -> Optional[Event]:
        """获取输出事件（非阻塞）"""
        try:
            event = self.output_queue.get_nowait()
            self._output_event.set() if not self.output_queue.empty() else None
            return event
        except asyncio.QueueEmpty:
            return None

    async def wait_for_output(self, timeout: Optional[float] = None) -> Optional[Event]:
        """等待输出事件（阻塞直到有输出或超时）"""
        if not self.output_queue.empty():
            return self.output_queue.get_nowait()

        self._output_event.clear()
        try:
            if timeout is None:
                await self._output_event.wait()
            else:
                await asyncio.wait_for(self._output_event.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            return None

        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    def _record_event(self, event: Event) -> None:
        """记录事件历史，用于幂等性检查"""
        self._event_history.append(event.event_id)
        if len(self._event_history) > self._max_history:
            self._event_history = self._event_history[-self._max_history :]

    def _is_event_processed(self, event: Event) -> bool:
        """检查事件是否已处理过（幂等性保证）"""
        return event.event_id in self._event_history

    async def _run(self) -> None:
        """会话主循环"""
        logger.debug(f"Session {self.session_id} started")
        while self._state != self.STATE_STOPPED:
            try:
                event = await self.input_queue.get()

                if self._is_event_processed(event) and event.type != "resume":
                    logger.debug(
                        f"Session {self.session_id}: skipped duplicate event {event.event_id}"
                    )
                    continue

                self._record_event(event)
                self._last_event_time = time.time()

                if event.type == "resume":
                    self._state = self.STATE_RUNNING
                    self._restore_from_checkpoint()
                    if self._current_node:
                        next_node = self._current_node
                    else:
                        next_node = event.payload.get("resume_node", self.graph.entry_point)
                else:
                    self._state = self.STATE_RUNNING
                    next_node = self.graph.entry_point

                logger.debug(f"Session {self.session_id}: processing event {event.type}")
                if next_node:
                    await self._execute_node_chain(next_node, event)

                if self._state == self.STATE_RUNNING:
                    self._state = self.STATE_IDLE

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Session {self.session_id} error: {e}")
                self._state = self.STATE_IDLE
                await self._put_output(
                    Event(
                        type="error",
                        payload={"error": str(e)},
                        session_id=self.session_id,
                    )
                )

    async def _execute_node_chain(self, start_node: str, event: Event) -> None:
        """执行节点链"""
        current_node = start_node
        iteration_count = 0
        max_iterations = (
            self.graph.max_iterations
            if hasattr(self.graph, "max_iterations")
            else Graph.DEFAULT_MAX_ITERATIONS
        )

        while current_node and current_node != Graph.END_NODE and self._state == self.STATE_RUNNING:
            iteration_count += 1
            if iteration_count > max_iterations:
                logger.error(
                    f"Session {self.session_id}: exceeded max iterations ({max_iterations}), possible infinite loop"
                )
                await self._put_output(
                    Event(
                        type="error",
                        payload={"error": f"Exceeded max iterations ({max_iterations})"},
                        session_id=self.session_id,
                    )
                )
                break

            self._current_node = current_node

            node = self.graph.get_node(current_node)
            if not node:
                logger.warning(f"Session {self.session_id}: node '{current_node}' not found")
                break

            try:
                if isinstance(node, Graph):
                    await self._execute_subgraph(node, event)
                else:
                    output_events = await self._execute_node(current_node, node, event)

                    for output_event in output_events:
                        if output_event.type == "interrupt":
                            self._save_checkpoint(current_node)
                            self._state = self.STATE_INTERRUPTED
                            self._interrupted = True
                            await self._put_output(output_event)
                            return
                        await self._put_output(output_event)

                current_node = self.graph.get_next_node(current_node, self.state, event)
            except Exception as e:
                logger.error(f"Session {self.session_id}: error in node {current_node}: {e}")
                await self._put_output(
                    Event(
                        type="error",
                        payload={"node": current_node, "error": str(e)},
                        session_id=self.session_id,
                    )
                )
                break

    async def _execute_subgraph(self, subgraph: Graph, event: Event) -> None:
        """执行子图"""
        next_node = subgraph.entry_point

        while next_node and next_node != Graph.END_NODE and self._state == self.STATE_RUNNING:
            node = subgraph.get_node(next_node)
            if not node:
                break

            try:
                output_events = await self._execute_node(next_node, node, event)
                for output_event in output_events:
                    await self._put_output(output_event)

                next_node = subgraph.get_next_node(next_node, self.state, event)
            except Exception as e:
                logger.error(f"Session {self.session_id}: error in subgraph node {next_node}: {e}")
                break

    async def _execute_node(
        self,
        node_name: str,
        node: Any,
        event: Event,
    ) -> list[Event]:
        """执行单个节点"""
        logger.debug(f"Session {self.session_id}: executing node {node_name}")
        try:
            if hasattr(node, "execute"):
                result = await node.execute(self.state, event)
            elif asyncio.iscoroutinefunction(node):
                result = await node(self.state, event)
            else:
                result = node(self.state, event)

            if isinstance(result, tuple):
                new_state, output_events = result
                self._merge_state(new_state)
                return output_events
            else:
                self._merge_state(result)
                return []

        except Exception as e:
            logger.error(f"Session {self.session_id}: node {node_name} error: {e}")
            return [
                Event(
                    type="error",
                    payload={"node": node_name, "error": str(e)},
                    session_id=self.session_id,
                )
            ]

    def _merge_state(self, new_state: dict) -> None:
        """合并新 state，保留特殊字段

        如果节点返回新 state，需要保留 _output_queue, _session_id 等特殊字段。
        """
        if not isinstance(new_state, dict):
            return

        reserved_keys = {"_output_queue", "_session_id"}
        for key in reserved_keys:
            if key in self.state and key not in new_state:
                new_state[key] = self.state[key]

        self.state = new_state

    def _save_checkpoint(self, current_node: str) -> None:
        """保存检查点（包含 state 和 pending events）"""
        self._checkpoint = {
            "state": copy.deepcopy(self.state),
            "current_node": current_node,
            "pending_events": list(self._pending_input_events),
        }

    def _restore_from_checkpoint(self) -> None:
        """恢复检查点"""
        if self._checkpoint:
            self.state = copy.deepcopy(self._checkpoint["state"])
            self._current_node = self._checkpoint.get("current_node")
            self._interrupted = False
            pending = self._checkpoint.get("pending_events", [])
            self._pending_input_events = deque(pending)


class Engine:
    """执行引擎

    管理所有会话，负责事件路由和图执行。
    """

    def __init__(self, app: FastMind):
        """初始化引擎

        Args:
            app: FastMind 应用实例
        """
        self.app = app
        self._sessions: dict[str, Session] = {}
        self._running = False
        self._tasks: list[asyncio.Task] = []

    async def start(self) -> None:
        """启动引擎"""
        self._running = True

    async def stop(self) -> None:
        """停止引擎"""
        self._running = False

        for task in self._tasks:
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
        self._tasks.clear()

        for session in self._sessions.values():
            await session.stop()

    def get_or_create_session(
        self,
        session_id: str,
        graph_name: str = "main",
    ) -> Session:
        """获取或创建会话

        Args:
            session_id: 会话 ID
            graph_name: 图名称，默认 "main"

        Returns:
            Session 实例
        """
        if session_id in self._sessions:
            session = self._sessions[session_id]
            if session.session_state == Session.STATE_STOPPED:
                session._state = Session.STATE_CREATED
            return session

        graph = self.app.get_graph(graph_name)
        if not graph:
            raise ValueError(f"Graph '{graph_name}' not found")

        session = Session(session_id, graph, self.app)
        self._sessions[session_id] = session
        return session

    async def push_event(
        self,
        session_id: str,
        event: Event,
        graph_name: str = "main",
    ) -> Session:
        """推送事件到会话

        Args:
            session_id: 会话 ID
            event: 事件
            graph_name: 图名称

        Returns:
            Session 实例
        """
        session = self.get_or_create_session(session_id, graph_name)

        if not session.is_alive:
            raise RuntimeError(f"Session {session_id} is stopped")

        if not session.is_running:
            await session.start()

        await session.push_event(event)
        return session

    def get_session(self, session_id: str) -> Optional[Session]:
        """获取会话"""
        return self._sessions.get(session_id)

    def get_session_state(self, session_id: str) -> Optional[dict]:
        """获取会话状态"""
        session = self._sessions.get(session_id)
        if session:
            return session.state.copy()
        return None

    async def get_session_output(self, session_id: str) -> Optional[Event]:
        """获取会话输出事件（非阻塞）"""
        session = self._sessions.get(session_id)
        if session:
            return await session.get_output()
        return None

    async def wait_for_session_output(
        self,
        session_id: str,
        timeout: Optional[float] = None,
    ) -> Optional[Event]:
        """等待会话输出事件（阻塞直到有输出或超时）"""
        session = self._sessions.get(session_id)
        if session:
            return await session.wait_for_output(timeout)
        return None

    def list_sessions(self) -> list[str]:
        """列出所有会话 ID"""
        return list(self._sessions.keys())

    async def delete_session(self, session_id: str) -> None:
        """删除会话"""
        if session_id in self._sessions:
            await self._sessions[session_id].stop()
            del self._sessions[session_id]

    async def resume_session(
        self,
        session_id: str,
        user_input: Any = None,
    ) -> None:
        """恢复中断的会话

        Args:
            session_id: 会话 ID
            user_input: 用户输入
        """
        session = self._sessions.get(session_id)
        if session and session.session_state == Session.STATE_INTERRUPTED:
            resume_event = Event(
                type="resume",
                payload={"user_input": user_input},
                session_id=session_id,
            )
            await session.push_event(resume_event)
        elif not session:
            raise ValueError(f"Session {session_id} not found")
        else:
            raise RuntimeError(
                f"Session {session_id} is not interrupted (state: {session.session_state})"
            )
