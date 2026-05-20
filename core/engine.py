"""执行引擎 - 事件驱动的图执行 + VLA 快循环"""

from typing import Any, Optional
import asyncio
import copy
from collections import deque
import time

from .event import Event
from .graph import Graph
from .app import FastMind
from .state import State
from .signal import SignalBus
from ..utils.logging import get_logger

logger = get_logger("fastmind.engine")


class EventBuffer:
    """只追加环形缓冲区，游标读取，不消费

    多个消费者可独立通过游标读取全部事件，互不干扰。
    替代 asyncio.Queue 解决"一个事件只能被一个消费者取走"的问题。
    """

    def __init__(self, maxlen: int = 5000):
        self._events: deque = deque()
        self._maxlen = maxlen
        self._notifier = asyncio.Event()
        self._base = 0

    def append(self, event) -> int:
        """追加事件，返回其逻辑索引"""
        self._events.append(event)
        if len(self._events) > self._maxlen:
            self._events.popleft()
            self._base += 1
        self._notifier.set()
        return self._base + len(self._events) - 1

    def read(self, cursor: int) -> list:
        """返回 cursor 之后的所有事件（不消费）"""
        if cursor < self._base:
            cursor = self._base
        idx = cursor - self._base
        if idx >= len(self._events):
            return []
        return list(self._events)[idx:]

    async def wait(self, cursor: int, timeout: float = None) -> list:
        """阻塞直到有 cursor 之后的新事件"""
        if cursor < self._base:
            cursor = self._base
        if cursor < self._base + len(self._events):
            return list(self._events)[cursor - self._base:]

        self._notifier.clear()
        if cursor < self._base + len(self._events):
            return list(self._events)[cursor - self._base:]

        try:
            await asyncio.wait_for(self._notifier.wait(), timeout=timeout)
        except asyncio.TimeoutError:
            pass

        if cursor < self._base:
            cursor = self._base
        if cursor < self._base + len(self._events):
            return list(self._events)[cursor - self._base:]
        return []

    @property
    def tail_cursor(self) -> int:
        return self._base + len(self._events)


class NotifyingQueue:
    """向后兼容包装：对外暴露 queue 接口，内部用 EventBuffer

    保留 .put() / .get() / .empty() 等传统 Queue 接口，
    默认消费者使用内部游标，其他消费者可直接读 EventBuffer。
    """

    def __init__(self, buffer: EventBuffer, event: asyncio.Event):
        self._buffer = buffer
        self._event = event
        self._cursor = 0

    async def put(self, item):
        self._buffer.append(item)
        self._event.set()

    def put_nowait(self, item):
        self._buffer.append(item)
        self._event.set()

    async def get(self):
        while True:
            self._event.clear()
            events = self._buffer.read(self._cursor)
            if events:
                self._cursor += len(events)
                return events[0]
            events = self._buffer.read(self._cursor)
            if events:
                self._cursor += len(events)
                return events[0]
            await self._event.wait()

    def get_nowait(self):
        events = self._buffer.read(self._cursor)
        if not events:
            raise asyncio.QueueEmpty()
        self._cursor += len(events)
        return events[0]

    def empty(self):
        return self._cursor >= self._buffer.tail_cursor

    def qsize(self):
        return self._buffer.tail_cursor - self._cursor

    def full(self):
        return False


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
        self._event_buffer: EventBuffer = EventBuffer(maxlen=5000)
        self._output_event: asyncio.Event = asyncio.Event()
        self.output_queue = NotifyingQueue(self._event_buffer, self._output_event)
        self._task: Optional[asyncio.Task] = None
        self._state = self.STATE_CREATED
        self._checkpoint: Optional[dict] = None
        self._interrupted = False
        self._current_node: Optional[str] = None
        self._pending_input_events: deque[Event] = deque()
        self._event_history: list[str] = []
        self._max_history = 100
        self._last_event_time: float = time.time()
        self.signal_bus = SignalBus()
        self._vla_task: Optional[asyncio.Task] = None
        self._signal_tasks: list[asyncio.Task] = []

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

        # 启动 VLA 快循环
        if self.app.has_vla() and (self._vla_task is None or self._vla_task.done()):
            self._vla_task = asyncio.create_task(self._vla_scheduler())

        # 启动信号源
        self._start_signals()

    def _start_signals(self) -> None:
        """启动所有注册的信号源（每个信号独立 task）"""
        signals = self.app.get_signals()
        running = {s.get_name() for s in self._signal_tasks}
        for name, sig in signals.items():
            if name not in running:
                task = asyncio.create_task(self._run_signal(name, sig))
                task.set_name(name)
                self._signal_tasks.append(task)

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

        if self._vla_task:
            self._vla_task.cancel()
            try:
                await self._vla_task
            except asyncio.CancelledError:
                pass
            self._vla_task = None

        for task in self._signal_tasks:
            task.cancel()
        if self._signal_tasks:
            await asyncio.gather(*self._signal_tasks, return_exceptions=True)
            self._signal_tasks = []

    async def push_event(self, event: Event) -> None:
        """推送事件到输入队列"""
        self._pending_input_events.append(event)
        if len(self._pending_input_events) > self._max_history:
            self._pending_input_events.popleft()
        await self.input_queue.put(event)

    async def _put_output(self, event: Event) -> None:
        """内部方法：推送输出事件并触发信号"""
        await self.output_queue.put(event)

    async def get_output(self) -> Optional[Event]:
        """获取输出事件（非阻塞）"""
        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None

    async def wait_for_output(self, timeout: Optional[float] = None) -> Optional[Event]:
        """等待输出事件（阻塞直到有输出或超时）"""
        self._output_event.clear()
        try:
            return self.output_queue.get_nowait()
        except asyncio.QueueEmpty:
            pass

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
                    user_input = str(event.payload.get("user_input", ""))
                    cancel_node = (
                        self._checkpoint.get("cancel_node") if self._checkpoint else None
                    )
                    resume_node = (
                        self._checkpoint.get("resume_node") if self._checkpoint else None
                    )
                    if cancel_node and user_input.lower() == "cancel":
                        next_node = cancel_node
                    elif resume_node:
                        next_node = resume_node
                    elif self._current_node:
                        next_node = self.graph.get_next_node(
                            self._current_node, self.state, event
                        )
                    else:
                        next_node = self.graph.entry_point
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

    async def _vla_scheduler(self) -> None:
        """VLA 快循环调度器

        遍历所有注册的 @app.vla，按各自频率调度执行。
        每个 VLA 的输出按通道名路由到对应的 @app.vla_action。
        """
        vlas = self.app.get_vlas()
        actions = self.app.get_vla_actions()
        last_ticks: dict[str, float] = {}

        logger.debug(
            f"Session {self.session_id}: VLA scheduler started "
            f"({len(vlas)} VLAs: {list(vlas.keys())})"
        )

        while self._state != self.STATE_STOPPED:
            now = time.time()

            for name, cfg in vlas.items():
                # 频率控制
                interval = 1.0 / cfg.frequency
                last = last_ticks.get(name, 0)
                if now - last < interval:
                    continue
                last_ticks[name] = now

                # 检查 LLM 暂停信号
                if self.state.get("llm", {}).get("vla_paused", False):
                    continue

                # VLA 推理
                try:
                    action_dict = await cfg.func(self.state, self.signal_bus)
                except asyncio.CancelledError:
                    raise
                except Exception as e:
                    logger.error(
                        f"Session {self.session_id}: VLA '{name}' inference error: {e}"
                    )
                    continue

                # 按通道名路由到 action executor
                if isinstance(action_dict, dict):
                    for channel, vector in action_dict.items():
                        if channel not in actions:
                            continue
                        try:
                            result = await actions[channel].execute(vector)
                            self.state.setdefault("vla_actions", {})[channel] = vector
                            if isinstance(result, dict):
                                self.state.setdefault("vla_action_results", {})[channel] = result
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            logger.error(
                                f"Session {self.session_id}: VLA action '{channel}' error: {e}"
                            )

            await asyncio.sleep(0.001)

    async def _run_signal(self, name: str, signal_cfg) -> None:
        """运行单个信号源

        按 interval 周期调用信号函数，结果写入 SignalBus。
        """
        while self._state != self.STATE_STOPPED:
            try:
                result = await signal_cfg.func()
                self.signal_bus.write(name, result)
                await asyncio.sleep(signal_cfg.interval)
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: signal '{name}' error: {e}"
                )
                await asyncio.sleep(signal_cfg.interval)

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
                await self._put_output(
                    Event(
                        type="error",
                        payload={"node": current_node, "error": f"Node '{current_node}' not found"},
                        session_id=self.session_id,
                    )
                )
                break

            logger.debug(
                f"Session {self.session_id}: executing node '{current_node}' "
                f"(iteration={iteration_count}, tool_calls={self.state.get('tool_calls')}, "
                f"tool_results={self.state.get('tool_results')})"
            )

            try:
                if isinstance(node, Graph):
                    await self._execute_subgraph(node, event)
                else:
                    output_events = await self._execute_node(current_node, node, event)

                    for output_event in output_events:
                        if output_event.type == "interrupt":
                            self._save_checkpoint(current_node, output_event.payload)
                            self._state = self.STATE_INTERRUPTED
                            self._interrupted = True
                            await self._put_output(output_event)
                            return
                        await self._put_output(output_event)

                logger.debug(
                    f"Session {self.session_id}: node '{current_node}' completed, "
                    f"tool_calls={self.state.get('tool_calls')}, tool_results={self.state.get('tool_results')}"
                )

                current_node = self.graph.get_next_node(current_node, self.state, event)

                logger.debug(f"Session {self.session_id}: next node is '{current_node}'")
            except Exception as e:
                logger.error(
                    f"Session {self.session_id}: error in node {current_node}: {e}",
                    exc_info=True,
                )
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
        iteration_count = 0
        max_iterations = (
            subgraph.max_iterations
            if hasattr(subgraph, "max_iterations")
            else Graph.DEFAULT_MAX_ITERATIONS
        )

        while next_node and next_node != Graph.END_NODE and self._state == self.STATE_RUNNING:
            iteration_count += 1
            if iteration_count > max_iterations:
                logger.error(
                    f"Session {self.session_id}: subgraph exceeded max iterations ({max_iterations})"
                )
                await self._put_output(
                    Event(
                        type="error",
                        payload={"error": f"Subgraph exceeded max iterations ({max_iterations})"},
                        session_id=self.session_id,
                    )
                )
                break

            node = subgraph.get_node(next_node)
            if not node:
                logger.warning(f"Session {self.session_id}: subgraph node '{next_node}' not found")
                await self._put_output(
                    Event(
                        type="error",
                        payload={"node": next_node, "error": f"Node '{next_node}' not found in subgraph"},
                        session_id=self.session_id,
                    )
                )
                break

            try:
                output_events = await self._execute_node(next_node, node, event)
                for output_event in output_events:
                    if output_event.type == "interrupt":
                        self._save_checkpoint(next_node, output_event.payload)
                        self._state = self.STATE_INTERRUPTED
                        self._interrupted = True
                        await self._put_output(output_event)
                        return
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
        """合并新 state，仅更新节点返回的 key，不替换整个 dict

        防止节点返回部分 state 时意外丢弃其他 key。
        """
        if not isinstance(new_state, dict):
            return

        self.state.update(new_state)

    def _safe_deepcopy(self, obj: Any, max_depth: int = 10) -> Any:
        """安全的 deepcopy，遇到不可序列化对象时转为字符串

        Args:
            obj: 要拷贝的对象
            max_depth: 递归最大深度，防止无限递归

        Returns:
            深拷贝结果，或不可序列化时的字符串表示
        """
        if max_depth <= 0:
            return str(obj)

        # 基本类型直接返回
        if obj is None or isinstance(obj, (bool, int, float, str, bytes)):
            return obj

        try:
            return copy.deepcopy(obj)
        except (TypeError, RuntimeError, AttributeError):
            pass

        # 逐项处理容器类型
        if isinstance(obj, dict):
            return {
                k: self._safe_deepcopy(v, max_depth - 1)
                for k, v in obj.items()
            }
        if isinstance(obj, (list, tuple)):
            return type(obj)(
                self._safe_deepcopy(item, max_depth - 1) for item in obj
            )
        if isinstance(obj, set):
            return {self._safe_deepcopy(item, max_depth - 1) for item in obj}

        # 兜底：转为字符串
        try:
            return str(obj)
        except Exception:
            return "<unpicklable>"

    def _save_checkpoint(
        self, current_node: str, interrupt_payload: dict = None
    ) -> None:
        """保存检查点（包含 state 和 pending events）"""
        state_snapshot = {
            k: v for k, v in self.state.items()
            if k not in ("_output_queue", "_session_id")
        }
        self._checkpoint = {
            "state": self._safe_deepcopy(state_snapshot),
            "current_node": current_node,
            "pending_events": list(self._pending_input_events),
        }
        if interrupt_payload:
            self._checkpoint["resume_node"] = interrupt_payload.get("resume_node")
            self._checkpoint["cancel_node"] = interrupt_payload.get("cancel_node")

    def _restore_from_checkpoint(self) -> None:
        """恢复检查点"""
        if self._checkpoint:
            self.state = State(self._safe_deepcopy(self._checkpoint["state"]))
            self._current_node = self._checkpoint.get("current_node")
            self._interrupted = False
            pending = self._checkpoint.get("pending_events", [])
            self._pending_input_events = deque(pending)
            self.state["_output_queue"] = self.output_queue
            self.state["_session_id"] = self.session_id


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
