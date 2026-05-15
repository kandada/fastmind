"""Engine 和 Session 单元测试"""

import pytest
import asyncio
import time
from fastmind import FastMind
from fastmind.core.graph import Graph
from fastmind.core.event import Event
from fastmind.core.engine import Session, Engine


class TestSession:
    """Session 测试"""

    @pytest.mark.asyncio
    async def test_session_initialization(self):
        """测试 Session 初始化"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        assert session.session_id == "test_session"
        assert session.graph == graph
        assert session.app == app
        assert isinstance(session.state, dict)
        assert isinstance(session.input_queue, asyncio.Queue)
        assert hasattr(session.output_queue, "put")
        assert hasattr(session.output_queue, "get")
        assert hasattr(session.output_queue, "put_nowait")
        assert hasattr(session.output_queue, "get_nowait")

    @pytest.mark.asyncio
    async def test_session_state_contains_queues(self):
        """测试 Session 初始化时 state 包含队列引用"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        assert "_output_queue" in session.state
        assert "_session_id" in session.state
        assert session.state["_output_queue"] is session.output_queue
        assert session.state["_session_id"] == "test_session"

    @pytest.mark.asyncio
    async def test_session_start_stop(self):
        """测试 Session 启动和停止"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            state["executed"] = True
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")

        session = Session("test_session", graph, app)
        await session.start()
        assert session.is_running is True
        assert session._task is not None

        await session.stop()
        assert session.session_state == Session.STATE_STOPPED

    @pytest.mark.asyncio
    async def test_session_push_event(self):
        """测试推送事件"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        event = Event("test.type", {"data": "value"}, "test_session")
        await session.push_event(event)

        pushed_event = await session.input_queue.get()
        assert pushed_event.type == "test.type"
        assert pushed_event.payload["data"] == "value"

    @pytest.mark.asyncio
    async def test_session_output_queue(self):
        """测试输出队列"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        event = Event("output.type", {"result": "test"}, "test_session")
        await session.output_queue.put(event)

        result = await asyncio.wait_for(session.output_queue.get(), timeout=0.1)
        assert result.type == "output.type"
        assert result.payload["result"] == "test"

    @pytest.mark.asyncio
    async def test_session_get_output_timeout(self):
        """测试获取输出超时"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        result = await session.get_output()
        assert result is None

    @pytest.mark.asyncio
    async def test_session_wait_for_output(self):
        """测试等待输出事件（事件驱动）"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        async def put_event_after_delay():
            await asyncio.sleep(0.1)
            await session._put_output(Event("test.type", {"data": "value"}, "test_session"))

        task = asyncio.create_task(put_event_after_delay())

        result = await session.wait_for_output(timeout=1.0)
        assert result is not None
        assert result.type == "test.type"
        assert result.payload["data"] == "value"

        await task

    @pytest.mark.asyncio
    async def test_session_wait_for_output_immediate(self):
        """测试队列已有事件时立即返回"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        await session._put_output(Event("immediate.type", {}, "test_session"))

        result = await session.wait_for_output(timeout=1.0)
        assert result is not None
        assert result.type == "immediate.type"

    @pytest.mark.asyncio
    async def test_session_wait_for_output_timeout(self):
        """测试等待输出超时"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        result = await session.wait_for_output(timeout=0.1)
        assert result is None

    @pytest.mark.asyncio
    async def test_session_checkpoint_save_restore(self):
        """测试检查点保存和恢复"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        session.state["data"] = "original"
        session.state["count"] = 10
        session._current_node = "test_node"

        session._save_checkpoint("test_node")

        assert session._checkpoint is not None
        assert session._checkpoint["state"]["data"] == "original"
        assert session._checkpoint["state"]["count"] == 10
        assert session._checkpoint["current_node"] == "test_node"

        session.state["data"] = "modified"
        session.state["count"] = 99

        session._restore_from_checkpoint()

        assert session.state["data"] == "original"
        assert session.state["count"] == 10
        assert session._current_node == "test_node"

    @pytest.mark.asyncio
    async def test_session_interrupt_flag(self):
        """测试中断标志"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        assert session._interrupted is False

        session._save_checkpoint("test_node")
        session._interrupted = True

        assert session._interrupted is True

    @pytest.mark.asyncio
    async def test_session_state_properties(self):
        """测试 Session 状态属性"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        assert session.session_state == Session.STATE_CREATED
        assert session.is_alive is True
        assert session.is_running is False

        await session.start()
        assert session.is_running is True
        assert session.is_alive is True

        await session.stop()
        assert session.session_state == Session.STATE_STOPPED
        assert session.is_alive is False

    @pytest.mark.asyncio
    async def test_session_idempotency(self):
        """测试 Session 幂等性保证"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        event1 = Event("test.type", {}, "test_session")
        assert session._is_event_processed(event1) is False

        session._record_event(event1)
        assert session._is_event_processed(event1) is True

        event2 = Event("test.type", {}, "test_session")
        assert session._is_event_processed(event2) is False

    @pytest.mark.asyncio
    async def test_session_has_signal_bus(self):
        """测试 Session 初始化时包含 SignalBus"""
        app = FastMind()
        graph = Graph()
        session = Session("test", graph, app)

        assert hasattr(session, "signal_bus")
        assert session.signal_bus is not None
        assert hasattr(session.signal_bus, "write")
        assert hasattr(session.signal_bus, "read")
        assert hasattr(session.signal_bus, "has")

    @pytest.mark.asyncio
    async def test_session_signal_bus_isolation(self):
        """测试每个 Session 拥有独立的 SignalBus"""
        app = FastMind()
        graph = Graph()

        s1 = Session("s1", graph, app)
        s2 = Session("s2", graph, app)
        s1.signal_bus.write("test", "value1")
        s2.signal_bus.write("test", "value2")

        assert s1.signal_bus.read("test") == "value1"
        assert s2.signal_bus.read("test") == "value2"

    @pytest.mark.asyncio
    async def test_vla_scheduler_runs_vla_functions(self):
        """测试 VLA 调度器自动运行 VLA 函数"""
        app = FastMind()

        @app.vla(name="test_vla", frequency=50.0)
        async def test_vla(state, signal_bus):
            state.setdefault("vla_count", 0)
            state["vla_count"] += 1
            return {"test_action": [1.0]}

        @app.vla_action(name="test_action")
        async def test_executor(action):
            return {"executed": True}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.12)

        assert session.state.get("vla_count", 0) >= 3

        await session.stop()

    @pytest.mark.asyncio
    async def test_vla_scheduler_routes_to_actions(self):
        """测试 VLA 输出按通道名路由到动作执行器"""
        app = FastMind()

        @app.vla(name="test_vla", frequency=50.0)
        async def test_vla(state, signal_bus):
            return {"body": [1.0, 2.0], "face": [0.5]}

        @app.vla_action(name="body")
        async def body_exec(action):
            return {"body_done": True}

        @app.vla_action(name="face")
        async def face_exec(action):
            return {"face_done": True}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.15)

        vla_actions = session.state.get("vla_actions", {})
        assert "body" in vla_actions
        assert "face" in vla_actions
        assert vla_actions["body"] == [1.0, 2.0]
        assert vla_actions["face"] == [0.5]

        vla_results = session.state.get("vla_action_results", {})
        assert "body" in vla_results
        assert vla_results["body"]["body_done"] is True
        assert "face" in vla_results
        assert vla_results["face"]["face_done"] is True

        await session.stop()

    @pytest.mark.asyncio
    async def test_vla_scheduler_respects_pause(self):
        """测试 VLA 调度器响应暂停标志"""
        app = FastMind()

        @app.vla(name="test_vla", frequency=50.0)
        async def test_vla(state, signal_bus):
            state.setdefault("count", 0)
            state["count"] += 1
            return {"act": [0.0]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.12)
        count_before = session.state.get("count", 0)
        assert count_before >= 3

        session.state.setdefault("llm", {})["vla_paused"] = True
        count_at_pause = session.state.get("count", 0)
        await asyncio.sleep(0.15)
        count_after = session.state.get("count", 0)
        assert count_after == count_at_pause

        session.state["llm"]["vla_paused"] = False
        await asyncio.sleep(0.15)
        count_resumed = session.state.get("count", 0)
        assert count_resumed > count_at_pause

        await session.stop()

    @pytest.mark.asyncio
    async def test_signal_source_writes_to_bus(self):
        """测试信号源写入 SignalBus"""
        app = FastMind()

        call_count = 0

        @app.signal(name="test_signal", interval=0.02)
        async def test_src():
            nonlocal call_count
            call_count += 1
            return f"data_{call_count}"

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.1)

        assert session.signal_bus.has("test_signal")
        value = session.signal_bus.read("test_signal")
        assert value.startswith("data_")

        await session.stop()

    @pytest.mark.asyncio
    async def test_vla_llm_dual_loop(self):
        """测试快慢循环共存：VLA 和 Event 驱动的 LLM 同时工作"""
        app = FastMind()

        @app.vla(name="fast_loop", frequency=30.0)
        async def fast_vla(state, signal_bus):
            state.setdefault("fast_count", 0)
            state["fast_count"] += 1
            return {"dummy": [0.0]}

        @app.vla_action(name="dummy")
        async def dummy_exec(action):
            return {}

        async def slow_agent(state, event):
            state.setdefault("slow_count", 0)
            state["slow_count"] += 1
            return state

        graph = Graph()
        graph.add_node("slow", slow_agent)
        graph.set_entry_point("slow")
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()

        # Push events to trigger LLM slow path
        for _ in range(3):
            await session.push_event(Event("test", {}, "test"))
            await asyncio.sleep(0.01)

        await asyncio.sleep(0.15)

        fast = session.state.get("fast_count", 0)
        slow = session.state.get("slow_count", 0)
        assert fast >= 3, f"fast={fast}"
        assert slow == 3, f"slow={slow}"

        await session.stop()

    @pytest.mark.asyncio
    async def test_vla_and_signal_cleaned_on_stop(self):
        """测试停止时清理 VLA 和信号任务"""
        app = FastMind()

        @app.vla(name="test", frequency=50.0)
        async def test_vla(state, sb):
            return {"act": [0.0]}

        @app.vla_action(name="act")
        async def act(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        assert session._vla_task is not None
        assert not session._vla_task.done()

        await session.stop()
        assert session._vla_task is None

    # ════════════════════════════════════════════════════════════════
    # VLA 可靠性测试
    # ════════════════════════════════════════════════════════════════

    @pytest.mark.asyncio
    async def test_vla_long_running_stability(self):
        """测试 VLA 长时间运行的稳定性（连续运行多次无异常）"""
        app = FastMind()
        count = 0

        @app.vla(name="stable", frequency=100.0)
        async def stable_vla(state, sb):
            nonlocal count
            count += 1
            return {"act": [float(count)]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {"received": action[0]}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.3)
        await session.stop()

        assert count > 15, f"VLA should run many times, got {count}"
        assert len(session.state.get("vla_actions", {})) > 0

    @pytest.mark.asyncio
    async def test_vla_error_recovery(self):
        """测试 VLA 函数抛出异常后能自动恢复"""
        app = FastMind()
        attempt = 0

        @app.vla(name="erratic", frequency=50.0)
        async def erratic_vla(state, sb):
            nonlocal attempt
            attempt += 1
            # Fail on first 3 attempts, recover on 4th
            if attempt <= 3:
                raise RuntimeError(f"simulated failure #{attempt}")
            return {"act": [float(attempt)]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.3)
        await session.stop()

        assert attempt >= 5, f"VLA should keep running after errors, got {attempt}"
        last_action = session.state.get("vla_actions", {}).get("act")
        assert last_action is not None, "VLA should recover and produce actions"

    @pytest.mark.asyncio
    async def test_vla_action_error_isolation(self):
        """测试一个 action executor 报错不影响其他 action"""
        app = FastMind()

        @app.vla(name="multi_action", frequency=50.0)
        async def multi_vla(state, sb):
            return {"good": [1.0], "bad": [2.0], "also_good": [3.0]}

        @app.vla_action(name="good")
        async def good_exec(action):
            return {"good_done": True}

        @app.vla_action(name="bad")
        async def bad_exec(action):
            raise RuntimeError("bad executor failed")

        @app.vla_action(name="also_good")
        async def also_good_exec(action):
            return {"also_done": True}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.2)
        await session.stop()

        actions = session.state.get("vla_actions", {})
        assert "good" in actions, "good action should be present"
        assert "bad" not in actions, "failed executor should not write to vla_actions"
        assert "also_good" in actions, "also_good action should not be blocked by bad executor"

        results = session.state.get("vla_action_results", {})
        assert results.get("good", {}).get("good_done") is True
        assert "bad" not in results, "failed executor should not write to vla_action_results"
        assert results.get("also_good", {}).get("also_done") is True

    @pytest.mark.asyncio
    async def test_vla_function_takes_longer_than_interval(self):
        """测试 VLA 函数执行时间超过 interval 时不会积压"""
        app = FastMind()
        overlap_count = 0

        @app.vla(name="slow_vla", frequency=100.0)  # 10ms interval
        async def slow_vla(state, sb):
            nonlocal overlap_count
            overlap_count += 1
            await asyncio.sleep(0.05)  # 50ms — slower than interval
            return {"act": [float(overlap_count)]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.2)
        await session.stop()

        # Should not crash or accumulate pending tasks
        assert overlap_count >= 1, "VLA should run at least once"

    @pytest.mark.asyncio
    async def test_signal_source_error_recovery(self):
        """测试信号源抛出异常后能自动恢复"""
        app = FastMind()
        call_idx = 0

        @app.signal(name="unstable", interval=0.02)
        async def unstable_signal():
            nonlocal call_idx
            call_idx += 1
            if call_idx <= 2:
                raise RuntimeError("signal failed")
            return f"data_{call_idx}"

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.15)
        await session.stop()

        value = session.signal_bus.read("unstable")
        assert value is not None, "Signal should recover and produce data"
        assert str(value).startswith("data_"), f"Got unexpected value: {value}"

    @pytest.mark.asyncio
    async def test_vla_llm_concurrent_state_access(self):
        """测试 VLA 和 LLM 同时写入 state 不互相破坏"""
        app = FastMind()

        @app.vla(name="vla_writer", frequency=50.0)
        async def vla_writer(state, sb):
            state.setdefault("vla_write_count", 0)
            state["vla_write_count"] += 1
            # VLA writes to its own namespace
            state.setdefault("vla", {})["frame"] = state["vla_write_count"]
            return {"act": [1.0]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {}

        async def llm_writer(state, event):
            state.setdefault("llm", {})

            # LLM writes goal
            if event.type == "user.message":
                state["llm"]["goal"] = event.payload.get("text", "")
                state["llm"]["version"] = state.get("llm", {}).get("version", 0) + 1

            # LLM reads VLA state
            vla_frame = state.get("vla", {}).get("frame", 0)
            state["llm"]["last_seen_vla_frame"] = vla_frame

            return state

        graph = Graph()
        graph.add_node("llm", llm_writer)
        graph.set_entry_point("llm")
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()

        # Push multiple LLM events while VLA runs
        for goal in ["go_castle", "go_market", "stop"]:
            await session.push_event(Event("user.message", {"text": goal}, "test"))
            await asyncio.sleep(0.05)

        await asyncio.sleep(0.2)
        await session.stop()

        # Both should have made progress
        assert session.state.get("vla_write_count", 0) > 0, "VLA should have written"
        assert session.state.get("llm", {}).get("version", 0) > 0, "LLM should have written"
        assert session.state.get("llm", {}).get("goal") == "stop", "LLM goal should be latest"

    @pytest.mark.asyncio
    async def test_vla_state_does_not_grow_unbounded(self):
        """测试 vla_actions 和 vla_action_results 不会无限制增长"""
        app = FastMind()

        @app.vla(name="growing", frequency=100.0)
        async def growing_vla(state, sb):
            return {"ch": [1.0]}

        @app.vla_action(name="ch")
        async def ch_exec(action):
            return {"ts": time.time()}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.25)
        await session.stop()

        # vla_actions and vla_action_results should retain only the latest value per channel
        vla_actions = session.state.get("vla_actions", {})
        vla_results = session.state.get("vla_action_results", {})
        assert len(vla_actions) <= 1, f"vla_actions should have 1 entry, has {len(vla_actions)}"
        assert len(vla_results) <= 1, f"vla_action_results should have 1 entry, has {len(vla_results)}"

    @pytest.mark.asyncio
    async def test_multiple_sessions_independent_vla(self):
        """测试多个 Session 各自独立运行 VLA，互不干扰"""
        app = FastMind()

        @app.vla(name="counter", frequency=50.0)
        async def counter_vla(state, sb):
            state.setdefault("vla_count", 0)
            state["vla_count"] += 1
            return {"act": [float(state["vla_count"])]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        s1 = Session("s1", graph, app)
        s2 = Session("s2", graph, app)
        await s1.start()
        await s2.start()
        await asyncio.sleep(0.2)
        await s1.stop()
        await s2.stop()

        c1 = s1.state.get("vla_count", 0)
        c2 = s2.state.get("vla_count", 0)
        assert c1 > 3, f"s1 VLA count too low: {c1}"
        assert c2 > 3, f"s2 VLA count too low: {c2}"
        # Each should be within reasonable range of each other
        assert abs(c1 - c2) <= max(c1, c2) * 0.5, f"VLA counts too different: {c1} vs {c2}"

    @pytest.mark.asyncio
    async def test_vla_resume_after_stop_clean_state(self):
        """测试 VLA 停止后重启，状态正确"""
        app = FastMind()

        @app.vla(name="resumable", frequency=50.0)
        async def resumable_vla(state, sb):
            state.setdefault("run_count", 0)
            state["run_count"] += 1
            return {"act": [float(state["run_count"])]}

        @app.vla_action(name="act")
        async def act_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)

        # Run 1
        await session.start()
        await asyncio.sleep(0.12)
        first_count = session.state.get("run_count", 0)
        assert first_count > 0, "VLA should have run in first session"
        await session.stop()

        # Reset state
        session.state["run_count"] = 0
        session._state = Session.STATE_CREATED

        # Run 2
        await session.start()
        await asyncio.sleep(0.12)
        second_count = session.state.get("run_count", 0)
        assert second_count > 0, "VLA should run again after restart"
        await session.stop()

    @pytest.mark.asyncio
    async def test_vla_action_result_overwritten(self):
        """测试 VLA 的 action_result 被新的覆盖（last-value 语义）"""
        app = FastMind()
        tick = 0

        @app.vla(name="ticker", frequency=50.0)
        async def ticker_vla(state, sb):
            nonlocal tick
            tick += 1
            return {"ch": [float(tick)]}

        @app.vla_action(name="ch")
        async def ch_exec(action):
            return {"tick": action[0]}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.15)
        await session.stop()

        last_action = session.state.get("vla_actions", {}).get("ch")
        last_result = session.state.get("vla_action_results", {}).get("ch", {})
        # Should be the latest tick, not the first one
        assert last_action == [float(tick)], f"Expected latest action, got {last_action}"
        assert last_result.get("tick") == float(tick), f"Expected latest result, got {last_result}"

    @pytest.mark.asyncio
    async def test_signal_concurrent_writes(self):
        """测试多个信号源并发写入 SignalBus"""
        app = FastMind()

        @app.signal(name="sig_a", interval=0.01)
        async def sig_a():
            return "a"

        @app.signal(name="sig_b", interval=0.015)
        async def sig_b():
            return "b"

        @app.signal(name="sig_c", interval=0.02)
        async def sig_c():
            return "c"

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.1)
        await session.stop()

        assert session.signal_bus.has("sig_a"), "sig_a should exist"
        assert session.signal_bus.has("sig_b"), "sig_b should exist"
        assert session.signal_bus.has("sig_c"), "sig_c should exist"
        assert session.signal_bus.read("sig_a") == "a"
        assert session.signal_bus.read("sig_b") == "b"
        assert session.signal_bus.read("sig_c") == "c"

    @pytest.mark.asyncio
    async def test_vla_with_llm_override_cycle(self):
        """测试 LLM override → clear → VLA 恢复正常运行的完整周期"""
        app = FastMind()
        inference_count = 0

        @app.vla(name="override_test", frequency=30.0)
        async def override_vla(state, sb):
            nonlocal inference_count
            inference_count += 1
            override = state.get("llm", {}).get("override_action", {}).get("body")
            if override is not None:
                return {"body": override}
            return {"body": [float(inference_count)]}

        @app.vla_action(name="body")
        async def body_exec(action):
            return {"body_val": action[0]}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.1)

        # Normal VLA running
        assert session.state.get("vla_actions", {}).get("body") is not None

        # LLM sets override
        session.state.setdefault("llm", {})["override_action"] = {"body": [999.0]}
        await asyncio.sleep(0.1)
        overridden = session.state.get("vla_actions", {}).get("body")
        assert overridden == [999.0], f"Expected override [999.0], got {overridden}"

        # LLM clears override
        del session.state["llm"]["override_action"]
        await asyncio.sleep(0.1)
        resumed = session.state.get("vla_actions", {}).get("body")
        assert resumed != [999.0], "VLA should resume normal inference after override cleared"

        await session.stop()

    @pytest.mark.asyncio
    async def test_vla_no_actions_registered(self):
        """测试 VLA 输出没有对应 action 时不会崩溃"""
        app = FastMind()

        @app.vla(name="orphan", frequency=30.0)
        async def orphan_vla(state, sb):
            state["orphan_ran"] = True
            return {"nonexistent_channel": [1.0]}  # no action registered for this channel

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.15)
        await session.stop()

        assert session.state.get("orphan_ran") is True, "VLA should run even without matching actions"

    @pytest.mark.asyncio
    async def test_vla_returns_non_dict_graceful(self):
        """测试 VLA 返回非 dict 时不会崩溃"""
        app = FastMind()

        @app.vla(name="bad_return", frequency=30.0)
        async def bad_return_vla(state, sb):
            state["bad_return_ran"] = True
            return "not a dict"  # invalid return type

        @app.vla_action(name="some_action")
        async def some_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.15)
        await session.stop()

        assert session.state.get("bad_return_ran") is True, "VLA should run without crashing on bad return"

    @pytest.mark.asyncio
    async def test_vla_pause_resume_cycle_multiple(self):
        """测试 VLA 多次暂停/恢复循环"""
        app = FastMind()
        tick = 0

        @app.vla(name="ping", frequency=50.0)
        async def ping_vla(state, sb):
            nonlocal tick
            tick += 1
            return {"pulse": [float(tick)]}

        @app.vla_action(name="pulse")
        async def pulse_exec(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.08)
        assert tick >= 2, f"Should have run before pause: {tick}"
        before = tick

        for _ in range(3):
            session.state.setdefault("llm", {})["vla_paused"] = True
            await asyncio.sleep(0.08)
            after_pause = tick
            assert after_pause == before, f"VLA should not advance while paused: {after_pause} vs {before}"

            session.state["llm"]["vla_paused"] = False
            await asyncio.sleep(0.08)
            after_resume = tick
            assert after_resume > before, f"VLA should advance after resume: {after_resume} vs {before}"
            before = after_resume

        await session.stop()
        """测试多个 VLA 不同频率"""
        app = FastMind()

        @app.vla(name="fast", frequency=100.0)
        async def fast_vla(state, sb):
            state.setdefault("fast", 0)
            state["fast"] += 1
            return {"act": [0.0]}

        @app.vla(name="slow", frequency=10.0)
        async def slow_vla(state, sb):
            state.setdefault("slow", 0)
            state["slow"] += 1
            return {"act": [0.0]}

        @app.vla_action(name="act")
        async def act(action):
            return {}

        graph = Graph()
        graph.set_entry_point(Graph.END_NODE)
        app.register_graph("main", graph)

        session = Session("test", graph, app)
        await session.start()
        await asyncio.sleep(0.25)

        fast = session.state.get("fast", 0)
        slow = session.state.get("slow", 0)
        assert fast > slow * 3, f"fast={fast} should be much larger than slow={slow}"

        await session.stop()

    @pytest.mark.asyncio
    async def test_resume_from_stopped(self):
        """测试从 stopped 状态恢复会话"""
        app = FastMind()
        graph = Graph()
        session = Session("test_session", graph, app)

        await session.start()
        assert session.is_running is True

        await session.stop()
        assert session.session_state == Session.STATE_STOPPED

        session._state = Session.STATE_CREATED
        assert session.is_alive is True


class TestEngine:
    """Engine 测试"""

    @pytest.mark.asyncio
    async def test_engine_initialization(self):
        """测试 Engine 初始化"""
        app = FastMind()
        engine = Engine(app)

        assert engine.app == app
        assert engine._sessions == {}
        assert engine._running is False
        assert engine._tasks == []

    @pytest.mark.asyncio
    async def test_engine_start_stop(self):
        """测试 Engine 启动和停止"""
        app = FastMind()
        engine = Engine(app)

        await engine.start()
        assert engine._running is True

        await engine.stop()
        assert engine._running is False

    @pytest.mark.asyncio
    async def test_engine_get_or_create_session(self):
        """测试获取或创建会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)

        session1 = engine.get_or_create_session("session_1")
        assert session1 is not None
        assert session1.session_id == "session_1"

        session2 = engine.get_or_create_session("session_1")
        assert session2 is session1

        session3 = engine.get_or_create_session("session_2", "main")
        assert session3.session_id == "session_2"
        assert session3 is not session1

    @pytest.mark.asyncio
    async def test_engine_get_or_create_session_invalid_graph(self):
        """测试获取不存在图时会话"""
        app = FastMind()
        engine = Engine(app)

        with pytest.raises(ValueError, match="not found"):
            engine.get_or_create_session("session_1", "nonexistent")

    @pytest.mark.asyncio
    async def test_engine_push_event_creates_session(self):
        """测试推送事件自动创建会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)

        event = Event("test", {}, "new_session")
        session = await engine.push_event("new_session", event)

        assert session is not None
        assert "new_session" in engine._sessions

    @pytest.mark.asyncio
    async def test_engine_get_session(self):
        """测试获取会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)

        session = engine.get_or_create_session("test_session")
        result = engine.get_session("test_session")
        assert result is session

        result_none = engine.get_session("nonexistent")
        assert result_none is None

    @pytest.mark.asyncio
    async def test_engine_get_session_state(self):
        """测试获取会话状态"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            state["value"] = 42
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)
        session = engine.get_or_create_session("test_session")
        await engine.push_event("test_session", Event("test", {}, "test_session"))

        await asyncio.sleep(0.2)

        state = engine.get_session_state("test_session")
        assert state is not None
        assert state["value"] == 42

        state_none = engine.get_session_state("nonexistent")
        assert state_none is None

    @pytest.mark.asyncio
    async def test_engine_list_sessions(self):
        """测试列出所有会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)
        engine.get_or_create_session("session_1")
        engine.get_or_create_session("session_2")

        sessions = engine.list_sessions()
        assert "session_1" in sessions
        assert "session_2" in sessions
        assert len(sessions) == 2

    @pytest.mark.asyncio
    async def test_engine_delete_session(self):
        """测试删除会话"""
        app = FastMind()

        async def simple_node(state: dict, event: Event) -> dict:
            return state

        graph = Graph()
        graph.add_node("test", simple_node)
        graph.set_entry_point("test")
        app.register_graph("main", graph)

        engine = Engine(app)
        engine.get_or_create_session("session_1")

        assert "session_1" in engine._sessions

        await engine.delete_session("session_1")

        assert "session_1" not in engine._sessions

    @pytest.mark.asyncio
    async def test_engine_multiple_sessions_isolated(self):
        """测试多会话状态隔离"""
        app = FastMind()

        async def counter_agent(state: dict, event: Event) -> dict:
            state.setdefault("count", 0)
            state["count"] += 1
            state["session_id"] = state["_session_id"]
            return state

        graph = Graph()
        graph.add_node("counter", counter_agent)
        graph.set_entry_point("counter")
        app.register_graph("main", graph)

        engine = Engine(app)
        session1 = engine.get_or_create_session("session_1")
        session2 = engine.get_or_create_session("session_2")

        await engine.start()
        await engine.push_event("session_1", Event("test", {}, "session_1"))
        await engine.push_event("session_2", Event("test", {}, "session_2"))

        await asyncio.sleep(0.3)

        state1 = engine.get_session_state("session_1")
        state2 = engine.get_session_state("session_2")

        assert state1["count"] == 1
        assert state2["count"] == 1
        assert state1["session_id"] == "session_1"
        assert state2["session_id"] == "session_2"

        await engine.push_event("session_1", Event("test", {}, "session_1"))
        await asyncio.sleep(0.2)

        state1_new = engine.get_session_state("session_1")
        assert state1_new["count"] == 2
        assert state2["count"] == 1

        await engine.stop()
