"""全面用户体验测试 —— 覆盖所有 example 场景。

不依赖交互式输入，端到端验证：
  - simple_chat, streaming_chat, human_in_loop, perception_loop
  - humanoid_robot, companion_bot, drone, comprehensive_assistant
  - sleep_assessment, npc_vla
  - Session.stop() / Engine.stop() 超时健壮性
  - stream_events 取消行为

用法:
    python -m fastmind.examples.test_all_examples
    LLM_API_KEY="sk-xxx" python fastmind/examples/test_all_examples.py
"""
import asyncio
import os
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastmind import FastMind, Graph, Event, ToolNode, ActionSpace
from fastmind.contrib import FastMindAPI
from fastmind.core.engine import Session, Engine

RESULTS = {"pass": 0, "fail": 0, "skip": 0, "errors": []}


def test(name, skip_if_no_llm=False):
    def decorator(func):
        async def wrapper():
            nonlocal name
            if skip_if_no_llm and not os.getenv("LLM_API_KEY"):
                RESULTS["skip"] += 1
                print(f"  ⏭️  {name} (no LLM_API_KEY)")
                return
            try:
                await func()
                RESULTS["pass"] += 1
                print(f"  ✅ {name}")
            except Exception as e:
                RESULTS["fail"] += 1
                import traceback
                msg = f"  ❌ {name}: {e}\n{traceback.format_exc()}"
                RESULTS["errors"].append(msg)
                print(msg)
        return wrapper
    return decorator


# ═══════════════════════════════════════════════════════════════
# 1. simple_chat 场景
# ═══════════════════════════════════════════════════════════════

@test("simple_chat - agent node executes and produces state")
async def test_simple_chat():
    app = FastMind()

    @app.agent(name="chat")
    async def chat(state, event):
        state.setdefault("msgs", [])
        state["msgs"].append(event.payload.get("text", ""))
        return state

    graph = Graph()
    graph.add_node("chat", chat)
    graph.set_entry_point("chat")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {"text": "hello"}, "u1"))
    await asyncio.sleep(0.1)
    s = api.get_state("u1")
    await api.stop()

    assert s["msgs"] == ["hello"]


@test("simple_chat - stream.end event on quit")
async def test_simple_chat_quit():
    app = FastMind()

    @app.agent(name="chat")
    async def chat(state, event):
        state.setdefault("msgs", [])
        text = event.payload.get("text", "")
        state["msgs"].append(text)
        if text.lower() == "quit":
            state["quit"] = True
            state["_output_queue"].put_nowait(
                Event("stream.end", {}, event.session_id)
            )
        return state

    graph = Graph()
    graph.add_node("chat", chat)
    graph.set_entry_point("chat")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {"text": "quit"}, "u1"))

    got_end = False
    async for ev in api.stream_events("u1"):
        if ev.type == "stream.end":
            got_end = True
            break

    await api.stop()
    assert got_end, "Should receive stream.end on quit"


@test("simple_chat - LLM conversation", skip_if_no_llm=True)
async def test_simple_chat_llm():
    api_key = os.getenv("LLM_API_KEY")
    app = FastMind()

    @app.agent(name="chat")
    async def chat(state, event):
        state.setdefault("messages", [])
        state["messages"].append({"role": "user", "content": event.payload.get("text", "")})

        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1")
        )
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            messages=state["messages"],
        )
        state["messages"].append({"role": "assistant", "content": resp.choices[0].message.content or ""})
        state["_output_queue"].put_nowait(Event("stream.end", {}, event.session_id))
        return state

    graph = Graph()
    graph.add_node("chat", chat)
    graph.set_entry_point("chat")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "Say hello in one word"}, "u1"))

    async for ev in api.stream_events("u1"):
        if ev.type == "stream.end":
            break

    s = api.get_state("u1")
    await api.stop()
    msgs = s.get("messages", [])
    assert any(m["role"] == "assistant" for m in msgs), "Should have assistant response"


# ═══════════════════════════════════════════════════════════════
# 2. streaming_chat 场景
# ═══════════════════════════════════════════════════════════════

@test("streaming_chat - synthetic chunk output")
async def test_streaming_chat_synthetic():
    app = FastMind()

    async def streaming_agent(state, event):
        events = []
        for c in "Hi!":
            events.append(Event("stream.chunk", {"delta": c}, event.session_id))
        events.append(Event("stream.end", {}, event.session_id))
        state["done"] = True
        return state, events

    graph = Graph()
    graph.add_node("stream", streaming_agent)
    graph.set_entry_point("stream")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("test", {}, "u1"))

    text = ""
    async for ev in api.stream_events("u1"):
        if ev.type == "stream.chunk":
            text += ev.payload.get("delta", "")
        elif ev.type == "stream.end":
            break

    await api.stop()
    assert text == "Hi!", f"Expected 'Hi!', got '{text}'"


@test("streaming_chat - LLM streaming", skip_if_no_llm=True)
async def test_streaming_chat_llm():
    api_key = os.getenv("LLM_API_KEY")
    app = FastMind()

    @app.agent(name="stream")
    async def stream_agent(state, event):
        state.setdefault("messages", [])
        state["messages"].append({"role": "user", "content": event.payload.get("text", "")})
        output_queue = state["_output_queue"]
        session_id = state["_session_id"]

        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1")
        )
        try:
            stream = await client.chat.completions.create(
                model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
                messages=state["messages"],
                stream=True,
            )
            full = ""
            async for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                full += delta
                output_queue.put_nowait(Event("stream.chunk", {"delta": delta}, session_id))
            state["messages"].append({"role": "assistant", "content": full})
        except Exception as e:
            output_queue.put_nowait(Event("stream.chunk", {"delta": str(e)}, session_id))
        output_queue.put_nowait(Event("stream.end", {}, session_id))
        return state

    graph = Graph()
    graph.add_node("stream", stream_agent)
    graph.set_entry_point("stream")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "Say just one word"}, "u1"))

    full_text = ""
    async for ev in api.stream_events("u1"):
        if ev.type == "stream.chunk":
            full_text += ev.payload.get("delta", "")
        elif ev.type == "stream.end":
            break

    await api.stop()
    assert len(full_text) > 0, "Should receive streamed text"


# ═══════════════════════════════════════════════════════════════
# 3. human_in_loop 场景
# ═══════════════════════════════════════════════════════════════

@test("human_in_loop - interrupt on high amount")
async def test_human_in_loop_interrupt():
    app = FastMind()

    @app.agent(name="order")
    async def order(state, event):
        amount = event.payload.get("amount", 0)
        state["amount"] = amount
        state["need_approval"] = amount > 100
        return state

    async def ask(state, event):
        return state, [Event("interrupt", {
            "prompt": "Approve?",
            "resume_node": "confirm",
            "cancel_node": "cancel",
        }, event.session_id)]

    async def confirm(state, event):
        state["confirmed"] = True
        return state

    async def cancel(state, event):
        state["confirmed"] = False
        return state

    graph = Graph()
    graph.add_node("order", order)
    graph.add_node("ask", ask)
    graph.add_node("confirm", confirm)
    graph.add_node("cancel", cancel)
    graph.add_conditional_edges("order",
        lambda s, e: "ask" if s.get("need_approval") else None,
        {"ask": "ask", None: "__end__"})
    graph.set_entry_point("order")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    await api.push_event("u1", Event("msg", {"amount": 150}, "u1"))
    await asyncio.sleep(0.1)
    s = api.get_session("u1")
    assert s.session_state == Session.STATE_INTERRUPTED

    await api.resume_session("u1", "confirm")
    await asyncio.sleep(0.1)
    assert api.get_state("u1")["confirmed"] is True

    await api.stop()


@test("human_in_loop - cancel route")
async def test_human_in_loop_cancel():
    app = FastMind()

    async def ask(state, event):
        return state, [Event("interrupt", {
            "prompt": "?", "resume_node": "ok", "cancel_node": "no",
        }, event.session_id)]

    async def ok(state, event):
        state["result"] = "ok"
        return state

    async def no(state, event):
        state["result"] = "cancelled"
        return state

    graph = Graph()
    graph.add_node("ask", ask)
    graph.add_node("ok", ok)
    graph.add_node("no", no)
    graph.set_entry_point("ask")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {}, "u1"))
    await asyncio.sleep(0.1)
    await api.resume_session("u1", "cancel")
    await asyncio.sleep(0.1)
    assert api.get_state("u1")["result"] == "cancelled"
    await api.stop()


# ═══════════════════════════════════════════════════════════════
# 4. tool calling 场景 (humanoid / companion / drone pattern)
# ═══════════════════════════════════════════════════════════════

@test("tool_node - single tool call")
async def test_tool_node_single():
    app = FastMind()

    @app.tool(name="echo")
    def echo(text: str) -> str:
        return text

    tn = ToolNode(app.get_tools())
    state = {"tool_calls": [{"id": "1", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}]}
    ns, _ = await tn.execute(state, Event("test", {}, "u1"))
    assert ns["tool_results"][0]["result"] == "hi"


@test("tool_node - multi tool calls")
async def test_tool_node_multi():
    app = FastMind()

    @app.tool(name="add")
    async def add(a: int, b: int) -> str:
        return str(a + b)

    @app.tool(name="sub")
    async def sub(a: int, b: int) -> str:
        return str(a - b)

    tn = ToolNode(app.get_tools())
    state = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "add", "arguments": '{"a": 3, "b": 4}'}},
            {"id": "c2", "function": {"name": "sub", "arguments": '{"a": 10, "b": 3}'}},
        ]
    }
    ns, _ = await tn.execute(state, Event("test", {}, "u1"))
    results = ns["tool_results"]
    assert len(results) == 2
    assert results[0]["result"] == "7"
    assert results[1]["result"] == "7"


@test("humanoid_robot - battery check tool flow")
async def test_humanoid_robot_flow():
    app = FastMind()

    @app.tool(name="get_battery")
    async def get_battery() -> str:
        return "85%"

    @app.tool(name="get_location")
    async def get_location() -> str:
        return "(0,0,0)"

    @app.agent(name="robot")
    async def robot(state, event):
        state.setdefault("msgs", [])
        if state.get("tool_results"):
            for r in state["tool_results"]:
                state["msgs"].append(f"{r['tool_name']}: {r['result']}")
            del state["tool_results"]
            return state
        text = event.payload.get("text", "")
        if "电池" in text:
            state["tool_calls"] = [{"id": "1", "function": {"name": "get_battery", "arguments": "{}"}}]
        elif "位置" in text:
            state["tool_calls"] = [{"id": "2", "function": {"name": "get_location", "arguments": "{}"}}]
        return state

    tn = ToolNode(app.get_tools())
    graph = Graph()
    graph.add_node("robot", robot)
    graph.add_node("tools", tn)
    graph.add_conditional_edges("robot",
        lambda s, e: "tools" if s.get("tool_calls") else None,
        {"tools": "tools", None: "__end__"})
    graph.add_edge("tools", "robot")
    graph.set_entry_point("robot")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {"text": "电池"}, "u1"))
    await asyncio.sleep(0.2)
    s = api.get_state("u1")
    await api.stop()

    msgs = s.get("msgs", [])
    assert any("get_battery" in m for m in msgs), f"Should have battery result: {msgs}"


@test("companion_bot - emotion detection")
async def test_companion_emotion():
    app = FastMind()

    @app.agent(name="companion")
    async def companion(state, event):
        state.setdefault("mood", "neutral")
        text = event.payload.get("text", "")
        if "难过" in text:
            state["mood"] = "sad"
        elif "开心" in text:
            state["mood"] = "happy"
        return state

    graph = Graph()
    graph.add_node("companion", companion)
    graph.set_entry_point("companion")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {"text": "好难过"}, "u1"))
    await asyncio.sleep(0.1)
    assert api.get_state("u1")["mood"] == "sad"

    await api.push_event("u1", Event("msg", {"text": "好开心"}, "u1"))
    await asyncio.sleep(0.1)
    assert api.get_state("u1")["mood"] == "happy"

    await api.stop()


@test("drone - sensor perception loop")
async def test_drone_sensor():
    app = FastMind()

    @app.perception(interval=0.05, name="sensor")
    async def sensor(app):
        while True:
            yield Event("sensor.flight_data",
                {"altitude": 100.0}, "drone_001")
            await asyncio.sleep(0.05)

    @app.agent(name="control")
    async def control(state, event):
        if event.type == "sensor.flight_data":
            state.setdefault("sensor_count", 0)
            state["sensor_count"] += 1
            state["altitude"] = event.payload.get("altitude")
        return state

    graph = Graph()
    graph.add_node("control", control)
    graph.set_entry_point("control")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await asyncio.sleep(0.25)

    s = api.get_state("drone_001")
    await api.stop()

    assert s is not None, "drone_001 session should exist"
    assert s.get("sensor_count", 0) >= 3, f"Sensor should fire multiple times: {s.get('sensor_count')}"


# ═══════════════════════════════════════════════════════════════
# 5. comprehensive_assistant 场景
# ═══════════════════════════════════════════════════════════════

@test("comprehensive_assistant - planner + tool chain")
async def test_comprehensive_flow():
    app = FastMind()

    @app.tool(name="get_time")
    async def get_time() -> str:
        return "12:00"

    @app.tool(name="search")
    async def search(query: str) -> str:
        return f"result for {query}"

    @app.agent(name="planner")
    async def planner(state, event):
        text = event.payload.get("text", "")
        if "时间" in text:
            state["tool_calls"] = [{"id": "1", "function": {"name": "get_time", "arguments": "{}"}}]
        return state

    tn = ToolNode(app.get_tools())
    graph = Graph()
    graph.add_node("planner", planner)
    graph.add_node("tools", tn)
    graph.add_conditional_edges("planner",
        lambda s, e: "tools" if s.get("tool_calls") else None,
        {"tools": "tools", None: "__end__"})
    graph.add_edge("tools", "planner")
    graph.set_entry_point("planner")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {"text": "现在时间"}, "u1"))
    await asyncio.sleep(0.2)
    s = api.get_state("u1")
    await api.stop()

    tr = s.get("tool_results", [])
    assert len(tr) >= 1, f"Should have tool result: {s}"


# ═══════════════════════════════════════════════════════════════
# 6. sleep_assessment 场景
# ═══════════════════════════════════════════════════════════════

@test("sleep_assessment - multi-stage HITL flow")
async def test_sleep_assessment():
    app = FastMind()

    @app.tool(name="save_report")
    async def save_report(report: str) -> str:
        return "saved"

    @app.tool(name="get_sleep_tips")
    async def get_sleep_tips(issue: str) -> str:
        return f"tips for {issue}"

    @app.agent(name="assess")
    async def assess(state, event):
        state.setdefault("stage", "ask")
        text = event.payload.get("text", "")
        if state.get("stage") == "ask":
            state["hours"] = text
            state["stage"] = "confirm"
            state["need_approval"] = True
        return state

    async def ask_confirm(state, event):
        return state, [Event("interrupt", {
            "prompt": "Generate?", "resume_node": "generate", "cancel_node": "skip",
        }, event.session_id)]

    async def generate(state, event):
        state["tool_calls"] = [
            {"id": "1", "function": {"name": "save_report", "arguments": '{"report": "test"}'}},
            {"id": "2", "function": {"name": "get_sleep_tips", "arguments": '{"issue": "bad"}'}},
        ]
        state["stage"] = "done"
        return state

    async def skip(state, event):
        state["stage"] = "done"
        state["skipped"] = True
        return state

    tn = ToolNode(app.get_tools())
    graph = Graph()
    graph.add_node("assess", assess)
    graph.add_node("ask", ask_confirm)
    graph.add_node("generate", generate)
    graph.add_node("skip", skip)
    graph.add_node("tools", tn)
    graph.add_conditional_edges("assess",
        lambda s, e: "ask" if s.get("need_approval") else None,
        {"ask": "ask", None: "__end__"})
    graph.add_edge("generate", "tools")
    graph.add_edge("tools", "__end__")
    graph.set_entry_point("assess")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    await api.push_event("u1", Event("msg", {"text": "6"}, "u1"))
    await asyncio.sleep(0.1)
    s = api.get_session("u1")
    assert s.session_state == Session.STATE_INTERRUPTED

    await api.resume_session("u1", "confirm")
    await asyncio.sleep(0.2)

    state = api.get_state("u1")
    await api.stop()

    assert state["stage"] == "done", f"Expected done, got {state.get('stage')}"
    tr = state.get("tool_results", [])
    assert len(tr) >= 2, f"Should have 2 tool results: {tr}"


# ═══════════════════════════════════════════════════════════════
# 7. npc_vla 场景
# ═══════════════════════════════════════════════════════════════

@test("npc_vla - fast loop VLA scheduler")
async def test_npc_vla():
    app = FastMind()

    @app.vla(name="nav", frequency=50.0)
    async def nav(state, sb):
        state.setdefault("vla_count", 0)
        state["vla_count"] += 1
        return {"body": [1.0]}

    @app.vla_action(name="body")
    async def body(action):
        pass

    @app.signal(name="vision", interval=0.03)
    async def vision():
        return {"frame": 1}

    graph = Graph()
    graph.set_entry_point(Graph.END_NODE)
    app.register_graph("main", graph)

    session = Session("npc", graph, app)
    await session.start()
    await asyncio.sleep(0.15)

    count = session.state.get("vla_count", 0)
    assert count >= 3, f"VLA should run multiple times: count={count}"
    assert session.signal_bus.has("vision")
    assert session.signal_bus.read("vision") is not None

    await session.stop()


@test("npc_vla - pause/resume cycle")
async def test_npc_vla_pause_resume():
    app = FastMind()
    tick = 0

    @app.vla(name="tick", frequency=50.0)
    async def tick_vla(state, sb):
        nonlocal tick
        tick += 1
        return {"a": [0.0]}

    @app.vla_action(name="a")
    async def a(action):
        pass

    graph = Graph()
    graph.set_entry_point(Graph.END_NODE)
    app.register_graph("main", graph)

    session = Session("test", graph, app)
    await session.start()
    await asyncio.sleep(0.1)
    assert tick >= 2
    before = tick

    session.state.setdefault("llm", {})["vla_paused"] = True
    await asyncio.sleep(0.1)
    assert tick == before, f"VLA should be paused: {tick} vs {before}"

    session.state["llm"]["vla_paused"] = False
    await asyncio.sleep(0.1)
    assert tick > before, f"VLA should resume: {tick} vs {before}"

    await session.stop()


@test("npc_vla - LLM overrides VLA action")
async def test_npc_vla_override():
    app = FastMind()

    @app.vla(name="ctrl", frequency=50.0)
    async def ctrl(state, sb):
        override = state.get("llm", {}).get("override_action", {}).get("body")
        if override is not None:
            return {"body": override}
        return {"body": [0.0]}

    @app.vla_action(name="body")
    async def body(action):
        pass

    graph = Graph()
    graph.set_entry_point(Graph.END_NODE)
    app.register_graph("main", graph)

    session = Session("test", graph, app)
    await session.start()
    await asyncio.sleep(0.1)

    session.state.setdefault("llm", {})["override_action"] = {"body": [999.0]}
    await asyncio.sleep(0.1)
    v = session.state.get("vla_actions", {}).get("body")
    assert v == [999.0], f"Override should work: {v}"

    del session.state["llm"]["override_action"]
    await asyncio.sleep(0.1)
    v2 = session.state.get("vla_actions", {}).get("body")
    assert v2 == [0.0], f"Should revert after override cleared: {v2}"

    await session.stop()


# ═══════════════════════════════════════════════════════════════
# 8. 条件路由
# ═══════════════════════════════════════════════════════════════

@test("conditional_routing - path selection")
async def test_conditional_routing():
    app = FastMind()

    async def start(state, event):
        state["path"] = event.payload.get("path", "a")
        return state

    async def a(state, event):
        state["result"] = "A"
        return state

    async def b(state, event):
        state["result"] = "B"
        return state

    graph = Graph()
    graph.add_node("start", start)
    graph.add_node("a", a)
    graph.add_node("b", b)
    graph.add_conditional_edges("start",
        lambda s, e: s.get("path"),
        {"a": "a", "b": "b"})
    graph.set_entry_point("start")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {"path": "a"}, "u1"))
    await api.push_event("u2", Event("msg", {"path": "b"}, "u2"))
    await asyncio.sleep(0.1)
    assert api.get_state("u1")["result"] == "A"
    assert api.get_state("u2")["result"] == "B"
    await api.stop()


@test("conditional_routing - fallback to regular edge")
async def test_conditional_fallback():
    app = FastMind()

    async def a(state, event):
        state["went"] = "a"
        return state

    async def fb(state, event):
        state["went"] = "fallback"
        return state

    graph = Graph()
    graph.add_node("a", a)
    graph.add_node("fb", fb)
    graph.add_conditional_edges("a",
        lambda s, e: "unknown",
        {"known": "__end__"})
    graph.add_edge("a", "fb")
    graph.set_entry_point("a")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {}, "u1"))
    await asyncio.sleep(0.1)
    assert api.get_state("u1")["went"] == "fallback"
    await api.stop()


# ═══════════════════════════════════════════════════════════════
# 9. 子图执行
# ═══════════════════════════════════════════════════════════════

@test("subgraph - child graph executes")
async def test_subgraph():
    app = FastMind()
    order = []

    async def child_task(state, event):
        order.append("child")
        state["child_done"] = True
        return state

    child = Graph()
    child.add_node("task", child_task)
    child.set_entry_point("task")

    async def parent(state, event):
        order.append("parent")
        return state

    graph = Graph()
    graph.add_node("parent", parent)
    graph.add_node("child", child)
    graph.add_edge("parent", "child")
    graph.set_entry_point("parent")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {}, "u1"))
    await asyncio.sleep(0.2)
    s = api.get_state("u1")
    await api.stop()

    assert order == ["parent", "child"]
    assert s["child_done"] is True


# ═══════════════════════════════════════════════════════════════
# 10. 错误处理
# ═══════════════════════════════════════════════════════════════

@test("error_handling - node exception emits error event")
async def test_node_error():
    app = FastMind()

    async def bad_node(state, event):
        raise ValueError("boom")

    graph = Graph()
    graph.add_node("bad", bad_node)
    graph.set_entry_point("bad")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {}, "u1"))
    await asyncio.sleep(0.1)

    s = api.get_session("u1")
    found = False
    while True:
        ev = await s.get_output()
        if ev is None:
            break
        if ev.type == "error":
            found = True

    await api.stop()
    assert found, "Should emit error event"


@test("error_handling - max iterations protection")
async def test_max_iterations():
    app = FastMind()

    async def loop(state, event):
        state.setdefault("cnt", 0)
        state["cnt"] += 1
        return state

    graph = Graph()
    graph.add_node("loop", loop)
    graph.add_edge("loop", "loop")
    graph.set_entry_point("loop")
    graph.max_iterations = 3
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {}, "u1"))
    await asyncio.sleep(0.2)

    s = api.get_session("u1")
    found = False
    while True:
        ev = await s.get_output()
        if ev is None:
            break
        if ev.type == "error":
            found = True

    await api.stop()
    assert found, "Should emit max iterations error"


# ═══════════════════════════════════════════════════════════════
# 11. 多 session 隔离
# ═══════════════════════════════════════════════════════════════

@test("session_isolation - independent state")
async def test_session_isolation():
    app = FastMind()

    async def node(state, event):
        state.setdefault("cnt", 0)
        state["cnt"] += 1
        state["sid"] = event.session_id
        return state

    graph = Graph()
    graph.add_node("n", node)
    graph.set_entry_point("n")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {}, "u1"))
    await api.push_event("u2", Event("msg", {}, "u2"))
    await asyncio.sleep(0.1)

    s1 = api.get_state("u1")
    s2 = api.get_state("u2")
    await api.stop()

    assert s1["cnt"] == 1
    assert s2["cnt"] == 1
    assert s1["sid"] == "u1"
    assert s2["sid"] == "u2"


# ═══════════════════════════════════════════════════════════════
# 12. 新优化: Session.stop() 超时
# ═══════════════════════════════════════════════════════════════

@test("stop_timeout - normal stop completes fast")
async def test_stop_timeout_normal():
    app = FastMind()
    graph = Graph()
    graph.set_entry_point(Graph.END_NODE)
    session = Session("test", graph, app)
    await session.start()

    t0 = time.monotonic()
    await session.stop()
    elapsed = time.monotonic() - t0

    assert elapsed < 2, f"Stop took {elapsed:.2f}s, expected < 2s"
    assert session._task is None


@test("stop_timeout - stop with VLA + signal")
async def test_stop_timeout_vla_signal():
    app = FastMind()

    @app.vla(name="v", frequency=30.0)
    async def vla(state, sb):
        return {"a": [0.0]}

    @app.vla_action(name="a")
    async def act(a):
        pass

    @app.signal(name="sig", interval=0.03)
    async def sig():
        return "x"

    graph = Graph()
    graph.set_entry_point(Graph.END_NODE)
    app.register_graph("main", graph)

    session = Session("test", graph, app)
    await session.start()
    await asyncio.sleep(0.1)

    t0 = time.monotonic()
    await session.stop()
    elapsed = time.monotonic() - t0

    assert elapsed < 3, f"VLA stop took {elapsed:.2f}s"
    assert session._vla_task is None
    assert session._signal_tasks == []


# ═══════════════════════════════════════════════════════════════
# 13. 新优化: Engine.stop() 并行
# ═══════════════════════════════════════════════════════════════

@test("engine_stop_parallel - multiple sessions")
async def test_engine_stop_parallel():
    app = FastMind()
    graph = Graph()
    graph.set_entry_point(Graph.END_NODE)
    app.register_graph("main", graph)

    engine = Engine(app)
    await engine.start()

    for i in range(5):
        s = engine.get_or_create_session(f"s{i}")
        await s.start()

    t0 = time.monotonic()
    await engine.stop()
    elapsed = time.monotonic() - t0

    assert elapsed < 5, f"Engine parallel stop took {elapsed:.2f}s"
    for s in engine._sessions.values():
        assert s.session_state == Session.STATE_STOPPED


# ═══════════════════════════════════════════════════════════════
# 14. 新优化: stream_events 取消不崩溃
# ═══════════════════════════════════════════════════════════════

@test("stream_events - cancel does not crash")
async def test_stream_events_cancel():
    app = FastMind()

    async def slow(state, event):
        await asyncio.sleep(0.1)
        return state

    graph = Graph()
    graph.add_node("slow", slow)
    graph.set_entry_point("slow")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {}, "u1"))

    async def consume():
        async for ev in api.stream_events("u1"):
            pass

    t = asyncio.create_task(consume())
    await asyncio.sleep(0.05)
    t.cancel()
    try:
        await t
    except asyncio.CancelledError:
        pass

    await api.stop()


# ═══════════════════════════════════════════════════════════════
# 15. ReAct tool calling with LLM
# ═══════════════════════════════════════════════════════════════

@test("react_loop - LLM tool calling", skip_if_no_llm=True)
async def test_react_with_llm():
    api_key = os.getenv("LLM_API_KEY")
    app = FastMind()

    @app.tool(name="get_weather", description="获取城市天气")
    async def get_weather(city: str) -> str:
        w = {"北京": "晴25度", "上海": "多云28度"}
        return w.get(city, "未知")

    @app.agent(name="react")
    async def react(state, event):
        state.setdefault("messages", [])
        if state.get("tool_results"):
            for r in state["tool_results"]:
                state["messages"].append({
                    "role": "tool", "tool_call_id": r["tool_call_id"],
                    "content": str(r["result"])
                })
            del state["tool_results"]
        elif event.payload.get("text"):
            state["messages"].append({"role": "user", "content": event.payload["text"]})

        from openai import AsyncOpenAI
        client = AsyncOpenAI(
            api_key=api_key,
            base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1")
        )
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            messages=state["messages"],
            tools=app.get_tool_schemas()
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            state["tool_calls"] = []
            for tc in msg.tool_calls:
                state["tool_calls"].append({
                    "id": tc.id, "type": "function",
                    "function": {"name": tc.function.name, "arguments": tc.function.arguments}
                })
            tc_list = list(state["tool_calls"])
            state["messages"].append({
                "role": "assistant", "content": msg.content or "",
                "tool_calls": tc_list
            })
        else:
            state["messages"].append({"role": "assistant", "content": msg.content or ""})
            state["_output_queue"].put_nowait(Event("stream.end", {}, event.session_id))
        return state

    tn = ToolNode(app.get_tools())
    graph = Graph()
    graph.add_node("react", react)
    graph.add_node("tools", tn)
    graph.add_conditional_edges("react",
        lambda s, e: "tools" if s.get("tool_calls") else None,
        {"tools": "tools", None: "__end__"})
    graph.add_edge("tools", "react")
    graph.set_entry_point("react")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "北京天气怎么样"}, "u1"))

    async for ev in api.stream_events("u1"):
        if ev.type == "stream.end":
            break

    s = api.get_state("u1")
    await api.stop()

    msgs = s.get("messages", [])
    has_tool = any(m.get("role") == "tool" for m in msgs)
    has_answer = any(
        m.get("role") == "assistant" and not m.get("tool_calls")
        for m in msgs
    )
    assert has_tool, f"Should have tool result: {[m['role'] for m in msgs]}"
    assert has_answer, f"Should have final answer: {[m['role'] for m in msgs]}"


@test("perception_loop - sensor data routed to session")
async def test_perception_data_routing():
    app = FastMind()

    @app.perception(interval=0.05, name="timer")
    async def timer(app):
        while True:
            yield Event("timer.tick", {"n": 1}, "user_001")
            await asyncio.sleep(0.05)

    @app.agent(name="handler")
    async def handler(state, event):
        if event.type == "timer.tick":
            state.setdefault("ticks", 0)
            state["ticks"] += 1
        return state

    graph = Graph()
    graph.add_node("handler", handler)
    graph.set_entry_point("handler")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await asyncio.sleep(0.2)

    s = api.get_state("user_001")
    await api.stop()

    assert s is not None, "user_001 session should be auto-created"
    assert s.get("ticks", 0) >= 2, f"Should receive timer ticks: {s.get('ticks')}"


# ═══════════════════════════════════════════════════════════════
# main
# ═══════════════════════════════════════════════════════════════

async def main():
    api_key = os.getenv("LLM_API_KEY")
    print("=" * 65)
    print(" FastMind 全方位用户体验测试")
    print(f" LLM: {'enabled' if api_key else 'disabled (set LLM_API_KEY)'}")
    print(f" Python: {sys.version.split()[0]}")
    print("=" * 65)
    print()

    # 基础
    await test_simple_chat()
    await test_simple_chat_quit()
    await test_streaming_chat_synthetic()
    await test_human_in_loop_interrupt()
    await test_human_in_loop_cancel()
    await test_tool_node_single()
    await test_tool_node_multi()
    await test_humanoid_robot_flow()
    await test_companion_emotion()
    await test_drone_sensor()
    await test_comprehensive_flow()
    await test_sleep_assessment()
    await test_npc_vla()
    await test_npc_vla_pause_resume()
    await test_npc_vla_override()

    # 路由 + 子图
    await test_conditional_routing()
    await test_conditional_fallback()
    await test_subgraph()

    # 错误处理
    await test_node_error()
    await test_max_iterations()

    # 隔离
    await test_session_isolation()

    # 新增 stop 超时
    print()
    await test_stop_timeout_normal()
    await test_stop_timeout_vla_signal()

    # 新增 engine 并行 stop
    await test_engine_stop_parallel()

    # 新增 stream_events 取消
    await test_stream_events_cancel()

    # 感知
    await test_perception_data_routing()

    # LLM 测试
    if api_key:
        print()
        await test_simple_chat_llm()
        await test_streaming_chat_llm()
        await test_react_with_llm()

    print()
    print("=" * 65)
    total = RESULTS["pass"] + RESULTS["fail"] + RESULTS["skip"]
    print(f" 结果: {RESULTS['pass']} passed, {RESULTS['fail']} failed, "
          f"{RESULTS['skip']} skipped ({total} total)")
    if RESULTS["errors"]:
        print(f" 失败详情:")
        for e in RESULTS["errors"]:
            print(f"   {e}")
    print("=" * 65)

    return RESULTS["fail"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
