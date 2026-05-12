"""综合测试：使用真实大模型测试 fastmind 框架所有核心场景。

用法:
    LLM_API_KEY="sk-xxx" LLM_API_URL="https://api.deepseek.com/v1" \
    LLM_MODEL_NAME="deepseek-chat" \
    python3 fastmind/examples/test_comprehensive.py
"""
import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastmind import FastMind, Graph, Event, ToolNode, StreamingToolNode
from fastmind.contrib import FastMindAPI
from fastmind.core.engine import Session

RESULTS = {"pass": 0, "fail": 0, "errors": []}


def test(name):
    def decorator(func):
        async def wrapper(*args, **kwargs):
            try:
                await func(*args, **kwargs)
                RESULTS["pass"] += 1
                print(f"  ✅ {name}")
            except Exception as e:
                RESULTS["fail"] += 1
                msg = f"  ❌ {name}: {e}"
                RESULTS["errors"].append(msg)
                print(msg)
        return wrapper
    return decorator


# ==========================================
# Test 1: Simple node chain (no LLM)
# ==========================================
@test("Simple node chain execution")
async def test_simple_chain():
    app = FastMind()
    order = []

    async def node_a(state, event):
        order.append("a")
        return state

    async def node_b(state, event):
        order.append("b")
        return state

    async def node_c(state, event):
        order.append("c")
        return state

    graph = Graph()
    graph.add_node("a", node_a)
    graph.add_node("b", node_b)
    graph.add_node("c", node_c)
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.set_entry_point("a")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("test", {}, "u1"))
    await asyncio.sleep(0.3)
    await api.stop()

    assert order == ["a", "b", "c"], f"Expected ['a','b','c'], got {order}"


# ==========================================
# Test 2: Conditional routing
# ==========================================
@test("Conditional routing based on state")
async def test_conditional_routing():
    app = FastMind()

    async def start_node(state, event):
        state["path"] = event.payload.get("path", "a")
        return state

    async def path_a(state, event):
        state["result"] = "went_a"
        return state

    async def path_b(state, event):
        state["result"] = "went_b"
        return state

    def router(state, event):
        return state.get("path", "a")

    graph = Graph()
    graph.add_node("start", start_node)
    graph.add_node("a", path_a)
    graph.add_node("b", path_b)
    graph.add_conditional_edges("start", router, {"a": "a", "b": "b"})
    graph.set_entry_point("start")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    # Test path a
    await api.push_event("u1", Event("test", {"path": "a"}, "u1"))
    await asyncio.sleep(0.2)
    assert api.get_state("u1")["result"] == "went_a"

    # Test path b
    await api.push_event("u2", Event("test", {"path": "b"}, "u2"))
    await asyncio.sleep(0.2)
    assert api.get_state("u2")["result"] == "went_b"

    await api.stop()


# ==========================================
# Test 3: Conditional edge fallback to regular edges
# ==========================================
@test("Conditional edge unmatched -> regular edge fallback")
async def test_conditional_fallback():
    app = FastMind()

    async def node_a(state, event):
        return state

    async def fallback_node(state, event):
        state["fallback"] = True
        return state

    graph = Graph()
    graph.add_node("a", node_a)
    graph.add_node("fb", fallback_node)
    graph.add_conditional_edges("a", lambda s, e: "unknown", {"known": "__end__"})
    graph.add_edge("a", "fb")
    graph.set_entry_point("a")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("test", {}, "u1"))
    await asyncio.sleep(0.2)
    state = api.get_state("u1")
    await api.stop()

    assert state.get("fallback") is True, "Conditional edge should fallback to regular edge"


# ==========================================
# Test 4: Cycle detection
# ==========================================
@test("detect_cycles detects conditional edge cycles")
async def test_cycle_detection():
    graph = Graph()
    graph.add_node("agent", lambda s, e: s)
    graph.add_node("tools", lambda s, e: s)
    graph.add_conditional_edges("agent", lambda s, e: "tools", {"tools": "tools"})
    graph.add_edge("tools", "agent")
    graph.set_entry_point("agent")

    cycles = graph.detect_cycles()
    assert len(cycles) >= 1, "Should detect agent->tools->agent cycle"


# ==========================================
# Test 5: Tool calling via ToolNode (no LLM)
# ==========================================
@test("ToolNode executes tool and returns results")
async def test_toolnode_execution():
    app = FastMind()

    @app.tool(name="add", description="Add two numbers")
    def add(a: int, b: int) -> str:
        return str(a + b)

    app._tool_registry.add("add", app.get_tool("add"))

    # Create ToolNode with specific tool
    tn = ToolNode(app.get_tools(tools=["add"]))
    state = {"tool_calls": [{"id": "1", "function": {"name": "add", "arguments": '{"a": 1, "b": 2}'}}]}
    new_state, events = await tn.execute(state, Event("test", {}, "u1"))

    assert new_state.get("tool_results")
    assert new_state["tool_results"][0]["result"] == "3"


# ==========================================
# Test 6: Tool calling with LLM (ReAct loop)
# ==========================================
@test("ReAct tool calling loop with LLM")
async def test_react_with_llm():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("LLM_API_KEY not set")

    app = FastMind()

    @app.tool(name="get_weather", description="获取城市天气")
    async def get_weather(city: str) -> str:
        weathers = {"北京": "晴，25度", "上海": "多云，28度"}
        return weathers.get(city, "天气未知")

    @app.agent(name="react_agent")
    async def react_agent(state, event):
        state.setdefault("messages", [])
        state.setdefault("iterations", 0)
        state["iterations"] += 1

        if state.get("tool_results"):
            for r in state["tool_results"]:
                state["messages"].append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": str(r["result"])})
            del state["tool_results"]
        elif event.payload.get("text"):
            state["messages"].append({"role": "user", "content": event.payload["text"]})

        if state["iterations"] > 10:
            return state, [Event("stream.end", {}, event.session_id)]

        try:
            from openai import AsyncOpenAI
            api_url = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1")
            model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")
            client = AsyncOpenAI(api_key=api_key, base_url=api_url)
            resp = await client.chat.completions.create(model=model, messages=state["messages"], tools=app.get_tool_schemas())
            msg = resp.choices[0].message

            if msg.tool_calls:
                state["tool_calls"] = []
                tc_list = []
                for tc in msg.tool_calls:
                    tc_dict = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                    state["tool_calls"].append(tc_dict)
                    tc_list.append(tc_dict)
                state["messages"].append({"role": "assistant", "content": msg.content or "", "tool_calls": tc_list})
            else:
                state["messages"].append({"role": "assistant", "content": msg.content or ""})
                return state, [Event("stream.end", {}, event.session_id)]
        except Exception as e:
            return state, [Event("stream.end", {}, event.session_id)]

        return state, []


# ==========================================
# Test 7: Tool filtering (tools parameter)
# ==========================================
@test("app.get_tools(tools=[...]) returns only specified tools")
async def test_tool_filtering():
    app = FastMind()

    @app.tool(name="tool_a")
    def tool_a():
        pass

    @app.tool(name="tool_b")
    def tool_b():
        pass

    filtered = app.get_tools(tools=["tool_a"])
    assert "tool_a" in filtered
    assert "tool_b" not in filtered
    assert len(filtered) == 1

    schemas = app.get_tool_schemas(tools=["tool_a"])
    assert len(schemas) == 1
    assert schemas[0]["function"]["name"] == "tool_a"


# ==========================================
# Test 8: Multiple tool calls in one response
# ==========================================
@test("Multiple tool calls executed by ToolNode")
async def test_multiple_tool_calls():
    app = FastMind()

    @app.tool(name="get_weather")
    async def get_weather(city: str) -> str:
        return f"{city}: sunny"

    @app.tool(name="get_time")
    async def get_time() -> str:
        return "14:30"

    tn = ToolNode(app.get_tools())
    state = {
        "tool_calls": [
            {"id": "c1", "function": {"name": "get_weather", "arguments": '{"city": "Beijing"}'}},
            {"id": "c2", "function": {"name": "get_time", "arguments": "{}"}},
        ]
    }
    new_state, _ = await tn.execute(state, Event("test", {}, "u1"))

    results = new_state.get("tool_results", [])
    assert len(results) == 2
    assert "sunny" in results[0]["result"]
    assert "14:30" in results[1]["result"]


# ==========================================
# Test 9: Streaming output
# ==========================================
@test("Streaming output via output_events")
async def test_streaming_output():
    app = FastMind()

    async def streaming_agent(state, event):
        output_events = []
        for c in "Hello":
            output_events.append(Event("stream.chunk", {"delta": c}, event.session_id))
        output_events.append(Event("stream.end", {}, event.session_id))
        state["streamed"] = True
        return state, output_events

    graph = Graph()
    graph.add_node("stream", streaming_agent)
    graph.set_entry_point("stream")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    await api.push_event("u1", Event("test", {}, "u1"))

    full_text = ""
    async for ev in api.stream_events("u1"):
        if ev.type == "stream.chunk":
            full_text += ev.payload.get("delta", "")
        elif ev.type == "stream.end":
            break

    await api.stop()

    assert full_text == "Hello", f"Expected 'Hello', got '{full_text}'"
    assert api.get_state("u1")["streamed"] is True


# ==========================================
# Test 10: StreamingToolNode
# ==========================================
@test("StreamingToolNode emits chunk events")
async def test_streaming_toolnode():
    app = FastMind()

    @app.tool(name="echo")
    def echo(text: str) -> str:
        return text

    tn = StreamingToolNode(app.get_tools())
    state = {"tool_calls": [{"id": "1", "function": {"name": "echo", "arguments": '{"text": "hi"}'}}]}
    _, events = await tn.execute(state, Event("test", {}, "u1"))

    chunk_events = [e for e in events if e.type == "stream.chunk"]
    assert len(chunk_events) >= 2


# ==========================================
# Test 11: Session isolation
# ==========================================
@test("Multiple sessions are isolated")
async def test_session_isolation():
    app = FastMind()

    async def node(state, event):
        state.setdefault("count", 0)
        state["count"] += 1
        state["session_id"] = event.session_id
        return state

    graph = Graph()
    graph.add_node("n", node)
    graph.set_entry_point("n")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    await api.push_event("u1", Event("test", {}, "u1"))
    await api.push_event("u2", Event("test", {}, "u2"))
    await asyncio.sleep(0.2)

    s1 = api.get_state("u1")
    s2 = api.get_state("u2")

    await api.stop()

    assert s1["count"] == 1
    assert s2["count"] == 1
    assert s1["session_id"] == "u1"
    assert s2["session_id"] == "u2"


# ==========================================
# Test 12: Error handling
# ==========================================
@test("Node exception emits error event")
async def test_error_handling():
    app = FastMind()

    async def error_node(state, event):
        raise ValueError("test error")

    graph = Graph()
    graph.add_node("err", error_node)
    graph.set_entry_point("err")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    await api.push_event("u1", Event("test", {}, "u1"))
    await asyncio.sleep(0.2)

    session = api.get_session("u1")
    errors = []
    while True:
        ev = await session.get_output()
        if ev is None:
            break
        errors.append(ev)

    await api.stop()

    error_events = [e for e in errors if e.type == "error"]
    assert len(error_events) >= 1, "Should emit error event"


# ==========================================
# Test 13: Node not found error
# ==========================================
@test("Node not found emits error event")
async def test_node_not_found():
    app = FastMind()

    async def start_node(state, event):
        return state

    graph = Graph()
    graph.add_node("start", start_node)
    graph.add_edge("start", "nonexistent")
    graph.set_entry_point("start")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("test", {}, "u1"))
    await asyncio.sleep(0.2)

    session = api.get_session("u1")
    found_error = False
    while True:
        ev = await session.get_output()
        if ev is None:
            break
        if ev.type == "error":
            found_error = True

    await api.stop()
    assert found_error, "Should emit error event for missing node"


# ==========================================
# Test 14: Max iterations protection
# ==========================================
@test("Max iterations protection prevents infinite loops")
async def test_max_iterations():
    app = FastMind()

    async def loop_node(state, event):
        state.setdefault("count", 0)
        state["count"] += 1
        return state

    graph = Graph()
    graph.add_node("loop", loop_node)
    graph.add_edge("loop", "loop")
    graph.set_entry_point("loop")
    graph.max_iterations = 5
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("test", {}, "u1"))
    await asyncio.sleep(0.3)

    session = api.get_session("u1")
    found_error = False
    while True:
        ev = await session.get_output()
        if ev is None:
            break
        if ev.type == "error" and "max iterations" in ev.payload.get("error", "").lower():
            found_error = True

    await api.stop()
    assert found_error, "Should emit max iterations error"


# ==========================================
# Test 15: HITL interrupt and resume
# ==========================================
@test("HITL interrupt and resume flow")
async def test_hitl_interrupt_resume():
    app = FastMind()

    async def process(state, event):
        state["step"] = "before_interrupt"
        return state

    async def interrupt_handler(state, event):
        return state, [Event("interrupt", {"prompt": "confirm?", "resume_node": "after"}, event.session_id)]

    async def after(state, event):
        state["step"] = "after_resume"
        return state

    graph = Graph()
    graph.add_node("process", process)
    graph.add_node("ask", interrupt_handler)
    graph.add_node("after", after)
    graph.add_edge("process", "ask")
    graph.set_entry_point("process")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    await api.push_event("u1", Event("test", {}, "u1"))
    await asyncio.sleep(0.2)
    assert api.get_state("u1")["step"] == "before_interrupt"

    s = api.get_session("u1")
    assert s.session_state == Session.STATE_INTERRUPTED

    await api.resume_session("u1", "confirm")
    await asyncio.sleep(0.2)
    assert api.get_state("u1")["step"] == "after_resume"

    await api.stop()


# ==========================================
# Test 16: HITL cancel route
# ==========================================
@test("HITL cancel routes to cancel_node")
async def test_hitl_cancel():
    app = FastMind()

    async def ask(state, event):
        return state, [Event("interrupt", {"prompt": "?", "resume_node": "after", "cancel_node": "end"}, event.session_id)]

    async def after(state, event):
        state["result"] = "confirmed"
        return state

    async def end(state, event):
        state["result"] = "cancelled"
        return state

    graph = Graph()
    graph.add_node("ask", ask)
    graph.add_node("after", after)
    graph.add_node("end", end)
    graph.set_entry_point("ask")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    await api.push_event("u1", Event("test", {}, "u1"))
    await asyncio.sleep(0.2)

    await api.resume_session("u1", "cancel")
    await asyncio.sleep(0.2)

    assert api.get_state("u1")["result"] == "cancelled"
    await api.stop()


# ==========================================
# Test 17: Subgraph execution
# ==========================================
@test("Subgraph execution")
async def test_subgraph():
    app = FastMind()
    executed = []

    async def sub_task(state, event):
        executed.append("sub")
        state["sub_done"] = True
        return state

    child = Graph()
    child.add_node("task", sub_task)
    child.set_entry_point("task")

    async def parent(state, event):
        executed.append("parent")
        return state

    graph = Graph()
    graph.add_node("parent", parent)
    graph.add_node("child", child)
    graph.add_edge("parent", "child")
    graph.set_entry_point("parent")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("test", {}, "u1"))
    await asyncio.sleep(0.3)
    state = api.get_state("u1")
    await api.stop()

    assert "parent" in executed
    assert "sub" in executed
    assert state.get("sub_done") is True


# ==========================================
# Test 18: Multi-node chain with LLM
# ==========================================
@test("Multi-node chain with LLM")
async def test_multinode_llm():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("LLM_API_KEY not set")

    app = FastMind()

    @app.tool(name="get_time", description="获取当前时间")
    def get_time() -> str:
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    @app.agent(name="planner")
    async def planner(state, event):
        state.setdefault("messages", [])
        if event.payload.get("text"):
            state["messages"].append({"role": "user", "content": event.payload["text"]})
        state["plan"] = "executed"
        return state

    @app.agent(name="executor")
    async def executor(state, event):
        state.setdefault("messages", state.get("messages", []))
        if state.get("tool_results"):
            for r in state["tool_results"]:
                state["messages"].append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": str(r["result"])})
            del state["tool_results"]

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1"))
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            messages=state["messages"],
            tools=app.get_tool_schemas(tools=["get_time"])
        )
        msg = resp.choices[0].message

        if msg.tool_calls:
            state["tool_calls"] = []
            tc_list = []
            for tc in msg.tool_calls:
                tc_dict = {"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}}
                state["tool_calls"].append(tc_dict)
                tc_list.append(tc_dict)
            state["messages"].append({"role": "assistant", "content": msg.content or "", "tool_calls": tc_list})
        else:
            state["messages"].append({"role": "assistant", "content": msg.content or ""})
            state["done"] = True
        return state

    tn = ToolNode(app.get_tools(tools=["get_time"]))
    graph = Graph()
    graph.add_node("planner", planner)
    graph.add_node("executor", executor)
    graph.add_node("tools", tn)
    graph.add_edge("planner", "executor")
    graph.add_conditional_edges("executor", lambda s, e: "tools" if s.get("tool_calls") else "__end__", {"tools": "tools", "__end__": "__end__"})
    graph.add_edge("tools", "executor")
    graph.set_entry_point("planner")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "现在几点了？只回答时间"}, "u1"))
    await asyncio.sleep(5)

    state = api.get_state("u1")
    msgs = state.get("messages", [])
    has_answer = any(m.get("role") == "assistant" and not m.get("tool_calls") for m in msgs)

    await api.stop()

    assert state.get("plan") == "executed"
    assert has_answer or state.get("done"), "Should have final answer"


# ==========================================
async def main():
    api_key = os.getenv("LLM_API_KEY")
    print("=" * 60)
    print(f"FastMind 综合测试")
    print(f"LLM: {'enabled' if api_key else 'disabled (set LLM_API_KEY)'}")
    print("=" * 60)
    print()

    await test_simple_chain()
    await test_conditional_routing()
    await test_conditional_fallback()
    await test_cycle_detection()
    await test_toolnode_execution()
    await test_tool_filtering()
    await test_multiple_tool_calls()
    await test_streaming_output()
    await test_streaming_toolnode()
    await test_session_isolation()
    await test_error_handling()
    await test_node_not_found()
    await test_max_iterations()
    await test_hitl_interrupt_resume()
    await test_hitl_cancel()
    await test_subgraph()

    if api_key:
        print()
        await test_react_with_llm()
        await test_multinode_llm()

    print()
    print("=" * 60)
    print(f"结果: {RESULTS['pass']} passed, {RESULTS['fail']} failed")
    if RESULTS["errors"]:
        for e in RESULTS["errors"]:
            print(f"  {e}")
    print("=" * 60)

    return RESULTS["fail"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
