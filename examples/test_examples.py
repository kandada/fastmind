# Copyright (c) 2024-2026 xiefujin <490021684@qq.com>
# Licensed under GNU GPLv3, see LICENSE file for full license terms.

"""测试所有 examples 的核心逻辑，用真实大模型运行。

用法:
    LLM_API_KEY="sk-xxx" LLM_API_URL="https://api.deepseek.com/v1" \
    LLM_MODEL_NAME="deepseek-chat" \
    python3 fastmind/examples/test_examples.py
"""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from dotenv import load_dotenv
load_dotenv(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), ".env"))

from fastmind import FastMind, Graph, Event, ToolNode
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
# Example 1: simple_chat.py
# ==========================================
@test("simple_chat - basic agent without LLM")
async def test_simple_chat():
    app = FastMind()

    @app.agent(name="chat")
    async def chat_agent(state, event):
        state.setdefault("messages", [])
        state["messages"].append(event.payload.get("text", ""))
        return state

    graph = Graph()
    graph.add_node("chat", chat_agent)
    graph.set_entry_point("chat")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("msg", {"text": "Hello"}, "u1"))
    await asyncio.sleep(0.2)
    state = api.get_state("u1")
    await api.stop()

    assert "Hello" in state.get("messages", [])


# ==========================================
# Example 2: simple_chat_with_tool.py (ReAct)
# ==========================================
@test("simple_chat_with_tool - ReAct loop with LLM")
async def test_react():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("LLM_API_KEY not set")

    app = FastMind()

    @app.tool(name="get_weather", description="获取城市天气")
    async def get_weather(city: str) -> str:
        return "晴，25度"

    @app.tool(name="get_time", description="获取当前时间")
    def get_time() -> str:
        from datetime import datetime
        return datetime.now().strftime("%H:%M:%S")

    @app.tool(name="calculate", description="计算数学表达式")
    def calculate(expression: str) -> str:
        return str(eval(expression))

    @app.agent(name="chat_agent")
    async def chat_agent(state, event):
        state.setdefault("messages", [])
        if state.get("tool_results"):
            for r in state["tool_results"]:
                state["messages"].append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": str(r["result"])})
            del state["tool_results"]
        elif event.payload.get("text"):
            state["messages"].append({"role": "user", "content": event.payload["text"]})

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1"))
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            messages=state["messages"],
            tools=app.get_tool_schemas()
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

    tn = ToolNode(app.get_tools())
    graph = Graph()
    graph.add_node("agent", chat_agent)
    graph.add_node("tools", tn)
    graph.add_conditional_edges("agent", lambda s, e: "tools" if s.get("tool_calls") else None, {"tools": "tools", None: "__end__"})
    graph.add_edge("tools", "agent")
    graph.set_entry_point("agent")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "北京天气怎么样？"}, "u1"))
    await asyncio.sleep(5)
    state = api.get_state("u1")
    await api.stop()

    msgs = state.get("messages", [])
    has_tool = any(m.get("role") == "tool" for m in msgs)
    has_answer = any(m.get("role") == "assistant" and not m.get("tool_calls") for m in msgs)
    assert has_tool, "Should have tool call result"
    assert has_answer, "Should have final answer"


# ==========================================
# Example 3: companion_bot.py - emotional companion
# ==========================================
@test("companion_bot - multi-tool companion")
async def test_companion_bot():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("LLM_API_KEY not set")

    app = FastMind()

    @app.tool(name="play_music", description="播放音乐")
    def play_music(song: str = "") -> str:
        return f"Playing: {song or 'relaxing music'}"

    @app.tool(name="tell_joke", description="讲个笑话")
    def tell_joke() -> str:
        return "Why did the chicken cross the road?"

    @app.tool(name="get_weather", description="获取天气")
    def get_weather(city: str) -> str:
        return "晴，舒适"

    @app.agent(name="companion")
    async def companion(state, event):
        state.setdefault("messages", [])
        if state.get("tool_results"):
            for r in state["tool_results"]:
                state["messages"].append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": str(r["result"])})
            del state["tool_results"]
        elif event.payload.get("text"):
            state["messages"].append({"role": "user", "content": event.payload["text"]})

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1"))
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            messages=state["messages"],
            tools=app.get_tool_schemas()
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

    tn = ToolNode(app.get_tools(tools=["play_music", "tell_joke", "get_weather"]))
    graph = Graph()
    graph.add_node("companion", companion)
    graph.add_node("tools", tn)
    graph.add_conditional_edges("companion", lambda s, e: "tools" if s.get("tool_calls") else None, {"tools": "tools", None: "__end__"})
    graph.add_edge("tools", "companion")
    graph.set_entry_point("companion")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "请使用 tell_joke 工具给我讲个笑话"}, "u1"))
    await asyncio.sleep(5)
    state = api.get_state("u1")
    await api.stop()

    msgs = state.get("messages", [])
    has_tool = any(m.get("role") == "tool" for m in msgs)
    has_answer = any(m.get("role") == "assistant" and not m.get("tool_calls") for m in msgs)
    assert has_tool or has_answer, "Should call tool or produce answer"


# ==========================================
# Example 4: drone.py - drone control
# ==========================================
@test("drone - multi-tool drone")
async def test_drone():
    app = FastMind()

    @app.tool(name="get_gps", description="获取 GPS 坐标")
    def get_gps() -> str:
        return "39.9042, 116.4074"

    @app.tool(name="get_battery_level", description="获取电量")
    def get_battery_level() -> str:
        return "85%"

    @app.agent(name="drone_agent")
    async def drone_agent(state, event):
        state.setdefault("messages", [])
        if state.get("tool_results"):
            for r in state["tool_results"]:
                state["messages"].append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": str(r["result"])})
            del state["tool_results"]
        elif event.payload.get("text"):
            state["messages"].append({"role": "user", "content": event.payload["text"]})

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1"))
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            messages=state["messages"],
            tools=app.get_tool_schemas()
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

    tn = ToolNode(app.get_tools(tools=["get_gps", "get_battery_level"]))
    graph = Graph()
    graph.add_node("agent", drone_agent)
    graph.add_node("tools", tn)
    graph.add_conditional_edges("agent", lambda s, e: "tools" if s.get("tool_calls") else None, {"tools": "tools", None: "__end__"})
    graph.add_edge("tools", "agent")
    graph.set_entry_point("agent")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "检查无人机状态（GPS和电量）"}, "u1"))
    await asyncio.sleep(5)
    state = api.get_state("u1")
    await api.stop()

    msgs = state.get("messages", [])
    has_tool_result = any(m.get("role") == "tool" for m in msgs)
    has_answer = any(m.get("role") == "assistant" and not m.get("tool_calls") for m in msgs)
    assert has_tool_result or has_answer, "Should execute drone workflow"


# ==========================================
# Example 5: human_in_loop.py
# ==========================================
@test("human_in_loop - HITL with real flow")
async def test_human_in_loop():
    app = FastMind()

    @app.agent(name="order_agent")
    async def order_agent(state, event):
        state.setdefault("orders", [])
        action = event.payload.get("action", "")
        amount = event.payload.get("amount", 0)
        if action == "下单":
            state["current_order"] = {"amount": amount, "status": "pending"}
            state["orders"].append(state["current_order"])
            state["need_approval"] = amount > 100
        elif action == "确认":
            state["current_order"]["status"] = "confirmed"
            state["confirmed"] = True
        elif action == "取消":
            state["current_order"]["status"] = "cancelled"
            state["confirmed"] = False
        return state

    async def ask_confirm(state, event):
        return state, [Event("interrupt", {"prompt": "确认?", "resume_node": "confirm", "cancel_node": "cancel"}, event.session_id)]

    def need_confirm(state, event):
        return "ask_confirm" if state.get("need_approval") else "__end__"

    graph = Graph()
    graph.add_node("order", order_agent)
    graph.add_node("ask_confirm", ask_confirm)
    graph.add_node("confirm", lambda s, e: {**s, "confirmed": True})
    graph.add_node("cancel", lambda s, e: {**s, "confirmed": False})
    graph.add_conditional_edges("order", need_confirm, {"ask_confirm": "ask_confirm", "__end__": "__end__"})
    graph.set_entry_point("order")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()

    # Test confirm
    await api.push_event("u1", Event("user.message", {"action": "下单", "amount": 150}, "u1"))
    await asyncio.sleep(0.2)
    s = api.get_session("u1")
    assert s.session_state == Session.STATE_INTERRUPTED, f"Expected INTERRUPTED, got {s.session_state}"

    await api.resume_session("u1", "confirm")
    await asyncio.sleep(0.2)
    state = api.get_state("u1")
    assert state.get("confirmed") is True, f"Confirm failed: {state}"

    # Test cancel
    await api.push_event("u2", Event("user.message", {"action": "下单", "amount": 200}, "u2"))
    await asyncio.sleep(0.2)
    await api.resume_session("u2", "cancel")
    await asyncio.sleep(0.2)
    state2 = api.get_state("u2")
    assert state2.get("confirmed") is False, f"Cancel failed: {state2}"

    await api.stop()


# ==========================================
# Example 6: streaming_chat.py - streaming with LLM
# ==========================================
@test("streaming_chat - streaming output with LLM")
async def test_streaming_chat():
    api_key = os.getenv("LLM_API_KEY")
    if not api_key:
        raise Exception("LLM_API_KEY not set")

    app = FastMind()

    @app.agent(name="streaming_agent")
    async def streaming_agent(state, event):
        output_events = []
        state.setdefault("messages", [])
        state["messages"].append({"role": "user", "content": event.payload.get("text", "")})

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=api_key, base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1"))
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            messages=state["messages"],
            stream=False
        )
        content = resp.choices[0].message.content or ""
        for char in content:
            output_events.append(Event("stream.chunk", {"delta": char}, event.session_id))
        output_events.append(Event("stream.end", {}, event.session_id))
        state["messages"].append({"role": "assistant", "content": content})
        return state, output_events

    graph = Graph()
    graph.add_node("agent", streaming_agent)
    graph.set_entry_point("agent")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "Say hello in 3 words"}, "u1"))

    text = ""
    async for ev in api.stream_events("u1"):
        if ev.type == "stream.chunk":
            text += ev.payload.get("delta", "")
        elif ev.type == "stream.end":
            break

    await api.stop()
    state = api.get_state("u1")
    assert len(text) > 0, "Should receive streamed text"
    assert state.get("messages", [])[-1]["role"] == "assistant"


# ==========================================
# Example 7: sleep_assessment.py - multi-state HITL
# ==========================================
@test("sleep_assessment - multi-state flow")
async def test_sleep_assessment():
    app = FastMind()

    @app.tool(name="save_report", description="保存报告")
    def save_report(content: str) -> str:
        return "报告已保存"

    @app.agent(name="evaluate_agent")
    async def evaluate_agent(state, event):
        state.setdefault("messages", [])
        if state.get("tool_results"):
            for r in state["tool_results"]:
                state["messages"].append({"role": "tool", "tool_call_id": r["tool_call_id"], "content": str(r["result"])})
            del state["tool_results"]
        elif event.payload.get("text"):
            state["messages"].append({"role": "user", "content": event.payload["text"]})

        from openai import AsyncOpenAI
        client = AsyncOpenAI(api_key=os.getenv("LLM_API_KEY"), base_url=os.getenv("LLM_API_URL", "https://api.deepseek.com/v1"))
        resp = await client.chat.completions.create(
            model=os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
            messages=state["messages"],
            tools=app.get_tool_schemas(tools=["save_report"])
        )
        msg = resp.choices[0].message
        if msg.tool_calls:
            state["tool_calls"] = []
            for tc in msg.tool_calls:
                state["tool_calls"].append({"id": tc.id, "type": "function", "function": {"name": tc.function.name, "arguments": tc.function.arguments}})
            state["messages"].append({"role": "assistant", "content": msg.content or "", "tool_calls": list(state["tool_calls"])})
        else:
            state["messages"].append({"role": "assistant", "content": msg.content or ""})
            state["done"] = True
        return state

    tn = ToolNode(app.get_tools(tools=["save_report"]))
    graph = Graph()
    graph.add_node("evaluate", evaluate_agent)
    graph.add_node("tools", tn)
    graph.add_conditional_edges("evaluate", lambda s, e: "tools" if s.get("tool_calls") else None, {"tools": "tools", None: "__end__"})
    graph.add_edge("tools", "evaluate")
    graph.set_entry_point("evaluate")
    app.register_graph("main", graph)

    api = FastMindAPI(app)
    await api.start()
    await api.push_event("u1", Event("user.message", {"text": "评估我的睡眠：昨晚睡了6小时，质量一般，帮我保存报告"}, "u1"))

    state = None
    for _ in range(20):
        await asyncio.sleep(1)
        state = api.get_state("u1")
        if state and state.get("done"):
            break

    await api.stop()

    assert state is not None, "State should exist"
    msgs = state.get("messages", [])
    has_answer = any(m.get("role") == "assistant" and not m.get("tool_calls") for m in msgs)
    assert has_answer or state.get("done"), f"Should have evaluation result. msgs={len(msgs)} done={state.get('done')}"


# ==========================================
# Run all
# ==========================================
async def main():
    api_key = os.getenv("LLM_API_KEY")
    print("=" * 60)
    print("FastMind Examples 用户测试")
    print(f"LLM: {'enabled' if api_key else 'disabled (set LLM_API_KEY)'}")
    print("=" * 60)
    print()

    await test_simple_chat()
    await test_human_in_loop()

    if api_key:
        await test_react()
        await test_companion_bot()
        await test_drone()
        await test_streaming_chat()
        await test_sleep_assessment()
    else:
        print("  ⏭️  LLM-dependent tests skipped (no API key)")

    print()
    print("=" * 60)
    print(f"结果: {RESULTS['pass']} passed, {RESULTS['fail']} failed")
    for e in RESULTS["errors"]:
        print(f"  {e}")
    print("=" * 60)
    return RESULTS["fail"] == 0


if __name__ == "__main__":
    success = asyncio.run(main())
    sys.exit(0 if success else 1)
