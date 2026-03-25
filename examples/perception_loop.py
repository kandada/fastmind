"""感知循环示例

可以直接运行：python perception_loop.py
演示定时器感知和事件路由
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI


app = FastMind()


@app.perception(interval=2.0, name="timer")
async def timer_perception(app: FastMind):
    """定时器感知，每2秒触发一次"""
    count = 0
    while True:
        count += 1
        yield Event(
            type="timer.tick",
            payload={"count": count, "time": datetime.now().strftime("%H:%M:%S")},
            session_id="monitor_session",
        )
        await asyncio.sleep(2.0)


@app.agent(name="monitor_agent")
async def monitor_agent(state: dict, event: Event) -> dict:
    """监控 Agent"""
    state.setdefault("events", [])
    state.setdefault("counts", {})

    if event.type == "timer.tick":
        count = event.payload.get("count", 0)
        time_str = event.payload.get("time", "")
        state["events"].append(f"[{time_str}] Timer #{count}")
        state["counts"]["timer"] = count
        state["last_update"] = datetime.now().strftime("%H:%M:%S")

    return state


@app.agent(name="chat_agent")
async def chat_agent(state: dict, event: Event) -> dict:
    """聊天 Agent，同时处理用户消息和定时器事件"""
    state.setdefault("messages", [])
    state.setdefault("events", [])
    state.setdefault("counts", {})

    if event.type == "timer.tick":
        count = event.payload.get("count", 0)
        time_str = event.payload.get("time", "")
        state["events"].append(f"[{time_str}] Timer #{count}")
        state["counts"]["timer"] = count
        state["last_update"] = datetime.now().strftime("%H:%M:%S")
        return state

    if event.type == "user.message":
        text = event.payload.get("text", "")

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or last_msg.get("content") != text or last_msg.get("role") != "user":
            state["messages"].append({"role": "user", "content": text})

        if text.lower() == "status":
            timer_count = state.get("counts", {}).get("timer", 0)
            response = f"Timer has ticked {timer_count} times"
            if state.get("last_update"):
                response += f", last update: {state['last_update']}"
            state["messages"].append({"role": "assistant", "content": response})
        elif text.lower() == "events":
            events = state.get("events", [])
            response = f"Events received: {len(events)}"
            if events:
                response += f"\nLast 3: {'; '.join(events[-3:])}"
            state["messages"].append({"role": "assistant", "content": response})
        elif text.lower() == "quit":
            state["quit"] = True
            state["messages"].append({"role": "assistant", "content": "再见！"})
        else:
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": "发送 'status' 查看状态，'events' 查看事件历史，'quit' 退出",
                }
            )

    return state


graph = Graph()
graph.add_node("chat", chat_agent)

graph.add_edge("__start__", "chat")

graph.set_entry_point("chat")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind 感知循环示例")
    print("=" * 50)
    print("功能: 定时器每2秒触发一次事件")
    print("提示:")
    print("  - 输入 'status' 查看定时器状态")
    print("  - 输入 'events' 查看事件历史")
    print("  - 输入 'quit' 退出")
    print("-" * 50)
    print("提示: 定时器每2秒自动触发，观察状态变化")
    print("-" * 50)

    session_id = "user_001"

    # 定期检查并推送事件到会话（事件驱动）
    async def event_forwarder():
        while True:
            ev = await fm_api.wait_for_output_event("monitor_session", timeout=2.0)
            if ev and ev.type == "timer.tick":
                await fm_api.push_event(session_id, ev)

    forwarder_task = asyncio.create_task(event_forwarder())

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue

            event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)

            await asyncio.sleep(0.3)

            state = fm_api.get_state(session_id)
            if state and "messages" in state:
                last_msg = state["messages"][-1]
                if last_msg["role"] == "assistant":
                    print(f"Bot: {last_msg['content']}")

                if state.get("quit"):
                    break

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\n退出...")
            break

    forwarder_task.cancel()
    try:
        await forwarder_task
    except asyncio.CancelledError:
        pass

    await fm_api.stop()


if __name__ == "__main__":
    asyncio.run(main())
