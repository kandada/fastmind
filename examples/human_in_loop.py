"""Human-in-the-loop 中断示例

可以直接运行：python human_in_loop.py
用于测试人工确认流程
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI


app = FastMind()


@app.agent(name="order_agent")
async def order_agent(state: dict, event: Event) -> dict:
    """订单处理 Agent"""
    state.setdefault("orders", [])

    if event.type in ("user.message", "msg", "resume"):
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
        else:
            state["messages"] = [f"未知动作: {action}"]

    return state


async def ask_confirm_node(state: dict, event: Event) -> tuple[dict, list[Event]]:
    """中断节点，等待用户确认"""
    amount = state.get("current_order", {}).get("amount", 0)
    return state, [
        Event(
            type="interrupt",
            payload={
                "prompt": f"确认执行订单（金额: {amount}元）？",
                "resume_node": "confirm",
                "cancel_node": "cancel",
            },
            session_id=event.session_id,
        )
    ]


async def confirm_node(state: dict, event: Event) -> dict:
    """确认节点"""
    user_input = event.payload.get("user_input", "")
    state["confirm_input"] = user_input
    state["confirmed"] = True
    if state.get("current_order"):
        state["current_order"]["status"] = "confirmed"
    return state


async def cancel_node(state: dict, event: Event) -> dict:
    """取消节点"""
    state["confirmed"] = False
    if state.get("current_order"):
        state["current_order"]["status"] = "cancelled"
    return state


def need_confirm(state: dict, event: Event) -> str:
    """判断是否需要确认"""
    return "ask_confirm" if state.get("need_approval") else "__end__"


graph = Graph()
graph.add_node("order", order_agent)
graph.add_node("ask_confirm", ask_confirm_node)
graph.add_node("confirm", confirm_node)
graph.add_node("cancel", cancel_node)

# 条件边：判断是否需要人工确认
graph.add_conditional_edges(
    "order", need_confirm, {"ask_confirm": "ask_confirm", "__end__": "__end__"}
)
graph.add_edge("ask_confirm", "confirm")
graph.add_edge("ask_confirm", "cancel")

graph.set_entry_point("order")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind Human-in-the-loop 示例")
    print("=" * 50)
    print("功能: 订单处理，金额>100需要人工确认")
    print("-" * 50)

    session_id = "user_001"

    # 触发高金额订单（需要确认）
    print("\n[测试1] 高金额订单（150元，需确认）")
    event = Event("user.message", {"action": "下单", "amount": 150}, session_id)
    await fm_api.push_event(session_id, event)

    # 监听中断事件（事件驱动）
    interrupted = False
    while True:
        ev = await fm_api.wait_for_output_event(session_id, timeout=5.0)
        if ev:
            if ev.type == "interrupt":
                print(f"\n⚠️  中断: {ev.payload['prompt']}")
                print("请选择 (confirm/cancel): ", end="")
                choice = input().strip()

                if choice.lower() == "confirm":
                    await fm_api.resume_session(session_id, "confirm")
                else:
                    await fm_api.resume_session(session_id, "cancel")
                interrupted = True
                break
        else:
            break

    if interrupted:
        await asyncio.sleep(0.3)
        state = fm_api.get_state(session_id)
        if state:
            order = state.get("current_order", {})
            print(f"\n订单状态: {order.get('status', 'unknown')}")

    # 触发低金额订单（不需要确认）
    print("\n" + "-" * 50)
    print("[测试2] 低金额订单（50元，无需确认）")
    event = Event("user.message", {"action": "下单", "amount": 50}, session_id)
    await fm_api.push_event(session_id, event)
    await asyncio.sleep(0.3)

    state = fm_api.get_state(session_id)
    if state:
        orders = state.get("orders", [])
        for i, order in enumerate(orders):
            print(f"订单{i + 1}: 金额={order['amount']}, 状态={order['status']}")

    await fm_api.stop()


if __name__ == "__main__":
    asyncio.run(main())
