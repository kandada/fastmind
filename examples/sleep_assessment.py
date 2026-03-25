"""睡眠评估示例

可以直接运行：python sleep_assessment.py
演示多状态管理和 Human-in-the-loop
"""

import asyncio
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmind import FastMind, Graph, Event, ToolNode
from fastmind.contrib import FastMindAPI


app = FastMind()


# ============ 工具定义 ============


@app.tool(name="save_report", description="保存睡眠报告")
async def save_report(report: str) -> str:
    """保存报告"""
    return f"报告已保存: {report[:50]}..."


@app.tool(name="get_sleep_tips", description="获取睡眠建议")
async def get_sleep_tips(issue: str) -> str:
    """获取睡眠建议"""
    tips = {
        "难以入睡": "建议: 1. 睡前1小时避免使用电子设备 2. 保持卧室黑暗和凉爽 3. 尝试深呼吸放松",
        "睡眠不足": "建议: 1. 固定作息时间 2. 避免咖啡因 3. 午睡不超过20分钟",
        "睡眠质量差": "建议: 1. 保持规律运动但避免睡前剧烈运动 2. 减少卧室噪音 3. 床垫枕头舒适度检查",
    }
    return tips.get(issue, "建议: 保持健康的生活方式，规律作息。")


# ============ Agent 定义 ============


@app.agent(name="assessment_agent")
async def assessment_agent(state: dict, event: Event) -> dict:
    """睡眠评估 Agent"""
    state.setdefault("answers", {})
    state.setdefault("stage", "start")

    if event.type == "user.message":
        text = event.payload.get("text", "")
        stage = state["stage"]

        if stage == "start":
            state["stage"] = "hours"
            state["messages"] = [
                "你好！我是睡眠评估助手。我会问你几个问题来评估你的睡眠状况。\n问题1: 你每天睡多少小时？"
            ]
        elif stage == "hours":
            state["answers"]["hours"] = text
            state["stage"] = "quality"
            state["messages"] = ["问题2: 你的睡眠质量如何？很好/一般/差"]
        elif stage == "quality":
            quality = text.lower()
            if "差" in quality or "不好" in quality:
                state["answers"]["quality"] = "差"
                state["issue"] = "睡眠质量差"
            elif "一般" in quality:
                state["answers"]["quality"] = "一般"
                state["issue"] = "睡眠不足"
            else:
                state["answers"]["quality"] = "很好"
            state["stage"] = "difficulty"
            state["messages"] = ["问题3: 你有入睡困难吗？有/没有"]
        elif stage == "difficulty":
            difficulty = text.lower()
            if "有" in difficulty or "困难" in difficulty or "难" in difficulty:
                state["answers"]["difficulty"] = "有"
                if not state.get("issue"):
                    state["issue"] = "难以入睡"
            else:
                state["answers"]["difficulty"] = "没有"
            state["stage"] = "assessment"
            state["need_approval"] = True
            state["messages"] = ["好的，我已经收集了足够信息。需要人工确认是否生成详细报告吗？"]
        elif stage == "confirm":
            if "是" in text or "确认" in text or "好" in text:
                state["generate_report"] = True
                state["stage"] = "generate"
                state["messages"] = ["正在生成报告..."]
            else:
                state["generate_report"] = False
                state["stage"] = "end"
                state["messages"] = ["好的，评估结束。"]
        elif stage == "quit":
            state["quit"] = True
            state["messages"] = ["再见！祝你有好睡眠~"]
        else:
            state["messages"] = ["请按流程回答问题，或输入 'quit' 退出"]

    return state


async def ask_confirm_node(state: dict, event: Event) -> tuple[dict, list[Event]]:
    """确认节点"""
    return state, [
        Event(
            type="interrupt",
            payload={
                "prompt": "是否生成详细的睡眠评估报告？",
                "resume_node": "confirm",
                "cancel_node": "skip",
            },
            session_id=event.session_id,
        )
    ]


async def skip_node(state: dict, event: Event) -> dict:
    """跳过报告"""
    state["stage"] = "end"
    state["messages"] = ["好的，跳过报告生成。"]
    return state


async def confirm_node(state: dict, event: Event) -> dict:
    """确认节点，处理中断恢复"""
    user_input = event.payload.get("user_input", "")
    if "是" in user_input or "确认" in user_input or "好" in user_input:
        state["generate_report"] = True
        state["stage"] = "generate"
        state["messages"] = ["正在生成报告..."]
    else:
        state["generate_report"] = False
        state["stage"] = "end"
        state["messages"] = ["好的，评估结束。"]
    return state


def need_confirm(state: dict, event: Event) -> str:
    """判断是否需要确认"""
    return "ask_confirm" if state.get("need_approval") else "evaluate"


@app.agent(name="evaluate_agent", tools=["get_sleep_tips", "save_report"])
async def evaluate_agent(state: dict, event: Event) -> dict:
    """评估和生成报告 Agent"""
    state.setdefault("messages", [])

    issue = state.get("issue", "一般")
    answers = state.get("answers", {})

    # 获取建议
    if state.get("tool_results"):
        for result in state["tool_results"]:
            state["messages"].append(f"[建议] {result['result']}")
        del state["tool_results"]
        return state

    # 生成评估
    report = f"""
=== 睡眠评估报告 ===
睡眠时长: {answers.get("hours", "未知")}
睡眠质量: {answers.get("quality", "未知")}
入睡困难: {answers.get("difficulty", "未知")}

问题: {issue}
    """

    state["report"] = report
    state["messages"].append(f"评估完成:\n{report}")

    if state.get("generate_report"):
        state["tool_calls"] = [
            {
                "id": "call_1",
                "function": {
                    "name": "save_report",
                    "arguments": f'{{"report": "{report.strip()}"}}',
                },
            },
            {
                "id": "call_2",
                "function": {"name": "get_sleep_tips", "arguments": f'{{"issue": "{issue}"}}'},
            },
        ]
    else:
        state["stage"] = "end"

    return state


tool_node = ToolNode(app.get_tools())


def route_to_evaluate(state: dict, event: Event) -> str:
    """路由到评估节点"""
    return "evaluate" if state.get("generate_report") else "end"


graph = Graph()
graph.add_node("assessment", assessment_agent)
graph.add_node("ask_confirm", ask_confirm_node)
graph.add_node("confirm", confirm_node)
graph.add_node("skip", skip_node)
graph.add_node("evaluate", evaluate_agent)
graph.add_node("tools", tool_node)

# 评估流程
graph.add_edge("__start__", "assessment")

# 判断是否需要确认
graph.add_conditional_edges(
    "assessment", need_confirm, {"ask_confirm": "ask_confirm", "evaluate": "evaluate"}
)

# 确认后
graph.add_edge("ask_confirm", "confirm")
graph.add_conditional_edges(
    "confirm",
    lambda s, e: "evaluate" if s.get("generate_report") else "end",
    {"evaluate": "evaluate", "end": "__end__"},
)

# 评估和工具
graph.add_edge("skip", "__end__")
graph.add_conditional_edges(
    "evaluate",
    lambda s, e: "tools" if s.get("tool_calls") else None,
    {"tools": "tools", None: "__end__"},
)
graph.add_edge("tools", "evaluate")

graph.set_entry_point("assessment")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind 睡眠评估示例")
    print("=" * 50)
    print("功能: 回答问题获取睡眠建议")
    print("-" * 50)

    session_id = "sleep_001"

    # 评估流程
    stages = [
        ("开始评估", "start"),
    ]

    print("\nBot: 你好！我是睡眠评估助手。")
    print("Bot: 我会问你几个问题来评估你的睡眠状况。")

    while True:
        state = fm_api.get_state(session_id)
        if not state:
            await asyncio.sleep(0.1)
            continue

        stage = state.get("stage", "")

        if stage == "hours":
            print("\nBot: 问题1: 你每天睡多少小时？")
            user_input = input("你: ").strip()
            if user_input.lower() == "quit":
                event = Event("user.message", {"text": "quit"}, session_id)
            else:
                event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)
            await asyncio.sleep(0.3)

        elif stage == "quality":
            print("\nBot: 问题2: 你的睡眠质量如何？很好/一般/差")
            user_input = input("你: ").strip()
            if user_input.lower() == "quit":
                event = Event("user.message", {"text": "quit"}, session_id)
            else:
                event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)
            await asyncio.sleep(0.3)

        elif stage == "difficulty":
            print("\nBot: 问题3: 你有入睡困难吗？有/没有")
            user_input = input("你: ").strip()
            if user_input.lower() == "quit":
                event = Event("user.message", {"text": "quit"}, session_id)
            else:
                event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)
            await asyncio.sleep(0.3)

        elif stage == "confirm":
            print("\nBot: 是否生成详细的睡眠评估报告？(是/否)")
            user_input = input("你: ").strip()
            event = Event("user.message", {"text": user_input, "action": "confirm"}, session_id)
            await fm_api.push_event(session_id, event)

            # 检查中断（事件驱动）
            ev = await fm_api.wait_for_output_event(session_id, timeout=5.0)
            if ev and ev.type == "interrupt":
                print(f"\n⚠️  中断: {ev.payload['prompt']}")
                print("输入 'confirm' 继续或 'skip' 跳过: ", end="")
                choice = input().strip()
                if choice.lower() == "confirm":
                    await fm_api.resume_session(session_id, "是")
                else:
                    await fm_api.resume_session(session_id, "否")

        elif state.get("messages"):
            last_msg = state["messages"][-1]
            if last_msg.startswith("[建议]"):
                print(f"\nBot: {last_msg}")

        if state.get("stage") == "end" or state.get("quit"):
            break

        await asyncio.sleep(0.1)

    print("\n" + "=" * 50)
    print("评估完成！")
    print("=" * 50)

    await fm_api.stop()


if __name__ == "__main__":
    asyncio.run(main())
