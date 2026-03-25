"""综合示例 - 任务助手

可以直接运行：python comprehensive_assistant.py

演示特性：
1. 多节点图（规划、执行、确认）
2. 工具调用（搜索、存储、提醒）
3. 流式输出（打字机效果）
4. Human-in-the-loop（人工确认）
5. 命令行交互
"""

import asyncio
import sys
import os
from datetime import datetime

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from fastmind import FastMind, Graph, Event, ToolNode
from fastmind.contrib import FastMindAPI


app = FastMind()


# ============ 工具定义 ============


@app.tool(name="search", description="搜索信息")
async def search(query: str) -> str:
    """模拟搜索功能"""
    results = {
        "python": "Python 3.12 是一种高级编程语言...",
        "fastmind": "FastMind 是一个事件驱动的 AI 框架...",
        "ai": "人工智能（AI）是让机器具有人类智能的技术...",
    }
    return results.get(query.lower(), f"关于 '{query}' 的搜索结果：这是一条模拟的搜索结果...")


@app.tool(name="save_note", description="保存笔记")
async def save_note(title: str, content: str) -> str:
    """保存笔记"""
    return f"笔记已保存：{title}"


@app.tool(name="set_reminder", description="设置提醒")
async def set_reminder(thing: str, time: str) -> str:
    """设置提醒"""
    return f"提醒已设置：{thing} 于 {time}"


@app.tool(name="get_time", description="获取当前时间")
async def get_time() -> str:
    """获取当前时间"""
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


# ============ Agent 定义 ============


@app.agent(name="planner")
async def planner_agent(state: dict, event: Event) -> tuple[dict, list[Event]]:
    """分析用户意图，决定是否需要工具调用或人工确认"""
    state.setdefault("messages", [])
    state.setdefault("tasks", [])
    output_events = []

    if event.type == "user.message":
        text = event.payload.get("text", "")
        state["messages"].append({"role": "user", "content": text})

        if "quit" in text.lower():
            state["quit"] = True
            state["messages"].append({"role": "assistant", "content": "再见！有什么需要随时叫我。"})
            output_events.append(
                Event(
                    type="stream.chunk",
                    payload={"delta": "再见！有什么需要随时叫我。\n"},
                    session_id=event.session_id,
                )
            )
            output_events.append(Event(type="stream.end", payload={}, session_id=event.session_id))
            return state, output_events

        # 分析意图
        if "搜索" in text or "查一下" in text or "帮我找" in text:
            query = text.replace("搜索", "").replace("查一下", "").replace("帮我找", "").strip()
            state["current_task"] = "search"
            state["task_data"] = {"query": query}
            state["need_approval"] = True
            msg = f"好的，我来帮你搜索 '{query}' 相关的信息。需要我继续吗？"
            state["messages"].append({"role": "assistant", "content": msg})
            output_events.append(
                Event(
                    type="stream.chunk", payload={"delta": msg + "\n"}, session_id=event.session_id
                )
            )
        elif "保存" in text and "笔记" in text:
            state["current_task"] = "save_note"
            state["task_data"] = {"title": "未命名", "content": text}
            state["need_approval"] = True
            msg = "好的，我来帮你保存这条笔记。"
            state["messages"].append({"role": "assistant", "content": msg})
            output_events.append(
                Event(
                    type="stream.chunk", payload={"delta": msg + "\n"}, session_id=event.session_id
                )
            )
        elif "提醒" in text:
            state["current_task"] = "set_reminder"
            state["task_data"] = {"thing": text, "time": "稍后"}
            state["need_approval"] = True
            msg = "好的，我来帮你设置提醒。"
            state["messages"].append({"role": "assistant", "content": msg})
            output_events.append(
                Event(
                    type="stream.chunk", payload={"delta": msg + "\n"}, session_id=event.session_id
                )
            )
        elif "时间" in text or "现在几点" in text:
            state["current_task"] = "get_time"
            state["tool_calls"] = [
                {"id": "call_time", "function": {"name": "get_time", "arguments": "{}"}}
            ]
            output_events.append(Event(type="stream.end", payload={}, session_id=event.session_id))
        else:
            msg = "我可以帮你：\n  1. 搜索信息（说'搜索xxx'）\n  2. 保存笔记（说'保存笔记...'）\n  3. 设置提醒（说'提醒我...'）\n  4. 查询时间（说'现在几点'）\n输入 'quit' 退出"
            state["messages"].append({"role": "assistant", "content": msg})
            output_events.append(
                Event(
                    type="stream.chunk", payload={"delta": msg + "\n"}, session_id=event.session_id
                )
            )
            output_events.append(Event(type="stream.end", payload={}, session_id=event.session_id))

    return state, output_events


@app.agent(name="executor")
async def executor_agent(state: dict, event: Event) -> tuple[dict, list[Event]]:
    """处理工具结果，生成最终回复"""
    state.setdefault("messages", [])
    output_events = []

    if state.get("tool_results"):
        for result in state["tool_results"]:
            tool_name = result.get("tool_name", "unknown")
            tool_result = result.get("result", "")

            if tool_name == "search":
                msg = f"搜索结果：{tool_result}"
                state["messages"].append({"role": "assistant", "content": msg})
            elif tool_name == "get_time":
                msg = f"现在是：{tool_result}"
                state["messages"].append({"role": "assistant", "content": msg})
            elif tool_name == "save_note":
                msg = tool_result
                state["messages"].append({"role": "assistant", "content": msg})
            elif tool_name == "set_reminder":
                msg = tool_result
                state["messages"].append({"role": "assistant", "content": msg})

            output_events.append(
                Event(
                    type="stream.chunk", payload={"delta": msg + "\n"}, session_id=event.session_id
                )
            )

        del state["tool_results"]
        state["task_completed"] = True
        output_events.append(Event(type="stream.end", payload={}, session_id=event.session_id))

    return state, output_events


async def ask_confirm_node(state: dict, event: Event) -> tuple[dict, list[Event]]:
    """等待用户确认的节点"""
    task = state.get("current_task", "unknown")
    return state, [
        Event(
            type="interrupt",
            payload={
                "prompt": f"确认执行任务：{task}？",
                "resume_node": "execute",
                "cancel_node": "cancel",
            },
            session_id=event.session_id,
        )
    ]


async def cancel_node(state: dict, event: Event) -> dict:
    """取消任务"""
    state["messages"].append({"role": "assistant", "content": "好的，已取消。有其他需要帮忙的吗？"})
    state["task_completed"] = True
    return state


def need_approval(state: dict, event: Event) -> str:
    """判断是否需要人工确认"""
    return "ask_confirm" if state.get("need_approval") and not state.get("tool_calls") else None


def has_tool_calls(state: dict, event: Event) -> str:
    """判断是否有待执行的工具调用"""
    return "tools" if state.get("tool_calls") else None


# ============ 工具节点 ============
tool_node = ToolNode(app.get_tools())


# ============ 图定义 ============
graph = Graph()
graph.add_node("planner", planner_agent)
graph.add_node("ask_confirm", ask_confirm_node)
graph.add_node("cancel", cancel_node)
graph.add_node("tools", tool_node)

# 主流程: __start__ -> planner
graph.add_edge("__start__", "planner")

# planner 执行后:
#   - need_approval=true (无 tool_calls) -> ask_confirm
#   - tool_calls 存在 -> tools
#   - 其他 -> __end__
graph.add_conditional_edges(
    "planner",
    lambda s, e: (
        "ask_confirm"
        if s.get("need_approval") and not s.get("tool_calls")
        else ("tools" if s.get("tool_calls") else None)
    ),
    {"ask_confirm": "ask_confirm", "tools": "tools", None: "__end__"},
)

# ask_confirm 确认后 -> tools
graph.add_edge("ask_confirm", "tools")

# ask_confirm 取消后 -> __end__
graph.add_edge("ask_confirm", "__end__")

# tools 执行后 -> __end__ (工具结果通过 stream 输出)
graph.add_edge("tools", "__end__")

# 设置入口点
graph.set_entry_point("planner")
app.register_graph("main", graph)


# ============ 流式工具节点 ============


class StreamingToolNodeWithOutput(ToolNode):
    """带流式输出的工具节点"""

    async def execute(self, state: dict, event: Event) -> tuple[dict, list[Event]]:
        tool_calls = state.get("tool_calls", [])
        if not tool_calls:
            return state, []

        results = []
        output_events = []

        for tool_call in tool_calls:
            tool_name = tool_call.get("function", {}).get("name")
            arguments = tool_call.get("function", {}).get("arguments", "{}")

            if isinstance(arguments, str):
                import json

                try:
                    arguments = json.loads(arguments)
                except json.JSONDecodeError:
                    arguments = {}

            output_events.append(
                Event(
                    type="stream.chunk",
                    payload={"delta": f"[思考] 正在调用 {tool_name}...\n"},
                    session_id=event.session_id,
                )
            )

            tool = self.tools.get(tool_name)
            if not tool:
                result = f"Tool '{tool_name}' not found"
            else:
                try:
                    if asyncio.iscoroutinefunction(tool.func):
                        result = await tool.func(**arguments)
                    else:
                        result = tool.func(**arguments)
                except Exception as e:
                    result = f"Error: {str(e)}"

            output_events.append(
                Event(
                    type="stream.chunk",
                    payload={"delta": f"[完成] {result}\n"},
                    session_id=event.session_id,
                )
            )

            results.append(
                {
                    "tool_call_id": tool_call.get("id"),
                    "tool_name": tool_name,
                    "result": result,
                }
            )

        state["tool_results"] = results
        if "tool_calls" in state:
            del state["tool_calls"]

        return state, output_events


# ============ 主程序 ============


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 60)
    print("FastMind 综合助手")
    print("=" * 60)
    print("功能:")
    print("  - 搜索信息：'搜索 python'")
    print("  - 保存笔记：'保存笔记 xxx'")
    print("  - 设置提醒：'提醒我 xxx'")
    print("  - 查询时间：'现在几点'")
    print("  - 退出：'quit'")
    print("-" * 60)
    print("提示: 部分操作需要人工确认")
    print("-" * 60)

    session_id = "assistant_session"

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue

            event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)

            # 收集流式输出
            full_response = []
            while True:
                try:
                    ev = await asyncio.wait_for(
                        fm_api.wait_for_output_event(session_id, timeout=3.0), timeout=5.0
                    )
                    if ev is None:
                        break

                    if ev.type == "stream.chunk":
                        delta = ev.payload.get("delta", "")
                        full_response.append(delta)
                        print(delta, end="", flush=True)

                    elif ev.type == "interrupt":
                        print(f"\n⚠️  {ev.payload.get('prompt', '需要确认')}")
                        print("请输入 'confirm' 继续 或 'cancel' 取消: ", end="")
                        choice = input().strip()

                        if choice.lower() == "confirm":
                            await fm_api.resume_session(session_id, "confirm")
                        else:
                            await fm_api.resume_session(session_id, "cancel")
                        break

                    elif ev.type == "error":
                        print(f"\n❌ 错误: {ev.payload.get('error', '未知错误')}")
                        break

                    else:
                        break

                except asyncio.TimeoutError:
                    break

            # 检查是否退出
            state = fm_api.get_state(session_id)
            if state and state.get("quit"):
                break

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\n退出...")
            break

    await fm_api.stop()


if __name__ == "__main__":
    asyncio.run(main())
