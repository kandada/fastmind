"""陪伴机器人示例

可以直接运行：python companion_bot.py
演示复杂对话和情感识别
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


@app.tool(name="play_music", description="播放音乐")
async def play_music(song: str) -> str:
    """播放音乐"""
    return f"正在播放: {song}"


@app.tool(name="set_alarm", description="设置闹钟")
async def set_alarm(time: str) -> str:
    """设置闹钟"""
    return f"闹钟已设置为: {time}"


@app.tool(name="tell_joke", description="讲笑话")
async def tell_joke() -> str:
    """讲笑话"""
    jokes = [
        "为什么程序员总是分不清万圣节和圣诞节？因为 Oct 31 = Dec 25",
        "一个SQL查询走进一家酒吧，看到两张表，问：我能 JOIN 你们吗？",
        "程序员最讨厌的事：写文档、别人不写文档",
    ]
    import random

    return random.choice(jokes)


@app.tool(name="get_weather", description="获取天气")
async def get_weather(city: str = "本地") -> str:
    """获取天气"""
    return f"{city} 今天天气晴朗，温度 22-28 度"


@app.tool(name="set_reminder", description="设置提醒")
async def set_reminder(thing: str, time: str) -> str:
    """设置提醒"""
    return f"提醒已设置: {thing} 于 {time}"


# ============ Agent 定义 ============


@app.agent(
    name="companion_agent",
    tools=["play_music", "set_alarm", "tell_joke", "get_weather", "set_reminder"],
)
async def companion_agent(state: dict, event: Event) -> dict:
    """陪伴机器人 Agent"""
    state.setdefault("messages", [])
    state.setdefault("mood", "happy")

    if event.type == "user.message":
        text = event.payload.get("text", "")

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or last_msg.get("content") != text or last_msg.get("role") != "user":
            state["messages"].append({"role": "user", "content": text})

        # 处理工具结果
        if state.get("tool_results"):
            for result in state["tool_results"]:
                state["messages"].append(
                    {"role": "assistant", "content": f"[系统] {result['result']}"}
                )
            del state["tool_results"]
            return state

        text_lower = text.lower()

        # 情感识别
        if any(w in text_lower for w in ["难过", "伤心", "抑郁", "累", "不开心"]):
            state["mood"] = "sad"
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": "我感觉到你可能有些不开心，想听个笑话吗？或者我可以放些轻松的音乐~",
                }
            )
        elif any(w in text_lower for w in ["开心", "高兴", "棒", "好"]):
            state["mood"] = "happy"
            state["messages"].append(
                {"role": "assistant", "content": "太好了！看到你开心我也开心 😊"}
            )

        # 命令处理
        if "音乐" in text_lower or "放歌" in text_lower:
            song = "轻音乐"
            state["tool_calls"] = [
                {
                    "id": "call_1",
                    "function": {"name": "play_music", "arguments": f'{{"song": "{song}"}}'},
                }
            ]
        elif "闹钟" in text_lower:
            import re

            times = re.findall(r"\d{1,2}[点:]?\d{0,2}", text)
            if times:
                state["tool_calls"] = [
                    {
                        "id": "call_2",
                        "function": {"name": "set_alarm", "arguments": f'{{"time": "{times[0]}"}}'},
                    }
                ]
            else:
                state["messages"].append(
                    {
                        "role": "assistant",
                        "content": "请告诉我想设置几点的闹钟，例如: 帮我设一个7点的闹钟",
                    }
                )
        elif "笑话" in text_lower or "讲个笑话" in text_lower:
            state["tool_calls"] = [
                {"id": "call_3", "function": {"name": "tell_joke", "arguments": "{}"}}
            ]
        elif "天气" in text_lower:
            state["tool_calls"] = [
                {"id": "call_4", "function": {"name": "get_weather", "arguments": "{}"}}
            ]
        elif "提醒" in text_lower:
            import re

            times = re.findall(r"\d{1,2}[点:]?\d{0,2}", text)
            thing = text.replace("提醒", "").replace("设", "").strip()
            if times:
                state["tool_calls"] = [
                    {
                        "id": "call_5",
                        "function": {
                            "name": "set_reminder",
                            "arguments": f'{{"thing": "{thing}", "time": "{times[0]}"}}',
                        },
                    }
                ]
            else:
                state["messages"].append(
                    {"role": "assistant", "content": "请告诉我提醒内容和时间，例如: 提醒我3点开会"}
                )
        elif "quit" in text_lower:
            state["quit"] = True
            state["messages"].append({"role": "assistant", "content": "再见！有需要随时叫我~"})
        else:
            responses = [
                "我听不太懂，可以告诉我你想做什么吗？",
                "嗯嗯，然后呢？",
                "好的，我可以帮你播放音乐、讲笑话、查天气、设闹钟和提醒~",
            ]
            import random

            state["messages"].append({"role": "assistant", "content": random.choice(responses)})

    return state


tool_node = ToolNode(app.get_tools())


def has_tool_calls(state: dict, event: Event) -> str:
    """检查是否有待执行的工具调用"""
    return "tools" if state.get("tool_calls") else None


graph = Graph()
graph.add_node("companion", companion_agent)
graph.add_node("tools", tool_node)

graph.add_conditional_edges("companion", has_tool_calls, {"tools": "tools", None: "__end__"})
graph.add_edge("tools", "companion")

graph.set_entry_point("companion")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind 陪伴机器人示例")
    print("=" * 50)
    print("功能:")
    print("  - 播放音乐: '放音乐'")
    print("  - 讲笑话: '讲个笑话'")
    print("  - 查天气: '今天天气怎么样'")
    print("  - 设闹钟: '帮我设7点闹钟'")
    print("  - 设提醒: '提醒我3点开会'")
    print("输入 'quit' 退出")
    print("-" * 50)

    session_id = "companion_001"

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

    await fm_api.stop()


if __name__ == "__main__":
    asyncio.run(main())
