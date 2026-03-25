"""人形机器人示例

可以直接运行：python humanoid_robot.py
演示多 Agent 协作和工具调用
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


@app.tool(name="move_to", description="控制机器人移动到指定位置")
async def move_to(x: float, y: float, z: float = 0) -> str:
    """移动机器人到指定坐标"""
    return f"Robot moved to ({x}, {y}, {z})"


@app.tool(name="say", description="让机器人说话")
async def say(text: str) -> str:
    """让机器人说话"""
    return f"Robot says: {text}"


@app.tool(name="get_battery", description="获取机器人电池电量")
async def get_battery() -> str:
    """获取电池电量"""
    return "85%"


@app.tool(name="get_location", description="获取机器人当前位置")
async def get_location() -> str:
    """获取当前位置"""
    return "(0.0, 0.0, 0.0)"


@app.tool(name="look_at", description="让机器人看向指定方向")
async def look_at(direction: str) -> str:
    """看向指定方向"""
    return f"Robot looking {direction}"


# ============ Agent 定义 ============


@app.agent(name="voice_agent", tools=["move_to", "say", "get_battery", "get_location", "look_at"])
async def voice_agent(state: dict, event: Event) -> dict:
    """语音交互 Agent"""
    state.setdefault("messages", [])

    if event.type == "user.message":
        text = event.payload.get("text", "")

        last_msg = state["messages"][-1] if state["messages"] else None
        if not last_msg or last_msg.get("content") != text or last_msg.get("role") != "user":
            state["messages"].append({"role": "user", "content": text})

        # 处理工具结果
        if state.get("tool_results"):
            for result in state["tool_results"]:
                state["messages"].append(
                    {"role": "system", "content": f"[{result['tool_name']}] {result['result']}"}
                )
            del state["tool_results"]
            return state

        # 简单的意图识别
        text_lower = text.lower()

        if "电池" in text_lower or "电量" in text_lower:
            state["tool_calls"] = [
                {"id": "call_1", "function": {"name": "get_battery", "arguments": "{}"}}
            ]
        elif "位置" in text_lower or "在哪里" in text_lower:
            state["tool_calls"] = [
                {"id": "call_2", "function": {"name": "get_location", "arguments": "{}"}}
            ]
        elif "移动" in text_lower or "走" in text_lower:
            # 简单解析坐标
            import re

            nums = re.findall(r"-?\d+\.?\d*", text)
            if len(nums) >= 2:
                x, y = float(nums[0]), float(nums[1])
                z = float(nums[2]) if len(nums) > 2 else 0
                state["tool_calls"] = [
                    {
                        "id": "call_3",
                        "function": {
                            "name": "move_to",
                            "arguments": f'{{"x": {x}, "y": {y}, "z": {z}}}',
                        },
                    }
                ]
                state["messages"].append(
                    {"role": "assistant", "content": f"正在移动到 ({x}, {y}, {z})..."}
                )
            else:
                state["messages"].append(
                    {"role": "assistant", "content": "请告诉我目标位置，例如: 移动到 1.0 2.0"}
                )
        elif "看" in text_lower or "look" in text_lower:
            direction = "前方"
            for d in ["左", "右", "前", "后"]:
                if d in text:
                    direction = d
                    break
            state["tool_calls"] = [
                {
                    "id": "call_4",
                    "function": {"name": "look_at", "arguments": f'{{"direction": "{direction}"}}'},
                }
            ]
        elif "说话" in text_lower or "说" in text_lower:
            speech = text.replace("说", "").replace("说话", "").strip()
            state["tool_calls"] = [
                {
                    "id": "call_5",
                    "function": {"name": "say", "arguments": f'{{"text": "{speech}"}}'},
                }
            ]
        elif "quit" in text_lower:
            state["quit"] = True
            state["messages"].append({"role": "assistant", "content": "再见！"})
        else:
            state["messages"].append(
                {
                    "role": "assistant",
                    "content": "我可以帮你:\n"
                    "  - 查询电池: '电池电量'\n"
                    "  - 查询位置: '你在哪'\n"
                    "  - 移动: '移动到 1.0 2.0'\n"
                    "  - 看向: '看向左方'\n"
                    "  - 说话: '说 Hello'\n"
                    "输入 'quit' 退出",
                }
            )

    return state


tool_node = ToolNode(app.get_tools())


def has_tool_calls(state: dict, event: Event) -> str:
    """检查是否有待执行的工具调用"""
    return "tools" if state.get("tool_calls") else None


graph = Graph()
graph.add_node("voice", voice_agent)
graph.add_node("tools", tool_node)

graph.add_edge("__start__", "voice")

graph.add_conditional_edges("voice", has_tool_calls, {"tools": "tools", None: "__end__"})

graph.add_edge("tools", "voice")

graph.set_entry_point("voice")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind 人形机器人示例")
    print("=" * 50)
    print("功能:")
    print("  - 查询电池: '电池电量'")
    print("  - 查询位置: '你在哪'")
    print("  - 移动: '移动到 1.0 2.0'")
    print("  - 看向: '看向左方'")
    print("  - 说话: '说 Hello'")
    print("输入 'quit' 退出")
    print("-" * 50)

    session_id = "robot_001"

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
                # 打印新的消息
                for msg in state["messages"][-(len(state.get("_printed", [])) + 1) :]:
                    if msg["role"] in ("assistant", "system"):
                        print(f"Robot: {msg['content']}")
                state["_printed"] = len(state["messages"])

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
