"""带工具调用的对话示例

可以直接运行：python -m fastmind.examples.simple_chat_with_tool
"""

import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastmind import FastMind, Graph, Event, ToolNode
from fastmind.contrib import FastMindAPI


app = FastMind()


@app.tool(name="get_weather", description="获取城市天气")
async def get_weather(city: str) -> str:
    """获取城市天气"""
    weathers = {
        "北京": "晴，25度",
        "上海": "多云，28度",
        "广州": "雨，30度",
        "深圳": "晴，29度",
    }
    return f"{city}: {weathers.get(city, '天气未知')}"


@app.tool(name="get_time", description="获取当前时间")
async def get_time() -> str:
    """获取当前时间"""
    from datetime import datetime

    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")


@app.tool(name="calculate", description="计算数学表达式")
async def calculate(expression: str) -> str:
    """计算数学表达式"""
    try:
        result = eval(expression)
        return f"{expression} = {result}"
    except Exception as e:
        return f"无法计算: {e}"


@app.agent(name="chat_agent", tools=["get_weather", "get_time", "calculate"])
async def chat_agent(state: dict, event: Event) -> tuple[dict, list[Event]]:
    """聊天 Agent，支持工具调用"""
    state.setdefault("messages", [])
    user_text = event.payload.get("text", "")
    output_events = []

    # 处理工具结果
    if state.get("tool_results"):
        for result in state["tool_results"]:
            state["messages"].append(
                {
                    "role": "tool",
                    "tool_call_id": result["tool_call_id"],
                    "content": str(result["result"]),
                }
            )
        del state["tool_results"]
    else:
        # 只有在没有工具结果时才添加用户消息（避免重复添加）
        state["messages"].append({"role": "user", "content": user_text})

    if user_text.lower() == "quit":
        state["messages"].append({"role": "assistant", "content": "再见！"})
        state["quit"] = True
        output_events.append(Event(type="stream.end", payload={}, session_id=event.session_id))
        return state, output_events

    # 调用 LLM
    # 注意：如果有待执行的 tool_calls，先不添加 LLM 的回复内容，等工具执行完再说
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

    if not api_key:
        state["messages"].append({"role": "assistant", "content": "错误: 未设置 LLM_API_KEY"})
        output_events.append(Event(type="stream.end", payload={}, session_id=event.session_id))
        return state, output_events

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=api_url)

        response = await client.chat.completions.create(
            model=model,
            messages=state["messages"],
            tools=app.get_tool_schemas(),
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            state["tool_calls"] = []
            tool_calls_for_msg = []
            for tc in msg.tool_calls:
                if hasattr(tc, "function"):
                    tc_dict = {
                        "id": tc.id,
                        "type": "function",
                        "function": {
                            "name": tc.function.name,
                            "arguments": tc.function.arguments,
                        },
                    }
                    state["tool_calls"].append(tc_dict)
                    tool_calls_for_msg.append(tc_dict)
                elif isinstance(tc, dict):
                    tc_dict = dict(tc)
                    tc_dict.setdefault("type", "function")
                    state["tool_calls"].append(tc_dict)
                    tool_calls_for_msg.append(tc_dict)
            # 添加带有 tool_calls 的 assistant 消息到历史
            assistant_msg = {"role": "assistant", "content": msg.content or ""}
            if tool_calls_for_msg:
                assistant_msg["tool_calls"] = tool_calls_for_msg
            state["messages"].append(assistant_msg)
            # 有 tool_calls 会继续到 tools 节点，所以这里不发 stream.end
        elif msg.content:
            # 只有在没有 tool_calls 时才添加回复内容
            state["messages"].append({"role": "assistant", "content": msg.content})
            # 发送流式事件
            for char in msg.content:
                output_events.append(
                    Event(type="stream.chunk", payload={"delta": char}, session_id=event.session_id)
                )
            output_events.append(Event(type="stream.end", payload={}, session_id=event.session_id))

    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"错误: {str(e)}"})
        output_events.append(Event(type="stream.end", payload={}, session_id=event.session_id))

    return state, output_events


tool_node = ToolNode(app.get_tools())


def has_tool_calls(state: dict, event: Event) -> str:
    """检查是否有待执行的工具调用"""
    return "tools" if state.get("tool_calls") else None


graph = Graph()
graph.add_node("agent", chat_agent)
graph.add_node("tools", tool_node)

graph.add_conditional_edges("agent", has_tool_calls, {"tools": "tools", None: "__end__"})
graph.add_edge("tools", "agent")

graph.set_entry_point("agent")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind 工具调用示例 (调用 LLM)")
    print("=" * 50)
    print("功能: 查天气、时间、计算")
    print("提示: 输入 'quit' 退出")
    print("-" * 50)

    session_id = "user_001"

    async def wait_for_response():
        """等待并收集 LLM 的回复"""
        response_text = ""
        async for ev in fm_api.stream_events(session_id):
            if ev.type == "stream.chunk":
                response_text += ev.payload.get("delta", "")
            elif ev.type == "stream.end":
                break
            elif ev.type == "error":
                response_text = f"错误: {ev.payload.get('error', '未知错误')}"
                break
        return response_text

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue

            event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)

            response = await wait_for_response()
            if response:
                print(f"Bot: {response}")

            if user_input.lower() == "quit":
                break

        except EOFError:
            break
        except KeyboardInterrupt:
            print("\n\n退出...")
            break

    await fm_api.stop()


if __name__ == "__main__":
    asyncio.run(main())
