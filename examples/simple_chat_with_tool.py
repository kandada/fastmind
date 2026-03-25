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
async def chat_agent(state: dict, event: Event) -> dict:
    """聊天 Agent，支持工具调用"""
    state.setdefault("messages", [])
    user_text = event.payload.get("text", "")

    # 避免在 ReAct 循环中重复添加 user 消息
    last_msg = state["messages"][-1] if state["messages"] else None
    if not last_msg or last_msg.get("content") != user_text or last_msg.get("role") != "user":
        state["messages"].append({"role": "user", "content": user_text})

    if user_text.lower() == "quit":
        state["messages"].append({"role": "assistant", "content": "再见！"})
        state["quit"] = True
        return state

    # 处理工具结果
    if state.get("tool_results"):
        for result in state["tool_results"]:
            state["messages"].append(
                {"role": "system", "content": f"[{result['tool_name']}] {result['result']}"}
            )
        del state["tool_results"]
        return state

    # 调用 LLM
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

    if not api_key:
        state["messages"].append({"role": "assistant", "content": "错误: 未设置 LLM_API_KEY"})
        return state

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=api_url)

        response = await client.chat.completions.create(
            model=model,
            messages=state["messages"],
            tools=app.get_tool_schemas(),
        )

        msg = response.choices[0].message

        # 优先使用 content（如果是有效回答），忽略 tool_calls
        # 因为 LLM 有时会同时返回 content 和 tool_calls
        if msg.content and msg.content.strip():
            state["messages"].append({"role": "assistant", "content": msg.content})
            # 清除可能存在的 tool_calls
            if "tool_calls" in state:
                del state["tool_calls"]
        elif msg.tool_calls:
            state["tool_calls"] = []
            for tc in msg.tool_calls:
                if hasattr(tc, "function"):
                    state["tool_calls"].append(
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
                elif isinstance(tc, dict):
                    state["tool_calls"].append(tc)

    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"错误: {str(e)}"})

    return state

    # 处理工具结果
    if state.get("tool_results"):
        for result in state["tool_results"]:
            state["messages"].append(
                {"role": "system", "content": f"[{result['tool_name']}] {result['result']}"}
            )
        del state["tool_results"]
        return state

    # 调用 LLM
    api_key = os.getenv("LLM_API_KEY")
    api_url = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1")
    model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

    if not api_key:
        state["messages"].append({"role": "assistant", "content": "错误: 未设置 LLM_API_KEY"})
        return state

    try:
        from openai import AsyncOpenAI

        client = AsyncOpenAI(api_key=api_key, base_url=api_url)

        # 清理之前连续的系统工具结果消息，避免 LLM 继续调用无关工具
        messages = state["messages"]
        while len(messages) >= 3:
            if (
                messages[-1].get("role") == "system"
                and messages[-2].get("role") == "assistant"
                and messages[-3].get("role") == "user"
            ):
                messages = messages[:-3]
            else:
                break
        state["messages"] = messages

        response = await client.chat.completions.create(
            model=model,
            messages=state["messages"],
            tools=app.get_tool_schemas(),
        )

        msg = response.choices[0].message

        if msg.tool_calls:
            state["tool_calls"] = []
            for tc in msg.tool_calls:
                if hasattr(tc, "function"):
                    state["tool_calls"].append(
                        {
                            "id": tc.id,
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                    )
                elif isinstance(tc, dict):
                    state["tool_calls"].append(tc)
        elif msg.content:
            state["messages"].append({"role": "assistant", "content": msg.content})

    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"错误: {str(e)}"})

    return state


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

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue

            event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)

            # 等待处理完成
            await asyncio.sleep(5)

            state = fm_api.get_state(session_id)
            if state and "messages" in state:
                last_msg = state["messages"][-1]
                if last_msg["role"] in ("assistant", "system"):
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
