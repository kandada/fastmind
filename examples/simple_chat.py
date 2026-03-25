"""简单对话示例

可以直接运行：python -m fastmind.examples.simple_chat
或从项目根目录：cd /path/to/project && python fastmind/examples/simple_chat.py
"""

import asyncio
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv

load_dotenv()

from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI


app = FastMind()


@app.agent(name="chat_agent")
async def chat_agent(state: dict, event: Event) -> dict:
    """聊天 Agent"""
    state.setdefault("messages", [])
    user_text = event.payload.get("text", "")

    last_msg = state["messages"][-1] if state["messages"] else None
    if not last_msg or last_msg.get("content") != user_text or last_msg.get("role") != "user":
        state["messages"].append({"role": "user", "content": user_text})

    if user_text.lower() == "quit":
        state["messages"].append({"role": "assistant", "content": "再见！"})
        state["quit"] = True
        return state

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
        )

        msg = response.choices[0].message
        state["messages"].append({"role": "assistant", "content": msg.content or ""})

    except Exception as e:
        state["messages"].append({"role": "assistant", "content": f"错误: {str(e)}"})

    return state


graph = Graph()
graph.add_node("chat", chat_agent)
graph.set_entry_point("chat")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind 简单对话示例 (调用 LLM)")
    print("=" * 50)
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

            await asyncio.sleep(3)

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
