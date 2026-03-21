"""流式输出示例

可以直接运行：python -m fastmind.examples.streaming_chat

特点：
- 无轮询，使用 asyncio.Queue.get() 阻塞等待
- 低CPU占用，事件驱动
- 支持字符级流式模拟（解决 API 批量返回的问题）
"""

import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastmind import FastMind, Graph, Event
from fastmind.contrib import FastMindAPI


app = FastMind()

STREAM_DELAY = 0.03


@app.agent(name="chat_agent")
async def chat_agent(state: dict, event: Event) -> dict:
    """流式输出 Agent"""
    state.setdefault("messages", [])
    user_text = event.payload.get("text", "")
    state["messages"].append({"role": "user", "content": user_text})

    if user_text.lower() == "quit":
        state["messages"].append({"role": "assistant", "content": "再见！"})
        state["quit"] = True
        state["_output_queue"].put_nowait(
            Event(
                type="stream.chunk",
                payload={"delta": "再见！"},
                session_id=event.session_id,
            )
        )
        state["_output_queue"].put_nowait(
            Event(
                type="stream.end",
                payload={"content": "再见！"},
                session_id=event.session_id,
            )
        )
        return state

    output_queue = state["_output_queue"]
    session_id = state["_session_id"]

    async def stream_llm():
        """后台任务：流式调用 LLM 并实时推送事件"""
        api_key = os.getenv("LLM_API_KEY")
        api_url = os.getenv("LLM_API_URL", "https://api.deepseek.com/v1")
        model = os.getenv("LLM_MODEL_NAME", "deepseek-chat")

        if not api_key:
            output_queue.put_nowait(
                Event(
                    type="stream.chunk",
                    payload={"delta": "错误: 未设置 LLM_API_KEY"},
                    session_id=session_id,
                )
            )
            output_queue.put_nowait(
                Event(
                    type="stream.end",
                    payload={"content": ""},
                    session_id=session_id,
                )
            )
            return

        from openai import OpenAI

        client = OpenAI(api_key=api_key, base_url=api_url)

        try:
            stream = client.chat.completions.create(
                model=model,
                messages=state["messages"],
                stream=True,
            )

            for chunk in stream:
                delta = chunk.choices[0].delta.content or ""
                if delta:
                    for char in delta:
                        output_queue.put_nowait(
                            Event(
                                type="stream.chunk",
                                payload={"delta": char},
                                session_id=session_id,
                            )
                        )
                        await asyncio.sleep(STREAM_DELAY)

            output_queue.put_nowait(
                Event(
                    type="stream.end",
                    payload={"content": ""},
                    session_id=session_id,
                )
            )

        except Exception as e:
            output_queue.put_nowait(
                Event(
                    type="stream.chunk",
                    payload={"delta": f"错误: {str(e)}"},
                    session_id=session_id,
                )
            )
            output_queue.put_nowait(
                Event(
                    type="stream.end",
                    payload={"content": ""},
                    session_id=session_id,
                )
            )

    asyncio.create_task(stream_llm())
    return state


graph = Graph()
graph.add_node("chat", chat_agent)
graph.set_entry_point("chat")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 50)
    print("FastMind 流式输出示例 (调用 LLM)")
    print("=" * 50)
    print("提示: 输入 'quit' 退出")
    print("-" * 50)

    session_id = "user_001"

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue

            print("Bot: ", end="", flush=True)

            full_text = await fm_api.run_streaming(
                session_id,
                user_input,
                on_chunk=lambda delta: print(delta, end="", flush=True),
            )
            print()

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
