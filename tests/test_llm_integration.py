"""LLM 集成测试 - 使用真实 API 测试 ReAct 循环"""

import pytest
import asyncio
import os
import sys

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from dotenv import load_dotenv

load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

from fastmind import FastMind, Graph, Event, ToolNode
from fastmind.contrib import FastMindAPI
from fastmind.core.engine import Engine


class TestLLMIntegration:
    """LLM 集成测试"""

    @pytest.fixture
    def llm_config(self):
        """LLM 配置"""
        return {
            "api_key": os.getenv("LLM_API_KEY", "sk-a64dc1190d3649fcbf53b3bd29219711"),
            "api_url": os.getenv("LLM_API_URL", "https://api.deepseek.com/v1"),
            "model": os.getenv("LLM_MODEL_NAME", "deepseek-chat"),
        }

    @pytest.fixture
    def react_app_with_llm(self):
        """创建带 LLM 的 ReAct 应用"""
        app = FastMind()

        @app.tool(name="get_weather", description="获取城市天气")
        async def get_weather(city: str) -> str:
            weathers = {"北京": "晴，25度", "上海": "多云，28度", "广州": "下雨，22度"}
            return weathers.get(city, f"{city}: 天气未知")

        return app, {
            "api_key": "sk-a64dc1190d3649fcbf53b3bd29219711",
            "api_url": "https://api.deepseek.com/v1",
            "model": "deepseek-chat",
        }

    @pytest.mark.asyncio
    async def test_react_with_deepseek(self, react_app_with_llm):
        """测试使用 DeepSeek 的 ReAct 循环"""
        app, config = react_app_with_llm
        execution_log = []

        @app.agent(name="weather_agent", tools=["get_weather"])
        async def weather_agent(state: dict, event: Event) -> tuple[dict, list[Event]]:
            state.setdefault("messages", [])
            state.setdefault("iterations", 0)
            state["iterations"] += 1

            execution_log.append(f"agent_iter_{state['iterations']}")

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

            if event.payload.get("text"):
                state["messages"].append({"role": "user", "content": event.payload["text"]})

            if state["iterations"] > 10:
                state["_end"] = True
                return state, [
                    Event("stream.chunk", {"delta": "超过最大迭代次数\n"}, event.session_id)
                ]

            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(api_key=config["api_key"], base_url=config["api_url"])
                response = await client.chat.completions.create(
                    model=config["model"],
                    messages=state["messages"],
                    tools=app.get_tool_schemas(),
                )
                msg = response.choices[0].message

                if msg.tool_calls:
                    state["tool_calls"] = []
                    for tc in msg.tool_calls:
                        state["tool_calls"].append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                        )
                    state["messages"].append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": state["tool_calls"],
                        }
                    )
                    execution_log.append(f"llm_requested_tool")
                elif msg.content:
                    state["messages"].append({"role": "assistant", "content": msg.content})
                    execution_log.append(f"llm_response: {msg.content[:50]}")
                    return state, [
                        Event("stream.chunk", {"delta": msg.content}, event.session_id),
                        Event("stream.end", {}, event.session_id),
                    ]
            except Exception as e:
                execution_log.append(f"error: {e}")
                return state, [
                    Event("stream.chunk", {"delta": f"错误: {e}"}, event.session_id),
                    Event("stream.end", {}, event.session_id),
                ]

            return state, []

        tool_node = ToolNode(app.get_tools())

        def route(state: dict, event: Event) -> str:
            if state.get("tool_calls"):
                return "tools"
            elif state.get("_end"):
                return "__end__"
            return "agent"

        graph = Graph()
        graph.add_node("agent", weather_agent)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges("agent", route, {"tools": "tools", None: "__end__"})
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")
        app.register_graph("weather", graph)

        api = FastMindAPI(app)
        await api.start()

        session_id = "test_weather"
        user_input = "北京天气怎么样？"

        event = Event("user.message", {"text": user_input}, session_id)
        await api.push_event(session_id, event, graph_name="weather")

        response_text = ""
        async for ev in api.stream_events(session_id):
            if ev.type == "stream.chunk":
                delta = ev.payload.get("delta", "")
                response_text += delta
            elif ev.type == "stream.end":
                break
            elif ev.type == "error":
                execution_log.append(f"stream_error: {ev.payload.get('error')}")

        await api.stop()

        assert "agent_iter_1" in execution_log, "Agent should execute at least once"
        assert "llm_requested_tool" in execution_log or "llm_response" in execution_log

        if "llm_requested_tool" in execution_log:
            assert "agent_iter_2" in execution_log, "Agent should execute again after tool"

        print(f"\nExecution log: {execution_log}")
        print(f"Response: {response_text}")

    @pytest.mark.asyncio
    async def test_tool_calls_not_duplicated(self, react_app_with_llm):
        """测试 tool_calls 不会被重复添加"""
        app, config = react_app_with_llm
        tool_call_ids = []

        async def tracking_agent(state: dict, event: Event) -> tuple[dict, list[Event]]:
            state.setdefault("messages", [])

            if event.payload.get("text"):
                state["messages"].append({"role": "user", "content": event.payload["text"]})

            if state.get("tool_results"):
                state["_end"] = True
                return state, []

            from openai import AsyncOpenAI

            client = AsyncOpenAI(api_key=config["api_key"], base_url=config["api_url"])
            response = await client.chat.completions.create(
                model=config["model"],
                messages=state["messages"],
                tools=app.get_tool_schemas(),
            )
            msg = response.choices[0].message

            if msg.tool_calls:
                if "tool_calls" not in state:
                    state["tool_calls"] = []
                    existing_ids = set()
                else:
                    existing_ids = {tc["id"] for tc in state.get("tool_calls", [])}

                for tc in msg.tool_calls:
                    if tc.id not in existing_ids:
                        tc_dict = {
                            "id": tc.id,
                            "type": "function",
                            "function": {
                                "name": tc.function.name,
                                "arguments": tc.function.arguments,
                            },
                        }
                        state["tool_calls"].append(tc_dict)
                        tool_call_ids.append(tc.id)
                        existing_ids.add(tc.id)

                state["messages"].append(
                    {
                        "role": "assistant",
                        "content": msg.content or "",
                        "tool_calls": state["tool_calls"],
                    }
                )

            return state, []

        tool_node = ToolNode(app.get_tools())

        def route(state: dict, event: Event) -> str:
            if state.get("tool_calls"):
                return "tools"
            elif state.get("_end"):
                return "__end__"
            return "agent"

        graph = Graph(max_iterations=5)
        graph.add_node("agent", tracking_agent)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges(
            "agent", route, {"tools": "tools", None: "__end__", "__end__": "__end__"}
        )
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")
        app.register_graph("tracking", graph)

        engine = Engine(app)
        session = engine.get_or_create_session("test_tracking", "tracking")
        await session.start()
        await session.push_event(Event("user.message", {"text": "天气？"}, "test_tracking"))

        import time

        timeout = 30
        start = time.time()
        while session.is_alive and time.time() - start < timeout:
            ev = await session.get_output()
            if ev and ev.type == "error" and "max iterations" in ev.payload.get("error", ""):
                break
            await asyncio.sleep(0.05)

        await session.stop()

        unique_ids = set(tool_call_ids)
        assert len(tool_call_ids) == len(unique_ids), (
            f"Tool call IDs should not be duplicated. Got: {tool_call_ids}"
        )


class TestMoonshotLLM:
    """Moonshot LLM 测试"""

    @pytest.fixture
    def moonshot_config(self):
        """Moonshot 配置"""
        return {
            "api_key": "sk-f2x8aZ4INh3NlCYLeMIYqgBa7e6uE6ICmt2Ihys2C4wbtr4F",
            "api_url": "https://api.moonshot.cn/v1",
            "model": "kimi-k2.5",
        }

    @pytest.mark.asyncio
    async def test_react_with_moonshot(self, moonshot_config):
        """测试使用 Moonshot 的 ReAct 循环"""
        app = FastMind()

        @app.tool(name="get_weather", description="获取城市天气")
        async def get_weather(city: str) -> str:
            weathers = {"北京": "晴，25度", "上海": "多云，28度"}
            return weathers.get(city, f"{city}: 天气未知")

        @app.agent(name="weather_agent", tools=["get_weather"])
        async def weather_agent(state: dict, event: Event) -> tuple[dict, list[Event]]:
            state.setdefault("messages", [])
            state.setdefault("iterations", 0)
            state["iterations"] += 1

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

            if event.payload.get("text"):
                state["messages"].append({"role": "user", "content": event.payload["text"]})

            if state["iterations"] > 10:
                return state, [
                    Event("stream.chunk", {"delta": "max iterations\n"}, event.session_id)
                ]

            try:
                from openai import AsyncOpenAI

                client = AsyncOpenAI(
                    api_key=moonshot_config["api_key"], base_url=moonshot_config["api_url"]
                )
                response = await client.chat.completions.create(
                    model=moonshot_config["model"],
                    messages=state["messages"],
                    tools=app.get_tool_schemas(),
                )
                msg = response.choices[0].message

                if msg.tool_calls:
                    state["tool_calls"] = []
                    for tc in msg.tool_calls:
                        state["tool_calls"].append(
                            {
                                "id": tc.id,
                                "type": "function",
                                "function": {
                                    "name": tc.function.name,
                                    "arguments": tc.function.arguments,
                                },
                            }
                        )
                    state["messages"].append(
                        {
                            "role": "assistant",
                            "content": msg.content or "",
                            "tool_calls": state["tool_calls"],
                        }
                    )
                elif msg.content:
                    state["messages"].append({"role": "assistant", "content": msg.content})
                    return state, [
                        Event("stream.chunk", {"delta": msg.content}, event.session_id),
                        Event("stream.end", {}, event.session_id),
                    ]
            except Exception as e:
                return state, [
                    Event("stream.chunk", {"delta": f"error: {e}"}, event.session_id),
                    Event("stream.end", {}, event.session_id),
                ]

            return state, []

        tool_node = ToolNode(app.get_tools())

        def route(state: dict, event: Event) -> str:
            if state.get("tool_calls"):
                return "tools"
            elif state.get("_end"):
                return "__end__"
            return "agent"

        graph = Graph()
        graph.add_node("agent", weather_agent)
        graph.add_node("tools", tool_node)
        graph.add_conditional_edges("agent", route, {"tools": "tools", None: "__end__"})
        graph.add_edge("tools", "agent")
        graph.set_entry_point("agent")
        app.register_graph("moonshot_weather", graph)

        api = FastMindAPI(app)
        await api.start()

        session_id = "test_moonshot"
        event = Event("user.message", {"text": "北京天气怎么样？"}, session_id)
        await api.push_event(session_id, event, graph_name="moonshot_weather")

        response_text = ""
        async for ev in api.stream_events(session_id):
            if ev.type == "stream.chunk":
                response_text += ev.payload.get("delta", "")
            elif ev.type == "stream.end":
                break

        await api.stop()

        print(f"\nMoonshot Response: {response_text}")
        assert len(response_text) > 0, "Should get some response"


if __name__ == "__main__":
    pytest.main([__file__, "-v", "-s"])
