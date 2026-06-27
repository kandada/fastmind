# Copyright (c) 2024-2026 xiefujin <490021684@qq.com>
# Licensed under GNU GPLv3, see LICENSE file for full license terms.

"""NPC 虚拟角色示例 - 快慢双循环架构

演示 VLA 快循环 + LLM 慢循环的完整协作。
VLA 部分用模拟数据，无需真实模型。

可以直接运行：python -m fastmind.examples.npc_vla
"""

import asyncio
import os
import sys
import math

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from dotenv import load_dotenv
load_dotenv()

from fastmind import FastMind, Graph, Event, ActionSpace
from fastmind.contrib import FastMindAPI


# ===========================================================
# 模拟游戏引擎（实际使用时替换为真实引擎 API）
# ===========================================================

class MockGameEngine:
    """模拟游戏引擎"""

    def __init__(self):
        self.npc_pos = [0.0, 0.0]
        self.npc_rot = 0.0
        self.npc_emotion = "neutral"
        self.npc_speed = 0.0
        self.frame_count = 0
        self.nearby_objects = [
            {"name": "城堡大门", "pos": [50.0, 0.0]},
            {"name": "商人",     "pos": [10.0, 5.0]},
            {"name": "树木",     "pos": [-5.0, 8.0]},
        ]

    def render_view(self):
        """模拟 NPC 视觉"""
        self.frame_count += 1
        return {
            "frame_id": self.frame_count,
            "npc_pos": self.npc_pos.copy(),
            "npc_rot": self.npc_rot,
            "objects": self.nearby_objects,
        }

    def get_state(self):
        """模拟 NPC 本体感觉"""
        return {
            "position": self.npc_pos.copy(),
            "rotation": self.npc_rot,
            "speed": self.npc_speed,
            "emotion": self.npc_emotion,
        }

    def move(self, dx, dy):
        """模拟移动"""
        self.npc_pos[0] += dx * 0.1
        self.npc_pos[1] += dy * 0.1
        self.npc_speed = math.sqrt(dx*dx + dy*dy)

    def set_emotion(self, emotion_id):
        """模拟表情"""
        emotions = ["neutral", "happy", "sad", "angry", "surprised"]
        idx = min(int(emotion_id), len(emotions) - 1)
        self.npc_emotion = emotions[idx]

    def speak(self, text):
        """模拟说话"""
        return f"[NPC] {text}"


engine = MockGameEngine()


# ===========================================================
# 创建应用
# ===========================================================

app = FastMind()


# ===========================================================
# 1. 高频信号——NPC 的感知器官
# ===========================================================

@app.signal(name="vision", interval=1/30)
async def npc_vision():
    """NPC 视觉——30fps"""
    return engine.render_view()

@app.signal(name="proprioception", interval=1/60)
async def npc_body():
    """NPC 本体感觉——60Hz"""
    return engine.get_state()


# ===========================================================
# 2. VLA 快循环——导航控制
# ===========================================================

@app.vla(name="navigation", frequency=30.0)
async def navigation_vla(state, signal_bus):
    """导航 VLA——控制 NPC 移动

    从 SignalBus 读视觉和本体感觉，
    从慢循环读 LLM 指令，
    输出身体动作向量。
    """
    vision = signal_bus.read("vision")
    body = signal_bus.read("proprioception")

    goal = state.get("llm", {}).get("goal", "idle")
    speed = state.get("llm", {}).get("speed", 1.0)
    _ = vision  # 真实场景: 传入 VLA 模型做视觉推理

    override = state.get("llm", {}).get("override_action", {}).get("body")
    if override is not None:
        return {"body": override}

    # 模拟 VLA 推理（目标导向的移动逻辑）
    if goal == "go_to_castle":
        target = [50.0, 0.0]
    elif goal == "go_to_merchant":
        target = [10.0, 5.0]
    else:
        target = None

    current = body.get("position", [0, 0]) if body else [0, 0]

    if target:
        dx = target[0] - current[0]
        dy = target[1] - current[1]
        dist = math.sqrt(dx*dx + dy*dy)

        if dist > 1.0:
            norm_dx = dx / dist * speed
            norm_dy = dy / dist * speed
            # 检测卡住（位置没有变化）
            state.setdefault("vla", {})
            if state["vla"].get("prev_pos") == current:
                state["vla"]["status"] = "stuck"
            state["vla"]["prev_pos"] = current.copy()
            return {"body": [norm_dx, norm_dy, speed]}
        else:
            state["vla"]["status"] = "arrived"
            return {"body": [0.0, 0.0, 0.0]}
    else:
        return {"body": [0.0, 0.0, 0.0]}


@app.vla(name="facial", frequency=10.0)
async def facial_vla(state, signal_bus):
    """表情 VLA——控制 NPC 面部表情"""
    emotion = state.get("llm", {}).get("emotion", "neutral")
    n_status = state.get("vla", {}).get("status", "idle")

    if n_status == "stuck":
        return {"face": [3]}  # angry
    elif n_status == "arrived":
        return {"face": [1]}  # happy
    elif emotion == "happy":
        return {"face": [1]}
    elif emotion == "sad":
        return {"face": [2]}
    else:
        return {"face": [0]}  # neutral


# ===========================================================
# 3. VLA 动作执行器
# ===========================================================

@app.vla_action(name="body", action_space=ActionSpace(3))
async def body_executor(action):
    """执行身体动作"""
    engine.move(action[0], action[1])

@app.vla_action(name="face", action_space=ActionSpace(1))
async def face_executor(action):
    """执行面部表情"""
    engine.set_emotion(action[0])


# ===========================================================
# 4. 慢循环——LLM 层（event-driven）
# ===========================================================

@app.agent(name="npc_brain")
async def npc_brain(state, event):
    """NPC 大脑——规划、决策、对话"""

    if event.type == "system.tick":
        vla_status = state.get("vla", {}).get("status")

        if vla_status == "stuck":
            state.setdefault("llm", {})["goal"] = "go_to_merchant"
            state["llm"]["speed"] = 1.0
            state["llm"]["emotion"] = "happy"
            del state["vla"]["status"]
            reply = "发现卡住了，改为前往商人处"
            state.setdefault("messages", []).append(
                {"role": "assistant", "content": reply}
            )

        elif vla_status == "arrived":
            state.setdefault("llm", {})["goal"] = "idle"
            state["llm"]["emotion"] = "happy"
            del state["vla"]["status"]
            reply = "到达目标位置！"
            state.setdefault("messages", []).append(
                {"role": "assistant", "content": reply}
            )

        else:
            body = state.get("llm", {}).get("goal", "idle")
            if body != "idle":
                state.setdefault("messages", []).append(
                    {"role": "assistant", "content": f"正在前往目标 (current_speed={state.get('llm', {}).get('speed', 1.0)})..."}
                )

        return state

    if event.type == "user.message":
        text = event.payload.get("text", "")

        # 简单的 NLP（真实场景用 LLM API）
        text_lower = text.lower()
        reply = ""
        new_goal = None

        if "城堡" in text_lower or "castle" in text_lower or "门" in text_lower:
            new_goal = "go_to_castle"
            reply = "好的，正在前往城堡！"
        elif "商人" in text_lower or "merchant" in text_lower:
            new_goal = "go_to_merchant"
            reply = "好的，正在前往商人处！"
        elif "停" in text_lower or "stop" in text_lower:
            state.setdefault("llm", {})["override_action"] = {"body": [0.0, 0.0, 0.0]}
            reply = "已暂停移动！"
        elif "继续" in text_lower or "resume" in text_lower:
            state.setdefault("llm", {}).pop("override_action", None)
            reply = "已恢复移动！"
        elif "速度" in text_lower:
            state.setdefault("llm", {})["speed"] = 2.0
            reply = "速度已加快！"
        elif "正常" in text_lower:
            state.setdefault("llm", {})["speed"] = 1.0
            reply = "速度已恢复正常！"
        elif "状态" in text_lower or "status" in text_lower:
            pos = engine.npc_pos
            emo = engine.npc_emotion
            v_stat = state.get("vla", {}).get("status", "moving")
            goal_s = state.get("llm", {}).get("goal", "idle")
            reply = f"位置: ({pos[0]:.1f}, {pos[1]:.1f}), 表情: {emo}, 状态: {v_stat}, 目标: {goal_s}"
        elif "quit" in text_lower:
            state["quit"] = True
            reply = "再见！"
        else:
            reply = f"收到: {text}（试试：'去城堡' / '去商人' / '停' / '继续' / '状态'）"

        if new_goal:
            state.setdefault("llm", {})["goal"] = new_goal
            state.setdefault("llm", {}).pop("override_action", None)

        state.setdefault("messages", []).append(
            {"role": "assistant", "content": reply}
        )
        return state

    return state


# ===========================================================
# 5. 低频感知——周期性触发 LLM 重新规划
# ===========================================================

@app.perception(interval=3.0, name="replan_trigger")
async def replan_trigger(app):
    """每 3 秒触发 LLM 检查 VLA 状态"""
    yield Event(type="system.tick", payload={}, session_id="npc_001")


# ===========================================================
# 6. 构建与启动
# ===========================================================

graph = Graph()
graph.add_node("brain", npc_brain)
graph.set_entry_point("brain")
app.register_graph("main", graph)


async def main():
    fm_api = FastMindAPI(app)
    await fm_api.start()

    print("=" * 60)
    print("FastMind NPC 虚拟角色示例（VLA 快循环 + LLM 慢循环）")
    print("=" * 60)
    print("VLA 快循环 (30Hz): 导航控制 + 面部表情")
    print("LLM 慢循环 (event): 指令理解 + 状态监控")
    print()
    print("命令:")
    print("  去城堡     - NPC 前往城堡")
    print("  去商人     - NPC 前往商人处")
    print("  停/继续    - 暂停/恢复移动")
    print("  速度/正常  - 切换速度")
    print("  状态       - 查看 NPC 状态")
    print("  quit       - 退出")
    print("-" * 60)
    print("NPC 正在后台运行 VLA 循环...")
    print("-" * 60)

    session_id = "npc_001"

    # 转发感知事件到 NPC 会话
    async def forwarder():
        while True:
            ev = await fm_api.wait_for_output_event("monitor", timeout=2.0)
            if ev and ev.type == "system.tick":
                await fm_api.push_event(session_id, ev)

    fwd_task = asyncio.create_task(forwarder())

    while True:
        try:
            user_input = input("\n你: ").strip()
            if not user_input:
                continue

            event = Event("user.message", {"text": user_input}, session_id)
            await fm_api.push_event(session_id, event)

            await asyncio.sleep(1.0)

            state = fm_api.get_state(session_id)
            if state and "messages" in state:
                last_msg = state["messages"][-1]
                if last_msg["role"] == "assistant":
                    print(f"NPC: {last_msg['content']}")

                # 显示 NPC 状态
                pos = engine.npc_pos
                vstat = state.get("vla", {}).get("status", "moving")
                goal = state.get("llm", {}).get("goal", "idle")
                print(f"     [状态] 位置=({pos[0]:.1f},{pos[1]:.1f}) 表情={engine.npc_emotion} 状态={vstat} 目标={goal}")

                if state.get("quit"):
                    break

        except (EOFError, KeyboardInterrupt):
            print("\n退出...")
            break

    fwd_task.cancel()
    try:
        await fwd_task
    except asyncio.CancelledError:
        pass

    await fm_api.stop()

    print("=" * 60)
    print("会话结束")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
