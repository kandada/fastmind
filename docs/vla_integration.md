# VLA 集成指南

## 概述

VLA（Vision-Language-Action）模型是一类能够直接**从视觉输入输出连续动作**的多模态模型。典型场景包括机器人控制、虚拟世界 NPC、自动驾驶等。VLA 的特点是高频（10-100Hz）、闭环（Observe → Reason → Act），但存在语言盲、灾难性遗忘等问题。

FastMind 的 VLA 集成提供了**快慢双循环架构**：

```
VLA 快循环（time-driven，30Hz）： 持续感知 → 推理 → 动作
LLM 慢循环（event-driven）：       规划 → 推理 → 语言交互
```

两者在同一个 Session 中并发运行，通过共享 State（黑板模式）通信，互不阻塞。

---

## 核心概念

### Event vs Signal

FastMind 有两个平行的通信元语：

| | Event | Signal |
|--|-------|--------|
| **哲学** | "发生了一次某事" | "当前值是什么" |
| **数据结构** | 队列（FIFO） | 最新值缓存（last-value） |
| **消费模式** | push（被推送） | pull（主动读取） |
| **语义** | 离散、有序、不丢 | 连续、覆盖旧值、可丢 |
| **触发 graph** | 是 | 否（零开销） |
| **适合** | 用户消息、LLM 回复、中断 | 摄像头帧、关节角、NPC 视觉 |

### @app.vla

VLA 推理节点，由框架按固定频率自动调度，**不走 Graph**。

```python
@app.vla(name="navigation", frequency=30.0)
async def navigation_vla(state, signal_bus):
    """签名: (state, signal_bus) -> dict[str, list[float]]
    返回:  { 动作通道名: 动作向量 }
    """
    vision = signal_bus.read("vision")
    goal = state.get("llm", {}).get("goal", "idle")
    action = await vla_model.predict(image=vision, instruction=goal)
    return {"body": action}   # "body" 是通道名
```

特点：
- 不经过 Event 队列，不经过 Graph 遍历
- 框架按 `frequency` 自动调度
- 返回 dict，key 是动作通道名，value 是动作向量
- 可以读取 `state["llm/*"]`，受慢循环影响

### @app.vla_action

动作执行器，接收 VLA 输出的动作向量并执行。

```python
@app.vla_action(name="body", action_space=ActionSpace(3))
async def body_executor(action):
    """接收 3 维动作向量 [dx, dy, speed]"""
    await game_engine.move(action[0], action[1], action[2])
```

### @app.signal

高频信号源，周期执行并直写 SignalBus，不经过 Event 队列。

```python
@app.signal(name="vision", interval=1/30)
async def npc_vision():
    """30fps 视觉信号"""
    return capture_camera_frame()
```

### Action Channel（动作通道）

VLA 和 VLA_Action 通过通道名解耦：

```
navigation_vla ── "body" ──────→ body_executor
animation_vla  ── "body" ──────→ body_executor    ← 多 VLA 写同一通道
facial_vla     ── "face" ──────→ face_executor
speech_vla     ── "speech" ────→ speech_executor

一个 VLA 可以写多个通道:  {"body": [...], "face": [...], "speech": [...]}
一个通道可以被多个 VLA 写
```

---

## 双循环交互

### LLM → VLA 影响路径

| 模式 | 说明 | 代码 |
|------|------|------|
| **Goal Setting** | VLA 每帧读取 LLM 设定的目标 | `state["llm"]["goal"] = "去城堡"` |
| **Param Modulation** | VLA 读取 LLM 调参 | `state["llm"]["speed"] = 2.0` |
| **Override** | LLM 绕过 VLA 直接指定动作 | `state["llm"]["override_action"] = {"body": [0,0,0]}` |
| **Pause/Resume** | LLM 暂停/恢复 VLA 循环 | `state["llm"]["vla_paused"] = True` |

### VLA → LLM 影响路径

| 模式 | 说明 | 代码 |
|------|------|------|
| **Status Feedback** | VLA 上报执行状态 | `state["vla"]["status"] = "stuck"` |
| **Scene Graph** | VLA 补充视觉信息（补语言盲） | `state["vla"]["scene_graph"] = {...}` |
| **Short-term Memory** | VLA 写入短期记忆 | `state["vla"]["memory"] = [...]` |
| **Novelty Detection** | VLA 上报异常 | `state["vla"]["novelty"] = "unexpected_object"` |

---

## 完整示例：NPC 虚拟角色

见 `fastmind/examples/npc_vla.py`，演示了一个完整的 NPC 控制场景：

- **高频信号**：`@app.signal` vision 30fps + proprioception 60Hz
- **VLA 快循环**：`navigation` 30Hz 导航 + `facial` 10Hz 表情
- **VLA Action**：`body` 3维动作 + `face` 1维表情
- **LLM 慢循环**：`npc_brain` 处理玩家指令、状态监控
- **低频感知**：`@app.perception` 3秒触发重规划

运行：

```bash
python -m fastmind.examples.npc_vla
```

---

## API 参考

### FastMind 装饰器

```python
# 高频信号
@app.signal(name="vision", interval=1/30)
async def signal_func(): ...

# VLA 推理节点
@app.vla(name="navigation", frequency=30.0, input_signals=["vision"])
async def vla_func(state, signal_bus): ...

# VLA 动作执行器
@app.vla_action(name="body", action_space=ActionSpace(3))
async def action_func(action): ...
```

### FastMind 注册表

```python
app.get_signals() -> dict[str, Signal]
app.get_signal(name) -> Optional[Signal]
app.get_vlas() -> dict[str, VLAConfig]
app.get_vla(name) -> Optional[VLAConfig]
app.get_vla_actions() -> dict[str, VLAActionNode]
app.get_vla_action(name) -> Optional[VLAActionNode]
app.has_vla() -> bool
```

### FastMindAPI 新增方法

```python
api.read_signal(session_id, name) -> Optional[Any]      # 读信号
api.write_signal(session_id, name, data)                 # 写信号
api.list_signals(session_id) -> list[str]                # 列信号
api.pause_vla(session_id)                                # 暂停 VLA
api.resume_vla(session_id)                               # 恢复 VLA
```

### 数据类

```python
class ActionSpace:
    dim: int                    # 动作向量维度
    low: Optional[list[float]]  # 各维下界
    high: Optional[list[float]] # 各维上界

class VLAActionNode:
    name: str
    func: Callable
    action_space: Optional[ActionSpace]

class VLAConfig:
    name: str
    func: Callable
    frequency: float
    input_signals: list[str]
```

---

## 设计原则

1. **Event 和 Signal 是两个平等的通信元语** — 前者处理离散事件，后者处理连续信号
2. **VLA 不走 Graph** — VLA 的控制流是确定的线性步骤（感知→推理→动作），不需要条件路由
3. **快慢循环通过 Blackboard（State）通信** — 不直接调用，不传递消息，通过共享命名空间解耦
4. **Action Channel 解耦 VLA 和 VLA_Action** — 通道名匹配，支持 N:M 映射
5. **多个 VLA 各自独立频率** — 导航 30Hz、表情 10Hz、语音 5Hz，由调度器统一协调
6. **LLM 通过 state["llm/*"] 写入目标/参数/干预指令，VLA 被动读取** — 控制反转
7. **修改现有代码最小化** — 所有新增模块都是附加的，不破坏现有的 Event/Graph/Tool/Agent 流程
