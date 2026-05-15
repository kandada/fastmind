# VLA Integration Guide

## Overview

VLA (Vision-Language-Action) models produce **continuous actions directly from visual input**. Typical applications include robot control, virtual world NPCs, autonomous driving, etc. VLA models run at high frequency (10-100Hz) in a closed loop (Observe → Reason → Act), but suffer from language blindness and catastrophic forgetting.

FastMind's VLA integration provides a **dual-loop architecture**:

```
VLA Fast Loop (time-driven, 30Hz):  continuous perception → inference → action
LLM Slow Loop (event-driven):        planning → reasoning → language interaction
```

Both loops run concurrently in the same Session, communicating through shared State (Blackboard pattern), without blocking each other.

---

## Core Concepts

### Event vs Signal

FastMind provides two parallel communication primitives:

| | Event | Signal |
|--|-------|--------|
| **Philosophy** | "Something happened" | "Current value is" |
| **Data structure** | Queue (FIFO) | Last-value cache |
| **Consumption** | Push-based | Pull-based |
| **Semantics** | Discrete, ordered | Continuous, overwrites |
| **Triggers graph** | Yes | No (zero overhead) |
| **Best for** | User messages, LLM replies, interrupts | Camera frames, joint angles, NPC vision |

### @app.vla

VLA inference node, scheduled by the framework at a fixed frequency — **does NOT go through the Graph**.

```python
@app.vla(name="navigation", frequency=30.0)
async def navigation_vla(state, signal_bus):
    """Signature: (state, signal_bus) -> dict[str, list[float]]
    Returns:  { action_channel_name: action_vector }
    """
    vision = signal_bus.read("vision")
    goal = state.get("llm", {}).get("goal", "idle")
    action = await vla_model.predict(image=vision, instruction=goal)
    return {"body": action}   # "body" is the channel name
```

Key points:
- Bypasses Event queue and Graph traversal entirely
- Framework schedules at `frequency` automatically
- Returns a dict mapping channel names to action vectors
- Can read `state["llm/*"]` — influenced by the slow loop

### @app.vla_action

Action executor that receives and executes VLA output action vectors.

```python
@app.vla_action(name="body", action_space=ActionSpace(3))
async def body_executor(action):
    """Receives a 3-DoF action vector [dx, dy, speed]"""
    await game_engine.move(action[0], action[1], action[2])
```

### @app.signal

High-frequency signal source. Writes directly to SignalBus (last-value cache), bypassing the Event queue.

```python
@app.signal(name="vision", interval=1/30)
async def npc_vision():
    """30fps vision signal"""
    return capture_camera_frame()
```

### Action Channel

VLA and VLA_Action are decoupled by channel names:

```
navigation_vla ── "body" ──────→ body_executor
animation_vla  ── "body" ──────→ body_executor    ← multiple VLAs → same channel
facial_vla     ── "face" ──────→ face_executor
speech_vla     ── "speech" ────→ speech_executor

One VLA can write to multiple channels:  {"body": [...], "face": [...], "speech": [...]}
One channel can be written by multiple VLAs
```

---

## Dual-Loop Interaction

### LLM → VLA Pathways

| Pattern | Description | Code |
|---------|-------------|------|
| **Goal Setting** | VLA reads LLM's goal every frame | `state["llm"]["goal"] = "go_to_castle"` |
| **Param Modulation** | VLA reads LLM parameters | `state["llm"]["speed"] = 2.0` |
| **Override** | LLM bypasses VLA entirely | `state["llm"]["override_action"] = {"body": [0,0,0]}` |
| **Pause/Resume** | LLM pauses/resumes VLA loop | `state["llm"]["vla_paused"] = True` |

### VLA → LLM Pathways

| Pattern | Description | Code |
|---------|-------------|------|
| **Status Feedback** | VLA reports execution status | `state["vla"]["status"] = "stuck"` |
| **Scene Graph** | VLA provides visual context | `state["vla"]["scene_graph"] = {...}` |
| **Short-term Memory** | VLA writes action history | `state["vla"]["memory"] = [...]` |
| **Novelty Detection** | VLA reports anomalies | `state["vla"]["novelty"] = "unexpected_object"` |

---

## Complete Example: Virtual NPC

See `fastmind/examples/npc_vla.py` for a complete NPC control scenario:

- **High-frequency signals**: `@app.signal` vision 30fps + proprioception 60Hz
- **VLA fast loop**: `navigation` 30Hz + `facial` 10Hz
- **VLA Actions**: `body` 3-DoF + `face` 1-DoF expression
- **LLM slow loop**: `npc_brain` handles player commands and status monitoring
- **Low-frequency perception**: `@app.perception` triggers replanning every 3s

```bash
python -m fastmind.examples.npc_vla
```

---

## API Reference

### FastMind Decorators

```python
# High-frequency signal
@app.signal(name="vision", interval=1/30)
async def signal_func(): ...

# VLA inference node
@app.vla(name="navigation", frequency=30.0, input_signals=["vision"])
async def vla_func(state, signal_bus): ...

# VLA action executor
@app.vla_action(name="body", action_space=ActionSpace(3))
async def action_func(action): ...
```

### FastMind Registry

```python
app.get_signals() -> dict[str, Signal]
app.get_signal(name) -> Optional[Signal]
app.get_vlas() -> dict[str, VLAConfig]
app.get_vla(name) -> Optional[VLAConfig]
app.get_vla_actions() -> dict[str, VLAActionNode]
app.get_vla_action(name) -> Optional[VLAActionNode]
app.has_vla() -> bool
```

### FastMindAPI New Methods

```python
api.read_signal(session_id, name) -> Optional[Any]      # read signal
api.write_signal(session_id, name, data)                 # write signal
api.list_signals(session_id) -> list[str]                # list signals
api.pause_vla(session_id)                                # pause VLA
api.resume_vla(session_id)                               # resume VLA
```

### Data Classes

```python
class ActionSpace:
    dim: int                    # action vector dimension
    low: Optional[list[float]]  # lower bounds per dimension
    high: Optional[list[float]] # upper bounds per dimension

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

## Design Principles

1. **Event and Signal are two equal primitives** — Event for discrete messages, Signal for continuous data
2. **VLA does NOT go through the Graph** — VLA control flow is a deterministic linear pipeline (sense→infer→act), no conditional routing needed
3. **Dual loops communicate via Blackboard (State)** — no direct calls, no message passing, just shared namespaces
4. **Action Channel decouples VLA from VLA_Action** — channel name matching, supports N:M mapping
5. **Multiple VLAs with independent frequencies** — navigation 30Hz, facial 10Hz, speech 5Hz, coordinated by scheduler
6. **LLM writes to state["llm/*"], VLA reads passively** — inversion of control
7. **Minimal changes to existing code** — all new modules are additive, zero breakage to existing Event/Graph/Tool/Agent workflow
