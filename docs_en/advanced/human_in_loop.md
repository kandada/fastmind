# Human-in-the-Loop

Human-in-the-loop (HITL) allows pausing execution at critical decision points for human confirmation.

## Interrupt Node

```python
async def ask_confirm(state, event):
    return state, [Event(
        type="interrupt",
        payload={
            "prompt": "Confirm transfer?",
            "resume_node": "execute",
            "cancel_node": "cancel"
        },
        session_id=event.session_id
    )]
```

## Conditional Trigger

```python
def need_approval(state, event):
    return "ask_confirm" if state.get("amount", 0) > 1000 else None

graph.add_conditional_edges("process", need_approval, {
    "ask_confirm": "ask_confirm",
    None: "__end__"
})
```

## Resume Execution

```python
api = FastMindAPI(app)
await api.start()

# Trigger interrupt flow
await api.push_event("user1", Event("transfer", {"amount": 2000}, "user1"))

# Listen for interrupt
async for ev in api.stream_events("user1"):
    if ev.type == "interrupt":
        print(f"Confirm: {ev.payload['prompt']}")
        await api.resume_session("user1", "confirm")
```
