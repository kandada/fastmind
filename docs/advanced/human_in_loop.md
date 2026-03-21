# Human-in-the-loop

Human-in-the-loop (HITL) 允许在关键决策点暂停执行，等待人工确认。

## 中断节点

```python
async def ask_confirm(state, event):
    return state, [Event(
        type="interrupt",
        payload={
            "prompt": "确认执行转账？",
            "resume_node": "execute",
            "cancel_node": "cancel"
        },
        session_id=event.session_id
    )]
```

## 条件触发

```python
def need_approval(state, event):
    return "ask_confirm" if state.get("amount", 0) > 1000 else None

graph.add_conditional_edges("process", need_approval, {
    "ask_confirm": "ask_confirm",
    None: "__end__"
})
```

## 恢复执行

```python
api = FastMindAPI(app)
await api.start()

# 触发中断流程
await api.push_event("user1", Event("transfer", {"amount": 2000}, "user1"))

# 监听中断
async for ev in api.stream_events("user1"):
    if ev.type == "interrupt":
        print(f"确认: {ev.payload['prompt']}")
        await api.resume_session("user1", "confirm")
```
