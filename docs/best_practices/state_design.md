# State 设计最佳实践

## 使用 setdefault 初始化

```python
async def agent(state, event):
    state.setdefault("messages", [])
    state.setdefault("count", 0)
    state["messages"].append(event.payload["text"])
    state["count"] += 1
    return state
```

## 避免直接修改深层嵌套

```python
# 不好
state["user"]["profile"]["name"] = "Alice"

# 好
state.setdefault("user", {})["profile"] = {"name": "Alice"}
```

## 使用约定 key

推荐使用约定的 key 名称，便于工具和框架交互：

- `messages`: 对话历史
- `tool_calls`: 待执行的工具调用
- `tool_results`: 工具执行结果
