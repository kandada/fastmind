# State Design Best Practices

## Use setdefault for Initialization

```python
async def agent(state, event):
    state.setdefault("messages", [])
    state.setdefault("count", 0)
    state["messages"].append(event.payload["text"])
    state["count"] += 1
    return state
```

## Avoid Directly Modifying Deep Nesting

```python
# Bad
state["user"]["profile"]["name"] = "Alice"

# Good
state.setdefault("user", {})["profile"] = {"name": "Alice"}
```

## Use State API Convenience Methods

```python
# Add message
state.add_message("user", "Hello")

# Get last message
last = state.get_last_message()

# Pop all messages
all_msgs = state.pop_messages()

# Get message count
count = state.get_message_count()
```

## Use Conventional Keys

Recommended to use conventional key names for easy interaction with tools and framework:

- `messages`: Conversation history
- `tool_calls`: Pending tool calls
- `tool_results`: Tool execution results
