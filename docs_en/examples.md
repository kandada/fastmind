# Examples

This directory contains various examples of the FastMind framework that can be run directly.

## Basic Examples

### simple_chat.py

Simple chat example demonstrating the most basic Agent usage.

```bash
PYTHONPATH=. python3 examples/simple_chat.py
```

### simple_chat_with_tool.py

Chat with tool calling example demonstrating Agent-Tool-Agent ReAct loop.

```bash
PYTHONPATH=. python3 examples/simple_chat_with_tool.py
```

### streaming_chat.py

Streaming output example demonstrating how to handle `stream.chunk` and `stream.end` events.

```bash
PYTHONPATH=. python3 examples/streaming_chat.py
```

## Advanced Examples

### human_in_loop.py

Human-in-the-loop example demonstrating interruption and resume mechanism.

```bash
PYTHONPATH=. python3 examples/human_in_loop.py
```

### perception_loop.py

Perception loop example demonstrating timer-based perception and event routing.

```bash
PYTHONPATH=. python3 examples/perception_loop.py
```

## Application Examples

### humanoid_robot.py

Humanoid robot example demonstrating multi-tool collaboration and voice interaction.

```bash
PYTHONPATH=. python3 examples/humanoid_robot.py
```

### companion_bot.py

Companion bot example demonstrating emotion recognition and complex conversation.

```bash
PYTHONPATH=. python3 examples/companion_bot.py
```

### sleep_assessment.py

Sleep assessment example demonstrating multi-state management and HITL flow.

```bash
PYTHONPATH=. python3 examples/sleep_assessment.py
```

### drone.py

Drone example demonstrating timer-based perception and flight control.

```bash
PYTHONPATH=. python3 examples/drone.py
```

### comprehensive_assistant.py

Full-featured assistant example demonstrating an integrated assistant with tool calling, streaming output, and human-in-the-loop.

```bash
PYTHONPATH=. python3 examples/comprehensive_assistant.py
```
