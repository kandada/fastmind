# Tool Design Best Practices

## Clear Tool Definition

```python
@app.tool(name="get_weather", description="Get current weather for a specified city")
async def get_weather(city: str) -> str:
    """Get city weather
    
    Args:
        city: City name
        
    Returns:
        Weather description string
    """
    return f"{city} is sunny"
```

## Use Type Annotations

```python
@app.tool(name="calculate", description="Evaluate a mathematical expression")
async def calculate(expression: str) -> str:
    return str(eval(expression))
```

## Handle Exceptions

```python
@app.tool(name="fetch_data", description="Fetch data from URL")
async def fetch_data(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.text()
    except Exception as e:
        return f"Error: {str(e)}"
```

## Tool Call Format

```python
state["tool_calls"] = [{
    "id": "call_1",
    "function": {
        "name": "get_weather",
        "arguments": '{"city": "Beijing"}'
    }
}]
```
