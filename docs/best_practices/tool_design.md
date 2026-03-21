# 工具设计最佳实践

## 清晰的工具定义

```python
@app.tool(name="get_weather", description="获取指定城市的当前天气")
async def get_weather(city: str) -> str:
    """获取城市天气
    
    Args:
        city: 城市名称
        
    Returns:
        天气描述字符串
    """
    return f"{city} 天气晴朗"
```

## 使用类型注解

```python
@app.tool(name="calculate", description="计算数学表达式")
async def calculate(expression: str) -> str:
    return str(eval(expression))
```

## 处理异常

```python
@app.tool(name="fetch_data", description="获取数据")
async def fetch_data(url: str) -> str:
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url) as resp:
                return await resp.text()
    except Exception as e:
        return f"Error: {str(e)}"
```

## 工具调用格式

```python
state["tool_calls"] = [{
    "id": "call_1",
    "function": {
        "name": "get_weather",
        "arguments": '{"city": "北京"}'
    }
}]
```
