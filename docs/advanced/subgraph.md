# 子图

子图允许将复杂的工作流拆分为可复用的模块。

## 定义子图

```python
# 子图定义
child_graph = Graph()
child_graph.add_node("task", task_node)
child_graph.set_entry_point("task")

# 父图使用子图
parent_graph = Graph()
parent_graph.add_node("sub_task", child_graph)
parent_graph.add_edge("start", "sub_task")
parent_graph.add_edge("sub_task", "end")
parent_graph.set_entry_point("start")
```

## 执行流程

子图执行时，会按照子图的入口点开始执行，直到子图结束，然后返回父图继续执行。
