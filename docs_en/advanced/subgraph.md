# Subgraph

Subgraphs allow splitting complex workflows into reusable modules.

## Define Subgraph

```python
# Subgraph definition
child_graph = Graph()
child_graph.add_node("task", task_node)
child_graph.set_entry_point("task")

# Parent graph uses subgraph
parent_graph = Graph()
parent_graph.add_node("sub_task", child_graph)
parent_graph.add_edge("start", "sub_task")
parent_graph.add_edge("sub_task", "end")
parent_graph.set_entry_point("start")
```

## Execution Flow

When executing a subgraph, it starts from the subgraph's entry point and executes until the subgraph ends, then returns to the parent graph to continue execution.
