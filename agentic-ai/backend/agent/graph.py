from langgraph.graph import StateGraph, END
from langgraph.checkpoint.base import BaseCheckpointSaver
from agent.state import AgentState
from agent.nodes import orchestrator_node, tool_node, memory_node, end_node, should_continue
from memory import mongo
from typing import Optional
from langchain_core.runnables import RunnableConfig
import asyncio

from memory.redis_checkpointer import AsyncRedisCheckpointer

graph = StateGraph(AgentState)

graph.add_node("orchestrator", orchestrator_node)
graph.add_node("tools", tool_node)
graph.add_node("memory", memory_node)
graph.add_node("end", end_node)

graph.set_entry_point("orchestrator")

graph.add_conditional_edges("orchestrator", should_continue, {
    "tools": "tools",
    "memory": "memory",
    "end": "end",
    "orchestrator": "orchestrator"
})

graph.add_edge("tools", "orchestrator")
graph.add_edge("memory", "orchestrator")
graph.add_edge("end", END)

redis_checkpointer = AsyncRedisCheckpointer()
app = graph.compile(checkpointer=redis_checkpointer)
