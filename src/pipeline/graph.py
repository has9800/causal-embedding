from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.pipeline.nodes import (
    compute_reward,
    critic_evaluate,
    extract_embeddings,
    generate_trace,
    human_checkpoint,
    local_filter,
    probe_score,
    update_model,
)
from src.pipeline.state import PipelineState


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("generate_trace", generate_trace)
    graph.add_node("local_filter", local_filter)
    graph.add_node("critic_evaluate", critic_evaluate)
    graph.add_node("extract_embeddings", extract_embeddings)
    graph.add_node("probe_score", probe_score)
    graph.add_node("compute_reward", compute_reward)
    graph.add_node("update_model", update_model)
    graph.add_node("human_checkpoint", human_checkpoint)

    graph.set_entry_point("generate_trace")
    graph.add_edge("generate_trace", "local_filter")
    graph.add_edge("local_filter", "critic_evaluate")
    graph.add_edge("critic_evaluate", "extract_embeddings")
    graph.add_edge("extract_embeddings", "probe_score")
    graph.add_edge("probe_score", "compute_reward")
    graph.add_edge("compute_reward", "update_model")
    graph.add_edge("update_model", "human_checkpoint")

    graph.add_conditional_edges(
        "human_checkpoint",
        lambda s: "end" if s.stop else "loop",
        {
            "loop": "generate_trace",
            "end": END,
        },
    )
    return graph.compile()
