from __future__ import annotations

from langgraph.graph import END, StateGraph

from src.pipeline.nodes import (
    generate_traces,
    human_checkpoint,
    log_kl_metrics,
    rank_and_pair,
    score_all,
    update_model,
)
from src.pipeline.state import PipelineState


def build_graph():
    graph = StateGraph(PipelineState)
    graph.add_node("generate_traces", generate_traces)
    graph.add_node("score_all", score_all)
    graph.add_node("rank_and_pair", rank_and_pair)
    graph.add_node("update_model", update_model)
    graph.add_node("log_kl_metrics", log_kl_metrics)
    graph.add_node("human_checkpoint", human_checkpoint)

    graph.set_entry_point("generate_traces")
    graph.add_edge("generate_traces", "score_all")
    graph.add_edge("score_all", "rank_and_pair")
    graph.add_edge("rank_and_pair", "update_model")
    graph.add_edge("update_model", "log_kl_metrics")
    graph.add_edge("log_kl_metrics", "human_checkpoint")
    graph.add_edge("human_checkpoint", END)

    return graph.compile()
