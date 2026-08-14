"""
graph/workflow.py

The LangGraph backbone: builds the stateful graph, wires up checkpointing
(so the run can pause and resume across a human approval gate), and defines
the conditional branch that retries content generation if the quality score
comes back too low.

Graph shape:

    research --> data_analysis --> content --> quality_check
                                                     |
                                          (score < threshold AND
                                           retries remaining)
                                              /          \\
                                           revise      human_approval
                                              \\            |
                                               \\---------> human_approval
                                                             |
                                                          finalize

`human_approval` is configured as an `interrupt_before` node: the graph
physically stops before running it, and execution resumes only when you
call .invoke() again on the same thread after injecting human_approved /
human_feedback into state.
"""

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from graph.state import PlatformState
from graph.nodes import (
    research_node,
    data_analysis_node,
    content_node,
    quality_check_node,
    revision_node,
    human_approval_node,
    finalize_node,
    route_after_quality_check,
)


def build_graph():
    graph = StateGraph(PlatformState)

    graph.add_node("research", research_node)
    graph.add_node("data_analysis", data_analysis_node)
    graph.add_node("content", content_node)
    graph.add_node("quality_check", quality_check_node)
    graph.add_node("revise", revision_node)
    graph.add_node("human_approval", human_approval_node)
    graph.add_node("finalize", finalize_node)

    graph.set_entry_point("research")
    graph.add_edge("research", "data_analysis")
    graph.add_edge("data_analysis", "content")
    graph.add_edge("content", "quality_check")

    # Conditional branch: retry-if-quality-score-is-low
    graph.add_conditional_edges(
        "quality_check",
        route_after_quality_check,
        {
            "revise": "revise",
            "human_approval": "human_approval",
        },
    )
    # After revision, go straight back through quality_check again
    graph.add_edge("revise", "quality_check")

    graph.add_edge("human_approval", "finalize")
    graph.add_edge("finalize", END)

    # Checkpointing: MemorySaver is the free, zero-setup option (in-process,
    # lost on restart). For persistence across restarts, swap in
    # langgraph.checkpoint.sqlite.SqliteSaver pointed at a local file --
    # still free, still no external service.
    checkpointer = MemorySaver()

    compiled = graph.compile(
        checkpointer=checkpointer,
        interrupt_before=["human_approval"],
    )
    return compiled


if __name__ == "__main__":
    # quick manual test: python graph/workflow.py
    app = build_graph()
    config = {"configurable": {"thread_id": "demo-thread-1"}}

    initial_state = {
        "topic": "the rise of multi-agent AI systems in 2026",
        "target_keyword": "multi-agent AI systems",
        "revision_count": 0,
    }

    # First call runs research -> data_analysis -> content -> quality_check
    # -> (maybe revise/quality_check again) -> stops before human_approval
    result = app.invoke(initial_state, config=config)
    print("\n=== PAUSED FOR HUMAN APPROVAL ===")
    print("Draft quality score:", result.get("quality_score"))
    print("Draft preview:", result.get("draft_article", "")[:500])

    # Simulate a human reviewing and approving
    app.update_state(config, {"human_approved": True, "human_feedback": "Looks good."})
    final_result = app.invoke(None, config=config)
    print("\n=== FINAL ARTICLE ===")
    print(final_result["final_article"])
