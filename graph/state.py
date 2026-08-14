"""
graph/state.py

Shared state schema for the LangGraph workflow. Every node reads from and
writes to this single TypedDict as it flows through the graph -- this is
the "stateful workflow" backbone LangGraph is responsible for.
"""

from typing import TypedDict, Optional


class PlatformState(TypedDict, total=False):
    # --- input ---
    topic: str
    target_keyword: str

    # --- research crew output ---
    research_brief: str

    # --- data analysis (AutoGen) output ---
    analysis_transcript: str
    chart_paths: list[str]

    # --- content crew output ---
    draft_article: str

    # --- quality control / retry loop ---
    quality_score: float          # 0.0 - 1.0, set by the quality-check node
    quality_feedback: str
    revision_count: int           # guards against infinite retry loops

    # --- human-in-the-loop ---
    human_approved: Optional[bool]
    human_feedback: Optional[str]

    # --- final output ---
    final_article: str
