"""
graph/nodes.py

Each function here is a LangGraph node. Nodes are the seams where LangGraph
hands off to CrewAI or AutoGen and gets a plain string/dict back -- there is
no shared object model between the frameworks, just plain Python data
crossing the boundary.
"""

from graph.state import PlatformState
from crews.research_crew import run_research_crew
from crews.content_crew import run_content_crew
from autogen_team.data_analysis_team import run_data_analysis
from tools.sql_tool import sql_query, seed_demo_database
from config.llm_config import get_langchain_llm

MAX_REVISIONS = 2
QUALITY_THRESHOLD = 0.7


# ---------------------------------------------------------------------------
# Node 1: Research crew (CrewAI: searcher -> analyst -> fact-checker)
# ---------------------------------------------------------------------------
def research_node(state: PlatformState) -> dict:
    print("\n[LangGraph] -> Entering research_node (delegating to CrewAI research crew)")
    brief = run_research_crew(state["topic"])
    return {"research_brief": brief}


# ---------------------------------------------------------------------------
# Node 2: Data analysis (AutoGen: UserProxy + AssistantAgent in sandbox)
# ---------------------------------------------------------------------------
def data_analysis_node(state: PlatformState) -> dict:
    print("\n[LangGraph] -> Entering data_analysis_node (delegating to AutoGen pair)")

    # Pull structured numbers from the SQL tool first (LangChain tool layer),
    # then hand them to AutoGen for deeper quantitative analysis + charting.
    seed_demo_database()
    sql_result = sql_query.invoke({
        "natural_language_question": f"Show monthly mentions and sentiment trends relevant to {state['topic']}"
    })

    task = (
        f"Here is structured data relevant to the topic '{state['topic']}':\n\n"
        f"{sql_result}\n\n"
        f"Analyze the trend, compute percent change over the period, and "
        f"produce a line chart saved as 'topic_trend.png'."
    )
    result = run_data_analysis(task)
    return {
        "analysis_transcript": result["transcript"],
        "chart_paths": result["charts"],
    }


# ---------------------------------------------------------------------------
# Node 3: Content crew (CrewAI: writer -> SEO analyst -> editor)
# ---------------------------------------------------------------------------
def content_node(state: PlatformState) -> dict:
    print("\n[LangGraph] -> Entering content_node (delegating to CrewAI content crew)")
    combined_brief = (
        f"{state['research_brief']}\n\n"
        f"--- Quantitative analysis ---\n{state.get('analysis_transcript', '')[:1500]}"
    )
    draft = run_content_crew(combined_brief, state["target_keyword"])
    return {"draft_article": draft}


# ---------------------------------------------------------------------------
# Node 4: Quality check -- this is what powers the conditional retry branch
# ---------------------------------------------------------------------------
def quality_check_node(state: PlatformState) -> dict:
    print("\n[LangGraph] -> Entering quality_check_node")
    llm = get_langchain_llm(temperature=0)

    prompt = (
        "Score this article draft from 0.0 to 1.0 on overall quality "
        "(clarity, structure, depth, and whether it actually reflects the "
        "research brief). Respond in EXACTLY this format on two lines:\n"
        "SCORE: <number>\nFEEDBACK: <one or two sentence critique>\n\n"
        f"DRAFT:\n{state['draft_article']}"
    )
    response = llm.invoke(prompt).content

    score = 0.5
    feedback = "Could not parse quality score; defaulting to 0.5."
    try:
        lines = response.strip().splitlines()
        score_line = next(l for l in lines if l.upper().startswith("SCORE"))
        feedback_line = next(l for l in lines if l.upper().startswith("FEEDBACK"))
        score = float(score_line.split(":", 1)[1].strip())
        feedback = feedback_line.split(":", 1)[1].strip()
    except (StopIteration, ValueError, IndexError):
        pass

    print(f"[LangGraph]    quality_score={score}  feedback={feedback}")
    return {
        "quality_score": score,
        "quality_feedback": feedback,
        "revision_count": state.get("revision_count", 0) + 1,
    }


# ---------------------------------------------------------------------------
# Node 5: Revision node -- only reached if quality check fails AND retries remain
# ---------------------------------------------------------------------------
def revision_node(state: PlatformState) -> dict:
    print("\n[LangGraph] -> Entering revision_node (re-running content crew with feedback)")
    combined_brief = (
        f"{state['research_brief']}\n\n"
        f"--- Quantitative analysis ---\n{state.get('analysis_transcript', '')[:1500]}\n\n"
        f"--- REVISION REQUIRED: the previous draft scored "
        f"{state['quality_score']} and needs improvement. "
        f"Specific feedback to address: {state['quality_feedback']} ---"
    )
    draft = run_content_crew(combined_brief, state["target_keyword"])
    return {"draft_article": draft}


# ---------------------------------------------------------------------------
# Node 6: Human-in-the-loop approval gate
# ---------------------------------------------------------------------------
def human_approval_node(state: PlatformState) -> dict:
    """
    This node deliberately does nothing but mark that we've arrived at the
    approval gate. The actual pause happens via LangGraph's `interrupt_before`
    mechanism configured on the compiled graph (see workflow.py) -- execution
    halts BEFORE this node runs, control returns to your calling code, and
    you resume the graph after setting human_approved/human_feedback on the
    checkpointed state.
    """
    print("\n[LangGraph] -> Entering human_approval_node (post-resume)")
    return {}


# ---------------------------------------------------------------------------
# Node 7: Finalize
# ---------------------------------------------------------------------------
def finalize_node(state: PlatformState) -> dict:
    print("\n[LangGraph] -> Entering finalize_node")
    if state.get("human_approved"):
        return {"final_article": state["draft_article"]}
    else:
        note = state.get("human_feedback") or "No specific feedback given."
        return {
            "final_article": (
                f"[NOT APPROVED -- human reviewer requested changes]\n"
                f"Reviewer feedback: {note}\n\n"
                f"--- Last draft for reference ---\n{state['draft_article']}"
            )
        }


# ---------------------------------------------------------------------------
# Conditional edge function: retry-if-low-quality branch
# ---------------------------------------------------------------------------
def route_after_quality_check(state: PlatformState) -> str:
    if state["quality_score"] >= QUALITY_THRESHOLD:
        return "human_approval"
    if state.get("revision_count", 0) >= MAX_REVISIONS:
        print(f"[LangGraph]    Max revisions ({MAX_REVISIONS}) reached, proceeding anyway.")
        return "human_approval"
    return "revise"
