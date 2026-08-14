"""
crews/research_crew.py

CrewAI research crew: searcher -> analyst -> fact-checker, sequential
process, with memory enabled so each agent retains context from earlier
runs within the crew's short-term memory store.

This is invoked as a subroutine from a LangGraph node (see graph/workflow.py)
-- it is NOT itself a LangGraph node. It takes a topic in, returns a
structured research brief out.
"""

from crewai import Agent, Task, Crew, Process
from config.llm_config import get_crewai_llm
from tools.search_tool import web_search
from tools.rag_tool import rag_query

llm = get_crewai_llm(temperature=0.3)


def build_research_crew(topic: str) -> Crew:
    searcher = Agent(
        role="Research Searcher",
        goal=f"Find the most relevant, current information about: {topic}",
        backstory=(
            "You are a meticulous research scout. You scour the web and the "
            "existing knowledge base for raw material, and you never editorialize "
            "-- you just gather facts and sources for others to analyze. You are "
            "decisive: once you have enough material, you stop searching and "
            "report your findings immediately."
        ),
        tools=[web_search, rag_query],
        llm=llm,
        memory=True,
        verbose=True,
        max_iter=3,   # hard cap: this model tends to loop re-deciding whether
                       # it has "enough" info rather than recognizing it does.
                       # 3 tool calls is enough for 4-5 facts; force a stop after.
    )

    analyst = Agent(
        role="Research Analyst",
        goal="Synthesize raw research into clear, structured insights",
        backstory=(
            "You are a sharp analyst who takes messy raw research and finds the "
            "signal: key themes, trends, contradictions, and what actually matters. "
            "You build on what the searcher found rather than re-researching from scratch."
        ),
        llm=llm,
        memory=True,
        verbose=True,
        max_iter=4,
    )

    fact_checker = Agent(
        role="Fact Checker",
        goal="Verify claims in the analysis using only the existing findings",
        backstory=(
            "You are a skeptical fact-checker. You cross-reference claims made by "
            "the analyst against the original sources the searcher found, and you "
            "flag (rather than silently fix) anything that looks unsupported, "
            "outdated, or overstated. You work entirely from the research and "
            "analysis already provided to you -- you never need to search "
            "independently, because everything required is already in context."
        ),
        # No tools given here intentionally. Small local models are unreliable
        # at the ReAct format (Thought/Action/Final Answer as strictly separate
        # blocks) once a tool is in play -- they tend to either skip the Action
        # name or try to act and answer in the same message. Since this task
        # only needs the searcher's and analyst's outputs (already passed via
        # `context=`), removing the tool removes the failure mode entirely
        # rather than trying to instruct around it.
        llm=llm,
        memory=True,
        verbose=True,
        max_iter=2,
    )

    search_task = Task(
        description=(
            f"Research the topic: '{topic}'. Follow these exact steps:\n"
            f"1. Call rag_query ONCE to check the local knowledge base.\n"
            f"2. Call web_search ONCE with a clear query about the topic.\n"
            f"3. STOP making tool calls. You now have enough information.\n"
            f"4. Write your Final Answer immediately: a bullet list of 4-5 "
            f"distinct facts from what you already retrieved, each with a "
            f"one-line source attribution.\n"
            f"Do not call the same tool with the same or similar input twice. "
            f"Do not analyze the findings -- just list them."
        ),
        expected_output=(
            "A bullet list of raw findings, each with a one-line source attribution."
        ),
        agent=searcher,
    )

    analysis_task = Task(
        description=(
            "Using the searcher's raw findings (in context), identify the 3-5 "
            "most important themes or trends. Note any contradictions between "
            "sources. Produce a structured analysis, not a restatement of facts."
        ),
        expected_output=(
            "A structured analysis with sections: Key Themes, Notable Contradictions "
            "(if any), and Implications."
        ),
        agent=analyst,
        context=[search_task],
    )

    fact_check_task = Task(
        description=(
            "Review the analyst's claims against the searcher's original findings "
            "(both already provided to you above). For each major claim, mark it "
            "Verified, Unsupported, or Needs More Evidence, based only on whether "
            "the searcher's findings actually support it. Do not use any tools -- "
            "everything you need is already in the text above."
        ),
        expected_output=(
            "A verification table/list: claim -> status -> brief justification. "
            "End with an overall confidence rating (High/Medium/Low) for the research brief."
        ),
        agent=fact_checker,
        context=[search_task, analysis_task],
    )

    return Crew(
        agents=[searcher, analyst, fact_checker],
        tasks=[search_task, analysis_task, fact_check_task],
        process=Process.sequential,
        memory=True,
        verbose=True,
    )


def run_research_crew(topic: str) -> str:
    """Entry point called from the LangGraph node. Returns the final fact-checked brief."""
    crew = build_research_crew(topic)
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    # quick manual test: python crews/research_crew.py
    print(run_research_crew("the impact of AI agents on knowledge work in 2026"))
