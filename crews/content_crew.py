"""
crews/content_crew.py

CrewAI content crew: writer -> SEO analyst -> editor, collaborating toward
a polished draft. Sequential process under the hood (CrewAI's "collaborate"
behavior comes from each task seeing prior tasks' context, not from a
separate collaboration mode), but framed so the SEO analyst and editor are
genuinely revising the writer's draft rather than working independently.

Invoked as a subroutine from a LangGraph node, same pattern as research_crew.
"""

from crewai import Agent, Task, Crew, Process
from config.llm_config import get_crewai_llm

llm = get_crewai_llm(temperature=0.6)  # a bit more creative for writing


def build_content_crew(research_brief: str, target_keyword: str) -> Crew:
    writer = Agent(
        role="Content Writer",
        goal="Turn a research brief into an engaging, well-structured first draft",
        backstory=(
            "You are a versatile content writer who translates research into "
            "clear, engaging prose for a general audience. You write a complete "
            "draft, not an outline."
        ),
        llm=llm,
        memory=True,
        verbose=True,
        max_iter=4,
    )

    seo_analyst = Agent(
        role="SEO Analyst",
        goal=f"Optimize the draft for the target keyword: '{target_keyword}'",
        backstory=(
            "You are an SEO specialist. You don't rewrite content wholesale -- "
            "you give specific, actionable edits: where to place the keyword "
            "naturally, what headers to add, what's missing for search intent, "
            "and where the draft is keyword-stuffed or off-target."
        ),
        llm=llm,
        memory=True,
        verbose=True,
        max_iter=4,
    )

    editor = Agent(
        role="Editor",
        goal="Produce the final polished draft incorporating SEO feedback",
        backstory=(
            "You are a meticulous editor. You take the writer's draft and the "
            "SEO analyst's recommendations and merge them into one coherent final "
            "piece -- fixing flow, tightening prose, and making sure the SEO "
            "suggestions were actually incorporated, not just listed."
        ),
        llm=llm,
        memory=True,
        verbose=True,
        max_iter=4,
    )

    write_task = Task(
        description=(
            f"Using this research brief:\n\n{research_brief}\n\n"
            f"Write ONE complete first-draft article (600-900 words). Structure:\n"
            f"1. Title (one line)\n"
            f"2. Intro (2-3 sentences)\n"
            f"3. 2-4 body sections, each with its own subheading\n"
            f"4. Short conclusion (2-3 sentences)\n"
            f"Write the full article in a single response. Do not ask questions "
            f"or produce an outline -- the article itself is the only output."
        ),
        expected_output="A complete first-draft article with title and subheadings.",
        agent=writer,
    )

    seo_task = Task(
        description=(
            f"Review the writer's draft for SEO targeting the keyword "
            f"'{target_keyword}'. Give specific edit recommendations: header "
            f"changes, keyword placement, meta description suggestion, and "
            f"anything missing for search intent. Do not rewrite the article "
            f"yourself -- give actionable recommendations."
        ),
        expected_output=(
            "A list of specific SEO recommendations plus a suggested meta description."
        ),
        agent=seo_analyst,
        context=[write_task],
    )

    edit_task = Task(
        description=(
            "Merge the writer's draft with the SEO analyst's recommendations "
            "into one final, polished article. Apply the SEO suggestions "
            "concretely (don't just append them as a list). Tighten prose, "
            "fix flow issues, keep the word count roughly the same."
        ),
        expected_output=(
            "The final polished article, ready to publish, with title, "
            "subheadings, and the suggested meta description at the top."
        ),
        agent=editor,
        context=[write_task, seo_task],
    )

    return Crew(
        agents=[writer, seo_analyst, editor],
        tasks=[write_task, seo_task, edit_task],
        process=Process.sequential,
        memory=True,
        verbose=True,
    )


def run_content_crew(research_brief: str, target_keyword: str) -> str:
    """Entry point called from the LangGraph node. Returns the final polished article."""
    crew = build_content_crew(research_brief, target_keyword)
    result = crew.kickoff()
    return str(result)


if __name__ == "__main__":
    # quick manual test: python crews/content_crew.py
    dummy_brief = (
        "Key Themes: AI agents are increasingly handling multi-step knowledge work. "
        "Implications: productivity gains, but new orchestration challenges."
    )
    print(run_content_crew(dummy_brief, target_keyword="AI agents knowledge work"))
