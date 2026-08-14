"""
main.py

Run this file in VS Code (or `python main.py` in a terminal) to execute the
full pipeline end to end:

    LangGraph orchestrates ->
        CrewAI research crew (searcher -> analyst -> fact-checker)
        AutoGen data analysis pair (UserProxy + AssistantAgent, sandboxed)
        CrewAI content crew (writer -> SEO analyst -> editor)
        quality check -> conditional retry loop
        human-in-the-loop approval gate (pauses here for real terminal input)
        finalize

Before running:
    1. python -m venv venv && source venv/bin/activate  (or venv\\Scripts\\activate on Windows)
    2. pip install -r requirements.txt
    3. cp .env.example .env   and fill in GROQ_API_KEY (free, see .env.example)
    4. python main.py
"""

import os
from rich.console import Console
from rich.markdown import Markdown

from graph.workflow import build_graph

console = Console()


def main():
    app = build_graph()
    config = {"configurable": {"thread_id": "platform-run-1"}}

    console.print("\n[bold cyan]=== Autonomous AI Research & Content Platform ===[/bold cyan]\n")

    topic = input("Enter a research topic: ").strip() or "the rise of multi-agent AI systems in 2026"
    keyword = input("Enter the target SEO keyword: ").strip() or topic

    initial_state = {
        "topic": topic,
        "target_keyword": keyword,
        "revision_count": 0,
    }

    console.print(f"\n[yellow]Starting pipeline for topic:[/yellow] {topic}\n")

    # Runs research -> data_analysis -> content -> quality_check loop,
    # and stops automatically right before the human_approval node.
    result = app.invoke(initial_state, config=config)

    console.print("\n[bold magenta]=== PIPELINE PAUSED: HUMAN APPROVAL REQUIRED ===[/bold magenta]")
    console.print(f"[green]Quality score:[/green] {result.get('quality_score')}")
    console.print(f"[green]Quality feedback:[/green] {result.get('quality_feedback')}")
    if result.get("chart_paths"):
        console.print(f"[green]Charts generated:[/green] {result['chart_paths']}")

    console.print("\n[bold]--- Draft Article ---[/bold]")
    console.print(Markdown(result.get("draft_article", "(no draft produced)")))

    decision = input("\nApprove this article? [y/n]: ").strip().lower()
    feedback = ""
    if decision != "y":
        feedback = input("Feedback for what should change: ").strip()

    # Inject the human decision into the checkpointed state, then resume.
    app.update_state(
        config,
        {"human_approved": decision == "y", "human_feedback": feedback},
    )

    final_result = app.invoke(None, config=config)

    console.print("\n[bold cyan]=== FINAL OUTPUT ===[/bold cyan]")
    console.print(Markdown(final_result["final_article"]))

    os.makedirs("outputs", exist_ok=True)
    out_path = os.path.join("outputs", "final_article.md")
    with open(out_path, "w") as f:
        f.write(final_result["final_article"])
    console.print(f"\n[green]Saved final article to {out_path}[/green]")


if __name__ == "__main__":
    main()
