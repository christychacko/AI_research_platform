"""
autogen_team/data_analysis_team.py

AutoGen pair: a UserProxyAgent that executes code in a sandboxed local
working directory, and an AssistantAgent that writes Python to analyze
data, produce charts, and surface quantitative insights.

"Sandboxed" here means a dedicated, isolated working directory
(autogen_team/sandbox/) where generated code actually runs -- it is NOT
a Docker container (Docker requires a daemon most VS Code setups don't
have configured, which would break "just run this" for most readers).
If you want real OS-level isolation, see the docker note at the bottom
of this file.

This is invoked as a subroutine from a LangGraph node, same pattern as
the CrewAI crews -- it is not itself a LangGraph node.
"""

import os
import warnings

# pyautogen optionally depends on flaml[automl] for AutoML features we don't
# use here (we only use AssistantAgent/UserProxyAgent for code-gen/execution).
# This warning is harmless and safe to suppress.
warnings.filterwarnings("ignore", message="flaml.automl is not available")

import autogen
from config.llm_config import get_autogen_llm_config

SANDBOX_DIR = os.path.join(os.path.dirname(__file__), "sandbox")
os.makedirs(SANDBOX_DIR, exist_ok=True)

llm_config = get_autogen_llm_config(temperature=0.2)


def build_data_analysis_team():
    assistant = autogen.AssistantAgent(
        name="DataAnalyst",
        llm_config=llm_config,
        system_message=(
            "You are a data analyst working in a back-and-forth conversation "
            "with a code executor. This takes TWO separate messages, never one:\n\n"
            "MESSAGE 1 (your first reply): write Python code only (pandas/"
            "matplotlib/numpy only) to load the data, compute the requested "
            "statistic, and save one chart as a PNG file. Wrap it in a single "
            "```python code block. Do NOT include the word TERMINATE in this "
            "message -- you have not seen the result yet, so you cannot be done.\n\n"
            "MESSAGE 2 (after the executor replies with the code's output): "
            "read the actual execution output you were just shown. If it shows "
            "an error, fix the code and send a new code block (no TERMINATE yet). "
            "If it shows success, summarize the quantitative insight in 2-3 "
            "plain-English sentences, THEN end that message with the word "
            "TERMINATE.\n\n"
            "Never put code and TERMINATE in the same message -- TERMINATE only "
            "ever appears after you have seen a real execution result."
        ),
    )

    user_proxy = autogen.UserProxyAgent(
        name="CodeExecutor",
        human_input_mode="NEVER",
        max_consecutive_auto_reply=4,  # one failed attempt + one retry should
                                        # be enough for this scoped task; a small
                                        # model that hasn't converged by then is
                                        # unlikely to converge with more rounds
        # Only treat a message as termination if it both contains TERMINATE
        # AND does not itself contain a code block -- this prevents the model
        # from ending the chat in the same turn it submits unexecuted code.
        is_termination_msg=lambda msg: (
            "TERMINATE" in (msg.get("content") or "")
            and "```" not in (msg.get("content") or "")
        ),
        code_execution_config={
            "work_dir": SANDBOX_DIR,
            "use_docker": False,  # set True if you have Docker running, for real isolation
        },
    )

    return assistant, user_proxy


def run_data_analysis(task_description: str, csv_data: str | None = None) -> dict:
    """
    Entry point called from the LangGraph node.

    Args:
        task_description: what analysis to perform, in plain English
        csv_data: optional raw CSV text to write to sandbox/input_data.csv
                  before the agents start, so the assistant has something
                  concrete to load with pandas.

    Returns:
        dict with keys: "transcript" (full conversation) and
        "charts" (list of PNG filepaths produced, if any).
    """
    if csv_data:
        input_path = os.path.join(SANDBOX_DIR, "input_data.csv")
        with open(input_path, "w") as f:
            f.write(csv_data)

        # Tell the model the EXACT column names AND the exact distinct values
        # in each column, instead of letting it guess either. A small model
        # will confidently hallucinate plausible-sounding names/values it has
        # seen in similar contexts (e.g. month values "Jan"/"Mar" instead of
        # the real "2026-01"/"2026-03", or column names "mentions_per_topic"
        # instead of "mentions") -- both cause a silent KeyError at runtime
        # that the model can't reason its way out of, because nothing is
        # actually wrong with its *logic*, only with values it invented.
        import csv
        import io

        reader = csv.DictReader(io.StringIO(csv_data))
        fieldnames = reader.fieldnames or []
        rows = list(reader)

        column_value_lines = []
        for col in fieldnames:
            distinct_vals = list(dict.fromkeys(row[col] for row in rows))
            # Only show distinct values for columns that look categorical
            # (few unique values) -- showing every value of a numeric column
            # would bloat the prompt for no benefit.
            if len(distinct_vals) <= 10:
                column_value_lines.append(f"  - '{col}' contains exactly these values: {distinct_vals}")
            else:
                column_value_lines.append(f"  - '{col}' is numeric/continuous (not categorical)")

        schema_block = "\n".join(column_value_lines)
        task_description += (
            f"\n\nThe data is available at './input_data.csv' (relative to "
            f"your working directory). Load it with pandas. It has EXACTLY "
            f"these columns: {fieldnames}. Their actual contents:\n"
            f"{schema_block}\n"
            f"Use these exact column names and exact values in your code. "
            f"Do not guess, abbreviate, or invent values that aren't listed "
            f"above (for example, if month values are '2026-01', '2026-02', "
            f"'2026-03', use those exact strings -- not 'Jan'/'Feb'/'Mar')."
        )

    assistant, user_proxy = build_data_analysis_team()

    chat_result = user_proxy.initiate_chat(
        assistant,
        message=task_description,
    )

    # collect any PNGs the assistant generated during this run
    charts = [
        os.path.join(SANDBOX_DIR, f)
        for f in os.listdir(SANDBOX_DIR)
        if f.lower().endswith(".png")
    ]

    transcript = "\n\n".join(
        f"[{m.get('name', m.get('role'))}]: {m.get('content')}"
        for m in chat_result.chat_history
    )

    if not charts:
        # The small local model may hit max_consecutive_auto_reply without
        # ever producing a working chart. Make that visible instead of
        # silently passing an empty chart list downstream.
        transcript += (
            "\n\n[NOTE: No chart PNG was found after this run. The analysis "
            "may have stalled before completing -- check the transcript above "
            "for errors, or consider re-running.]"
        )

    return {"transcript": transcript, "charts": charts}


if __name__ == "__main__":
    # quick manual test: python autogen_team/data_analysis_team.py
    sample_csv = (
        "topic,month,mentions,sentiment_score\n"
        "AI Agents,2026-01,1200,0.62\n"
        "AI Agents,2026-02,1850,0.58\n"
        "AI Agents,2026-03,2400,0.65\n"
        "Vector Databases,2026-01,400,0.71\n"
        "Vector Databases,2026-02,520,0.69\n"
        "Vector Databases,2026-03,610,0.73\n"
    )
    result = run_data_analysis(
        "Analyze the monthly trend in mentions per topic, compute the percent "
        "growth from the first month to the second month for each topic, and "
        "save a line chart comparing the two topics as trend_chart.png.",
        csv_data=sample_csv,
    )
    print(result["transcript"])
    print("Charts produced:", result["charts"])

# --- Optional real sandboxing note -----------------------------------------
# To get actual OS-level isolation instead of "just a folder", install Docker
# Desktop, make sure the daemon is running, then set use_docker=True above.
# AutoGen will run generated code inside a throwaway container per execution.
