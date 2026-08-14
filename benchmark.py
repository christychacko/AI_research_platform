"""
benchmark.py

Runs the full pipeline N times and measures three things per run:
  1. SPEED    -- how long each node took (research, data_analysis, content,
                 quality_check, revise if triggered)
  2. RELIABILITY -- did the run crash? which node? how many quality-check
                 retries did it need before passing?
  3. QUALITY  -- the final quality_score, draft word count, and whether it
                 cleared QUALITY_THRESHOLD without hitting MAX_REVISIONS

Usage:
    python benchmark.py                       # 3 runs, default topic
    python benchmark.py --runs 5
    python benchmark.py --runs 5 --topic "AI agents in healthcare"

Output:
    outputs/benchmark_results.json   (raw data, every run)
    outputs/benchmark_report.md      (human-readable summary)

Design note: this benchmark auto-approves the human-in-the-loop gate (so it
can run unattended). That's the right call for measuring the AUTOMATED part
of the pipeline -- the human approval step has no "speed" or "quality" of
its own to measure, since it's just waiting on you.
"""

import argparse
import json
import os
import statistics
import time
import traceback
from datetime import datetime, timezone

from graph import nodes as nodes_module

OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")


# ---------------------------------------------------------------------------
# Step 1: wrap each node function so every call records its own timing and
# any exception, WITHOUT changing what the node actually does.
#
# IMPORTANT ordering note: this wrapping MUST happen, and `build_graph` MUST
# be imported, only AFTER the wrapping is installed on the nodes module.
# workflow.py does `from graph.nodes import research_node, ...` -- that line
# captures a direct reference to whatever function object exists on the
# nodes module AT THE MOMENT IT RUNS. If we imported workflow first and
# patched nodes_module afterward, workflow's captured references would
# still point at the original, unwrapped functions, and this benchmark
# would silently record nothing. Patching first, then importing
# graph.workflow, is what makes workflow.py's import line pick up the
# wrapped versions instead.
# ---------------------------------------------------------------------------
_timing_log = []   # list of {"node": str, "seconds": float, "error": str|None}


def _wrap_node(name, fn):
    def wrapped(state):
        start = time.monotonic()
        try:
            result = fn(state)
            elapsed = time.monotonic() - start
            _timing_log.append({"node": name, "seconds": elapsed, "error": None})
            return result
        except Exception as e:
            elapsed = time.monotonic() - start
            _timing_log.append({
                "node": name,
                "seconds": elapsed,
                "error": f"{type(e).__name__}: {e}",
            })
            raise  # re-raise so the graph run still fails the way it normally would
    return wrapped


def _install_timed_nodes():
    for name in ["research_node", "data_analysis_node", "content_node",
                 "quality_check_node", "revision_node"]:
        original = getattr(nodes_module, name)
        setattr(nodes_module, name, _wrap_node(name, original))


# Patch BEFORE importing build_graph (see ordering note above).
_install_timed_nodes()
from graph.workflow import build_graph  # noqa: E402  (intentionally delayed import)


# ---------------------------------------------------------------------------
# Step 2: run the pipeline once, fully unattended (auto-approve at the
# human gate), and collect everything the benchmark cares about.
# ---------------------------------------------------------------------------
def run_once(topic: str, keyword: str, run_index: int) -> dict:
    global _timing_log
    _timing_log = []

    app = build_graph()
    config = {"configurable": {"thread_id": f"benchmark-run-{run_index}-{time.time()}"}}

    initial_state = {"topic": topic, "target_keyword": keyword, "revision_count": 0}

    run_record = {
        "run_index": run_index,
        "topic": topic,
        "crashed": False,
        "crash_node": None,
        "crash_error": None,
        "node_timings": [],
        "quality_score": None,
        "revision_count": None,
        "draft_word_count": None,
        "passed_quality_threshold": None,
        "total_seconds": None,
    }

    overall_start = time.monotonic()
    try:
        result = app.invoke(initial_state, config=config)

        # Auto-approve so the run completes without waiting on a human --
        # this benchmark measures the automated pipeline, not your reaction time.
        app.update_state(config, {"human_approved": True, "human_feedback": "auto-approved by benchmark"})
        final_result = app.invoke(None, config=config)

        run_record["quality_score"] = final_result.get("quality_score")
        run_record["revision_count"] = final_result.get("revision_count")
        draft = final_result.get("draft_article", "") or ""
        run_record["draft_word_count"] = len(draft.split())
        run_record["passed_quality_threshold"] = (
            (final_result.get("quality_score") or 0) >= nodes_module.QUALITY_THRESHOLD
        )

    except Exception as e:
        run_record["crashed"] = True
        run_record["crash_error"] = f"{type(e).__name__}: {e}"
        # the last entry in _timing_log is the node that was running when it crashed
        if _timing_log:
            run_record["crash_node"] = _timing_log[-1]["node"]
        print(f"  [run {run_index}] CRASHED: {run_record['crash_error']}")
        traceback.print_exc()

    run_record["total_seconds"] = time.monotonic() - overall_start
    run_record["node_timings"] = list(_timing_log)  # copy out before next run resets it
    return run_record


# ---------------------------------------------------------------------------
# Step 3: aggregate N runs into summary statistics.
# ---------------------------------------------------------------------------
def aggregate(runs: list[dict]) -> dict:
    n = len(runs)
    crashed = [r for r in runs if r["crashed"]]
    completed = [r for r in runs if not r["crashed"]]

    # per-node timing stats, pooled across all runs that reached that node
    per_node_times = {}
    for r in runs:
        for t in r["node_timings"]:
            per_node_times.setdefault(t["node"], []).append(t["seconds"])

    node_stats = {}
    for node_name, times in per_node_times.items():
        node_stats[node_name] = {
            "calls": len(times),
            "avg_seconds": round(statistics.mean(times), 1),
            "min_seconds": round(min(times), 1),
            "max_seconds": round(max(times), 1),
        }

    scores = [r["quality_score"] for r in completed if r["quality_score"] is not None]
    word_counts = [r["draft_word_count"] for r in completed if r["draft_word_count"] is not None]
    revisions = [r["revision_count"] for r in completed if r["revision_count"] is not None]
    total_times = [r["total_seconds"] for r in runs]

    return {
        "total_runs": n,
        "crash_count": len(crashed),
        "crash_rate": round(len(crashed) / n, 2) if n else None,
        "crash_nodes": [r["crash_node"] for r in crashed],
        "avg_total_seconds": round(statistics.mean(total_times), 1) if total_times else None,
        "node_stats": node_stats,
        "avg_quality_score": round(statistics.mean(scores), 2) if scores else None,
        "min_quality_score": round(min(scores), 2) if scores else None,
        "avg_revisions_needed": round(statistics.mean(revisions), 2) if revisions else None,
        "avg_word_count": round(statistics.mean(word_counts)) if word_counts else None,
        "pass_rate": (
            round(sum(1 for r in completed if r["passed_quality_threshold"]) / len(completed), 2)
            if completed else None
        ),
    }


# ---------------------------------------------------------------------------
# Step 4: write JSON (machine-readable, every run) + Markdown (human summary)
# ---------------------------------------------------------------------------
def write_reports(runs: list[dict], summary: dict, topic: str):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")

    json_path = os.path.join(OUTPUT_DIR, "benchmark_results.json")
    with open(json_path, "w") as f:
        json.dump({"timestamp": timestamp, "topic": topic, "summary": summary, "runs": runs}, f, indent=2)

    md_lines = [
        f"# Benchmark report",
        f"",
        f"Generated: {timestamp}  ",
        f"Topic used: \"{topic}\"  ",
        f"Runs: {summary['total_runs']}",
        f"",
        f"## Reliability",
        f"",
        f"- Crash rate: **{summary['crash_rate']*100:.0f}%** ({summary['crash_count']}/{summary['total_runs']} runs)",
    ]
    if summary["crash_nodes"]:
        md_lines.append(f"- Crashes happened in: {', '.join(summary['crash_nodes'])}")
    if summary["avg_revisions_needed"] is not None:
        md_lines.append(f"- Average quality-check revisions needed: {summary['avg_revisions_needed']}")

    md_lines += [
        f"",
        f"## Speed",
        f"",
        f"- Average total run time: **{summary['avg_total_seconds']}s**",
        f"",
        f"| Node | Calls | Avg (s) | Min (s) | Max (s) |",
        f"|---|---|---|---|---|",
    ]
    for node_name, stats in summary["node_stats"].items():
        md_lines.append(
            f"| {node_name} | {stats['calls']} | {stats['avg_seconds']} | "
            f"{stats['min_seconds']} | {stats['max_seconds']} |"
        )

    md_lines += [
        f"",
        f"## Quality",
        f"",
        f"- Average quality score: **{summary['avg_quality_score']}** "
        f"(threshold to pass: {nodes_module.QUALITY_THRESHOLD})",
        f"- Lowest quality score seen: {summary['min_quality_score']}",
        f"- Pass rate (score >= threshold by the time the run finished): "
        f"**{(summary['pass_rate'] or 0)*100:.0f}%**",
        f"- Average final draft length: {summary['avg_word_count']} words",
        f"",
        f"## Per-run detail",
        f"",
        f"| Run | Crashed? | Quality score | Revisions | Word count | Total time (s) |",
        f"|---|---|---|---|---|---|",
    ]
    for r in runs:
        md_lines.append(
            f"| {r['run_index']} | {'YES - ' + str(r['crash_node']) if r['crashed'] else 'no'} | "
            f"{r['quality_score']} | {r['revision_count']} | {r['draft_word_count']} | "
            f"{round(r['total_seconds'], 1)} |"
        )

    md_path = os.path.join(OUTPUT_DIR, "benchmark_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(md_lines))

    return json_path, md_path


def main():
    parser = argparse.ArgumentParser(description="Benchmark the AI research/content pipeline")
    parser.add_argument("--runs", type=int, default=3, help="Number of full pipeline runs (default 3)")
    parser.add_argument("--topic", type=str, default="the rise of multi-agent AI systems in 2026")
    parser.add_argument("--keyword", type=str, default=None, help="Defaults to the topic if omitted")
    args = parser.parse_args()

    keyword = args.keyword or args.topic

    print(f"Running benchmark: {args.runs} run(s) on topic '{args.topic}'\n")
    runs = []
    for i in range(1, args.runs + 1):
        print(f"--- Run {i}/{args.runs} ---")
        record = run_once(args.topic, keyword, i)
        runs.append(record)
        status = "CRASHED" if record["crashed"] else f"OK (score={record['quality_score']})"
        print(f"  Result: {status}  |  time: {round(record['total_seconds'], 1)}s\n")

    summary = aggregate(runs)
    json_path, md_path = write_reports(runs, summary, args.topic)

    print("=" * 60)
    print(f"Crash rate: {summary['crash_rate']*100:.0f}%")
    print(f"Avg total time: {summary['avg_total_seconds']}s")
    print(f"Avg quality score: {summary['avg_quality_score']}")
    print(f"Pass rate: {(summary['pass_rate'] or 0)*100:.0f}%")
    print("=" * 60)
    print(f"\nFull report: {md_path}")
    print(f"Raw data: {json_path}")


if __name__ == "__main__":
    main()
