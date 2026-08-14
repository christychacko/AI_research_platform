# Autonomous AI Research & Content Platform

LangGraph orchestrates two CrewAI crews and one AutoGen pair, with a shared
LangChain tool layer (web search, RAG, self-correcting SQL agent).

## Architecture (how it actually runs, not just how it's described)

LangGraph, CrewAI, and AutoGen each have their own agent/execution model —
there's no native way for them to call into each other. So the real shape
of this project is:

- **LangGraph is the only orchestrator.** It owns the state graph,
  checkpointing, the conditional retry branch, and the human-approval
  interrupt.
- **CrewAI and AutoGen are subroutines.** Each LangGraph node calls a plain
  Python function (`run_research_crew()`, `run_data_analysis()`, etc.) that
  spins up a crew or an AutoGen pair, runs it to completion, and returns a
  string/dict. Nothing crosses the boundary except plain data.
- **LangChain is the tool layer**, not a fourth independent framework —
  `web_search`, `rag_query`, and `sql_query` are `@tool`-decorated functions
  that CrewAI agents and LangGraph nodes both use directly.

```
research_node (CrewAI: searcher -> analyst -> fact-checker)
       |
data_analysis_node (AutoGen: UserProxy + AssistantAgent, sandboxed code exec)
       |
content_node (CrewAI: writer -> SEO analyst -> editor)
       |
quality_check_node --score < 0.7 & retries left?--> revise_node --back to quality_check
       |  (else)
human_approval_node   <-- graph PAUSES here (interrupt_before), resumes on your input
       |
finalize_node
```

## Setup (free, ~10 minutes)

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env
```

Edit `.env` and add a free Groq API key:
1. Go to https://console.groq.com → sign up (no credit card) → create an API key.
2. Paste it into `GROQ_API_KEY=` in `.env`.

That's it — DuckDuckGo search needs no key, embeddings run locally via
`sentence-transformers`, and SQLite/Chroma are just local files.

### Alternative: fully local, zero API key
Install [Ollama](https://ollama.com), run `ollama pull llama3.1:8b`, then in
`.env` set `USE_OLLAMA=true`. Slower than Groq depending on your hardware,
but makes zero network calls to any LLM provider.

## Run it

```bash
python main.py
```

You'll be prompted for a topic and an SEO keyword, then watch the pipeline
run through research → data analysis → content drafting → quality check
(with automatic retry if the draft scores below 0.7) → it will **pause and
print the draft**, asking you to approve or reject it in the terminal. That
pause is real — it's LangGraph's `interrupt_before`, not a sleep() — the
graph's state is checkpointed and execution resumes exactly where it left
off once you respond.

The final article is saved to `outputs/final_article.md`. Any charts from
the AutoGen analysis land in `autogen_team/sandbox/`.

## Testing individual pieces

Every module has a `if __name__ == "__main__"` block so you can test each
layer in isolation before running the full pipeline:

```bash
python tools/search_tool.py        # test free web search
python tools/rag_tool.py           # test local embeddings + Chroma
python -m tools.sql_tool          # test self-correcting SQL agent
python crews/research_crew.py      # test the research crew alone
python crews/content_crew.py       # test the content crew alone
python autogen_team/data_analysis_team.py   # test AutoGen pair alone
python graph/workflow.py           # test the full graph with a stubbed approval
```

## Known rough edges

- **Cost of going through 3 frameworks per run:** a full run makes a lot of
  LLM calls (3 CrewAI agents × 2 crews + AutoGen's back-and-forth + the
  quality scorer). On Groq's free tier this is fine but not instant —
  expect a couple of minutes per run.
- **CrewAI memory** is short-term/in-process by default; it doesn't persist
  across separate `python main.py` runs. If you need persistent memory
  across sessions, look at CrewAI's long-term memory config.
- **`use_docker=False`** in the AutoGen config means generated code runs
  directly in `autogen_team/sandbox/` on your machine, not in a container.
  Fine for a demo with a trusted LLM and free-tier rate limits; if you want
  real isolation, install Docker Desktop and flip that flag.
- **MemorySaver checkpointing** is in-process and lost when the script
  exits — fine for a single interactive run. For checkpoints that survive
  a restart, swap in `langgraph.checkpoint.sqlite.SqliteSaver`.
