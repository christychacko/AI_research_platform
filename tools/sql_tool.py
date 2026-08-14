"""
tools/sql_tool.py

A self-correcting SQL agent over a local SQLite database (free, no server,
just a file on disk). "Self-correcting" means: if the generated SQL fails,
the error message is fed back to the LLM to produce a fixed query, up to
N retries, before giving up.

Includes a small seed script that creates a demo `research_metrics` table
so the tool is testable out of the box.
"""

import os
import sqlite3
from langchain_core.tools import tool
from config.llm_config import get_langchain_llm

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "research.db")
MAX_RETRIES = 3


def seed_demo_database():
    """Creates a small demo table so sql_query has something to query against."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""
        CREATE TABLE IF NOT EXISTS research_metrics (
            id INTEGER PRIMARY KEY,
            topic TEXT,
            month TEXT,
            mentions INTEGER,
            sentiment_score REAL
        )
    """)
    cur.execute("SELECT COUNT(*) FROM research_metrics")
    if cur.fetchone()[0] == 0:
        sample_rows = [
            ("AI Agents", "2026-01", 1200, 0.62),
            ("AI Agents", "2026-02", 1850, 0.58),
            ("AI Agents", "2026-03", 2400, 0.65),
            ("Vector Databases", "2026-01", 400, 0.71),
            ("Vector Databases", "2026-02", 520, 0.69),
            ("Vector Databases", "2026-03", 610, 0.73),
        ]
        cur.executemany(
            "INSERT INTO research_metrics (topic, month, mentions, sentiment_score) "
            "VALUES (?, ?, ?, ?)",
            sample_rows,
        )
    conn.commit()
    conn.close()


def _get_schema() -> str:
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table'")
    schema = "\n".join(row[0] for row in cur.fetchall() if row[0])
    conn.close()
    return schema


def _run_sql(query: str):
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    try:
        cur.execute(query)
        if query.strip().lower().startswith("select"):
            cols = [d[0] for d in cur.description]
            rows = cur.fetchall()
            return True, (cols, rows)
        else:
            conn.commit()
            return True, f"OK, {cur.rowcount} rows affected."
    except Exception as e:
        return False, str(e)
    finally:
        conn.close()


@tool
def sql_query(natural_language_question: str) -> str:
    """
    Answer a question about structured research data by generating and
    running SQL against the local database, self-correcting on errors.

    The database has a `research_metrics` table tracking topic mentions
    and sentiment over time. Ask things like "average sentiment for AI
    Agents by month" and it will write, run, and fix SQL as needed.

    Args:
        natural_language_question: the question to answer using the DB
    """
    if not os.path.exists(DB_PATH):
        seed_demo_database()

    llm = get_langchain_llm(temperature=0)
    schema = _get_schema()

    history = []
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        if last_error is None:
            prompt = (
                f"You are a SQLite expert. Given this schema:\n{schema}\n\n"
                f"Write ONE SQL query (no markdown, no explanation, just the "
                f"raw SQL) to answer: {natural_language_question}"
            )
        else:
            prompt = (
                f"The previous SQL query failed.\nSchema:\n{schema}\n\n"
                f"Previous query:\n{history[-1]}\n\n"
                f"Error:\n{last_error}\n\n"
                f"Write a corrected SQL query (raw SQL only, no markdown) "
                f"to answer: {natural_language_question}"
            )

        response = llm.invoke(prompt)
        sql = response.content.strip().strip("```sql").strip("```").strip()
        history.append(sql)

        ok, result = _run_sql(sql)
        if ok:
            if isinstance(result, tuple):
                cols, rows = result
                if not rows:
                    return f"Query ran successfully but returned no rows.\nSQL used: {sql}"
                formatted_rows = "\n".join(str(r) for r in rows)
                return f"SQL used: {sql}\n\nColumns: {cols}\nResults:\n{formatted_rows}"
            return f"SQL used: {sql}\n\n{result}"
        else:
            last_error = result
            if attempt == MAX_RETRIES:
                return (
                    f"Failed after {MAX_RETRIES} attempts.\n"
                    f"Last query tried: {sql}\nLast error: {last_error}"
                )
    return "Unexpected failure."


if __name__ == "__main__":
    # quick manual test: python tools/sql_tool.py
    seed_demo_database()
    print(sql_query.invoke({
        "natural_language_question": "What is the average sentiment score per topic?"
    }))
