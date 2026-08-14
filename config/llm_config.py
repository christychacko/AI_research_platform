"""
config/llm_config.py

Single source of truth for which LLM backend powers every framework in this
stack (LangGraph nodes, CrewAI agents, AutoGen agents).

Why this matters: LangGraph/LangChain, CrewAI, and AutoGen each expect the
LLM connection described in a *different* shape (a LangChain ChatModel object,
a CrewAI LLM string/object, and an AutoGen config dict, respectively). This
module builds all three from the same .env values so you only configure your
API key once.

Free options supported:
  1. Groq (default) - free tier, no credit card, fast Llama 3.x inference.
     Get a key at https://console.groq.com
  2. Ollama - fully local, zero API key, zero network calls. Slower, needs
     a decent machine, but truly free with no rate limits.
"""

import os
from dotenv import load_dotenv

load_dotenv()

USE_OLLAMA = os.getenv("USE_OLLAMA", "false").lower() == "true"
OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "llama3.1:8b")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")
GROQ_MODEL = "llama-3.3-70b-versatile"  # free-tier model on Groq as of writing

if not USE_OLLAMA and not GROQ_API_KEY:
    raise RuntimeError(
        "No LLM backend configured. Either set GROQ_API_KEY in your .env "
        "(free, sign up at https://console.groq.com) or set USE_OLLAMA=true "
        "and install Ollama locally. See .env.example."
    )


# ---------------------------------------------------------------------------
# 1. LangChain ChatModel (used directly by LangGraph nodes)
# ---------------------------------------------------------------------------
def get_langchain_llm(temperature: float = 0.3):
    if USE_OLLAMA:
        from langchain_community.chat_models import ChatOllama
        return ChatOllama(model=OLLAMA_MODEL, temperature=temperature)
    else:
        # Groq exposes an OpenAI-compatible API, so langchain-openai works
        # as long as we point base_url at Groq.
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model=GROQ_MODEL,
            temperature=temperature,
            api_key=GROQ_API_KEY,
            base_url="https://api.groq.com/openai/v1",
        )


# ---------------------------------------------------------------------------
# 2. CrewAI LLM object
# ---------------------------------------------------------------------------
def get_crewai_llm(temperature: float = 0.3):
    from crewai import LLM
    if USE_OLLAMA:
        return LLM(
            model=f"ollama/{OLLAMA_MODEL}",
            base_url="http://localhost:11434",
            temperature=temperature,
        )
    else:
        return LLM(
            model=f"groq/{GROQ_MODEL}",
            api_key=GROQ_API_KEY,
            temperature=temperature,
        )


# ---------------------------------------------------------------------------
# 3. AutoGen LLM config dict
# ---------------------------------------------------------------------------
def get_autogen_llm_config(temperature: float = 0.2):
    if USE_OLLAMA:
        return {
            "config_list": [
                {
                    "model": OLLAMA_MODEL,
                    "base_url": "http://localhost:11434/v1",
                    "api_key": "ollama",  # required placeholder, unused
                }
            ],
            "temperature": temperature,
            "cache_seed": None,
        }
    else:
        return {
            "config_list": [
                {
                    "model": GROQ_MODEL,
                    "base_url": "https://api.groq.com/openai/v1",
                    "api_key": GROQ_API_KEY,
                }
            ],
            "temperature": temperature,
            "cache_seed": None,
        }
