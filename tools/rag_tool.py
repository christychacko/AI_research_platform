"""
tools/rag_tool.py

A small RAG pipeline over a local Chroma vector store, using free local
embeddings (sentence-transformers, runs on CPU, no API key needed).

Workflow:
  1. ingest_documents(): chunk + embed + store text documents
  2. rag_query tool: retrieve relevant chunks for a query (used by agents)

This is intentionally simple (no reranking, no hybrid search) so it's easy
to read and modify. Swap in a different embedding model or add a reranker
later if you need better retrieval quality.
"""

import os
from langchain_core.tools import tool
from langchain_community.vectorstores import Chroma
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document

PERSIST_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "chroma_db")

# all-MiniLM-L6-v2: small, fast, free, runs locally on CPU. Good enough for
# a demo RAG pipeline; swap to a bigger model if you need higher recall.
_embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

_vectorstore = None  # lazy-initialized singleton


def _get_vectorstore() -> Chroma:
    global _vectorstore
    if _vectorstore is None:
        _vectorstore = Chroma(
            collection_name="research_kb",
            embedding_function=_embeddings,
            persist_directory=PERSIST_DIR,
        )
    return _vectorstore


def ingest_documents(texts: list[str], metadatas: list[dict] | None = None) -> int:
    """
    Chunk and embed raw text documents into the vector store.
    Call this once up front (or whenever new source material arrives)
    before agents start querying the knowledge base.

    Returns the number of chunks stored.
    """
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    docs = []
    metadatas = metadatas or [{} for _ in texts]
    for text, meta in zip(texts, metadatas):
        for chunk in splitter.split_text(text):
            docs.append(Document(page_content=chunk, metadata=meta))

    vs = _get_vectorstore()
    vs.add_documents(docs)
    return len(docs)


@tool
def rag_query(query: str, k: int = 4) -> str:
    """
    Retrieve the most relevant passages from the local knowledge base
    (vector DB of previously ingested research material) for a given query.
    Use this before web_search when checking if the answer already exists
    in material you've already collected.

    Args:
        query: what you're trying to find out
        k: number of chunks to retrieve (default 4)
    """
    vs = _get_vectorstore()
    if vs._collection.count() == 0:
        return "Knowledge base is empty. Nothing has been ingested yet."

    results = vs.similarity_search(query, k=k)
    if not results:
        return "No relevant passages found."

    return "\n\n---\n\n".join(
        f"[source: {d.metadata.get('source', 'unknown')}]\n{d.page_content}"
        for d in results
    )


if __name__ == "__main__":
    # quick manual test: python tools/rag_tool.py
    n = ingest_documents(
        ["LangGraph is a library for building stateful, multi-actor applications "
         "with LLMs, built on top of LangChain. It models workflows as graphs with "
         "nodes and edges, supports checkpointing, and allows human-in-the-loop "
         "interrupts at any node."],
        metadatas=[{"source": "manual_test"}],
    )
    print(f"Ingested {n} chunks")
    print(rag_query.invoke({"query": "what is langgraph used for"}))
