"""Reference solutions for the Session 2 (Build) notebook exercises (preparation.md §5).

chunk_string, vectorize_text, top_k_search and the ReAct loop assembly.

Every function here takes what it needs as a parameter, so they can be imported and tested on
their own. The notebook wraps `search_filings` in a small tool that supplies the model and the
vectors, since those cannot be chosen by the LLM.
"""

from collections.abc import Callable

import numpy as np
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_core.tools import BaseTool, tool
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from langgraph.graph import END, START, StateGraph
from langgraph.graph.state import CompiledStateGraph
from langgraph.prebuilt import ToolNode, tools_condition
from numpy.typing import NDArray

from utils.build import State


def chunk_string(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into overlapping, non-blank chunks.

    Parameters
    ----------
    text : str
        The text to split.
    chunk_size : int
        Maximum number of characters in a chunk.
    overlap : int
        Number of characters each chunk shares with the previous one.

    Returns
    -------
    list[str]
        The chunks, in order. Blank chunks are dropped.
    """
    # separator="" means: do not look for a character to break on, just cut on length.
    splitter = CharacterTextSplitter(separator="", chunk_size=chunk_size, chunk_overlap=overlap)
    return [chunk for chunk in splitter.split_text(text) if chunk.strip()]


def embed_texts(texts: list[str], model: HuggingFaceEmbeddings) -> NDArray[np.float32]:
    """Turn each text into a vector.

    Parameters
    ----------
    texts : list[str]
        The texts to embed.
    model : HuggingFaceEmbeddings
        The embedding model, already loaded.

    Returns
    -------
    NDArray[np.float32]
        One row per text, of shape (len(texts), embedding dimension).
    """
    return np.asarray(model.embed_documents(texts), dtype=np.float32)


def top_k_search(query_vector: NDArray[np.float32], chunk_vectors: NDArray[np.float32], k: int) -> list[int]:
    """Find the k chunks whose vectors are closest to the query vector.

    Parameters
    ----------
    query_vector : NDArray[np.float32]
        The vector of the question, of shape (embedding dimension,).
    chunk_vectors : NDArray[np.float32]
        One row per chunk, of shape (number of chunks, embedding dimension).
    k : int
        Number of chunks to return.

    Returns
    -------
    list[int]
        The row indices of the k closest chunks, closest first.
    """
    # The vectors are normalised, so this dot product is the cosine similarity.
    scores = chunk_vectors @ query_vector
    return np.argsort(scores)[::-1][:k].tolist()


SYSTEM_PROMPT = (
    "You answer questions about NVIDIA using the company's own filings and earnings calls. "
    "Search the filings before answering a question about the company. "
    "Base your answer only on the passages you retrieve, and cite the chunk ids you used. "
    "If the passages do not contain the answer, say so plainly instead of guessing. "
    "If a question makes no sense, say so."
)


def search_filings(
    query: str,
    k: int,
    embedding_model: HuggingFaceEmbeddings,
    chunk_vectors: NDArray[np.float32],
    chunks: list[dict],
) -> str:
    """Find the passages of the corpus most relevant to a question.

    Parameters
    ----------
    query : str
        What to look for, in plain English.
    k : int
        Number of passages to return.
    embedding_model : HuggingFaceEmbeddings
        The model used to turn the query into a vector.
    chunk_vectors : NDArray[np.float32]
        One row per chunk, in the same order as `chunks`.
    chunks : list[dict]
        The chunk records, each with an `id` and a `text`.

    Returns
    -------
    str
        The matching passages, each preceded by the id of the chunk it comes from.
    """
    query_vector = embed_texts([query], embedding_model)[0]
    indices = top_k_search(query_vector, chunk_vectors, k)
    return "\n\n".join(f"[{chunks[i]['id']}] {chunks[i]['text']}" for i in indices)


@tool
def calculator(a: float, b: float, operation: str) -> float:
    """Apply an arithmetic operation to two numbers.

    Parameters
    ----------
    a : float
        The left operand.
    b : float
        The right operand.
    operation : str
        One of "add", "subtract", "multiply", "divide".

    Returns
    -------
    float
        The result of the operation.
    """
    if operation == "add":
        return a + b
    if operation == "subtract":
        return a - b
    if operation == "multiply":
        return a * b
    if operation == "divide":
        return a / b
    raise ValueError(f"Unknown operation: {operation}")


def make_agentic_rag(node: Callable[..., State], tools: list[BaseTool]) -> CompiledStateGraph:
    """Wire the model and its tools into a ReAct loop.

    Parameters
    ----------
    node : Callable[..., State]
        The node that calls the model, as written above.
    tools : list[BaseTool]
        The tools the model is allowed to call.

    Returns
    -------
    CompiledStateGraph
        The compiled agent, ready to be invoked.
    """
    # ty reports a false positive here: even LangGraph's own MessagesState fails
    # this bound. See https://github.com/astral-sh/ty (as of ty 0.0.83).
    graph = StateGraph(State)  # ty: ignore[invalid-argument-type]

    graph.add_node("answer_or_call_tool", node)
    graph.add_node("tools", ToolNode(tools))
    graph.add_edge(START, "answer_or_call_tool")
    graph.add_conditional_edges("answer_or_call_tool", tools_condition, {"tools": "tools", END: END})
    graph.add_edge("tools", "answer_or_call_tool")

    return graph.compile()


def answer_question(question: str, agent: CompiledStateGraph) -> str:
    """Ask the agent one question and return its final answer.

    Parameters
    ----------
    question : str
        The question to ask.
    agent : CompiledStateGraph
        The compiled agent, as returned by `make_agentic_rag`.

    Returns
    -------
    str
        The text of the agent's last message.
    """
    messages = [SystemMessage(SYSTEM_PROMPT), HumanMessage(question)]
    return agent.invoke({"messages": messages})["messages"][-1].content
