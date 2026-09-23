"""Reference solutions for the Session 2 (Build) notebook exercises (preparation.md §5).

chunk_string, vectorize_text, top_k_search and the ReAct loop assembly.
"""

import numpy as np
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_text_splitters import CharacterTextSplitter
from numpy.typing import NDArray


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
