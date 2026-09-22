"""Reference solutions for the Session 2 (Build) notebook exercises (preparation.md §5).

chunk_string, vectorize_text, top_k_search and the ReAct loop assembly.
"""

from langchain_text_splitters import CharacterTextSplitter


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
