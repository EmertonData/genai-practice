"""Helpers given to students for the Session 2 notebook.

These are deliberately not exercises: reading a PDF and writing JSON are plumbing,
so they come ready-made and the exercises stay on the RAG concepts.
"""

import json
import re
from pathlib import Path

import numpy as np
from numpy.typing import NDArray
from pypdf import PdfReader


def load_document(input_path: str) -> tuple[str, int]:
    """Read a document as one string, along with the fiscal year it covers.

    Parameters
    ----------
    input_path : str
        Path to the document, either a `.pdf` filing or a `.txt` transcript.

    Returns
    -------
    tuple[str, int]
        The full text of the document, and the fiscal year read from its file name.
    """
    path = Path(input_path)
    year = int(re.search(r"\d{4}", path.stem).group())

    if path.suffix == ".pdf":
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
        text = "\n".join(pages)
    else:
        text = path.read_text()

    return text, year


def build_chunk_records(text_chunks: list[str], document_name: str, year: int) -> list[dict]:
    """Attach an id and its provenance to each chunk of a document.

    Parameters
    ----------
    text_chunks : list[str]
        The chunks of one document, in order, as returned by `chunk_string`.
    document_name : str
        File name of the document the chunks come from, e.g. `10K-NVDA-2026.pdf`.
    year : int
        Fiscal year the document covers.

    Returns
    -------
    list[dict]
        One record per chunk, with the keys `id`, `text`, `document` and `year`.
    """
    stem = Path(document_name).stem
    return [
        {
            # Zero-padded so the ids sort in order as text: c0002 before c0010.
            "id": f"{stem}_c{i:04d}",
            "text": text_chunk,
            "document": document_name,
            "year": year,
        }
        for i, text_chunk in enumerate(text_chunks)
    ]


def save_chunks(chunks: list[dict], output_path: str) -> None:
    """Write chunks to a JSON file, replacing any existing content.

    Parameters
    ----------
    chunks : list[dict]
        The chunks to save.
    output_path : str
        Path to the JSON file to write.

    Returns
    -------
    None
    """
    Path(output_path).write_text(json.dumps(chunks, indent=2))


def save_vectors(vectors: NDArray[np.float32], chunk_ids: list[str], output_path: str) -> None:
    """Write vectors and their chunk ids to one `.npz` file.

    Parameters
    ----------
    vectors : NDArray[np.float32]
        One row per chunk, of shape (len(chunk_ids), embedding dimension).
    chunk_ids : list[str]
        The chunk ids, in the same order as the rows of `vectors`.
    output_path : str
        Path to the `.npz` file to write.

    Returns
    -------
    None
    """
    np.savez(output_path, vectors=vectors, chunk_ids=np.array(chunk_ids))


def load_vectors(input_path: str) -> tuple[NDArray[np.float32], list[str]]:
    """Read back a file written by `save_vectors`.

    Parameters
    ----------
    input_path : str
        Path to the `.npz` file.

    Returns
    -------
    tuple[NDArray[np.float32], list[str]]
        The vectors, and the chunk ids they encode in the same order.
    """
    loaded = np.load(input_path)
    return loaded["vectors"], loaded["chunk_ids"].tolist()
