"""Reference solutions for the Session 2 (Build) notebook exercises (preparation.md §5).

chunk_text, vectorize_text, save_vectors, top_k_search, hybrid (BM25 + semantic) search,
and the ReAct loop assembly.
"""

import json
import re
from pathlib import Path

from pypdf import PdfReader


def chunk_text(input_path: str, output_path: str, chunk_size: int, overlap: int) -> None:
    """Split a document into overlapping chunks and save them with their metadata.

    Parameters
    ----------
    input_path : str
        Path to the document to chunk, either a `.pdf` filing or a `.txt` transcript.
    output_path : str
        Path to the JSON file the chunks are written to. Existing content is replaced.
    chunk_size : int
        Maximum number of characters in a chunk.
    overlap : int
        Number of characters each chunk shares with the previous one.

    Returns
    -------
    None
        The chunks are written to `output_path` as a list of dictionaries, each with
        the keys `id`, `text`, `document`, `page` and `year`.
    """
    path = Path(input_path)
    # The fiscal year is the four-digit number in the file name, e.g. 10K-NVDA-2026.
    year = int(re.search(r"\d{4}", path.stem).group())

    # A transcript has no pages, so treat the whole file as a single one.
    if path.suffix == ".pdf":
        pages = [page.extract_text() or "" for page in PdfReader(path).pages]
    else:
        pages = [path.read_text()]

    # Join the pages: a sentence crossing a page break must still appear whole in a chunk.
    text = "\n".join(pages)

    # Save where each page begins, so a chunk can report which page it starts on.
    page_starts = []
    offset = 0
    for page in pages:
        page_starts.append(offset)
        offset += len(page) + 1  # + 1 for the separator added by join

    # Create the chunks.
    chunks = []
    step = chunk_size - overlap
    start = 0
    while start < len(text):
        piece = text[start : start + chunk_size]
        if piece.strip():  # blank pages produce empty chunks, which would pollute search.
            page = sum(1 for page_start in page_starts if page_start <= start)
            chunks.append(
                {
                    "id": f"{path.stem}_p{page:03d}_c{len(chunks):04d}",
                    "text": piece,
                    "document": path.name,
                    "page": page,
                    "year": year,
                }
            )
        # Break if this chunk reached the end.
        if start + chunk_size >= len(text):
            break
        start += step

    # Write the chunks to the output file.
    Path(output_path).write_text(json.dumps(chunks, indent=2))
