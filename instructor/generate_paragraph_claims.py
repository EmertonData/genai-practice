"""Assemble the paragraph-level faithfulness claim dataset (preparation.md §6.1).

Instructor-only. The claims themselves are written by Claude Code subagents, one batch
of 50 sampled chunks each, following instructor/prompts/faithful_claim.md. This script
merges their batch files, joins each claim back to the text of its chunk, checks the
result and writes data/05_questions/paragraph_claims.json.

Usage: uv run python instructor/generate_paragraph_claims.py <batch_dir>
"""

import json
import re
import sys
from pathlib import Path

SAMPLED_CHUNKS = Path("data/05_questions/sampled_chunks.json")
PARAGRAPH_CLAIMS = Path("data/05_questions/paragraph_claims.json")

MAX_WORDS = 20
CLAIM_KEYS = {"claim_id", "chunk_id", "claim", "true_faithfulness_label", "error_type"}
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")


def numbers_in(text: str) -> set[str]:
    """Return the numbers in a text, with thousands separators and trailing commas removed."""
    return {match.rstrip(",").replace(",", "") for match in NUMBER.findall(text)}


def load_batches(batch_dir: Path) -> list[dict]:
    rows = []
    for batch_path in sorted(batch_dir.glob("faithful_[0-9][0-9].json")):
        rows.extend(json.loads(batch_path.read_text()))
    return rows


def check_rows(rows: list[dict], chunks: dict[str, dict]) -> None:
    """Raise on any row that breaks the dataset's rules."""
    chunk_ids = [row["chunk_id"] for row in rows]
    if sorted(chunk_ids) != sorted(chunks):
        raise ValueError(f"expected one claim per sampled chunk ({len(chunks)}), found {len(rows)}")
    for row in rows:
        if set(row) != CLAIM_KEYS:
            raise ValueError(f"{row['chunk_id']}: unexpected keys {sorted(row)}")
        if row["claim_id"] != f"{row['chunk_id']}_f":
            raise ValueError(f"{row['chunk_id']}: claim_id {row['claim_id']} does not match")
        if not 0 < len(row["claim"].split()) <= MAX_WORDS:
            raise ValueError(f"{row['chunk_id']}: claim is empty or over {MAX_WORDS} words")
        if row["true_faithfulness_label"] != 1 or row["error_type"] is not None:
            raise ValueError(f"{row['chunk_id']}: a faithful claim needs label 1 and no error_type")


def number_suspects(rows: list[dict]) -> list[tuple[str, set[str]]]:
    """List the claims holding a number their paragraph does not contain, such as a computed figure."""
    suspects = []
    for row in rows:
        missing = numbers_in(row["claim"]) - numbers_in(row["paragraph"])
        if missing:
            suspects.append((row["claim_id"], missing))
    return suspects


def main() -> None:
    batch_dir = Path(sys.argv[1])
    sampled = json.loads(SAMPLED_CHUNKS.read_text())
    chunks = {chunk["id"]: chunk for chunk in sampled}

    rows = load_batches(batch_dir)
    check_rows(rows, chunks)

    # Order the rows like the sample, and join each claim to its paragraph
    by_chunk = {row["chunk_id"]: row for row in rows}
    rows = [{**by_chunk[chunk["id"]], "paragraph": chunk["text"]} for chunk in sampled]
    PARAGRAPH_CLAIMS.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} claims -> {PARAGRAPH_CLAIMS}")

    suspects = number_suspects(rows)
    print(f"{len(suspects)} claims hold a number absent from their paragraph:")
    for claim_id, missing in suspects:
        print(f"  {claim_id}: {sorted(missing)}")


if __name__ == "__main__":
    main()
