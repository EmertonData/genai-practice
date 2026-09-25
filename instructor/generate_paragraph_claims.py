"""Assemble the paragraph-level faithfulness claim dataset (preparation.md §6.1).

Instructor-only. The claims themselves are written by Claude Code subagents, one batch
of 50 sampled chunks each: one faithful claim per chunk following
instructor/prompts/faithful_claim.md, and one unfaithful claim per chunk following
instructor/prompts/unfaithful_claim.md. This script merges their batch files, joins
each claim back to the text of its chunk, checks the result and writes
data/05_questions/paragraph_claims.json.

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
# Each kind of batch: the claim_id suffix, the label, and any key beyond CLAIM_KEYS
KINDS = {
    "faithful": ("_f", 1, set()),
    "unfaithful": ("_u", 0, {"distortion_note"}),
}
ERROR_TYPES = {
    "Acronym",
    "Truncation",
    "Synecdoche",
    "Context",
    "Agglomeration",
    "Invention",
    "Simplification",
    "Conflation",
    "Reversal",
    "Exaggeration",
}
NUMBER = re.compile(r"\d[\d,]*(?:\.\d+)?")
# Surface features a judge could use to tell the classes apart without reading the chunk
SURFACE_FEATURES = {
    "contains a digit": re.compile(r"\d"),
    "mentions a year": re.compile(r"\b20\d\d\b"),
    "mentions a call or report": re.compile(r"\b(call|report)\b", re.I),
    "starts with NVIDIA": re.compile(r"^NVIDIA"),
    "says or said": re.compile(r"\b(says|said)\b"),
    "absolute wording": re.compile(r"\b(never|only|all|always|complete|consists? of|each|most|every)\b", re.I),
}


def numbers_in(text: str) -> set[str]:
    """Return the numbers in a text, with thousands separators and trailing commas removed."""
    return {match.rstrip(",").replace(",", "") for match in NUMBER.findall(text)}


def load_batches(batch_dir: Path, kind: str) -> list[dict]:
    rows = []
    for batch_path in sorted(batch_dir.glob(f"{kind}_[0-9][0-9].json")):
        rows.extend(json.loads(batch_path.read_text()))
    return rows


def check_rows(rows: list[dict], chunks: dict[str, dict], kind: str) -> None:
    """Raise on any row that breaks the dataset's rules for its kind."""
    suffix, label, extra_keys = KINDS[kind]
    chunk_ids = [row["chunk_id"] for row in rows]
    if sorted(chunk_ids) != sorted(chunks):
        raise ValueError(f"expected one {kind} claim per sampled chunk ({len(chunks)}), found {len(rows)}")
    for row in rows:
        if set(row) != CLAIM_KEYS | extra_keys:
            raise ValueError(f"{row['chunk_id']}: unexpected keys {sorted(row)}")
        if row["claim_id"] != f"{row['chunk_id']}{suffix}":
            raise ValueError(f"{row['chunk_id']}: claim_id {row['claim_id']} does not match")
        if not 0 < len(row["claim"].split()) <= MAX_WORDS:
            raise ValueError(f"{row['claim_id']}: claim is empty or over {MAX_WORDS} words")
        if row["true_faithfulness_label"] != label:
            raise ValueError(f"{row['claim_id']}: a {kind} claim needs label {label}")
        if (row["error_type"] is None) != (kind == "faithful") or row["error_type"] not in ERROR_TYPES | {None}:
            raise ValueError(f"{row['claim_id']}: error_type {row['error_type']} does not fit a {kind} claim")


def number_suspects(rows: list[dict]) -> list[tuple[str, set[str]]]:
    """List the claims holding a number their chunk does not contain, such as a computed figure."""
    suspects = []
    for row in rows:
        missing = numbers_in(row["claim"]) - numbers_in(row["chunk"])
        if missing:
            suspects.append((row["claim_id"], missing))
    return suspects


def class_stats(rows: list[dict]) -> None:
    """Print each surface feature's rate in both classes, so a gap between them shows up."""
    classes = {
        kind: [row["claim"] for row in rows if row["true_faithfulness_label"] == label]
        for kind, (_, label, _) in KINDS.items()
    }
    print(f"{'':28}" + "".join(f"{kind:>12}" for kind in classes))
    mean_words = {kind: sum(len(claim.split()) for claim in claims) / len(claims) for kind, claims in classes.items()}
    print(f"{'mean words':28}" + "".join(f"{mean_words[kind]:>12.1f}" for kind in classes))
    for name, pattern in SURFACE_FEATURES.items():
        rates = {
            kind: sum(bool(pattern.search(claim)) for claim in claims) / len(claims) for kind, claims in classes.items()
        }
        print(f"{name:28}" + "".join(f"{rates[kind]:>12.0%}" for kind in classes))


def main() -> None:
    batch_dir = Path(sys.argv[1])
    sampled = json.loads(SAMPLED_CHUNKS.read_text())
    chunks = {chunk["id"]: chunk for chunk in sampled}

    by_kind = {}
    for kind in KINDS:
        rows = load_batches(batch_dir, kind)
        check_rows(rows, chunks, kind)
        by_kind[kind] = {row["chunk_id"]: row for row in rows}

    # For each chunk in sample order, its faithful then its unfaithful claim, joined to the chunk text
    rows = [
        {
            **by_kind[kind][chunk["id"]],
            "distortion_note": by_kind[kind][chunk["id"]].get("distortion_note"),
            "chunk": chunk["text"],
        }
        for chunk in sampled
        for kind in KINDS
    ]
    PARAGRAPH_CLAIMS.write_text(json.dumps(rows, indent=2))
    print(f"wrote {len(rows)} claims -> {PARAGRAPH_CLAIMS}")

    for kind, (_, label, _) in KINDS.items():
        suspects = number_suspects([row for row in rows if row["true_faithfulness_label"] == label])
        print(f"{len(suspects)} {kind} claims hold a number absent from their chunk:")
        for claim_id, missing in suspects:
            print(f"  {claim_id}: {sorted(missing)}")

    class_stats(rows)


if __name__ == "__main__":
    main()
