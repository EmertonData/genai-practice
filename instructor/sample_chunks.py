"""Draw the chunks the faithfulness claims are written from (preparation.md §6.1).

Reads data/03_chunks/all_chunks.json, as produced by the Session 2 notebook with
chunk_size=1000 and overlap=200, and writes data/05_questions/sampled_chunks.json.
The sample is tracked in git because data/03_chunks/ is not, which pins the exact
paragraph text every claim is checked against.
"""

import json
import random
from pathlib import Path

ALL_CHUNKS = Path("data/03_chunks/all_chunks.json")
SAMPLED_CHUNKS = Path("data/05_questions/sampled_chunks.json")

SEED = 42
SAMPLE_SIZE = 400
# The corpus size under the notebook's chunking settings; a different count means
# the chunking changed, and so would every sampled paragraph
EXPECTED_CHUNK_COUNT = 922


def main() -> None:
    chunks = json.loads(ALL_CHUNKS.read_text())
    if len(chunks) != EXPECTED_CHUNK_COUNT:
        raise ValueError(f"expected {EXPECTED_CHUNK_COUNT} chunks, found {len(chunks)}: has the chunking changed?")
    sample = random.Random(SEED).sample(chunks, SAMPLE_SIZE)
    SAMPLED_CHUNKS.write_text(json.dumps(sample, indent=2))
    print(f"sampled {len(sample)} of {len(chunks)} chunks -> {SAMPLED_CHUNKS}")


if __name__ == "__main__":
    main()
