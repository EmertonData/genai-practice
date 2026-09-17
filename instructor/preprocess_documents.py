"""Trim the raw filings down to their substantive pages (preparation.md §4.2-4.3).

Reads data/01_raw/, writes data/02_processed/. Dropped pages are the cover and
table of contents at the front, and the exhibit index, signatures, insider-trading
policy, equity-plan boilerplate and certifications at the back — everything an
analyst would actually read is kept, so retrieval has a real corpus to search.

Page numbers below are 1-indexed and inclusive, matching a PDF viewer.
"""

import shutil
from pathlib import Path

from pypdf import PdfReader, PdfWriter

RAW = Path("data/01_raw")
PROCESSED = Path("data/02_processed")

# filename -> (first_page, last_page) of the content to keep
PDF_RANGES = {
    "10K-NVDA-2025.pdf": (4, 82),
    "10K-NVDA-2026.pdf": (4, 81),
}


def trim_pdf(name: str, first: int, last: int) -> None:
    reader = PdfReader(RAW / name)
    writer = PdfWriter()
    for page in reader.pages[first - 1 : last]:
        writer.add_page(page)
    out = PROCESSED / name
    with out.open("wb") as fh:
        writer.write(fh)
    print(f"{name}: kept pp. {first}-{last} ({len(writer.pages)}/{len(reader.pages)} pages) -> {out}")


def main() -> None:
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, (first, last) in PDF_RANGES.items():
        trim_pdf(name, first, last)
    for transcript in sorted(RAW.glob("*.txt")):
        shutil.copy2(transcript, PROCESSED / transcript.name)
        print(f"{transcript.name}: copied as-is -> {PROCESSED / transcript.name}")


if __name__ == "__main__":
    main()
