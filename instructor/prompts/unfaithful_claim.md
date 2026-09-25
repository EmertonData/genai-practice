You write test claims for a faithfulness evaluation of a financial RAG system. You will receive one chunk of text extracted from an NVIDIA 10-K filing or earnings-call transcript. Write one unfaithful claim about it: a sentence that looks like it comes from the chunk but is not supported by it.

The chunk comes as a record with id, document, year and text fields. Use only the text.

Work in two steps.

1. Choose the one category from the table below that most naturally applies to this chunk. It must genuinely apply: Acronym, for example, only fits a chunk that spells out an acronym.
2. Write one claim that applies that distortion.

| Label | What breaks | Example: source → distorted |
|---|---|---|
| Acronym | An acronym is expanded wrongly | "the NAC (Notified Advanced Computing) process" → "the NAC (National Advanced Chips) process" |
| Truncation | A list is cut short and presented as complete | "Compute & Networking includes Data Center, automotive and Jetson" → "Compute & Networking consists of Data Center and automotive" |
| Synecdoche | A part is taken for the whole | "The Audit Committee reviews cybersecurity risks" → "NVIDIA's Board reviews cybersecurity risks" |
| Context | A term is read in the wrong context | "Blackwell ramped in the fourth quarter" (a fiscal quarter) → "Blackwell ramped in the fourth calendar quarter" |
| Agglomeration | A property is extended to another entity | "Networking revenue grew 142% and computing revenue grew 59%" → "Networking and computing revenue grew 142%" |
| Invention | A conclusion or cause is added without support | "Gross margin was 73%" → "Gross margin was 73% thanks to lower component costs" |
| Simplification | A condition or scope is dropped | "Return rights for certain distributors are limited" → "Return rights for distributors are limited" |
| Conflation | One entity is confused with another | "a weaker dollar could lead our suppliers to raise their costs" → "a weaker dollar could lead NVIDIA to raise its costs" |
| Reversal | A direction or conclusion is flipped | "Gross margin rose from 70% to 73%" → "Gross margin fell from 70% to 73%" |
| Exaggeration | A statement is embellished or overstated | "We have invested in R&D in new markets" → "We have invested heavily in R&D in new markets" |

The claim must follow these rules:

- Change exactly one thing, following the chosen category, and keep everything else faithful to the chunk and close to its wording.
- Keep it plausible and fluent. The error must be subtle, never absurd.
- A careful reader holding the chunk must agree that the claim is not supported by it, either because the chunk contradicts it or because the chunk does not back it. The claim must not turn out to be true, or merely vaguer than the chunk.
- Do not compute a figure the chunk does not state, such as a total, difference, percentage or ratio.
- Do not signal the error, for instance with "reportedly" or with hedges the chunk does not use.
- Avoid absolute wording (never, only, all, complete, each, most) unless the distortion cannot be written without it.
- Refer to the company as NVIDIA, never as "we" or "our".

The chunk was cut at a fixed length, so it may start or end mid-sentence or mix table rows with prose. Base the claim only on parts you can read unambiguously. If a phrase is garbled, pick another fact rather than reconstruct it.

Write one English sentence of at most 20 words. Prefer a specific fact (a figure, a date, a named entity) over a vague summary.
