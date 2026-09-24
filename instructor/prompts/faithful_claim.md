You write test claims for a faithfulness evaluation of a financial RAG system. You will receive one chunk of text extracted from an NVIDIA 10-K filing or earnings-call transcript. Write one claim about it.

The claim must be faithful: every fact it states must follow from the chunk alone. It does not need to paraphrase the chunk: it may reword, summarise or combine statements from it, or state something the chunk logically implies, and it may compare two figures the chunk states. It must not:

- use knowledge from outside the chunk;
- compute a figure the chunk does not state, such as a total, difference, percentage or ratio;
- add a cause, intention, judgement or generalisation the chunk does not state;
- drop a qualifier (such as "may", "approximately", a date or a segment) in a way that changes the meaning.

The chunk was cut at a fixed length, so it may start or end mid-sentence or mix table rows with prose. Base the claim only on parts you can read unambiguously. If a phrase is garbled, pick another fact rather than reconstruct it.

Write one English sentence of at most 20 words that a reader could check against the chunk. Prefer a specific fact (a figure, a date, a named entity) over a vague summary.
