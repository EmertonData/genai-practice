These are the messages that launch the claim-writing subagents, one per batch of 50 chunks. Each subagent is a Claude Code `general-purpose` agent run on Opus. Replace `<START>` and `<END>` with the batch's 0-based indices in `data/06_claims/sampled_chunks.json` (0 and 49 for batch `00`, up to 350 and 399 for batch `07`), `<NN>` with the batch number, and `<BATCH_DIR>` with a folder outside the repository.

## Faithful claims

```
You are writing rows of a faithfulness evaluation dataset. Do not modify any file in the repository; the only file you write is the output file below.

1. Read the claim-writing instructions in instructor/prompts/faithful_claim.md. Follow them exactly for every chunk.
2. Read data/06_claims/sampled_chunks.json, a JSON list of chunk records with keys id, document, year, text. Your chunks are the ones at 0-based list indices <START> to <END> inclusive (50 chunks). Use Bash with python3 to print them if the file is too large to read directly.
3. For each chunk, write one claim following the instructions. Treat each chunk independently: use nothing from other chunks or from your own knowledge. Before writing each claim, locate the exact span(s) of the chunk that support it, and check every figure and date in the claim against that span. The claim must be at most 20 words.
4. Write a JSON list to <BATCH_DIR>/faithful_<NN>.json (create the folder if needed), one object per chunk, in index order, with exactly these keys:
   {"claim_id": "<chunk id>_f", "chunk_id": "<chunk id>", "claim": "<your sentence>", "true_faithfulness_label": 1, "error_type": null}
   Do not copy the chunk text into the file. Validate the file afterwards with python3 (valid JSON, 50 objects, ids in index order).

Your final message should be short: the number of claims written, and a list of any chunk ids where following the instructions was difficult, with one line on why for each.
```

## Unfaithful claims

```
You are writing rows of a faithfulness evaluation dataset. Do not modify any file in the repository; the only file you write is the output file below.

1. Read the claim-writing instructions in instructor/prompts/unfaithful_claim.md. Follow them exactly for every chunk.
2. Read data/06_claims/sampled_chunks.json, a JSON list of chunk records with keys id, document, year, text. Your chunks are the ones at 0-based list indices <START> to <END> inclusive (50 chunks). Use Bash with python3 to print them if the file is too large to read directly.
3. For each chunk, choose the category and write one claim following the instructions. Treat each chunk independently: use nothing from other chunks or from your own knowledge. Before finalising each claim, re-read the chunk and check that the claim is clearly not supported by it, that it contains exactly one distortion, and that it is at most 20 words. Write every claim yourself; do not generate claims programmatically.
4. Write a JSON list to <BATCH_DIR>/unfaithful_<NN>.json, one object per chunk, in index order, with exactly these keys:
   {"claim_id": "<chunk id>_u", "chunk_id": "<chunk id>", "claim": "<your sentence>", "true_faithfulness_label": 0, "error_type": "<label from the table>", "distortion_note": "<one sentence on what was changed>"}
   Do not copy the chunk text into the file. Validate the file afterwards with python3 (valid JSON, 50 objects, ids in index order, error_type values from the table, at most 20 words per claim).

Your final message should be short: the number of claims written, the count per error_type, and any chunk ids where following the instructions was difficult, with one line on why for each.
```
