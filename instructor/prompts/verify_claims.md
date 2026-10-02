These are the messages that launch the verification subagents, once `instructor/generate_paragraph_claims.py` has merged the batches into `data/06_claims/paragraph_claims.json`. Each subagent is a Claude Code `general-purpose` agent run on Opus and checks 50 claims of one class. Replace `<START>` and `<END>` with the positions of its claims among the claims of that class (0 and 49, up to 350 and 399), `<NN>` with the batch number, and `<VERIFY_DIR>` with a folder outside the repository.

## Faithful claims

```
You are auditing rows of a faithfulness evaluation dataset. Every row pairs a chunk of an NVIDIA 10-K filing or earnings-call transcript with a claim. Your job is to find any claim labelled faithful whose label is wrong. Be strict and skeptical: you are the last check before these labels become ground truth. Do not modify any file in the repository.

1. Read the rules the claim writers had to follow in instructor/prompts/faithful_claim.md.
2. Load data/06_claims/paragraph_claims.json with python3 and keep only the rows whose true_faithfulness_label is 1, in file order. Your rows are positions <START> to <END> inclusive of that filtered list. Use only each row's "chunk" and "claim" fields.
3. For each claim, check it against its chunk alone. Flag the claim if any of these holds:
   - a fact, figure, date, entity or speaker in the claim is not stated in the chunk, or is attached to the wrong thing (wrong row, column, year, segment or speaker);
   - the claim computes a figure the chunk does not state;
   - the claim adds a cause, intention, judgement or generalisation the chunk does not state;
   - the claim drops or changes a qualifier ("may", "approximately", a date, a segment, a condition) in a way that changes the meaning, or turns a hedged statement into a firm one;
   - the claim reconstructs a garbled or truncated phrase into a statement the text does not clearly make;
   - the claim needs knowledge from outside the chunk to be verified.
   Do not flag: rewording, unit or date formatting ("$10 billion" for "$10,000,000,000", "January" for "Jan"), common abbreviations (CEO, CFO, R&D, U.S.), calling the company "NVIDIA" where the text says "we", or referring to the source as "the call", "the earnings call" or "the report".
4. Write a JSON list to <VERIFY_DIR>/verify_f_<NN>.json (create the folder if needed) containing ONLY the flagged claims, each as {"claim_id": ..., "problem": "<one sentence>", "severity": "wrong" | "borderline"}. "wrong" means the claim is not supported; "borderline" means it is arguably supported but a strict judge could reasonably reject it. Write [] if nothing is flagged.

Your final message: how many claims you checked, and the flagged claim ids with severity.
```

## Unfaithful claims

```
You are auditing rows of a faithfulness evaluation dataset. Each row you check pairs a chunk of an NVIDIA 10-K filing or earnings-call transcript with a claim labelled UNFAITHFUL, plus the distortion category (error_type) it is supposed to illustrate. Your job is to find claims whose label or category is wrong. Be strict and skeptical: you are the last check before these labels become ground truth. Do not modify any file in the repository.

1. Read the category table and rules the claim writers followed in instructor/prompts/unfaithful_claim.md.
2. Load data/06_claims/paragraph_claims.json with python3 and keep only the rows whose true_faithfulness_label is 0, in file order. Your rows are positions <START> to <END> inclusive of that filtered list. Use only each row's "chunk", "claim" and "error_type" fields. Do NOT read "distortion_note": judge independently.
3. For each claim, check it against its chunk alone and flag it if:
   - LABEL_WRONG: the claim is actually supported by the chunk (true, or only a harmless rewording/vaguer version), so it should not be labelled unfaithful;
   - WRONG_CATEGORY: the claim is unsupported, but its distortion clearly belongs to a different category of the table than its error_type (give the right one);
   - MULTIPLE: the claim contains more than one independent distortion;
   - TOO_OBVIOUS: the error would be caught instantly by any reader (absurd, or flatly contradicting a sentence stated in almost the same words).
   Do not flag number formatting ("$41 billion" for "$41,000,000,000"), calling the company "NVIDIA", or subtle but real distortions: subtle is the goal.
4. Write a JSON list to <VERIFY_DIR>/verify_u_<NN>.json containing ONLY the flagged claims, each as {"claim_id": ..., "flag": "LABEL_WRONG" | "WRONG_CATEGORY" | "MULTIPLE" | "TOO_OBVIOUS", "problem": "<one sentence>", "suggested_category": "<label or null>"}. Write [] if nothing is flagged.

Final message: how many claims you checked, and the flagged claim ids with their flag.
```
