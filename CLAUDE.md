# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It's tracked in git and shared between both maintainers — keep it accurate and useful for either of them, not written to just one person.

## Memory

Claude Code sessions working in this repo should proactively update the auto-memory system (project/feedback/user memories) whenever a step is completed, a decision gets locked in, or just periodically during a long session — don't wait for an explicit "update your memory" request before doing so.

## Current state of this repo

**Session 2's notebook is complete and verified against the real API.** It runs in two parts. Part 1 builds the retrieval pipeline: `chunk_string` (§5.1), `embed_texts` (§5.2) and `top_k_search` (§5.4). Part 2 builds the agent: the two tools (`search_filings` and `calculator`) and the LangGraph loop in `make_agentic_rag` (§5.6). §5.3 ships as the given helpers `save_vectors`/`load_vectors`, and §5.5 (BM25) was dropped. Every exercise has its correction in `corrections/correction_build.py`.

The agent is a hand-built `StateGraph` rather than LangChain's prebuilt `create_agent`, so the conditional edge and the loop-back edge stay visible. `State` lives in `utils/build.py` and is given. `search_filings` takes its model, vectors and chunks as parameters, and the notebook wraps it in a small `@tool` so the LLM only ever chooses `query` and `k` — a `functools.partial` cannot be used here, because `@tool` rejects an object with no `__name__`.

**Session 4 is under way.** The claims dataset (§6.1) is merged (PR #13): `data/06_claims/paragraph_claims.json` holds 800 claims, one faithful and one unfaithful for each of 400 chunks. See "Ground truth for faithfulness" below for how the claims are made. The Session 4 PRs are planned as follows: (1) this dataset, (2) the judge (§7.1 and §7.2), (3) the quality pass on the unfaithful claims (§6.2), done after the judge because its scores show which claims are too easy, and (4) the GLIDE exercise (§7.3 to §7.5).

**The judge is validated and merged (PR #14).** `score_faithfulness(chunk, claim, llm, system_prompt)` is in `corrections/correction_eval.py`: one claim per call, Haiku at temperature 0, returning `{"reasoning", "verdict"}`. The prompt is adapted from Fig. 6 of Grégoire's paper (arXiv 2507.21753), which the notebook does not cite. It is `JUDGE_PROMPT` in `corrections/correction_eval.py`, its only copy: students write their own prompt in the notebook, and the instructor scripts import this one. It is zero-shot, on Grégoire's advice that examples bias the judge, and it forbids outside knowledge, unlike the paper's step 5. `instructor/measure_judge.py` checks it on 20 claims; `instructor/run_judge_scores.py` scores all 800 into `data/06_claims/judge_scores.json`. Don't tune the prompt on the judge's errors: its leniency is the bias GLIDE corrects.

**Session 4's notebook, part 1 (the judge), is on `feat/7.1-notebook-judge` (PR #15), rewritten after Grégoire's review.** It follows an engineer evaluating their own agent (see the narrative convention below), and runs as follows:
1. The intro: the client notices hallucinations, the answers are cut into claims, human annotations are slow, so an LLM-as-Judge annotates every claim and PPI (GLIDE) combines both.
2. One hallucinated claim, shown with its chunk and its human annotation.
3. The judge's model.
4. Exercise 1, writing `JUDGE_PROMPT`, from rules given as sentences and an imposed JSON output format (`reasoning` first, then `verdict`). The correction is our prompt.
5. Exercise 2, `score_faithfulness`, checked for free by a `FakeJudge` that records the messages it receives.
6. The judge on 5 random claims, showing verdict and reasoning, with no label.
7. `judge_all` from `utils/evaluate.py`, which runs the student's function in parallel with a `tqdm` bar and returns a float array (NaN for a failed call). `save_verdicts` writes it to `data/07_verdicts/verdicts.json`, so the GLIDE part reloads it with `load_verdicts`; the same function reads the reference scores in `data/06_claims/judge_scores.json`, the fallback if a student's run failed.
8. A teaser: how good the judge is will show when GLIDE shrinks the error bar.

The notebook warns that the full run costs real money, without giving a figure, and reuses the Session 2 key as-is.

**Session 4's notebook, part 2 (GLIDE), is written on `feat/7.3-glide`, stacked on PR #15.** Details in preparation.md §7.3 to §7.5.

**Measured on the judge, so don't re-derive** (all 800 claims, 2026-09-25, after Grégoire's review of PR #14): 399/400 faithful claims judged correctly and 361/400 unfaithful ones, so the judge is lenient. It says 54.8% of claims are faithful where the truth is 50.0%. It misses mostly Simplification (38% caught), then Acronym and Truncation (80%) and Exaggeration (85%); everything else is caught at 97-100%. The run cost $0.90 and took 2.7 minutes with 8 calls in parallel (about 600 input and 105 output tokens per claim), with no failed call. The prompt asks for reasoning first, under 100 words, and `max_tokens=200`: an earlier cap of 150 truncated 23 replies. Dropping the `reasoning` field does not save tokens: on 20 claims Haiku then reasoned in free text before the JSON, which broke 15 replies and raised output to 131 tokens a claim. The shared key's limits are 10,000 requests, 10M input and 2M output tokens per minute: one student uses about 330 requests a minute, so 70 students starting in the same minute would need 2.3× the request limit. Don't add descriptions ahead of the actual exercise content — describe an exercise when it's authored.

**Measured, so don't re-derive:** the notebook runs top to bottom in 43 seconds, and its five agent calls cost $0.033 per run (24,259 input / 1,659 output tokens at Haiku 4.5 rates).

**Dense retrieval is weak on this corpus.** Measured while writing §5.4: on the 7 questions whose reference answer carries a distinctive number, `all-MiniLM-L6-v2` finds the answer chunk in 2/7 at k=5 against 5/7 for a throwaway BM25, and the chunk holding NVIDIA's FY2026 revenue ranks 23rd of 922 for the question that asks for it. It works on thematic questions and fails on table lookups, which is why the §5.4 demo cell uses question 12 rather than question 1. This is the open input to the §5.5 decision.

Dependencies are added as each exercise needs them (see the dependency policy below). The ReAct loop uses **LangGraph**: issue #7 specifies `make_agentic_rag(...) -> StateGraph`, which settles the §5.6 question. Once a test setup exists, add the commands here.

### Notebook conventions 

- **One cell = one job**, each preceded by a short markdown cell. Keep both short, avoid the "wall of text" risk once eight exercises each carry commentary.
- **Exercise statements follow a fixed order**: motivation and stakes → objective → task ("complete the function, respecting its signature and docstring") → hint, placed after the code cell.
- **One exercise = one function**, signature typed and numpy docstring given, body left to the student. An exercise may group two closely related functions under one heading when they are one idea (§5.6 does this for the two tools). Ship the simplest thing that works; improvements go in a numbered synthesis at the very end of the finished notebook, not after each exercise.
- **As much as possible, no code with blanks.** A stub is the signature, the numpy docstring and `# YOUR CODE HERE`. This is Grégoire's preference, applied from notebook 2 on.
- **When an exercise has steps, they go as comments in the function body, never as a list in the markdown** (Grégoire's review of PR #15). Each step is a `# STEP n - ...` comment saying what to do, followed by `# YOUR CODE HERE`, with no partial code. The markdown keeps the motivation, the objective and the task. This keeps every exercise in the same shape. Never `raise NotImplementedError`.
- **Tell the story of an engineer evaluating their own system, never of the dataset's designer** (Grégoire's review of PR #15). Students don't learn how the claims were built. Human annotations are treated as rare and costly, as they are in real life: never compute statistics over all the true labels, and never state the hallucination rate, which is an output of the protocol.
- **Plumbing is given, not exercised.** Anything that teaches Python rather than RAG (reading a PDF, building a dict, writing JSON) lives in `utils/build.py` and is imported. Note the constraint: a helper in that module cannot call a function the student writes in the notebook, and putting the student's function in the module would hand them the solution — so composition happens in the notebook.
- **Checks are inline `assert`s** on a small, representative example, ending in a `print("OK: …")`. No assert message: `assert x == y, x` reads like a stray tuple.
- **Write prose, not telegrams, and avoid the tells of AI-written text.** Grégoire's PR #8 review named three: em-dashes, the binary "X, not Y" rhythm, and bullet lists of bare fragments. A section should read like a book, so introduce a list with a full sentence ("The next step is to orchestrate chunking across a list of documents. It proceeds as follows:") and make each item a sentence. Prefer the established term over an approximation (an `index`, not "one list"). Never address the instructor in student-facing text.
- Notebooks are committed with **outputs cleared** and with `source` stored as a list of lines. Both are enforced by the pre-commit hooks in `prek.toml` (`nbstripout` and `normalize-notebooks`), adopted from the GLIDE repo on Grégoire's suggestion. They exist because editing a cell programmatically rewrites `source` as one long string, which turns the GitHub diff into an unreadable blob. `nbstripout` also renumbers cell ids to sequential integers, so inserting a cell mid-notebook renumbers everything after it; appending at the end does not.

## Commands

Environment is managed with [uv](https://docs.astral.sh/uv/); Python **>=3.12 is required** (`glide-py`, needed later for §6-§7, doesn't support earlier versions).

- `uv sync` — install/update the environment from `uv.lock`.
- `uv run python <script>` — run a script (e.g. `instructor/generate_paragraph_claims.py`) inside the project environment.
- `uv add <package>` / `uv remove <package>` — change dependencies (updates `pyproject.toml` and `uv.lock` together; don't hand-edit the dependency list and forget to re-lock).
- `uv lock` — re-resolve and refresh `uv.lock` after a manual `pyproject.toml` edit.
- `uv sync --group dev` — **what the two of us run.** Adds the tooling in the `dev` dependency group (`prek`, `ruff`, `ty`) on top of the student environment.
- `uv run --group dev prek install` — **install the git pre-commit hooks, once per clone.** Without this the hooks never run and the problems they prevent come back.
- `uv run --group dev prek run --all-files` — run every hook over the whole repo, rather than only on staged files.
- `make lint` / `make type-check` — `ruff check --fix` and `ty check`. **Both must pass with no `# noqa` and no suppressions**, since a suppression hides a real problem rather than fixing it. The one documented exception is `StateGraph(State)` in `corrections/correction_build.py`: `ty` rejects it, and rejects LangGraph's own `MessagesState` identically, so no spelling of that line can satisfy it.

**Students run plain `uv sync` and get none of the tooling.** `[tool.uv] default-groups = []` overrides uv's habit of installing the `dev` group by default, which keeps the student install limited to what the notebooks actually import. Anything added for our own workflow belongs in the `dev` group, never in `[project] dependencies`.

**Dependency policy: add packages incrementally, not upfront.** Introduce each dependency via `uv add <package>` only when the script/exercise that actually needs it is being written (e.g. add `sentence-transformers` when writing §5.2 `vectorize_text`, `rank_bm25` when writing §5.5, `glide-py` when writing §6.3/§7.3). This was explicit review feedback from Grégoire on PR #3 — the goal is avoiding a bloated, slow-to-resolve venv full of packages nothing uses yet. Don't front-load the full anticipated dependency list again.

Notes:
- **`utils/` is a real local Python package**, installed into the venv by `uv sync` via hatchling (`[build-system]` + `[tool.hatch.build.targets.wheel] packages = ["utils"]`). Notebooks therefore import it as `from utils.build import ...` from anywhere. Never go back to `sys.path.append("../src")`: Grégoire flagged that on PR #8 as the thing to avoid.
- The PyPI package `glide-py` imports as `import glide`, and is a dependency since part 2, with `matplotlib`. Its verified API is noted in preparation.md §7.5.
- `uv.lock` is committed (unlike `.venv/`, which is gitignored) so both maintainers and every student get identical resolved versions of whatever's actually been added — this is what satisfies preparation.md §8.1's "pin dependencies" ticket.

## Who is doing what

**Benjamin** and his manager **Grégoire** are co-preparing this workshop for HEC students, at Emerton Data. Benjamin is principally responsible for the practical part — designing and building the notebooks, the environment, and the evaluation dataset. Grégoire reviews PRs and co-decides cross-cutting calls (source document, spend cap, correction-release policy, ReAct framework choice, target platform, dependency-management style). One open contingency worth tracking: the Session 2 exercises (§5) currently assume *low* student GenAI usage; if HEC's policy on student AI use turns out to be more permissive, the "complete this function" exercise framing may need to shift toward "explain/debug this code" so the exercises stay meaningful.

## Work plan (draft, subject to refinement)

**Week 1 — Foundations + locking decisions with Grégoire.** Nail down with him, in a short discussion, the choices that commit everything downstream: the source document for the RAG (§4.1, e.g. NVIDIA 10-K), the spend cap on the shared API key (§2.2), the correction-release policy (§1.5 — recommendation: ship everything from the start, matches the stated intent and removes a moving part on the day), the ReAct-loop framework choice (LangGraph vs. manual loop, §5.6), and the target platform for the 70 students (conditions §8.2-8.3) — students will use their own laptops regardless of the workshop being held on HEC's campus. In parallel, advance solo on whatever doesn't depend on any of these decisions: create the repo and its skeleton (§1.1-1.4), get the API key (§2.1), and select/clean the source document pages (§4.2-4.4). None of this is sensitive to HEC's policy on student AI usage.

**Week 2 — Build notebook end to end.** Write the 15-20 due-diligence questions (§4.5), then work through exercises §5.1 to §5.6 (chunking, vectorization, storage, top-k search, ReAct assembly) with their corrections and tests — this is the only block of the plan genuinely sensitive to the "low GenAI usage" assumption: if the exercise prompt stays a plain "complete this function," students who used AI heavily would solve it trivially with no pedagogical effort. For now the assumption is low usage, so no change needed — but revisit if HEC's policy changes the picture (one option if needed: add an "explain/debug this code" dimension rather than "write it"). Finish the week with the full dry run (§5.7) against the real API.

**Week 3 — Faithfulness dataset and notebook.** Once §5 is locked, tackle §6: generate the claims dataset with the hallucination taxonomy (§6.1, script `instructor/generate_paragraph_claims.py`), quality-review the "unfaithful" claims so they stay subtle (§6.2 — important so the naive-vs-GLIDE gap is visible in §7.4), then the notebook itself: `score_faithfulness` (§7.1), centralized precomputation of judge scores (§7.2), the GLIDE exercise (§7.3, to verify against the real `glide-py` API, §7.5). In parallel, follow up with Grégoire on the Sessions 1 and 3 slide progress: the certification quiz (§11) can only be written once that content is locked, so the sooner a stable version exists, the better for keeping to the final-stretch schedule.

**Last days before the day.** Timed solo rehearsal of both sessions (§9.1), beta-test with colleagues seeing the content for the first time (§9.2), a cost dry run to check the key's budget against actually observed spend (§9.3), then day-of logistics (§10: spend dashboard kept open, organizers' Slack channel, student support channel) and quiz finalization (§11.1-11.4).

## Git workflow

Both Benjamin and Grégoire push commits to this repo, and the goal is to keep it clean and easy for either to review — so use branches and PRs, not direct commits to `main`:

- **Never commit straight to `main`.** Create a branch for each unit of work and open a PR.
- **Branch granularity**: not yet fixed. Default to one branch/PR per exercise-with-correction-and-test (e.g. §5.1 `chunk_text`) when a ticket is independently reviewable; group into one branch/PR per section when splitting would be artificial (e.g. tightly coupled steps in §6). If unsure which fits a given piece of work, ask before opening the branch.
- **Branch naming**: `<type>/<section-num>-<slug>`, e.g. `feat/5.1-chunk-text`, `docs/1.4-readme`, matching the Conventional Commits type used for the branch's main commit.
- **Commit messages: Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`), scoped to what actually changed.
- **Commit at natural checkpoints**, not mid-edit — e.g. after an exercise + its correction + its test all pass, after a script produces valid output, after a self-contained doc update. Proactively suggest a commit at these points rather than waiting to be asked, and suggest opening a PR once a branch's unit of work is complete.
- Since Grégoire may also be branching off `main` concurrently, pull/rebase on latest `main` before starting new work on a branch.
- Creating branches, committing, pushing, and opening PRs are all done by asking first each time (push and PR creation are shared/visible actions) — per the standing rule of confirming before actions that affect shared state.
- Commit messages, PR titles/descriptions, and code comments should not mention Claude, Anthropic, or AI-generated authorship — no `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" footer, nothing equivalent. (This file itself is fine to be tracked and visible — that's a separate question from what goes in commit/PR text.)

## What this project is

Course materials for a one-day "GenAI in Practice" workshop (70 students) held at HEC, prepared by Emerton Data. Two hands-on practice sessions, each an instructor-authored Jupyter notebook with fill-in-the-blank exercises plus a separate "correction" script:

- **Session 2 — Build** (`notebooks/session_2_build_agentic_rag.ipynb`): students build a RAG agent over a financial document (chunking → embedding → vector search → a ReAct agent loop). BM25 and hybrid fusion are likely dropped — the guiding rule is the simplest RAG that works (§5.5).
- **Session 4 — Evaluate Faithfulness** (`notebooks/session_4_evaluate_faithfulness.ipynb`): students use LLM-as-judge scoring plus `glide-py` (prediction-powered inference / GLIDE) to produce a debiased faithfulness-rate estimate with a confidence interval, and compare it against the naive judge-mean baseline.

Session 4 depends on Session 2 being fully finished (exercises + corrections) first — there is a deliberate ordering constraint: don't touch afternoon material until the morning build, corrections included, is done.

## Target repo skeleton (from preparation.md §1.2)

```
instructor/                     # instructor-only scripts (not shown to students)
  sample_chunks.py              # draw the 400 chunks the claims are written from
  prompts/faithful_claim.md     # instructions given to the faithful-claim subagents
  prompts/unfaithful_claim.md   # same for unfaithful claims, with the English taxonomy table
  generate_paragraph_claims.py  # merge and check the subagents' claim batches
  run_judge_scores.py           # precompute judge faithfulness scores
utils/                          # local package, installed by uv sync
  __init__.py
  build.py                      # helpers given to students: load_document, build_chunk_records, save_chunks
notebooks/
  session_2_build_agentic_rag.ipynb    # imports the helpers with `from utils.build import ...`
  session_4_evaluate_faithfulness.ipynb
corrections/
  correction_build.py
  correction_eval.py
data/                           # numbered pipeline stages
  01_raw/                       # full filings + earnings calls, 2025 and 2026 (tracked)
  02_processed/                 # trimmed to substantive pages (tracked)
  03_chunks/                    # generated in the workshop (gitignored)
  04_vectors/                   # generated in the workshop (gitignored)
  05_questions/
    questions.json              # 20 due-diligence questions (build session)
  06_claims/                    # the faithfulness session's data (tracked)
    sampled_chunks.json         # the 400 chunks claims are written from (output of instructor/sample_chunks.py)
    paragraph_claims.json       # output of instructor/generate_paragraph_claims.py
    judge_scores.json           # output of instructor/run_judge_scores.py
  07_verdicts/                  # the students' own judge verdicts (gitignored)
pyproject.toml
uv.lock
.env.example
README.md
```

`instructor/` scripts are one-time, instructor-run batch jobs (the claims dataset is the exception: subagents write it, and the scripts only sample and merge) — they exist so dataset regeneration (e.g. a source PDF or taxonomy change) is reproducible instead of manual. They are never distributed to students. (Named `instructor/`, not `build/`, to avoid colliding with the conventional meaning of a `build/` directory in Python packaging.)

**`data/` stages 01, 02, 05 and 06 are tracked; 03, 04 and 07 are gitignored** apart from their `.gitkeep`, since chunks, vectors and verdicts are produced during the workshop. Only 2025 and 2026 documents are in scope — the 2024 files were moved out of the repo to `~/Documents/code/genai-practice-archive/`.

## Key architectural decisions to preserve

- **Single shared Anthropic API key** for all 70 students plus instructor prep/judge runs (`ANTHROPIC_API_KEY`, the only env var any notebook needs). There's no per-key model restriction, so notebooks must pin students to Haiku by default; a console-side spend limit is the backstop, not the primary control.
- **Ground truth for faithfulness is constructed, not annotated.** Each claim is written to be faithful or unfaithful, so its `true_faithfulness_label` is known by construction. That is a deliberately different, cheaper source of ground truth than judging real end-to-end RAG answers. The pipeline runs as follows:
  1. `instructor/sample_chunks.py` draws 400 of the 922 chunks with a fixed seed and pins them in `data/06_claims/sampled_chunks.json`, since `03_chunks/` is gitignored.
  2. Claude Code subagents, not an API script, write the claims (8 agents of 50 chunks), so the shared key stays for students. Their instructions are committed in `instructor/prompts/faithful_claim.md`: a faithful claim must be *deducible* from the chunk alone, not necessarily a paraphrase. It may compare two stated figures but never compute a new one, and it is at most 20 words.
  3. `instructor/generate_paragraph_claims.py <batch_dir>` merges the agents' batch files, joins each claim to its chunk text and checks the result. The batches live outside the repo and are deleted afterwards.
  4. A second set of subagents verifies every claim against its chunk, and flagged claims are reviewed by hand. Both halves came in under the 2% threshold that would have meant revising the prompt: 3 wrong labels out of 400 faithful claims, 4 out of 400 unfaithful ones, all corrected.

  Each unfaithful claim applies one distortion from the taxonomy (preparation.md §6, translated to English in `unfaithful_claim.md`), chosen by the writer as the most natural for its chunk; `error_type` holds the English label and `distortion_note` says what was changed. `generate_paragraph_claims.py` also prints how often each class shows surface features a judge could use without reading the chunk (length, years, absolute wording, "says"). Watch that table: the first unfaithful prompt produced absolute wording ("never", "consists of") in 45% of claims against 7% of faithful ones, which is why the prompt now discourages it.


- **Students run the judge on all the claims themselves** (Grégoire's review of PR #14), through `judge_all`. `instructor/run_judge_scores.py` still produces reference scores in `judge_scores.json`, used for the §6.2 quality pass and as a fallback on the day.
- **The labeled/proxy split for GLIDE is drawn live in the notebook** with `UniformSampler`, not precomputed, so students see GLIDE's sampling API in action.
- Unfaithful claims must stay *subtly* wrong (in the spirit of the Contresens/Troncature/Simplification taxonomy categories), not absurdly wrong — an easy claim the judge always catches produces no bias for GLIDE to visibly correct, which kills the intended "aha" moment when comparing the naive judge-mean estimate to the GLIDE debiased estimate.
- Retriever evaluation (precision@k / MAP@k) is explicitly out of scope for this iteration — do not reintroduce per-chunk relevance labeling.
- Per-student API keys, a LiteLLM proxy, and automated per-key budget enforcement are explicitly deferred to a future edition — don't build them into this year's version.
