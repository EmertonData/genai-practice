# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It's shared between both maintainers — keep it accurate and useful for either of them, not written to just one person. It was kept local during the course so students wouldn't read it, and has been tracked again since the course ended.

## Memory

Claude Code sessions working in this repo should proactively update the auto-memory system (project/feedback/user memories) whenever a step is completed, a decision gets locked in, or just periodically during a long session — don't wait for an explicit "update your memory" request before doing so.

## What this project is

Course materials for a one-day "GenAI in Practice" course (70 students, X-HEC DSAIB M2) prepared by Emerton Data. Two hands-on practice sessions, each a Jupyter notebook of exercises with a separate correction script:

- **Session 2 — Build** (`notebooks/session_2_build_agentic_rag.ipynb`): students build a RAG agent over NVIDIA's 10-K filings and earnings calls (chunking → embedding → vector search → a LangGraph ReAct agent).
- **Session 4 — Evaluate Faithfulness** (`notebooks/session_4_evaluate_faithfulness.ipynb`): students write an LLM-as-Judge, run it on 800 claims, and use `glide-py` (prediction-powered inference) to get a debiased faithfulness estimate with a confidence interval, compared against the judge alone and humans alone.

## Current state

**The first edition was delivered on 2026-09-30.** Both notebooks and their corrections are on `main`. The shared API key has been revoked, so a new edition needs a new key. The quiz ran on Menti, and no certification was given. The hand-over for whoever runs the next edition is in `instructor/README.md`; the original preparation checklist, with what was done and what was dropped, is `instructor/preparation.md`.

The one preparation item not done is the §6.2 quality pass on the unfaithful claims (not done in detail, for lack of time). It turned out not to be needed for the GLIDE gap to be visible.

## Session 2 notebook

It runs in two parts. Part 1 builds the retrieval pipeline: `chunk_string` (§5.1), `embed_texts` (§5.2) and `top_k_search` (§5.4). Part 2 builds the agent: the two tools (`search_filings` and `calculator`) and the LangGraph loop in `make_agentic_rag` (§5.6). §5.3 ships as the given helpers `save_vectors`/`load_vectors`, and §5.5 (BM25) was dropped. Every exercise has its correction in `corrections/correction_build.py`.

The agent is a hand-built `StateGraph` rather than LangChain's prebuilt `create_agent`, so the conditional edge and the loop-back edge stay visible. `State` lives in `utils/build.py` and is given. `search_filings` takes its model, vectors and chunks as parameters, and the notebook wraps it in a small `@tool` so the LLM only ever chooses `query` and `k` — a `functools.partial` cannot be used here, because `@tool` rejects an object with no `__name__`.

**Measured, so don't re-derive:** the notebook runs top to bottom in 43 seconds, and its five agent calls cost $0.033 per run (24,259 input / 1,659 output tokens at Haiku 4.5 rates).

**Dense retrieval is weak on this corpus, a known limitation.** On the 7 questions whose reference answer carries a distinctive number, `all-MiniLM-L6-v2` finds the answer chunk in 2/7 at k=5 against 5/7 for a throwaway BM25, and the chunk holding NVIDIA's FY2026 revenue ranks 23rd of 922 for the question that asks for it. It works on thematic questions and fails on table lookups, which is why the §5.4 demo cell uses question 12 rather than question 1. BM25 was dropped anyway, to keep the simplest RAG that works.

## Session 4 notebook

It follows an engineer evaluating their own agent (see the narrative convention below). Part 1 is the judge: Exercise 1 writes `JUDGE_PROMPT` from rules given as sentences and an imposed JSON format (`reasoning` first, then `verdict`); Exercise 2 is `score_faithfulness(chunk, claim, llm, system_prompt)`, checked by a `FakeJudge` that records the messages it receives. The given `judge_all` (`utils/evaluate.py`) runs it on all 800 claims in parallel and returns a float array (NaN for a failed call); `save_verdicts`/`load_verdicts` store it in `data/07_verdicts/verdicts.json`, with `data/06_claims/judge_scores.json` as the fallback. Part 2 is GLIDE: Exercise 3 `choose_claims_to_annotate` (`UniformSampler`), Exercise 4 `estimate_faithfulness` (the judge alone, humans alone and PPI), a given plotly plot, and Exercise 5 `is_above_threshold`, a one-sided test for a 50% deployment rule. Details and the verified `glide-py` 0.11.0 API are in `instructor/preparation.md` §7.

The judge's prompt is adapted from Fig. 6 of Grégoire's paper (arXiv 2507.21753), which the notebook does not cite. It is `JUDGE_PROMPT` in `corrections/correction_eval.py`, its only copy: the instructor scripts import it. It is zero-shot, on Grégoire's advice that examples bias the judge, and it forbids outside knowledge, unlike the paper's step 5. Don't tune the prompt on the judge's errors: its leniency is the bias GLIDE corrects.

**Measured on the judge, so don't re-derive** (all 800 claims, Haiku at temperature 0): 399/400 faithful claims judged correctly and 361/400 unfaithful ones, so the judge is lenient. It says 54.8% of claims are faithful where the truth is 50.0%. It misses mostly Simplification (38% caught), then Acronym and Truncation (80%) and Exaggeration (85%); everything else is caught at 97-100%. A run costs $0.90 and takes 2.7 minutes with 8 calls in parallel (about 600 input and 105 output tokens per claim). The prompt asks for reasoning first, under 100 words, with `max_tokens=200`: a cap of 150 truncated 23 replies. Dropping the `reasoning` field does not save tokens: Haiku then reasons in free text before the JSON, which broke 15 replies out of 20. The shared key's limits were 10,000 requests, 10M input and 2M output tokens per minute; one student uses about 330 requests a minute.

## Notebook conventions

- **One cell = one job**, each preceded by a short markdown cell. Keep both short, avoid the "wall of text" risk once eight exercises each carry commentary.
- **Exercise statements follow a fixed order**: motivation and stakes → objective → task ("complete the function, respecting its signature and docstring") → hint, placed after the code cell.
- **One exercise = one function**, signature typed and numpy docstring given, body left to the student. An exercise may group two closely related functions under one heading when they are one idea (§5.6 does this for the two tools). Ship the simplest thing that works; improvements go in a numbered synthesis at the very end of the notebook, not after each exercise.
- **As much as possible, no code with blanks.** A stub is the signature, the numpy docstring and `# YOUR CODE HERE`. This is Grégoire's preference.
- **When an exercise has steps, they go as comments in the function body, never as a list in the markdown** (Grégoire's review of PR #15). Each step is a `# STEP n - ...` comment saying what to do, followed by `# YOUR CODE HERE`, with no partial code. The markdown keeps the motivation, the objective and the task. Never `raise NotImplementedError`.
- **Tell the story of an engineer evaluating their own system, never of the dataset's designer** (Grégoire's review of PR #15). Students don't learn how the claims were built. Human annotations are treated as rare and costly, as they are in real life: never compute statistics over all the true labels, and never state the hallucination rate, which is an output of the protocol.
- **Plumbing is given, not exercised.** Anything that teaches Python rather than RAG (reading a PDF, building a dict, writing JSON) lives in `utils/` and is imported. A helper in that package cannot call a function the student writes in the notebook, and putting the student's function in the package would hand them the solution — so composition happens in the notebook.
- **Checks are inline `assert`s** on a small, representative example, ending in a `print("OK: …")`. No assert message: `assert x == y, x` reads like a stray tuple.
- **Write prose, not telegrams, and avoid the tells of AI-written text.** Grégoire's PR #8 review named three: em-dashes, the binary "X, not Y" rhythm, and bullet lists of bare fragments. A section should read like a book, so introduce a list with a full sentence and make each item a sentence. Prefer the established term over an approximation (an `index`, not "one list"). Never address the instructor in student-facing text.
- Notebooks are committed with **outputs cleared** and with `source` stored as a list of lines. Both are enforced by the pre-commit hooks in `prek.toml` (`nbstripout` and `normalize-notebooks`), adopted from the GLIDE repo on Grégoire's suggestion. They exist because editing a cell programmatically rewrites `source` as one long string, which turns the GitHub diff into an unreadable blob. `nbstripout` also renumbers cell ids to sequential integers, so inserting a cell mid-notebook renumbers everything after it; appending at the end does not.

## Commands

Environment is managed with [uv](https://docs.astral.sh/uv/); Python **>=3.12 is required** (`glide-py` doesn't support earlier versions).

- `uv sync` — install/update the environment from `uv.lock`.
- `uv run python <script>` — run a script (e.g. `instructor/run_judge_scores.py`) inside the project environment.
- `uv add <package>` / `uv remove <package>` — change dependencies (updates `pyproject.toml` and `uv.lock` together; don't hand-edit the dependency list and forget to re-lock).
- `uv lock` — re-resolve and refresh `uv.lock` after a manual `pyproject.toml` edit.
- `uv sync --group dev` — **what maintainers run.** Adds the tooling in the `dev` dependency group (`prek`, `ruff`, `ty`) on top of the student environment.
- `uv run --group dev prek install` — **install the git pre-commit hooks, once per clone.** Without this the hooks never run and the problems they prevent come back.
- `uv run --group dev prek run --all-files` — run every hook over the whole repo, rather than only on staged files.
- `make lint` / `make type-check` — `ruff check --fix` and `ty check`. **Both must pass with no `# noqa` and no suppressions**, since a suppression hides a real problem rather than fixing it. The one documented exception is `StateGraph(State)` in `corrections/correction_build.py`: `ty` rejects it, and rejects LangGraph's own `MessagesState` identically, so no spelling of that line can satisfy it.

There is no test suite: the checks are the inline `assert`s in the notebooks.

**Students run plain `uv sync` and get none of the tooling.** `[tool.uv] default-groups = []` overrides uv's habit of installing the `dev` group by default, which keeps the student install limited to what the notebooks actually import. Anything added for our own workflow belongs in the `dev` group, never in `[project] dependencies`.

**Dependency policy: add packages only when the code that needs them is written** (Grégoire's review of PR #3), to avoid a bloated, slow-to-resolve venv full of packages nothing uses.

Notes:
- **`utils/` is a real local Python package**, installed into the venv by `uv sync` via hatchling (`[build-system]` + `[tool.hatch.build.targets.wheel] packages = ["utils"]`). Notebooks therefore import it as `from utils.build import ...` from anywhere. Never go back to `sys.path.append("../src")`: Grégoire flagged that on PR #8 as the thing to avoid.
- The PyPI package `glide-py` imports as `import glide`.
- `uv.lock` is committed (unlike `.venv/`, which is gitignored) so every maintainer and student gets identical resolved versions.

## Who is doing what

**Benjamin** and his manager **Grégoire** co-prepared this course at Emerton Data. Benjamin was principally responsible for the practical part — the notebooks, the environment, and the evaluation dataset. Grégoire reviews PRs and co-decides cross-cutting calls (source document, spend cap, correction-release policy, framework choice, distribution).

## Git workflow

Both maintainers push to this repo, so use branches and PRs, not direct commits to `main`:

- **Never commit straight to `main`.** Create a branch for each unit of work and open a PR.
- **Branch naming**: `<type>/<section-num>-<slug>`, e.g. `feat/5.1-chunk-text`, `docs/1.4-readme`, matching the Conventional Commits type used for the branch's main commit.
- **Commit messages: Conventional Commits** (`feat:`, `fix:`, `docs:`, `chore:`, `test:`, `refactor:`), scoped to what actually changed.
- **Commit at natural checkpoints**, not mid-edit. Proactively suggest a commit at these points rather than waiting to be asked, and suggest opening a PR once a branch's unit of work is complete.
- Pull/rebase on latest `main` before starting new work on a branch.
- Creating branches, committing, pushing, and opening PRs are all done by asking first each time (push and PR creation are shared/visible actions).
- Commit messages, PR titles/descriptions, and code comments should not mention Claude, Anthropic, or AI-generated authorship — no `Co-Authored-By: Claude` trailer, no "Generated with Claude Code" footer, nothing equivalent.

## Repo layout

```
instructor/                     # maintainers' scripts and notes (not part of the student flow)
  README.md                     # hand-over for the next edition
  preparation.md                # the preparation checklist, with what was done and dropped
  preprocess_documents.py       # trim data/01_raw/ into data/02_processed/
  sample_chunks.py              # draw the 400 chunks the claims are written from
  prompts/faithful_claim.md     # instructions given to the faithful-claim subagents
  prompts/unfaithful_claim.md   # same for unfaithful claims, with the English taxonomy table
  prompts/write_claims.md       # the messages that launch the claim-writing subagents
  prompts/verify_claims.md      # the messages that launch the verification subagents
  generate_paragraph_claims.py  # merge and check the subagents' claim batches
  measure_judge.py              # check the judge on 20 claims
  run_judge_scores.py           # score all 800 claims into judge_scores.json
utils/                          # local package, installed by uv sync
  build.py                      # Session 2 helpers: loading documents, chunk records, vectors, State
  evaluate.py                   # Session 4 helpers: judge_all, save/load_verdicts, plot_estimates
notebooks/
  session_2_build_agentic_rag.ipynb
  session_4_evaluate_faithfulness.ipynb
  assets/                       # diagrams (PNG shipped; SVG sources gitignored)
corrections/
  correction_build.py
  correction_eval.py            # also holds JUDGE_PROMPT
data/                           # numbered pipeline stages
  01_raw/                       # full filings + earnings calls, 2025 and 2026 (tracked)
  02_processed/                 # trimmed to substantive pages (tracked)
  03_chunks/                    # generated in the course (gitignored)
  04_vectors/                   # generated in the course (gitignored)
  05_questions/questions.json   # 20 due-diligence questions
  06_claims/                    # sampled_chunks.json, paragraph_claims.json, judge_scores.json (tracked)
  07_verdicts/                  # the students' own judge verdicts (gitignored)
```

`instructor/` scripts are one-time, maintainer-run batch jobs, so that regenerating the data (e.g. after a source PDF or taxonomy change) is reproducible instead of manual. The claims themselves were written by Claude Code subagents, not by a script: the scripts only sample and merge. Only 2025 and 2026 documents are in scope.

## Key architectural decisions to preserve

- **A single shared Anthropic API key** for all students plus maintainer runs (`ANTHROPIC_API_KEY`, the only env var any notebook needs). There's no per-key model restriction, so notebooks pin students to Haiku; a console-side spend limit is the backstop, not the primary control. Per-student keys, a LiteLLM proxy and automated per-key budgets were deferred to a future edition.
- **Ground truth for faithfulness is constructed, not annotated.** Each claim is written to be faithful or unfaithful, so its `true_faithfulness_label` is known by construction. The pipeline runs as follows:
  1. `instructor/sample_chunks.py` draws 400 of the 922 chunks with a fixed seed and pins them in `data/06_claims/sampled_chunks.json`, since `03_chunks/` is gitignored.
  2. Claude Code subagents, not an API script, write the claims (8 agents of 50 chunks), so the shared key stays for students. Their instructions are in `instructor/prompts/`: a faithful claim must be *deducible* from the chunk alone, not necessarily a paraphrase. It may compare two stated figures but never compute a new one, and it is at most 20 words.
  3. `instructor/generate_paragraph_claims.py <batch_dir>` merges the agents' batch files, joins each claim to its chunk text and checks the result. The batches live outside the repo and were deleted afterwards.
  4. A second set of subagents verified every claim against its chunk (`instructor/prompts/verify_claims.md`), and flagged claims were reviewed by hand: 3 wrong labels out of 400 faithful claims, 4 out of 400 unfaithful ones, all corrected.

  Each unfaithful claim applies one distortion from the taxonomy (`instructor/preparation.md` §6, translated to English in `unfaithful_claim.md`); `error_type` holds the English label and `distortion_note` says what was changed. `generate_paragraph_claims.py` also prints how often each class shows surface features a judge could use without reading the chunk (length, years, absolute wording, "says"). Watch that table: the first unfaithful prompt produced absolute wording in 45% of claims against 7% of faithful ones, which is why the prompt now discourages it.
- Unfaithful claims must stay *subtly* wrong, not absurdly wrong — an easy claim the judge always catches produces no bias for GLIDE to visibly correct.
- **Students run the judge on all the claims themselves** (Grégoire's review of PR #14), through `judge_all`. `instructor/run_judge_scores.py` produces the reference scores in `judge_scores.json`, used as a fallback on the day.
- **The labeled/proxy split for GLIDE is drawn live in the notebook** with `UniformSampler`, not precomputed, so students see GLIDE's sampling API in action.
- Retriever evaluation (precision@k / MAP@k) is out of scope — do not reintroduce per-chunk relevance labeling.
