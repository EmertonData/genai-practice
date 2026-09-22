# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository. It's tracked in git and shared between both maintainers — keep it accurate and useful for either of them, not written to just one person.

## Memory

Claude Code sessions working in this repo should proactively update the auto-memory system (project/feedback/user memories) whenever a step is completed, a decision gets locked in, or just periodically during a long session — don't wait for an explicit "update your memory" request before doing so.

## Current state of this repo

Session 2's notebook has its first exercise (`chunk_string`, §5.1) plus the cells that chunk the corpus; the rest of §5 is unwritten, and `notebooks/02_evaluate_faithfulness.ipynb` is still just a title cell. `instructor/*.py` are still stubs. Don't add descriptions ahead of the actual exercise content — describe an exercise when it's authored.

Dependencies are added as each exercise needs them (see the dependency policy below). The ReAct loop uses **LangGraph**: issue #7 specifies `make_agentic_rag(...) -> StateGraph`, which settles the §5.6 question. Once a test setup exists, add the commands here.

### Notebook conventions 

- **One cell = one job**, each preceded by a short markdown cell. Keep both short, avoid the "wall of text" risk once eight exercises each carry commentary.
- **Exercise statements follow a fixed order**: motivation and stakes → objective → task ("complete the function, respecting its signature and docstring") → hint, placed after the code cell.
- **One exercise = one function**, signature typed and numpy docstring given, body left to the student, marked with `# YOUR CODE HERE` and nothing else (no `raise NotImplementedError`). Ship the simplest thing that works; improvements go in a 3-bullet synthesis at the very end of the finished notebook, not after each exercise.
- **Plumbing is given, not exercised.** Anything that teaches Python rather than RAG (reading a PDF, building a dict, writing JSON) lives in `utils/build.py` and is imported. Note the constraint: a helper in that module cannot call a function the student writes in the notebook, and putting the student's function in the module would hand them the solution — so composition happens in the notebook.
- **Checks are inline `assert`s** on a small, representative example, ending in a `print("OK: …")`. No assert message: `assert x == y, x` reads like a stray tuple.
- **Write prose, not telegrams, and avoid the tells of AI-written text.** Grégoire's PR #8 review named three: em-dashes, the binary "X, not Y" rhythm, and bullet lists of bare fragments. A section should read like a book, so introduce a list with a full sentence ("The next step is to orchestrate chunking across a list of documents. It proceeds as follows:") and make each item a sentence. Prefer the established term over an approximation (an `index`, not "one list"). Never address the instructor in student-facing text.
- Notebooks are committed with **outputs cleared**.

## Commands

Environment is managed with [uv](https://docs.astral.sh/uv/); Python **>=3.12 is required** (`glide-py`, needed later for §6-§7, doesn't support earlier versions).

- `uv sync` — install/update the environment from `uv.lock`.
- `uv run python <script>` — run a script (e.g. `instructor/generate_paragraph_claims.py`) inside the project environment.
- `uv add <package>` / `uv remove <package>` — change dependencies (updates `pyproject.toml` and `uv.lock` together; don't hand-edit the dependency list and forget to re-lock).
- `uv lock` — re-resolve and refresh `uv.lock` after a manual `pyproject.toml` edit.

**Dependency policy: add packages incrementally, not upfront.** Introduce each dependency via `uv add <package>` only when the script/exercise that actually needs it is being written (e.g. add `sentence-transformers` when writing §5.2 `vectorize_text`, `rank_bm25` when writing §5.5, `glide-py` when writing §6.3/§7.3). This was explicit review feedback from Grégoire on PR #3 — the goal is avoiding a bloated, slow-to-resolve venv full of packages nothing uses yet. Don't front-load the full anticipated dependency list again.

Notes:
- **`utils/` is a real local Python package**, installed into the venv by `uv sync` via hatchling (`[build-system]` + `[tool.hatch.build.targets.wheel] packages = ["utils"]`). Notebooks therefore import it as `from utils.build import ...` from anywhere. Never go back to `sys.path.append("../src")`: Grégoire flagged that on PR #8 as the thing to avoid.
- The PyPI package `glide-py` imports as `import glide` (not `import glide_py`); the sampler classes referenced in preparation.md §6.3 live at `glide.samplers` (`StratifiedSampler`, `UniformSampler`, etc.) — confirmed present when it was briefly installed to verify this, but it's not currently a dependency (see policy above) — add it back with `uv add glide-py` when actually implementing §6.3/§7.3.
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

- **Session 2 — Build** (`notebooks/01_build_agentic_rag.ipynb`): students build a RAG agent over a financial document (chunking → embedding → vector search → a ReAct agent loop). BM25 and hybrid fusion are likely dropped — the guiding rule is the simplest RAG that works (§5.5).
- **Session 4 — Evaluate Faithfulness** (`notebooks/02_evaluate_faithfulness.ipynb`): students use LLM-as-judge scoring plus `glide-py` (prediction-powered inference / GLIDE) to produce a debiased faithfulness-rate estimate with a confidence interval, and compare it against the naive judge-mean baseline.

Session 4 depends on Session 2 being fully finished (exercises + corrections) first — there is a deliberate ordering constraint: don't touch afternoon material until the morning build, corrections included, is done.

## Target repo skeleton (from preparation.md §1.2)

```
instructor/                     # instructor-only scripts (not shown to students)
  generate_paragraph_claims.py  # taxonomy-driven claim generation
  run_judge_scores.py           # precompute judge faithfulness scores
utils/                          # local package, installed by uv sync
  __init__.py
  build.py                      # helpers given to students: load_document, build_chunk_records, save_chunks
notebooks/
  01_build_agentic_rag.ipynb    # imports the helpers with `from utils.build import ...`
  02_evaluate_faithfulness.ipynb
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
    paragraph_claims.json       # output of instructor/generate_paragraph_claims.py
    judge_scores.json           # output of instructor/run_judge_scores.py
pyproject.toml
uv.lock
.env.example
README.md
```

`instructor/` scripts are one-time, instructor-run batch jobs against the shared API key (see below) — they exist so dataset regeneration (e.g. a source PDF or taxonomy change) is reproducible instead of manual. They are never distributed to students. (Named `instructor/`, not `build/`, to avoid colliding with the conventional meaning of a `build/` directory in Python packaging.)

**`data/` stages 01 and 02 are tracked; 03 and 04 are gitignored** apart from their `.gitkeep`, since chunks and vectors are produced during the workshop. Only 2025 and 2026 documents are in scope — the 2024 files were moved out of the repo to `~/Documents/code/genai-practice-archive/`.

## Key architectural decisions to preserve

- **Single shared Anthropic API key** for all 70 students plus instructor prep/judge runs (`ANTHROPIC_API_KEY`, the only env var any notebook needs). There's no per-key model restriction, so notebooks must pin students to Haiku by default; a console-side spend limit is the backstop, not the primary control.
- **Ground truth for faithfulness is constructed, not annotated.** `paragraph_claims.json` is built by prompting a strong LLM (Opus) to (1) paraphrase each chunk faithfully and (2) rewrite it unfaithfully per applicable category from the fixed 10-category hallucination taxonomy in preparation.md §6. This yields several hundred (paragraph, claim, true_label, error_type) pairs with `true_faithfulness_label` known by construction — a deliberately different, cheaper source of ground truth than judging real end-to-end RAG answers.
- **Judge scores are precomputed centrally** (`instructor/run_judge_scores.py` → `judge_scores.json`) rather than having every student call the judge on the same fixed data, to protect the shared spend limit and keep everyone's proxy labels identical. Students only make one or two *live* judge calls themselves, for illustration.
- **The labeled/proxy split for GLIDE is drawn live in the notebook**, not precomputed — students call a `glide.samplers` sampler (e.g. `StratifiedSampler` stratified on `error_type`, or `UniformSampler`) against `paragraph_claims.json` to pick which ids get their true label "revealed." This is intentional: students should see GLIDE's sampling API in action, not just its estimators.
- Unfaithful claims must stay *subtly* wrong (in the spirit of the Contresens/Troncature/Simplification taxonomy categories), not absurdly wrong — an easy claim the judge always catches produces no bias for GLIDE to visibly correct, which kills the intended "aha" moment when comparing the naive judge-mean estimate to the GLIDE debiased estimate.
- Retriever evaluation (precision@k / MAP@k) is explicitly out of scope for this iteration — do not reintroduce per-chunk relevance labeling.
- Per-student API keys, a LiteLLM proxy, and automated per-key budget enforcement are explicitly deferred to a future edition — don't build them into this year's version.
