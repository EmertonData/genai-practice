# 🧑‍🏫 GenAI in Practice: instructor guide

This guide is for whoever runs the next edition of the course or maintains this repository. The student-facing setup is in the root `README.md`; this one covers everything around it.

## 📚 The course

GenAI in Practice is a one-day course by Emerton Data. Its first edition was held on September 30th, 2026 for 70 X-HEC DSAIB M2 students. Sessions 1 and 3 are lectures, whose slides are not in this repository. Sessions 2 and 4 are the practical sessions, each a notebook of exercises:

- **Session 2, Build** (`notebooks/session_2_build_agentic_rag.ipynb`): students chunk NVIDIA's 10-K filings and earnings calls, embed the chunks with a local model, write a top-k search, and assemble a ReAct agent in LangGraph with a search tool and a calculator.
- **Session 4, Evaluate Faithfulness** (`notebooks/session_4_evaluate_faithfulness.ipynb`): students write an LLM-as-Judge prompt and function, run the judge on 800 claims, then use [GLIDE](https://github.com/EmertonData/glide) to combine the judge's verdicts with 100 human annotations into a debiased faithfulness estimate, and decide whether the agent can be deployed.

Every exercise has its correction in `corrections/`. The corrections were available from the start, with a warning in the student README asking students not to read them during the day.

The day closed with a quiz on Menti, mixing questions on the course with feedback questions. Students would recommend the course to next year's students at 9.5/10.

## ⚙️ Setup

You need [uv](https://docs.astral.sh/uv/getting-started/installation/); it installs Python 3.12 or later if needed (`glide-py` requires it). Clone the repository and install the environment with the maintainers' tooling:

```bash
git clone https://github.com/EmertonData/genai-practice.git
cd genai-practice
uv sync --group dev
uv run --group dev prek install
```

The `dev` group adds `prek`, `ruff` and `ty`, which students don't get with a plain `uv sync`. The second command installs the git pre-commit hooks, once per clone. They strip the notebooks' outputs and keep each cell's source as a list of lines, which keeps notebook diffs readable on GitHub.

Copy `.env.example` to `.env` and set `ANTHROPIC_API_KEY`. The key of the first edition has been revoked, so create a new one in the Anthropic Console (see "Preparing an edition" below).

Before committing, run:

```bash
make lint         # ruff check --fix
make type-check   # ty check
uv run --group dev prek run --all-files
```

Lint and type checks must pass without `# noqa` or other suppressions. The one known exception is `StateGraph(State)` in `corrections/correction_build.py`, which `ty` rejects whatever the spelling. There is no test suite: each exercise is checked by inline `assert`s in its notebook.

## 🤖 Working with Claude Code

The repository was co-built with Claude. `CLAUDE.md`, at the root,  holds the project's state, the notebook conventions agreed with reviewers (how an exercise is written, the engineer's narrative, the writing style), the commands, the git workflow and the measurements not to recompute.

If you do use it, keep `CLAUDE.md` up to date as decisions change, since a stale instruction there gets applied in every later session. During the first edition it was kept out of git so students wouldn't read it; it is tracked again now.

`instructor/preparation.md` is the preparation checklist of the first edition, section by section, with what was done, how, and what was dropped. The section numbers used in the code's docstrings (§5.1, §7.2…) refer to it.

## 🗂️ What is in the repository

```
instructor/      # this guide, the preparation checklist, and the scripts that build the data
corrections/     # reference solutions; correction_eval.py also holds the judge prompt
utils/           # helpers given to students (build.py for Session 2, evaluate.py for Session 4)
notebooks/       # the two notebooks, and their diagrams in assets/
data/
  01_raw/        # full 10-K filings and earnings calls, 2025 and 2026
  02_processed/  # the same, trimmed to their substantive pages
  03_chunks/     # chunks, produced by students in Session 2 (gitignored)
  04_vectors/    # embeddings, produced by students in Session 2 (gitignored)
  05_questions/  # 20 due-diligence questions for the agent
  06_claims/     # the 400 sampled chunks, the 800 claims and the reference judge scores
  07_verdicts/   # the students' own judge verdicts (gitignored)
```

## 🔁 Regenerating the data

The tracked data is ready to use, so this is only needed if the source documents, the chunking or the claims change. The steps run in this order:

1. `uv run python instructor/preprocess_documents.py` trims `data/01_raw/` into `data/02_processed/`. The page ranges kept are set at the top of the script.
2. Run the Session 2 notebook with the reference solutions in place, which writes `data/03_chunks/all_chunks.json`.
3. `uv run python instructor/sample_chunks.py` draws 400 of the 922 chunks with a fixed seed into `data/06_claims/sampled_chunks.json`. It stops if the chunk count is not 922, since that means the chunking changed.
4. The claims are written by Claude Code subagents, not by a script, so that the course's API key stays for students. Each subagent takes a batch of 50 chunks and follows `instructor/prompts/faithful_claim.md` or `instructor/prompts/unfaithful_claim.md`. The second file holds the taxonomy of distortions.
5. `uv run python instructor/generate_paragraph_claims.py <batch_dir>` merges the batch files into `data/06_claims/paragraph_claims.json`, checks them, and prints which surface features could give a claim away without reading its chunk. In the first edition, a second set of subagents then checked every claim against its chunk, and the claims they flagged were reviewed by hand.
6. `uv run python -m instructor.measure_judge` checks the judge on 20 claims before a full run.
7. `uv run python -m instructor.run_judge_scores` scores all 800 claims into `data/06_claims/judge_scores.json`. Students use these reference scores if their own run fails.

Steps 6 and 7 call the API, and are run as modules from the repository root so that they can import the judge prompt from `corrections/`.

## 🔑 Preparing an edition

The steps below are the ones that mattered in the first edition:

1. Create one API key for the room in the Anthropic Console, with a spend limit for the day. The notebooks pin students to Claude Haiku, but the key itself doesn't restrict the model, so the spend limit is the safeguard. Give the key to the students on the day, never in the repository, which is public.
2. Update the date and audience in the root `README.md`, and ask students to do its setup before the day, since the first download of the embedding model is the slowest part.
3. Run both notebooks top to bottom with the reference solutions, from a fresh clone.
4. On the day, keep the Console's usage dashboard open, and have a channel with the organizers to send a new key to the whole room if the first one has to be revoked.
5. Revoke the key after the course.

## 📏 Reference figures

These were measured in the first edition, with Claude Haiku 4.5:

- The Session 2 notebook runs top to bottom in 43 seconds, and its five agent calls cost $0.033 per run.
- In Session 4, one run of the judge on the 800 claims costs about $0.90 and takes 2.7 minutes with 8 calls in parallel.
- The judge is lenient: it gets 399 of the 400 faithful claims right but only 361 of the 400 unfaithful ones, so it puts the faithfulness rate at 54.8% where the truth is 50.0%. It misses mostly Simplification claims.
- With 100 human labels and the notebook's seed, humans alone give 47.0% ± 9.8 points and GLIDE gives 50.5% ± 4.9 points, so GLIDE halves the error bar.

## 🔭 Ideas for the next edition

- **Anticipating Environment Setup Issues.** Students might have anaconda installed on their computer which prevents the uv env to be seen by VSCode apparently.
- **Review the unfaithful claims.** The quality pass (`preparation.md` §6.2) was not done in detail, for lack of time. The judge's reference scores show which claims are too easy to catch.
- **Improve retrieval.** Dense retrieval with `all-MiniLM-L6-v2` is weak on table lookups: on 7 questions with a numerical answer it finds the right chunk in 2 at k=5, against 5 for BM25. BM25 was dropped to keep the session short.
- **Give each student a key**, or route calls through a proxy such as LiteLLM, with a budget per key, so that one runaway loop can't use up the whole room's budget.
- **Revisit the exercise format if students use AI assistants heavily.** "Complete this function" exercises assume they don't; "explain or debug this code" exercises would hold up better.
