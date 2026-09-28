<p align="center">
  <img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/docs/assets/logo-ed.jpg" alt="Emerton Data" height="50" align="middle">
  &nbsp;&nbsp;&nbsp;
  <img src="https://raw.githubusercontent.com/EmertonData/glide/refs/heads/main/docs/assets/logo-glide-white-bg.png" alt="GLIDE" height="100" align="middle">
</p>

# 🤖 GenAI in Practice

This repository contains the two practical sessions of the GenAI in Practice course, held on September 30th, 2026 for you, the X-HEC DSAIB M2 students. Each session has its own notebook:

- 🛠️ **Session 2, Build** (`notebooks/01_build_agentic_rag.ipynb`): you build a RAG agent that answers due-diligence questions on NVIDIA's annual reports and earnings calls.
- 🔍 **Session 4, Evaluate Faithfulness** (`notebooks/02_evaluate_faithfulness.ipynb`): you measure how often an agent's answers are faithful to its sources, with an LLM-as-Judge and prediction-powered inference ([GLIDE](https://github.com/EmertonData/glide)).

## ⚙️ Setup, before the course

This takes about ten minutes. Please do it before the day, since the first download is the slowest part.

**1. Install uv**, the tool that manages Python and the project's packages. On macOS or Linux:

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
```

On Windows, in PowerShell:

```powershell
powershell -ExecutionPolicy ByPass -c "irm https://astral.sh/uv/install.ps1 | iex"
```

Then open a new terminal. Other ways to install it are listed in the [uv documentation](https://docs.astral.sh/uv/getting-started/installation/). You don't need to install Python yourself: uv downloads Python 3.12 or later if your machine doesn't have it.

**2. Get the project and install its packages:**

```bash
git clone https://github.com/EmertonData/genai-practice.git
cd genai-practice
uv sync
```

`uv sync` creates a `.venv` folder in the project, holding the exact package versions listed in `uv.lock`.

**3. Download the embedding model** used in Session 2. This also checks that the installation worked:

```bash
uv run python -c "from sentence_transformers import SentenceTransformer; SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')"
```

**4. Open the notebooks** in your IDE, and select the Python interpreter of the project's `.venv` as the kernel.

## 🔑 On the day: the API key

Both sessions call Claude Haiku through one API key, shared by the whole room and valid only for the day. Your instructors will give it to you. Copy `.env.example` to `.env` (`cp .env.example .env` on macOS or Linux, `copy .env.example .env` on Windows), then paste the key after `ANTHROPIC_API_KEY=`. Please use it sparingly: every call costs real money, and the budget is shared.

## 🧗 About the corrections

You are going to do several exercises, all of them in the notebooks. The astute reader may notice that corrections are provided too. If you really want to learn something today, please do not consult them. This is your skill after all, take care of it. A few quotes to convince you that looking at corrections is the worst thing you can do during this day:

> "What I cannot create, I do not understand."
> Richard Feynman, Nobel Prize in Physics 1965

> "During your studies, you have to work hard at reshaping your mind, your brain, and that is what makes the beauty of any field... The time spent searching, the time spent checking the solution, the time spent understanding: the path matters more than the result. Give our students the ability to decide, after they have accepted the effort (or the intellectual discomfort without which no thinking happens) and sharpened their critical mind."
> Cédric Villani, Fields Medal 2010

> "Using AI to solve a problem is like asking a helicopter to drop you at the top of a mountain. No effort needed, great, you can take a photo at the summit. But the only thing you will have learned is how to get into a helicopter... We want to teach them to climb the mountain."
> Hugo Duminil-Copin, Fields Medal 2022

Our recommendation is to look at the corrections only once this course is finished, so that during this day you are entirely focused on thinking by yourself. 💪
