"""Precompute LLM-as-judge faithfulness scores (preparation.md §7.2).

Instructor-only, run once against the shared API key over every claim in
data/06_claims/paragraph_claims.json, with the judge prompt in
instructor/prompts/judge_system_prompt.md. Writes data/06_claims/judge_scores.json and prints
the judge's accuracy per class and per error type, the naive judge estimate of the faithfulness
rate next to the true one, and the cost and time of the run.

Usage, from the repository root so that `corrections` can be imported:
uv run python -m instructor.run_judge_scores
"""

import json
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic

from instructor.measure_judge import judge

PARAGRAPH_CLAIMS = Path("data/06_claims/paragraph_claims.json")
JUDGE_SYSTEM_PROMPT = Path("instructor/prompts/judge_system_prompt.md")
JUDGE_SCORES = Path("data/06_claims/judge_scores.json")

MODEL = "claude-haiku-4-5-20251001"
# Haiku 4.5 prices, in dollars per million tokens
INPUT_PRICE = 1.00
OUTPUT_PRICE = 5.00
# Calls sent to the API at the same time
WORKERS = 8


def judge_all(rows: list[dict], llm: ChatAnthropic, system_prompt: str, workers: int) -> list[dict]:
    """Judge every claim, several calls at a time.

    Parameters
    ----------
    rows : list[dict]
        The rows of paragraph_claims.json.
    llm : ChatAnthropic
        The judge model.
    system_prompt : str
        The judge's instructions.
    workers : int
        Number of calls sent to the API at the same time.

    Returns
    -------
    list[dict]
        The results of `judge`, in the same order as `rows`.
    """
    with ThreadPoolExecutor(max_workers=workers) as pool:
        return list(pool.map(lambda row: judge(row, llm, system_prompt), rows))


def to_scores(results: list[dict]) -> list[dict]:
    """Keep what later steps need from each result: the claim id, the verdict and its reasoning.

    Parameters
    ----------
    results : list[dict]
        The results of `judge`.

    Returns
    -------
    list[dict]
        One row per claim with "claim_id", "verdict" (None if the call failed), "reasoning",
        "input_tokens" and "output_tokens".
    """
    return [
        {
            "claim_id": r["row"]["claim_id"],
            "verdict": r["verdict"],
            "reasoning": r["reasoning"],
            "input_tokens": r["input_tokens"],
            "output_tokens": r["output_tokens"],
        }
        for r in results
    ]


def summarize(results: list[dict], prices: tuple[float, float], seconds: float) -> None:
    """Print the judge's accuracy per class and per error type, its naive estimate, and the run's cost.

    Parameters
    ----------
    results : list[dict]
        The results of `judge`.
    prices : tuple[float, float]
        Input and output prices of the judge model, in dollars per million tokens.
    seconds : float
        Wall-clock duration of the run.

    Returns
    -------
    None
    """
    scored = [r for r in results if r["verdict"] is not None]
    print(f"{len(scored)}/{len(results)} claims scored, {len(results) - len(scored)} failed calls")

    for kind, label in [("faithful", 1), ("unfaithful", 0)]:
        judged = [r for r in scored if r["row"]["true_faithfulness_label"] == label]
        correct = sum(r["verdict"] == label for r in judged)
        print(f"{kind}: {correct}/{len(judged)} judged correctly ({correct / len(judged):.1%})")

    print("\nunfaithful claims caught, per error type:")
    error_types = sorted({r["row"]["error_type"] for r in scored if r["row"]["error_type"]})
    for error_type in error_types:
        judged = [r for r in scored if r["row"]["error_type"] == error_type]
        caught = sum(r["verdict"] == 0 for r in judged)
        print(f"  {error_type:16} {caught:>3}/{len(judged):<3} ({caught / len(judged):.0%})")

    true_rate = sum(r["row"]["true_faithfulness_label"] for r in scored) / len(scored)
    judge_rate = sum(r["verdict"] for r in scored) / len(scored)
    print(f"\nfaithfulness rate: judge says {judge_rate:.1%}, truth is {true_rate:.1%}")

    input_price, output_price = prices
    input_tokens = sum(r["input_tokens"] for r in results)
    output_tokens = sum(r["output_tokens"] for r in results)
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    print(f"run: {input_tokens} input + {output_tokens} output tokens, ${cost:.2f}, {seconds / 60:.1f} min")


def main() -> None:
    """Judge every claim, write the scores and print the summary.

    Returns
    -------
    None
    """
    load_dotenv()
    rows = json.loads(PARAGRAPH_CLAIMS.read_text())
    system_prompt = JUDGE_SYSTEM_PROMPT.read_text()
    llm = ChatAnthropic(model=MODEL, temperature=0.0, max_tokens=150)

    start = time.perf_counter()
    results = judge_all(rows, llm, system_prompt, WORKERS)
    seconds = time.perf_counter() - start

    JUDGE_SCORES.write_text(json.dumps(to_scores(results), indent=2))
    print(f"wrote {len(results)} scores -> {JUDGE_SCORES}\n")
    summarize(results, (INPUT_PRICE, OUTPUT_PRICE), seconds)


if __name__ == "__main__":
    main()
