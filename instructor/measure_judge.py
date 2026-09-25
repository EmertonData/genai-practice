"""Measure the LLM judge on a small sample of claims before it goes into the notebook (preparation.md §7.1).

Instructor-only, a one-off run of about 20 calls against the shared API key. It judges 10
unfaithful claims, one per error type, and 10 faithful claims, then prints the judge's accuracy
per class, its errors with their reasoning, and the measured cost and time projected to the
full dataset and the whole room.

Usage, from the repository root so that `corrections` can be imported:
uv run python -m instructor.measure_judge
"""

import json
import random
import time
from pathlib import Path

from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_core.callbacks import get_usage_metadata_callback

from corrections.correction_eval import score_faithfulness

PARAGRAPH_CLAIMS = Path("data/06_claims/paragraph_claims.json")
JUDGE_SYSTEM_PROMPT = Path("instructor/prompts/judge_system_prompt.md")

MODEL = "claude-haiku-4-5-20251001"
# Haiku 4.5 prices, in dollars per million tokens
INPUT_PRICE = 1.00
OUTPUT_PRICE = 5.00

SEED = 42
N_FAITHFUL = 10
DATASET_SIZE = 800
STUDENTS = 70


def draw_sample(rows: list[dict], n_faithful: int, seed: int) -> list[dict]:
    """Draw one unfaithful claim per error type, plus faithful claims from other chunks.

    Parameters
    ----------
    rows : list[dict]
        The rows of paragraph_claims.json.
    n_faithful : int
        Number of faithful claims to draw.
    seed : int
        Seed of the random draw, so the sample is the same on every run.

    Returns
    -------
    list[dict]
        The unfaithful claims, one per error type, followed by the faithful claims.
    """
    rng = random.Random(seed)
    unfaithful = [row for row in rows if row["true_faithfulness_label"] == 0]
    error_types = sorted({row["error_type"] for row in unfaithful})
    sample = [rng.choice([row for row in unfaithful if row["error_type"] == error_type]) for error_type in error_types]
    # Faithful claims come from chunks not already drawn, so no claim is judged next to its twin
    drawn_chunks = {row["chunk_id"] for row in sample}
    faithful = [row for row in rows if row["true_faithfulness_label"] == 1 and row["chunk_id"] not in drawn_chunks]
    return sample + rng.sample(faithful, n_faithful)


def judge(row: dict, llm: ChatAnthropic, system_prompt: str) -> dict:
    """Judge one claim, recording the verdict, the tokens used and the time taken.

    Parameters
    ----------
    row : dict
        A row of paragraph_claims.json.
    llm : ChatAnthropic
        The judge model.
    system_prompt : str
        The judge's instructions.

    Returns
    -------
    dict
        The judgement ("reasoning", "verdict"), the judged row, the error message if the call
        failed, the time taken in seconds, and the input and output tokens.
    """
    start = time.perf_counter()
    with get_usage_metadata_callback() as usage:
        try:
            judgement = score_faithfulness(row["chunk"], row["claim"], llm, system_prompt)
            error = None
        except ValueError as exc:  # a truncated reply or unparsable JSON
            judgement, error = {"reasoning": None, "verdict": None}, str(exc)
    tokens = usage.usage_metadata[llm.model]
    return {
        **judgement,
        "row": row,
        "error": error,
        "seconds": time.perf_counter() - start,
        "input_tokens": tokens["input_tokens"],
        "output_tokens": tokens["output_tokens"],
    }


def report(results: list[dict], prices: tuple[float, float], dataset_size: int, students: int) -> None:
    """Print the judge's accuracy, its errors, and its cost and time projected to the full run.

    Parameters
    ----------
    results : list[dict]
        The judgements, as returned by `judge`.
    prices : tuple[float, float]
        Input and output prices of the judge model, in dollars per million tokens.
    dataset_size : int
        Number of claims in the full dataset, to project one full run.
    students : int
        Number of students, to project the whole room.

    Returns
    -------
    None
    """
    for kind, label in [("faithful", 1), ("unfaithful", 0)]:
        judged = [r for r in results if r["row"]["true_faithfulness_label"] == label]
        correct = sum(r["verdict"] == label for r in judged)
        print(f"{kind}: {correct}/{len(judged)} judged correctly")
    correct = sum(r["verdict"] == r["row"]["true_faithfulness_label"] for r in results)
    failed = sum(r["error"] is not None for r in results)
    print(f"overall: {correct}/{len(results)} correct, {failed} failed calls")

    print("\nerrors:")
    for r in results:
        if r["verdict"] != r["row"]["true_faithfulness_label"]:
            row = r["row"]
            print(f"- {row['claim_id']} [{row['error_type']}] judged {r['verdict']}: {row['claim']}")
            print(f"  reasoning: {r['reasoning'] or r['error']}")

    input_price, output_price = prices
    input_tokens = sum(r["input_tokens"] for r in results) / len(results)
    output_tokens = sum(r["output_tokens"] for r in results) / len(results)
    cost = (input_tokens * input_price + output_tokens * output_price) / 1_000_000
    seconds = sum(r["seconds"] for r in results) / len(results)
    run_cost = cost * dataset_size
    print(f"\nper claim: {input_tokens:.0f} input + {output_tokens:.0f} output tokens, ${cost:.5f}, {seconds:.1f} s")
    print(f"{dataset_size} claims: ${run_cost:.2f}, {seconds * dataset_size / 60:.0f} min if sequential")
    print(f"{students} students: ${run_cost * students:.0f} per run, ${run_cost * students * 3:.0f} for 3 runs")


def main() -> None:
    """Judge the sample claims and print the report.

    Returns
    -------
    None
    """
    load_dotenv()
    rows = json.loads(PARAGRAPH_CLAIMS.read_text())
    system_prompt = JUDGE_SYSTEM_PROMPT.read_text()
    llm = ChatAnthropic(model=MODEL, temperature=0.0, max_tokens=200)
    results = [judge(row, llm, system_prompt) for row in draw_sample(rows, N_FAITHFUL, SEED)]
    report(results, (INPUT_PRICE, OUTPUT_PRICE), DATASET_SIZE, STUDENTS)


if __name__ == "__main__":
    main()
