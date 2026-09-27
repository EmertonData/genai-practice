"""Helpers for the Session 4 notebook."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from langchain_anthropic import ChatAnthropic
from numpy.typing import NDArray
from tqdm import tqdm


def judge_all(
    score_fn: Callable[[str, str, ChatAnthropic, str], dict],
    claims: list[dict],
    llm: ChatAnthropic,
    system_prompt: str,
    max_workers: int = 8,
) -> NDArray[np.float64]:
    """Judge every claim with `score_fn`, several calls at a time, showing a progress bar.

    Parameters
    ----------
    score_fn : Callable[[str, str, ChatAnthropic, str], dict]
        The function that judges one claim, called as `score_fn(chunk, claim, llm, system_prompt)`
        and returning a dict with a "verdict".
    claims : list[dict]
        The claims to judge, each with a "chunk" and a "claim".
    llm : ChatAnthropic
        The judge model.
    system_prompt : str
        The judge's instructions.
    max_workers : int, optional
        Number of calls sent to the API at the same time, by default 8.

    Returns
    -------
    NDArray[np.float64]
        One verdict per claim, in the same order as `claims`: 1 if faithful, 0 if not, and NaN
        if the call failed.
    """

    def verdict(claim: dict) -> float:
        try:
            return float(score_fn(claim["chunk"], claim["claim"], llm, system_prompt)["verdict"])
        except Exception:  # one failed call must not stop the whole run
            return np.nan

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return np.array(list(tqdm(pool.map(verdict, claims), total=len(claims), desc="judging")))


def save_verdicts(verdicts: NDArray[np.float64], claims: list[dict], output_path: str) -> None:
    """Write each claim's id and verdict to a JSON file, replacing any existing content.

    Parameters
    ----------
    verdicts : NDArray[np.float64]
        One verdict per claim, as returned by `judge_all`. NaN is written as null.
    claims : list[dict]
        The judged claims, each with a "claim_id", in the same order as `verdicts`.
    output_path : str
        Path to the JSON file to write.

    Returns
    -------
    None
    """
    rows = [
        {"claim_id": claim["claim_id"], "verdict": None if np.isnan(verdict) else int(verdict)}
        for claim, verdict in zip(claims, verdicts)
    ]
    Path(output_path).write_text(json.dumps(rows, indent=2))


def load_verdicts(input_path: str, claims: list[dict]) -> NDArray[np.float64]:
    """Read verdicts written by `save_verdicts`, or the reference scores in `judge_scores.json`.

    Parameters
    ----------
    input_path : str
        Path to the JSON file to read.
    claims : list[dict]
        The claims, each with a "claim_id", in the order the verdicts should follow.

    Returns
    -------
    NDArray[np.float64]
        One verdict per claim, in the same order as `claims`: 1 if faithful, 0 if not, and NaN
        if the file has no verdict for it.
    """
    verdict_of = {row["claim_id"]: row["verdict"] for row in json.loads(Path(input_path).read_text())}
    verdicts = [verdict_of.get(claim["claim_id"]) for claim in claims]
    return np.array([np.nan if verdict is None else verdict for verdict in verdicts], dtype=float)


def plot_estimates(results: list, labels: list[str]) -> None:
    """Draw each estimate of the faithfulness rate as a point inside its confidence interval.

    Parameters
    ----------
    results : list
        GLIDE results, each with a `mean` and a `confidence_interval`.
    labels : list[str]
        The name of each estimate, in the same order as `results`.

    Returns
    -------
    None
    """
    colors = ["red", "steelblue", "purple"]
    fig, ax = plt.subplots(figsize=(10, 4.5))
    for y, (result, label, color) in enumerate(zip(results, labels, colors)):
        low, high = result.confidence_interval.lower_bound, result.confidence_interval.upper_bound
        ax.plot([low, high], [y, y], color=color, linewidth=4, solid_capstyle="round")
        ax.scatter(result.mean, y, s=150, color=color, edgecolors="white", linewidths=2, zorder=3)
        ax.text(result.mean, y - 0.25, f"{result.mean:.1%}", ha="center", color=color, fontweight="bold")
        ax.text(result.mean, y + 0.4, f"[{low:.1%}, {high:.1%}]", ha="center", color="gray")
    ax.set_yticks(range(len(labels)), labels)
    ax.set_ylim(-0.8, len(labels) - 0.2)
    ax.invert_yaxis()
    ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda value, _: f"{value:.0%}"))
    ax.set_xlabel("Faithfulness rate")
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.tick_params(left=False)
    fig.tight_layout()
    plt.show()
