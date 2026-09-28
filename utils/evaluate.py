"""Helpers for the Session 4 notebook."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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


def load_verdicts(input_path: str) -> dict[str, int | None]:
    """Read verdicts written by `save_verdicts`, or the reference scores in `judge_scores.json`.

    Parameters
    ----------
    input_path : str
        Path to the JSON file to read.

    Returns
    -------
    dict[str, int | None]
        The verdict of each claim, keyed by claim id.
    """
    return {row["claim_id"]: row["verdict"] for row in json.loads(Path(input_path).read_text())}
