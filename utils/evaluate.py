"""Helpers for the Session 4 notebook."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from langchain_anthropic import ChatAnthropic
from tqdm import tqdm


def judge_all(
    score_fn: Callable[[str, str, ChatAnthropic, str], dict],
    claims: list[dict],
    llm: ChatAnthropic,
    system_prompt: str,
    max_workers: int = 8,
) -> list[int | None]:
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
    list[int | None]
        One verdict per claim, in the same order as `claims`: 1 if faithful, 0 if not, and
        None if the call failed.
    """

    def verdict(claim: dict) -> int | None:
        try:
            return score_fn(claim["chunk"], claim["claim"], llm, system_prompt)["verdict"]
        except Exception:  # one failed call must not stop the whole run
            return None

    with ThreadPoolExecutor(max_workers=max_workers) as pool:
        return list(tqdm(pool.map(verdict, claims), total=len(claims), desc="judging"))


def save_verdicts(verdicts: list[int | None], claim_ids: list[str], output_path: str) -> None:
    """Write the verdicts and their claim ids to a JSON file, replacing any existing content.

    Parameters
    ----------
    verdicts : list[int | None]
        One verdict per claim, 1, 0 or None.
    claim_ids : list[str]
        The claim ids, in the same order as `verdicts`.
    output_path : str
        Path to the JSON file to write.

    Returns
    -------
    None
    """
    rows = [{"claim_id": claim_id, "verdict": verdict} for claim_id, verdict in zip(claim_ids, verdicts)]
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
