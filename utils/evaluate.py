"""Helpers for the Session 4 notebook."""

import json
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from glide.mean_inference_results import MeanInferenceResult
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


def plot_estimates(results: list[MeanInferenceResult], labels: list[str]) -> None:
    """Draw each estimate of the faithfulness rate as a point with its confidence interval.

    Parameters
    ----------
    results : list[MeanInferenceResult]
        GLIDE results, each with a `mean` and a `confidence_interval`.
    labels : list[str]
        The name of each estimate, in the same order as `results`. Use `<br>` for a line break.

    Returns
    -------
    None
    """
    colors = ["red", "steelblue", "purple"]
    fig = go.Figure()
    for result, label, color in zip(results, labels, colors):
        low, high = result.confidence_interval.lower_bound, result.confidence_interval.upper_bound
        name = label.split("<br>")[0]
        half_width = (high - low) / 2
        fig.add_trace(
            go.Scatter(
                x=[result.mean],
                y=[label],
                mode="markers+text",
                marker={"color": color, "size": 14},
                error_x={
                    "type": "data",
                    "symmetric": False,
                    "array": [high - result.mean],
                    "arrayminus": [result.mean - low],
                    "color": color,
                    "thickness": 4,
                    "width": 10,
                },
                text=[f"{result.mean:.1%}"],
                textposition="top center",
                textfont={"color": color, "size": 16},
                showlegend=False,
                hovertemplate=f"{name}: %{{x:.1%}} ± {half_width:.1%}<extra></extra>",
            )
        )
    fig.update_layout(
        hovermode="closest",
        font={"family": "Times New Roman", "color": "dimgray", "size": 15},
        plot_bgcolor="whitesmoke",
        xaxis={"title": "Faithfulness rate", "tickformat": ".0%", "gridcolor": "white", "range": [0, 1]},
        yaxis={"range": [len(labels) - 0.5, -0.8], "gridcolor": "white"},
        height=400,
        margin={"l": 20, "r": 20, "t": 30, "b": 50},
    )
    fig.show()
