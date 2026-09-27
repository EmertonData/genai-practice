"""Reference solutions for the Session 4 (Evaluate Faithfulness) notebook exercises
(preparation.md §7).

JUDGE_PROMPT, score_faithfulness, choose_claims_to_annotate and estimate_faithfulness.

Every function here takes what it needs as a parameter, so they can be imported and tested on
their own.
"""

import json

import numpy as np
from glide.estimators import ClassicalMeanEstimator, PPIMeanEstimator
from glide.mean_inference_results.classical import ClassicalMeanInferenceResult
from glide.mean_inference_results.prediction_powered import PredictionPoweredMeanInferenceResult
from glide.samplers import UniformSampler
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage
from numpy.typing import NDArray

JUDGE_PROMPT = (
    "You are a judge evaluating the faithfulness of claims produced by a financial RAG system. You "
    "will receive a chunk of text extracted from an NVIDIA 10-K filing or earnings-call transcript, "
    'and one claim about it. In these chunks, "we" and "our" refer to NVIDIA.\n'
    "\n"
    "Your task is to decide whether the claim is deducible from the chunk. A claim is deducible "
    "only if every piece of information it contains follows from the chunk. As soon as one piece of "
    "information cannot be deduced from the chunk, the whole claim is not deducible.\n"
    "\n"
    "Proceed as follows:\n"
    "\n"
    "1. Read the chunk in full.\n"
    "2. Read the claim in full.\n"
    "3. Check whether each piece of information in the claim follows from the chunk, and write this "
    "reasoning down in under 100 words.\n"
    "4. Only then, give the verdict that follows from your reasoning.\n"
    "\n"
    "Judge the claim against the chunk alone, without outside knowledge: a claim that is true in "
    "the world but not supported by the chunk is not deducible. The chunk was cut at a fixed "
    "length, so it may start or end mid-sentence or mix table rows with prose.\n"
    "\n"
    "The verdict is a label:\n"
    "\n"
    "- 1: every piece of information in the claim is deducible from the chunk.\n"
    "- 0: at least one piece of information is not deducible from the chunk.\n"
    "\n"
    "Return only a JSON object, with no text before or after it:\n"
    "\n"
    '{"reasoning": "<your reasoning, in under 100 words>", "verdict": <0 or 1>}\n'
)


def score_faithfulness(chunk: str, claim: str, llm: ChatAnthropic, system_prompt: str) -> dict:
    """Ask an LLM judge whether a claim is deducible from a chunk.

    Parameters
    ----------
    chunk : str
        The text the claim should be deducible from.
    claim : str
        The claim to judge.
    llm : ChatAnthropic
        The judge model.
    system_prompt : str
        The judge's instructions, which ask for a JSON object with a reasoning and a verdict.

    Returns
    -------
    dict
        The judgement: "reasoning", its explanation, and "verdict", 1 if the claim is deducible
        from the chunk and 0 if not.
    """
    messages = [SystemMessage(system_prompt), HumanMessage(f"Chunk:\n{chunk}\n\nClaim:\n{claim}")]
    response = llm.invoke(messages)
    # A reply cut off by max_tokens is incomplete JSON, so fail loudly rather than misparse it
    if response.response_metadata.get("stop_reason") == "max_tokens":
        raise ValueError(f"the judge's reply was truncated: {response.content!r}")
    return json.loads(response.text)


def choose_claims_to_annotate(y_proxy: NDArray[np.float64], n_samples: int, random_seed: int) -> NDArray[np.float64]:
    """Choose, uniformly at random, the claims to hand to the human expert.

    Parameters
    ----------
    y_proxy : NDArray[np.float64]
        The LLM-as-Judge annotation of every claim.
    n_samples : int
        Number of claims the expert has time to annotate.
    random_seed : int
        Seed of the random draw, so the choice is the same on every run.

    Returns
    -------
    NDArray[np.float64]
        One value per claim: 1 if the claim goes to the expert, 0 if not.
    """
    return UniformSampler().sample(n_total=len(y_proxy), n_samples=n_samples, random_seed=random_seed)


def estimate_faithfulness(
    y_true: NDArray[np.float64], y_proxy: NDArray[np.float64], confidence_level: float
) -> tuple[ClassicalMeanInferenceResult, ClassicalMeanInferenceResult, PredictionPoweredMeanInferenceResult]:
    """Estimate the faithfulness rate from the judge alone, from the human annotations alone, and with PPI.

    Parameters
    ----------
    y_true : NDArray[np.float64]
        The human annotation of each claim, NaN where the claim was not annotated.
    y_proxy : NDArray[np.float64]
        The LLM-as-Judge annotation of every claim.
    confidence_level : float
        Target coverage of the confidence intervals, e.g. 0.95.

    Returns
    -------
    tuple[ClassicalMeanInferenceResult, ClassicalMeanInferenceResult, PredictionPoweredMeanInferenceResult]
        The estimates from the judge alone, from the human annotations alone, and with PPI.
    """
    result_proxy_only = ClassicalMeanEstimator().estimate(y_proxy, confidence_level=confidence_level)
    result_true_only = ClassicalMeanEstimator().estimate(y_true, confidence_level=confidence_level)
    result_ppi = PPIMeanEstimator().estimate(y_true, y_proxy, confidence_level=confidence_level)
    return result_proxy_only, result_true_only, result_ppi
