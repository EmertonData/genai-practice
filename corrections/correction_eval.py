"""Reference solutions for the Session 4 (Evaluate Faithfulness) notebook exercises
(preparation.md §7).

score_faithfulness and the GLIDE prediction-powered mean estimator exercise.

Every function here takes what it needs as a parameter, so they can be imported and tested on
their own.
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage


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
        The judgement: "reasoning", one sentence, and "verdict", 1 if the claim is deducible
        from the chunk and 0 if not.
    """
    messages = [SystemMessage(system_prompt), HumanMessage(f"Chunk:\n{chunk}\n\nClaim:\n{claim}")]
    response = llm.invoke(messages)
    # A reply cut off by max_tokens is incomplete JSON, so fail loudly rather than misparse it
    if response.response_metadata.get("stop_reason") == "max_tokens":
        raise ValueError(f"the judge's reply was truncated: {response.content!r}")
    return json.loads(response.text)
