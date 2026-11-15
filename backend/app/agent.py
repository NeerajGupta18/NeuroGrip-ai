"""
agent.py

The decision pipeline, as a LangGraph state graph with two real branches:

    check_quality ──fault────────────────────────────────► abstain ─► END
          │ clean
          ▼
    extract_features ─► classify ──not calibrated────────► abstain ─► END
                            │      or confidence < floor
                            │ calibrated and confident
                            ▼
                        explain ─► END

A bionic hand that guesses on a bad signal can drop what it is holding, so the
graph has an explicit "do nothing" outcome. The agent abstains, and the hand
holds its last safe grip, in three cases: a sensor fault, a wearer who has not
yet calibrated every grasp (an uncalibrated model is barely better than chance
on a new person), or a low-confidence prediction. The confidence floor is chosen from the measured accuracy/coverage
trade-off in docs/EVALUATION.md.

The explanation is not written by a language model. It is computed: the
classifier is linear, so "grasp A rather than B" splits exactly into one
contribution per feature (model.explain). An optional LLM (GROQ_API_KEY) may
only re-word those computed numbers.
"""

import os
from typing import TypedDict

import numpy as np
from langgraph.graph import END, StateGraph

from .anomaly import detect_anomalies
from .calibration import wearer_means
from .dataset import source_model
from .features import CHANNEL_LABELS, FEATURE_LABELS, FEATURE_NAMES, extract_features
from .model import GRASP_LABELS, predict

CONFIDENCE_FLOOR = 0.7
GROQ_API_KEY = os.environ.get("GROQ_API_KEY")
GROQ_MODEL = os.environ.get("GROQ_MODEL", "llama-3.3-70b-versatile")


class AgentState(TypedDict, total=False):
    signal: np.ndarray            # (T, 2)
    profile: dict
    anomalies: list
    features: np.ndarray          # (n_windows, F)
    prediction: object            # model.Prediction
    decision: str                 # "act" | "abstain"
    abstain_reason: str           # "sensor_fault" | "not_calibrated" | "low_confidence"
    reasoning: str
    reasoning_source: str         # "computed" | "llm"


def _split(feature_name: str) -> tuple[str, str]:
    kind, channel = feature_name.rsplit("_", 1)
    return kind, channel


def _check_quality(state: AgentState) -> AgentState:
    thresholds = source_model(state["profile"]["source"])["thresholds"]
    return {"anomalies": detect_anomalies(state["signal"], thresholds)}


def _extract_features(state: AgentState) -> AgentState:
    return {"features": extract_features(state["signal"])}


def _classify(state: AgentState) -> AgentState:
    model = source_model(state["profile"]["source"])
    prediction = predict(state["features"], wearer_means(state["profile"]),
                         model["precision"], model["evidence_scale"])
    return {"prediction": prediction}


def _route_after_quality(state: AgentState) -> str:
    return "abstain" if state["anomalies"] else "extract_features"


def _is_calibrated(profile: dict) -> bool:
    return all(n > 0 for n in profile["recordings_learned"].values())


def _route_after_classify(state: AgentState) -> str:
    if not _is_calibrated(state["profile"]):
        return "abstain"
    return "explain" if state["prediction"].confidence >= CONFIDENCE_FLOOR else "abstain"


def _evidence_sentences(prediction, limit: int = 3) -> tuple[list, list]:
    """Top features for and against the winning grasp, as plain sentences."""
    total = sum(abs(c["contribution"]) for c in prediction.contributions) or 1.0
    supporting, opposing = [], []
    for c in prediction.contributions:
        kind, channel = _split(c["feature"])
        direction = "higher" if c["value"] > c["midpoint"] else "lower"
        share = round(100 * abs(c["contribution"]) / total)
        sentence = f"{FEATURE_LABELS[kind]} on {CHANNEL_LABELS[channel]} was {direction} ({share}% of the evidence)"
        (supporting if c["contribution"] > 0 else opposing).append(sentence)
    return supporting[:limit], opposing[:1]


def _computed_reasoning(state: AgentState, lead: bool = True) -> str:
    p = state["prediction"]
    winner, rival = GRASP_LABELS[p.grasp], GRASP_LABELS[p.runner_up]
    supporting, opposing = _evidence_sentences(p)
    opening = f"{winner} at {round(100 * p.confidence)}% confidence; the" if lead else "The"
    text = (
        f"{opening} closest alternative was {rival}. "
        f"Compared with the midpoint between this wearer's {winner} and {rival} profiles: "
        + "; ".join(supporting) + "."
    )
    if opposing:
        text += f" Pulling the other way, toward {rival}: {opposing[0]}."
    votes = sum(v == p.grasp for v in p.window_votes)
    text += f" {votes} of {len(p.window_votes)} half-second windows voted {winner}."
    return text


def _llm_reasoning(state: AgentState) -> str:
    from groq import Groq  # optional dependency, imported lazily

    p = state["prediction"]
    facts = _computed_reasoning(state)
    prompt = (
        "You are the explanation layer of a prosthetic-hand grip classifier. Below is the exact, "
        "computed evidence for a decision. Rewrite it in 2-3 plain sentences a clinician could read. "
        "Use only the facts given; do not add numbers, causes or claims that are not there.\n\n"
        f"Decision: {GRASP_LABELS[p.grasp]}\nEvidence: {facts}"
    )
    completion = Groq(api_key=GROQ_API_KEY).chat.completions.create(
        model=GROQ_MODEL, messages=[{"role": "user", "content": prompt}],
        max_tokens=180, temperature=0.2,
    )
    return completion.choices[0].message.content.strip()


def _explain(state: AgentState) -> AgentState:
    if GROQ_API_KEY:
        try:
            return {"decision": "act", "reasoning": _llm_reasoning(state), "reasoning_source": "llm"}
        except Exception:
            pass                       # any LLM failure falls back to the computed text
    return {"decision": "act", "reasoning": _computed_reasoning(state), "reasoning_source": "computed"}


def _abstain(state: AgentState) -> AgentState:
    if state.get("anomalies"):
        faults = " ".join(a["message"] for a in state["anomalies"])
        return {
            "decision": "abstain", "abstain_reason": "sensor_fault", "reasoning_source": "computed",
            "reasoning": f"No grip command issued; the hand holds its last safe grip. {faults} "
                         "A corrupted recording is never classified and never used for calibration.",
        }
    p = state["prediction"]
    if not _is_calibrated(state["profile"]):
        missing = [GRASP_LABELS[g] for g, n in state["profile"]["recordings_learned"].items() if n == 0]
        return {
            "decision": "abstain", "abstain_reason": "not_calibrated", "reasoning_source": "computed",
            "reasoning": (
                "No grip command issued; the hand holds its last safe grip. This wearer has not been "
                f"calibrated for: {', '.join(missing)}. A model built from other people is barely better "
                "than chance on a new wearer, so its guess is shown but not acted on. Uncalibrated best "
                f"guess: {GRASP_LABELS[p.grasp]} at {round(100 * p.confidence)}% confidence."
            ),
        }
    return {
        "decision": "abstain", "abstain_reason": "low_confidence", "reasoning_source": "computed",
        "reasoning": (
            f"No grip command issued; the hand holds its last safe grip. The best guess was "
            f"{GRASP_LABELS[p.grasp]} at {round(100 * p.confidence)}% confidence, below the "
            f"{round(100 * CONFIDENCE_FLOOR)}% floor required to act. " + _computed_reasoning(state, lead=False)
        ),
    }


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("check_quality", _check_quality)
    graph.add_node("extract_features", _extract_features)
    graph.add_node("classify", _classify)
    graph.add_node("explain", _explain)
    graph.add_node("abstain", _abstain)

    graph.set_entry_point("check_quality")
    graph.add_conditional_edges("check_quality", _route_after_quality,
                                {"abstain": "abstain", "extract_features": "extract_features"})
    graph.add_edge("extract_features", "classify")
    graph.add_conditional_edges("classify", _route_after_classify,
                                {"explain": "explain", "abstain": "abstain"})
    graph.add_edge("explain", END)
    graph.add_edge("abstain", END)
    return graph.compile()


_GRAPH = _build_graph()


def run_agent(signal: np.ndarray, profile: dict) -> AgentState:
    return _GRAPH.invoke({"signal": np.asarray(signal, dtype=float), "profile": profile})


def feature_table(prediction) -> list:
    """Contributions enriched with readable labels, for the API response."""
    rows = []
    for c in prediction.contributions:
        kind, channel = _split(c["feature"])
        rows.append({**c, "label": FEATURE_LABELS[kind], "channel": channel,
                     "favours": prediction.grasp if c["contribution"] > 0 else prediction.runner_up})
    return rows


assert all(_split(n)[0] in FEATURE_LABELS for n in FEATURE_NAMES)
