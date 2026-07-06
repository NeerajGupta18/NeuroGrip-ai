"""
agent.py

The core "AI brain" of NeuroGrip AI, built as a LangGraph state graph with
three nodes:

    extract_features -> classify_intent -> explain_prediction

1. extract_features: turns the raw signal window into classic EMG features
   (see features.py).
2. classify_intent: compares those features against the user's calibrated
   per-intent centroids (see calibration.py) using normalized distance, and
   converts distances into a softmax confidence distribution.
3. explain_prediction: produces a plain-English explanation of *why* the
   agent predicted what it did. If a GROQ_API_KEY environment variable is
   set, it uses an LLM (Groq-hosted Llama 3.3 70B, same setup as this
   author's ShelfVue AI project) to phrase the explanation naturally.
   Otherwise it falls back to a deterministic template so the project runs
   with zero external dependencies or API keys.

This explicit "show your reasoning" step is the project's main
differentiator versus a black-box classifier, and mirrors why explainable
AI matters in a medical-adjacent product: a clinician or user needs to
trust *why* the hand made a decision, not just the decision itself.
"""

import math
import os
from typing import TypedDict

from langgraph.graph import StateGraph, END

from .anomaly import detect_anomalies
from .calibration import FEATURE_KEYS, get_profile
from .features import extract_features
from .signal_simulator import INTENTS

# Rough per-feature normalization scale, derived once from the spread of
# base centroids, so no single feature (e.g. zero-crossing count) dominates
# the distance calculation just because of its numeric range.
_FEATURE_SCALE = {
    "mav": 0.4,
    "var": 0.15,
    "rms": 0.4,
    "wl": 20.0,
    "zc": 15.0,
}

GROQ_API_KEY = os.environ.get("GROQ_API_KEY")


class AgentState(TypedDict, total=False):
    signal: list[float]
    user_id: str
    features: dict
    anomalies: list[dict]
    distances: dict
    predicted_intent: str
    confidence: float
    confidence_by_intent: dict
    reasoning: str


def _extract_features_node(state: AgentState) -> AgentState:
    state["features"] = extract_features(state["signal"])
    return state


def _detect_anomaly_node(state: AgentState) -> AgentState:
    state["anomalies"] = detect_anomalies(state["signal"])
    return state


def _classify_intent_node(state: AgentState) -> AgentState:
    profile = get_profile(state["user_id"])
    centroids = profile["centroids"]
    features = state["features"]

    distances = {}
    for intent in INTENTS:
        centroid = centroids[intent]
        dist_sq = 0.0
        for key in FEATURE_KEYS:
            scale = _FEATURE_SCALE[key]
            diff = (features[key] - centroid[key]) / scale
            dist_sq += diff ** 2
        distances[intent] = math.sqrt(dist_sq)

    # Convert distances -> softmax confidence (smaller distance = higher confidence)
    neg_distances = {k: -v for k, v in distances.items()}
    max_val = max(neg_distances.values())
    exp_vals = {k: math.exp(v - max_val) for k, v in neg_distances.items()}
    total = sum(exp_vals.values())
    confidence_by_intent = {k: round(v / total, 4) for k, v in exp_vals.items()}

    predicted_intent = max(confidence_by_intent, key=confidence_by_intent.get)

    state["distances"] = {k: round(v, 4) for k, v in distances.items()}
    state["predicted_intent"] = predicted_intent
    state["confidence"] = confidence_by_intent[predicted_intent]
    state["confidence_by_intent"] = confidence_by_intent
    return state


def _template_reasoning(state: AgentState) -> str:
    f = state["features"]
    intent = state["predicted_intent"]
    confidence_pct = round(state["confidence"] * 100, 1)

    descriptors = []
    if f["mav"] > 0.4:
        descriptors.append("strong overall muscle activation")
    elif f["mav"] > 0.2:
        descriptors.append("moderate muscle activation")
    else:
        descriptors.append("light muscle activation")

    if f["var"] < 0.05:
        descriptors.append("a steady, low-variance signal (suggests a sustained, held contraction)")
    else:
        descriptors.append("higher signal variance (suggests a quick or dynamic motion)")

    if f["zc"] > 45:
        descriptors.append(f"a high zero-crossing rate ({f['zc']}), typical of rapid, pulse-like activity")
    elif f["zc"] < 15:
        descriptors.append(f"a low zero-crossing rate ({f['zc']}), typical of a smooth, sustained signal")
    else:
        descriptors.append(f"a moderate zero-crossing rate ({f['zc']})")

    detail = ", ".join(descriptors)
    base_reasoning = (
        f"Predicted '{intent}' with {confidence_pct}% confidence. "
        f"The signal window showed {detail}. This pattern is closest to this "
        f"user's calibrated '{intent}' profile compared to the other four grip intents."
    )

    anomalies = state.get("anomalies") or []
    if anomalies:
        caution = " ".join(a["message"] for a in anomalies)
        return (
            f"⚠ Signal quality issue detected before classification: {caution} "
            f"Treat the following prediction as provisional. {base_reasoning}"
        )
    return base_reasoning


def _explain_prediction_node(state: AgentState) -> AgentState:
    if GROQ_API_KEY:
        try:
            state["reasoning"] = _groq_reasoning(state)
            return state
        except Exception:
            # If the LLM call fails for any reason (no network, bad key,
            # rate limit), fall back to the deterministic template rather
            # than breaking the response.
            pass
    state["reasoning"] = _template_reasoning(state)
    return state


def _groq_reasoning(state: AgentState) -> str:
    from groq import Groq  # imported lazily so it's optional

    client = Groq(api_key=GROQ_API_KEY)
    f = state["features"]
    anomalies = state.get("anomalies") or []
    anomaly_note = (
        " Signal quality issues were also detected: "
        + " ".join(a["message"] for a in anomalies)
        + " Mention this caution before the prediction reasoning."
        if anomalies
        else ""
    )
    prompt = (
        "You are the reasoning layer of a bionic hand's grip-intent classifier. "
        f"Predicted intent: {state['predicted_intent']} "
        f"(confidence {round(state['confidence'] * 100, 1)}%). "
        f"Extracted EMG features: MAV={f['mav']}, VAR={f['var']}, RMS={f['rms']}, "
        f"WL={f['wl']}, ZC={f['zc']}.{anomaly_note} "
        "In 2-3 short sentences, explain in plain language why these feature "
        "values support this prediction. Be concrete and reference the actual "
        "numbers. Do not mention that you are an LLM."
    )
    completion = client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}],
        max_tokens=150,
        temperature=0.4,
    )
    return completion.choices[0].message.content.strip()


def _build_graph():
    graph = StateGraph(AgentState)
    graph.add_node("extract_features", _extract_features_node)
    graph.add_node("detect_anomaly", _detect_anomaly_node)
    graph.add_node("classify_intent", _classify_intent_node)
    graph.add_node("explain_prediction", _explain_prediction_node)

    graph.set_entry_point("extract_features")
    graph.add_edge("extract_features", "detect_anomaly")
    graph.add_edge("detect_anomaly", "classify_intent")
    graph.add_edge("classify_intent", "explain_prediction")
    graph.add_edge("explain_prediction", END)

    return graph.compile()


_COMPILED_GRAPH = _build_graph()


def run_agent(signal: list[float], user_id: str) -> AgentState:
    initial_state: AgentState = {"signal": signal, "user_id": user_id}
    result = _COMPILED_GRAPH.invoke(initial_state)
    return result
