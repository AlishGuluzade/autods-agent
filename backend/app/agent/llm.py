"""
Thin LLM wrapper used by the planner and explainer nodes.

Design choice (see the blueprint doc): the LLM is only ever asked to make
DECISIONS (which problem type, which columns to drop, how to phrase an
explanation) -- it never generates the feature-engineering or modelling
code itself. That code is fixed, tested, and lives in app/ml/. This keeps
the agent fast, cheap, debuggable, and safe (no LLM-generated code is ever
executed).

If GROQ_API_KEY is not set, both functions fall back to deterministic
heuristics so the whole pipeline still runs end-to-end for local dev/demo
without requiring an API key.
"""
import json
import os
from typing import Any, Dict, List

from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-70b-versatile")

_client = None
if GROQ_API_KEY:
    from groq import Groq

    _client = Groq(api_key=GROQ_API_KEY)


def _call_groq(system_prompt: str, user_prompt: str) -> str:
    assert _client is not None
    resp = _client.chat.completions.create(
        model=GROQ_MODEL,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.1,
        response_format={"type": "json_object"} if "JSON" in system_prompt else None,
    )
    return resp.choices[0].message.content


PLANNER_SYSTEM_PROMPT = """You are a senior data scientist planning an ML pipeline.
You will be given a JSON profile of a dataset. Decide:
- problem_type: "classification" or "regression"
- target_column: best guess for the target column name
- drop_columns: list of columns to drop (IDs, free text, near-constant columns)
- models_to_try: subset of ["xgboost", "lightgbm", "random_forest"]
Respond ONLY with a JSON object with exactly these four keys. No prose, no markdown."""

EXPLAIN_SYSTEM_PROMPT = """You are a data scientist explaining a model to a non-technical
business stakeholder. You will get the top SHAP features and model metrics.
Write 3-5 short sentences in plain language: what the model predicts, which
factors matter most and why that makes intuitive sense, and one caveat.
No jargon like 'SHAP value' -- say 'impact' or 'influence' instead."""


def plan_pipeline(profile: Dict[str, Any]) -> Dict[str, Any]:
    if _client is not None:
        try:
            raw = _call_groq(PLANNER_SYSTEM_PROMPT, json.dumps(profile))
            parsed = json.loads(raw)
            if all(k in parsed for k in ("problem_type", "target_column", "drop_columns", "models_to_try")):
                return parsed
        except Exception:
            pass  # fall through to heuristic

    return _heuristic_plan(profile)


def _heuristic_plan(profile: Dict[str, Any]) -> Dict[str, Any]:
    """Rule-based fallback planner -- no API key required."""
    columns = profile["columns"]

    # Guess the target: last column, or a column literally called
    # target/label/y/outcome/churn/default/price, whichever exists.
    candidate_names = ["target", "label", "y", "outcome", "churn", "default", "price", "class"]
    target_column = None
    for name in candidate_names:
        for col in columns:
            if col["name"].lower() == name:
                target_column = col["name"]
                break
        if target_column:
            break
    if target_column is None:
        target_column = columns[-1]["name"]

    target_info = next(c for c in columns if c["name"] == target_column)
    problem_type = "classification" if target_info["n_unique"] <= 20 else "regression"

    n_rows = profile.get("n_rows", 0)

    def _looks_like_identifier(col: Dict[str, Any]) -> bool:
        name = col["name"].lower()
        name_flags_id = name == "id" or name == "index" or name.endswith("_id") or "unnamed" in name
        near_unique = n_rows > 0 and col["kind"] == "categorical" and col["n_unique"] / n_rows > 0.95
        return name_flags_id or near_unique

    drop_columns = [
        c["name"]
        for c in columns
        if c["name"] != target_column
        and (c["n_unique"] <= 1 or c["missing_pct"] > 0.6 or _looks_like_identifier(c))
    ]

    return {
        "problem_type": problem_type,
        "target_column": target_column,
        "drop_columns": drop_columns,
        "models_to_try": ["xgboost", "lightgbm", "random_forest"],
    }


def explain_results(top_features: List[Dict[str, Any]], metrics: Dict[str, float], problem_type: str) -> str:
    if _client is not None:
        try:
            payload = json.dumps({"top_features": top_features, "metrics": metrics, "problem_type": problem_type})
            return _call_groq(EXPLAIN_SYSTEM_PROMPT, payload).strip()
        except Exception:
            pass

    return _heuristic_explanation(top_features, metrics, problem_type)


def _heuristic_explanation(top_features: List[Dict[str, Any]], metrics: Dict[str, float], problem_type: str) -> str:
    names = [f["feature"] for f in top_features[:3]]
    metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    kind = "predicting the outcome class" if problem_type == "classification" else "predicting the numeric target"
    return (
        f"This model is {kind}. The factors with the biggest influence on its predictions are "
        f"{', '.join(names)}, in that order. Performance on held-out data: {metric_str}. "
        "As with any model trained on historical data, results should be validated on new data "
        "before being used for high-stakes decisions."
    )
