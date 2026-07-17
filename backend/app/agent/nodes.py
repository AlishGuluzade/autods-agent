"""
Each function here is one node in the LangGraph graph (see graph.py).
Every node: reads what it needs from `state`, does its job, logs progress
to the job store (so the frontend can poll live status), and returns a
partial state dict that LangGraph merges in.
"""
import os
import time

import joblib
import pandas as pd

from app.agent.llm import explain_results, plan_pipeline
from app.agent.state import AgentState
from app.job_store import append_log, update_job
from app.ml.explain import compute_shap_summary
from app.ml.feature_eng import build_features
from app.ml.profiling import profile_dataframe
from app.ml.train import train_and_select_best
from app.reports.generator import generate_report

ARTIFACTS_DIR = os.getenv("ARTIFACTS_DIR", "/tmp/autods_artifacts")


def _log(job_id: str, step: str, message: str) -> None:
    append_log(job_id, step, message)
    update_job(job_id, {"status": step})


def _sync(job_id: str, result: dict) -> dict:
    """Push a node's public fields (anything not prefixed with '_') into the
    job store, then return the result unchanged so nodes can do
    `return _sync(job_id, {...})`. Without this, the job store used by
    /status and /report never sees metrics/report_path/etc., since those
    only otherwise live in the LangGraph state returned from `invoke`."""
    public_fields = {k: v for k, v in result.items() if not k.startswith("_")}
    update_job(job_id, public_fields)
    return result


def profile_node(state: AgentState) -> AgentState:
    job_id = state["job_id"]
    _log(job_id, "profile", "Reading dataset and computing column-level statistics...")

    df = pd.read_csv(state["dataset_path"])
    profile = profile_dataframe(df)

    _log(job_id, "profile", f"Profiled {profile['n_rows']} rows x {profile['n_columns']} columns.")
    return _sync(job_id, {"profile": profile})


def plan_node(state: AgentState) -> AgentState:
    job_id = state["job_id"]
    _log(job_id, "plan", "Deciding problem type, target column, and modelling strategy...")

    plan = plan_pipeline(state["profile"])

    _log(
        job_id,
        "plan",
        f"Plan ready: {plan['problem_type']} task, target='{plan['target_column']}', "
        f"trying models: {', '.join(plan['models_to_try'])}.",
    )
    return _sync(job_id, {"plan": plan})


def feature_eng_node(state: AgentState) -> AgentState:
    job_id = state["job_id"]
    _log(job_id, "feature_eng", "Imputing missing values, encoding categoricals, splitting train/test...")

    df = pd.read_csv(state["dataset_path"])
    x_train, x_test, y_train, y_test, feature_columns = build_features(df, state["plan"])

    run_dir = os.path.join(ARTIFACTS_DIR, job_id)
    os.makedirs(run_dir, exist_ok=True)
    x_train.assign(**{state["plan"]["target_column"]: y_train}).to_csv(os.path.join(run_dir, "train.csv"), index=False)
    x_test.assign(**{state["plan"]["target_column"]: y_test}).to_csv(os.path.join(run_dir, "test.csv"), index=False)

    _log(job_id, "feature_eng", f"Built {len(feature_columns)} features for {len(x_train)} training rows.")

    # The split frames are handed to the next node via the returned state dict
    # (kept in-memory since the graph runs in a single process). For a real
    # multi-process deployment you'd persist to disk/object storage instead
    # and reload by path.
    return _sync(job_id, {
        "feature_columns": feature_columns,
        "target_column": state["plan"]["target_column"],
        "train_path": os.path.join(run_dir, "train.csv"),
        "test_path": os.path.join(run_dir, "test.csv"),
        "_x_train": x_train,
        "_x_test": x_test,
        "_y_train": y_train,
        "_y_test": y_test,
    })


def train_node(state: AgentState) -> AgentState:
    job_id = state["job_id"]
    models = state["plan"]["models_to_try"]
    _log(job_id, "train", f"Tuning {len(models)} candidate models with Optuna (this is the slow step)...")

    start = time.time()
    model, best_name, metrics, all_results = train_and_select_best(
        state["_x_train"], state["_x_test"], state["_y_train"], state["_y_test"],
        models, state["plan"]["problem_type"],
    )
    elapsed = round(time.time() - start, 1)

    run_dir = os.path.join(ARTIFACTS_DIR, job_id)
    os.makedirs(run_dir, exist_ok=True)
    model_path = os.path.join(run_dir, "model.joblib")
    joblib.dump(model, model_path)

    metric_str = ", ".join(f"{k}={v:.3f}" for k, v in metrics.items())
    _log(job_id, "train", f"Best model: {best_name} ({metric_str}) -- tuned in {elapsed}s.")

    return _sync(job_id, {
        "best_model_name": best_name,
        "best_model_path": model_path,
        "metrics": metrics,
        "all_model_results": all_results,
        "_model": model,
    })


def explain_node(state: AgentState) -> AgentState:
    job_id = state["job_id"]
    _log(job_id, "explain", "Computing SHAP values and translating them into plain language...")

    run_dir = os.path.join(ARTIFACTS_DIR, job_id)
    top_features, chart_path = compute_shap_summary(state["_model"], state["_x_train"], run_dir, job_id)
    explanation = explain_results(top_features, state["metrics"], state["plan"]["problem_type"])

    _log(job_id, "explain", "Explanation ready.")
    return _sync(job_id, {
        "top_features": top_features,
        "shap_summary_path": chart_path,
        "explanation_text": explanation,
    })


def report_node(state: AgentState) -> AgentState:
    job_id = state["job_id"]
    _log(job_id, "report", "Generating PDF report...")

    run_dir = os.path.join(ARTIFACTS_DIR, job_id)
    report_path = os.path.join(run_dir, "report.pdf")
    generate_report(
        report_path,
        job_id,
        state["plan"],
        state["best_model_name"],
        state["metrics"],
        state["top_features"],
        state["explanation_text"],
        state["shap_summary_path"],
    )

    result = _sync(job_id, {"report_path": report_path})
    _log(job_id, "done", "Analysis complete.")
    return result
