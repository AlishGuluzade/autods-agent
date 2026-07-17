"""
Shared state object that flows through every node of the LangGraph pipeline.

Keeping this as a single TypedDict (rather than passing loose arguments
between functions) is what makes the graph inspectable and debuggable --
at any point you can print `state` and see exactly what the agent knows
and has decided so far.
"""
from typing import TypedDict, Optional, Any, List, Dict


class AgentState(TypedDict, total=False):
    # --- input ---
    job_id: str
    dataset_path: str
    user_question: Optional[str]

    # --- filled in by profile_node ---
    profile: Dict[str, Any]

    # --- filled in by plan_node ---
    plan: Dict[str, Any]

    # --- filled in by feature_eng_node ---
    train_path: str
    test_path: str
    feature_columns: List[str]
    target_column: str

    # --- internal (in-memory dataframes passed between feature_eng -> train -> explain) ---
    _x_train: Any
    _x_test: Any
    _y_train: Any
    _y_test: Any
    _model: Any

    # --- filled in by train_node ---
    best_model_name: str
    best_model_path: str
    metrics: Dict[str, float]
    all_model_results: List[Dict[str, Any]]

    # --- filled in by explain_node ---
    shap_summary_path: str
    top_features: List[Dict[str, Any]]
    explanation_text: str

    # --- filled in by report_node ---
    report_path: str

    # --- bookkeeping / progress ---
    logs: List[Dict[str, str]]
    status: str
    error: Optional[str]
