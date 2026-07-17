"""
Feature engineering executed by fixed, tested code.

The planner LLM only decides WHICH columns to drop and what the target is
-- this module is the deterministic implementation of HOW columns get
transformed. Keeping this LLM-free is a safety and reliability choice
(see app/agent/llm.py docstring).
"""
from typing import Any, Dict, Tuple

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


def build_features(
    df: pd.DataFrame, plan: Dict[str, Any]
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, list]:
    target_column = plan["target_column"]
    drop_columns = [c for c in plan.get("drop_columns", []) if c in df.columns and c != target_column]

    work = df.drop(columns=drop_columns).copy()

    # Separate target
    y = work.pop(target_column)

    # Impute
    for col in work.columns:
        if pd.api.types.is_numeric_dtype(work[col]):
            work[col] = work[col].fillna(work[col].median())
        else:
            work[col] = work[col].fillna(work[col].mode().iloc[0] if not work[col].mode().empty else "missing")

    # Encode categoricals (label encoding -- simple and tree-model-friendly,
    # since XGBoost/LightGBM/RandomForest don't need one-hot to perform well)
    for col in work.columns:
        if not pd.api.types.is_numeric_dtype(work[col]):
            le = LabelEncoder()
            work[col] = le.fit_transform(work[col].astype(str))

    # Encode target if classification and non-numeric
    if plan["problem_type"] == "classification" and not pd.api.types.is_numeric_dtype(y):
        y = LabelEncoder().fit_transform(y.astype(str))
        y = pd.Series(y)

    feature_columns = list(work.columns)

    x_train, x_test, y_train, y_test = train_test_split(
        work, y, test_size=0.2, random_state=42,
        stratify=y if plan["problem_type"] == "classification" else None,
    )

    return x_train, x_test, y_train, y_test, feature_columns
