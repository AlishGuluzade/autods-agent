"""
Trains the candidate models the planner selected, tunes each with Optuna,
and returns the best one by cross-validated score.

This is a direct evolution of the approach used in the Credit Risk
Intelligence Pipeline project -- same libraries, now wrapped so the agent
can call it generically for classification OR regression.
"""
import warnings
from typing import Any, Dict, List, Tuple

import lightgbm as lgb
import numpy as np
import optuna
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.metrics import accuracy_score, f1_score, mean_absolute_error, r2_score
from sklearn.model_selection import cross_val_score

warnings.filterwarnings("ignore")
optuna.logging.set_verbosity(optuna.logging.WARNING)

N_TRIALS = 15  # kept modest so a demo run finishes in well under a minute


def _objective_xgboost(trial, x, y, problem_type):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.6, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.6, 1.0),
    }
    model = (
        xgb.XGBClassifier(**params, eval_metric="logloss")
        if problem_type == "classification"
        else xgb.XGBRegressor(**params)
    )
    scoring = "f1_weighted" if problem_type == "classification" else "r2"
    return cross_val_score(model, x, y, cv=3, scoring=scoring).mean()


def _objective_lightgbm(trial, x, y, problem_type):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "num_leaves": trial.suggest_int("num_leaves", 15, 63),
        "verbose": -1,
    }
    model = (
        lgb.LGBMClassifier(**params) if problem_type == "classification" else lgb.LGBMRegressor(**params)
    )
    scoring = "f1_weighted" if problem_type == "classification" else "r2"
    return cross_val_score(model, x, y, cv=3, scoring=scoring).mean()


def _objective_random_forest(trial, x, y, problem_type):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 100, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
    }
    model = (
        RandomForestClassifier(**params, random_state=42)
        if problem_type == "classification"
        else RandomForestRegressor(**params, random_state=42)
    )
    scoring = "f1_weighted" if problem_type == "classification" else "r2"
    return cross_val_score(model, x, y, cv=3, scoring=scoring).mean()


_OBJECTIVES = {
    "xgboost": _objective_xgboost,
    "lightgbm": _objective_lightgbm,
    "random_forest": _objective_random_forest,
}


def _build_final_model(name: str, params: Dict[str, Any], problem_type: str):
    if name == "xgboost":
        return (
            xgb.XGBClassifier(**params, eval_metric="logloss")
            if problem_type == "classification"
            else xgb.XGBRegressor(**params)
        )
    if name == "lightgbm":
        params = {**params, "verbose": -1}
        return lgb.LGBMClassifier(**params) if problem_type == "classification" else lgb.LGBMRegressor(**params)
    if name == "random_forest":
        return (
            RandomForestClassifier(**params, random_state=42)
            if problem_type == "classification"
            else RandomForestRegressor(**params, random_state=42)
        )
    raise ValueError(f"Unknown model name: {name}")


def train_and_select_best(
    x_train: pd.DataFrame,
    x_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
    models_to_try: List[str],
    problem_type: str,
) -> Tuple[Any, str, Dict[str, float], List[Dict[str, Any]]]:
    results = []

    for name in models_to_try:
        if name not in _OBJECTIVES:
            continue
        study = optuna.create_study(direction="maximize")
        study.optimize(
            lambda trial: _OBJECTIVES[name](trial, x_train, y_train, problem_type),
            n_trials=N_TRIALS,
            show_progress_bar=False,
        )
        results.append(
            {
                "model": name,
                "cv_score": study.best_value,
                "best_params": study.best_params,
            }
        )

    results.sort(key=lambda r: r["cv_score"], reverse=True)
    best = results[0]

    final_model = _build_final_model(best["model"], best["best_params"], problem_type)
    final_model.fit(x_train, y_train)
    preds = final_model.predict(x_test)

    if problem_type == "classification":
        metrics = {
            "accuracy": float(accuracy_score(y_test, preds)),
            "f1_weighted": float(f1_score(y_test, preds, average="weighted")),
        }
    else:
        metrics = {
            "r2": float(r2_score(y_test, preds)),
            "mae": float(mean_absolute_error(y_test, preds)),
        }

    return final_model, best["model"], metrics, results
