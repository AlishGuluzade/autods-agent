"""
Computes SHAP values for the winning model and produces:
1. A ranked list of top features (fed to the LLM explainer node)
2. A summary bar chart image (embedded in the final PDF report)
"""
import os
from typing import Any, List, Tuple

import matplotlib

matplotlib.use("Agg")  # headless -- no display needed on a server
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap


def compute_shap_summary(
    model: Any, x_sample: pd.DataFrame, output_dir: str, job_id: str
) -> Tuple[List[dict], str]:
    # Cap the sample so SHAP stays fast on larger datasets during a demo
    sample = x_sample.sample(min(200, len(x_sample)), random_state=42)

    explainer = shap.TreeExplainer(model)
    shap_values = explainer.shap_values(sample)

    # SHAP returns different shapes depending on the model/version:
    # - list of arrays, one per class (older API)
    # - a single 3D array (n_samples, n_features, n_classes) for some binary/
    #   multi-class classifiers in newer shap versions
    # - a plain 2D array (n_samples, n_features) for regressors / some models
    if isinstance(shap_values, list):
        shap_values = np.mean(np.abs(np.array(shap_values)), axis=0)
    else:
        shap_values = np.abs(np.asarray(shap_values))
        if shap_values.ndim == 3:
            shap_values = shap_values.mean(axis=2)

    mean_abs_shap = shap_values.mean(axis=0)
    mean_abs_shap = np.asarray(mean_abs_shap).reshape(-1)
    order = np.argsort(mean_abs_shap)[::-1]

    top_features = [
        {"feature": sample.columns[i], "importance": round(float(mean_abs_shap[i]), 4)}
        for i in order[:10]
    ]

    fig, ax = plt.subplots(figsize=(7, 4))
    top10 = top_features[:10][::-1]
    ax.barh([f["feature"] for f in top10], [f["importance"] for f in top10], color="#4C72B0")
    ax.set_xlabel("Mean |SHAP value| (impact on prediction)")
    ax.set_title("Top features driving model predictions")
    fig.tight_layout()

    os.makedirs(output_dir, exist_ok=True)
    chart_path = os.path.join(output_dir, f"{job_id}_shap_summary.png")
    fig.savefig(chart_path, dpi=150)
    plt.close(fig)

    return top_features, chart_path
