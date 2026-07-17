"""
Fast, dependency-light dataset profiler.

We deliberately don't pull in a heavy library like ydata-profiling here --
we only need enough structure to hand the LLM planner a compact JSON
summary it can reason over.
"""
from typing import Any, Dict

import pandas as pd


def profile_dataframe(df: pd.DataFrame) -> Dict[str, Any]:
    columns = []
    for col in df.columns:
        series = df[col]
        n_unique = int(series.nunique(dropna=True))
        missing_pct = float(series.isna().mean())
        dtype = str(series.dtype)

        col_info = {
            "name": col,
            "dtype": dtype,
            "n_unique": n_unique,
            "missing_pct": round(missing_pct, 4),
        }

        if pd.api.types.is_numeric_dtype(series):
            col_info.update(
                {
                    "kind": "numeric",
                    "min": float(series.min()) if series.notna().any() else None,
                    "max": float(series.max()) if series.notna().any() else None,
                    "mean": float(series.mean()) if series.notna().any() else None,
                }
            )
        else:
            col_info["kind"] = "categorical"
            top_values = series.value_counts(dropna=True).head(5)
            col_info["top_values"] = {str(k): int(v) for k, v in top_values.items()}

        columns.append(col_info)

    return {
        "n_rows": int(len(df)),
        "n_columns": int(len(df.columns)),
        "columns": columns,
    }
