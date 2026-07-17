# Architecture notes

## Why LangGraph over CrewAI/AutoGen

This pipeline is a **deterministic, linear sequence of steps** where each step's output
is precisely defined (a profile dict, a plan dict, a trained model, ...). That's exactly
the case LangGraph is built for: explicit state, explicit edges, and a graph you can
reason about and debug node-by-node. CrewAI's role-based "crew of agents negotiating a
task" model and AutoGen's conversational multi-agent loop both solve a different
problem — open-ended tasks without a fixed shape — and would add overhead and
unpredictability here without buying anything.

## Why the LLM never writes or executes code

The planner LLM only ever returns a small, schema-constrained JSON object
(`problem_type`, `target_column`, `drop_columns`, `models_to_try`). The explainer LLM
only ever returns prose. Neither is allowed to generate code that gets executed. All of
the actual data manipulation (imputation, encoding, splitting, training, SHAP) lives in
plain, tested Python in `app/ml/`.

This is a deliberate reliability and safety boundary, not a limitation:
- **Reliability**: a hallucinated column name fails a `KeyError` immediately and
  visibly; a hallucinated *function* would fail silently or unpredictably.
- **Debuggability**: you can unit test `app/ml/feature_eng.py` without ever calling an
  LLM.
- **Safety**: no LLM output is ever passed to `eval`, `exec`, or a subprocess.

## Why heuristic fallbacks exist for both LLM calls

`plan_pipeline()` and `explain_results()` in `app/agent/llm.py` both work with **zero**
API keys, using rule-based logic (e.g. guessing the target column by name and
cardinality, guessing "classification" vs "regression" from the number of distinct
target values). This means:
- The project is fully runnable and demo-able without asking anyone for a Groq key.
- CI or automated tests can exercise the whole graph deterministically.
- If Groq's API is down or rate-limited during a live demo, the agent degrades
  gracefully instead of crashing.

## Why the job store is a plain in-memory dict

For a single-instance portfolio deployment, a `threading.Lock`-guarded dict is enough to
support polling-based progress updates. The three-function interface
(`create_job` / `get_job` / `update_job`) in `app/job_store.py` is intentionally the
entire contract — swapping in Redis (for multi-worker deployments) or a database (for
persistence across restarts) means rewriting one file, not touching the agent or API
layer at all.
