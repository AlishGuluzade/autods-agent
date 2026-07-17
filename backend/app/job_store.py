"""
Very small in-memory job store.

For a portfolio project this is intentionally simple (a dict, no Redis).
It maps job_id -> current AgentState so the frontend can poll
GET /status/{job_id} and render a live progress list while the graph runs
in a background thread.

If you ever need this to survive a server restart or run across multiple
workers, swap this out for Redis / a database table -- the interface
(get/set/append_log) is deliberately tiny so that swap is a one-file change.
"""
import threading
from typing import Any, Dict, Optional

_lock = threading.Lock()
_jobs: Dict[str, Dict[str, Any]] = {}


def create_job(job_id: str, initial_state: Dict[str, Any]) -> None:
    with _lock:
        _jobs[job_id] = initial_state


def get_job(job_id: str) -> Optional[Dict[str, Any]]:
    with _lock:
        return _jobs.get(job_id)


def update_job(job_id: str, patch: Dict[str, Any]) -> None:
    with _lock:
        if job_id in _jobs:
            _jobs[job_id].update(patch)


def append_log(job_id: str, step: str, message: str, level: str = "info") -> None:
    with _lock:
        job = _jobs.get(job_id)
        if job is None:
            return
        job.setdefault("logs", [])
        job["logs"].append({"step": step, "message": message, "level": level})
