"""Persists full run_agent() results to disk so the dashboard can browse past
runs, not just watch the current one.

run_agent()'s return value only lives in memory for the process that called
it — there's no history without something writing it down. This module is
that something: a thin, dashboard-facing layer that never touches agent or
detector logic directly, just serializes whatever run_agent() already
produces.

Each run is one JSON file under RUNS_DIR, named by run_id (a sortable
timestamp + a short slug of the task). Deliberately files-on-disk rather than
a database — run volume here is "however many times you click Run in the
dashboard," nowhere near the scale where a database would earn its
complexity.
"""

import json
import os
import re
import time
from datetime import datetime, timezone

RUNS_DIR = "runs"

MAX_SLUG_WORDS = 6


def _slugify(task: str) -> str:
    """Turn a task description into a short, filename-safe slug.

    Not trying to be a general-purpose slugifier — just enough to make run_id
    filenames recognizable at a glance in a file listing, e.g.
    "fix-the-bug-in-buggyaddpy" rather than an opaque timestamp alone.
    """
    words = re.findall(r"[a-zA-Z0-9]+", task.lower())[:MAX_SLUG_WORDS]
    return "-".join(words) if words else "task"


def _new_run_id(task: str) -> str:
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
    return f"{timestamp}_{_slugify(task)}"


def save_run(task: str, result: dict) -> str:
    """Save a completed run_agent() result to disk.

    Args:
        task: The original task description passed to run_agent().
        result: The full dict returned by run_agent() — status, message,
                trajectory, cost_summary, anomalies_by_step, rollback_history,
                and warning if present.

    Returns:
        The run_id this run was saved under (also the filename, minus the
        .json extension) — pass this to load_run() to retrieve it again.
    """
    os.makedirs(RUNS_DIR, exist_ok=True)

    run_id = _new_run_id(task)
    record = {
        "run_id": run_id,
        "task": task,
        "saved_at": time.time(),
        **result,
    }

    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    with open(path, "w") as f:
        json.dump(record, f, indent=2, default=str)

    return run_id


def list_runs() -> list:
    """List all saved runs, most recent first.

    Returns lightweight summaries (not full trajectories) — enough to
    populate a history list/table without loading every run's full detail
    just to render an index.

    Returns:
        A list of dicts: run_id, task, saved_at, status, step_count,
        total_tokens, anomaly_count, rollback_count.
    """
    if not os.path.isdir(RUNS_DIR):
        return []

    summaries = []
    for filename in os.listdir(RUNS_DIR):
        if not filename.endswith(".json"):
            continue
        try:
            with open(os.path.join(RUNS_DIR, filename)) as f:
                record = json.load(f)
        except (json.JSONDecodeError, OSError):
            continue  # skip corrupt/partially-written files rather than crash the whole list

        cost_summary = record.get("cost_summary") or {}
        anomalies_by_step = record.get("anomalies_by_step") or {}
        summaries.append({
            "run_id": record.get("run_id", filename.removesuffix(".json")),
            "task": record.get("task", ""),
            "saved_at": record.get("saved_at", 0),
            "status": record.get("status", "unknown"),
            "step_count": len(record.get("trajectory", [])),
            "total_tokens": cost_summary.get("total_tokens", 0),
            "anomaly_count": sum(len(v) for v in anomalies_by_step.values()),
            "rollback_count": len(record.get("rollback_history", [])),
        })

    summaries.sort(key=lambda s: s["saved_at"], reverse=True)
    return summaries


def load_run(run_id: str) -> dict:
    """Load the full saved record for one run.

    Args:
        run_id: The run_id returned by save_run() (matches the filename).

    Returns:
        The full saved record: run_id, task, saved_at, plus everything
        run_agent() originally returned.

    Raises:
        FileNotFoundError: If no run with this run_id was ever saved.
    """
    path = os.path.join(RUNS_DIR, f"{run_id}.json")
    with open(path) as f:
        return json.load(f)
