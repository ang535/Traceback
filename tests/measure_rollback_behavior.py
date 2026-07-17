"""
Measures real rollback behavior via the actual, full production entrypoint
(agent.run_agent()) to ground MAX_ROLLBACKS_PER_TASK (monitor/rollback.py).

Why this needs a live run: MAX_ROLLBACKS_PER_TASK is a retry-budget policy,
not a threshold on a signal — there's no offline scenario to label the way
severity or token thresholds could be. The real question — "how many
rollback attempts does a genuinely recoverable failure typically need before
it either succeeds or should be escalated" — only shows up by actually
watching a real failure get detected, rolled back, and retried.

Task design: reuses the EXACT task shape (a simple one-line bug fix on
tasks/buggy_add.py or tasks/buggy_multiply.py) that already produced a real
redundant-loop trajectory in a live dashboard run — recorded as the "REAL
scenario" in tests/fixtures/severity_scenarios.py (infinite_loop +
token_explosion at step 11, after the agent had already fixed the bug and
kept re-verifying). With ROLLBACK_SEVERITY_THRESHOLD now fixed at 0.45 (it
was 0.8, unvalidated, when that trajectory was recorded — the anomaly never
actually triggered a rollback at the time), this same failure mode should
now be caught for real.

This calls agent.run_agent() directly — the actual production path,
including the real RollbackManager and its MAX_ROLLBACKS_PER_TASK
enforcement — not a simplified stream loop, so whatever is observed here is
exactly how the live system behaves, not an approximation of it.

CAVEAT going in (flagged before running, not after): the earlier redundant-
loop behavior was observed in ONE live dashboard run. The same task shape
run via tests/measure_token_baseline.py (a simpler, more mechanical
stream-and-count harness, not agent.run_agent()) completed cleanly in 3
steps with no looping, across 4 separate trials. This suggests the looping
behavior may not be reliably reproducible on demand — this experiment might
observe zero rollbacks across every trial, which is itself a valid (if
less informative) result: it would mean 3 has margin over what's actually
been observed, not proof of the exact right number.

Run with:
    python3 -m tests.measure_rollback_behavior

Requires GROQ_API_KEY (or GOOGLE_API_KEY, depending on agent.agent.PROVIDER)
in .env. Results saved to tests/fixtures/rollback_behavior_results.json.
"""

import json
import os
import time
from dotenv import load_dotenv

load_dotenv()

BUGGY_ADD_PATH = "tasks/buggy_add.py"
BUGGY_ADD_FIXTURE = '''# This function should return the sum of two numbers
# but it has a bug — can you find and fix it?

def add(a, b):
    return a - b


if __name__ == "__main__":
    result = add(3, 5)
    print(f"3 + 5 = {result}")
    assert result == 8, f"Expected 8 but got {result}"
    print("All tests passed!")'''

BUGGY_MULTIPLY_PATH = "tasks/buggy_multiply.py"
BUGGY_MULTIPLY_FIXTURE = '''# This function should return the product of two numbers
# but it has a bug — can you find and fix it?

def multiply(a, b):
    return a + b


if __name__ == "__main__":
    result = multiply(3, 5)
    print(f"3 * 5 = {result}")
    assert result == 15, f"Expected 15 but got {result}"
    print("All tests passed!")'''

TASKS = [
    ("Read the file tasks/buggy_add.py, find the bug, fix it, and run it to confirm it works.",
     "buggy_add"),
    ("Read the file tasks/buggy_multiply.py, find the bug, fix it, and run it to confirm it works.",
     "buggy_multiply"),
]


def reset_fixtures():
    with open(BUGGY_ADD_PATH, "w") as f:
        f.write(BUGGY_ADD_FIXTURE)
    with open(BUGGY_MULTIPLY_PATH, "w") as f:
        f.write(BUGGY_MULTIPLY_FIXTURE)


def is_daily_quota_error(exc: Exception) -> bool:
    msg = str(exc)
    return "tokens per day" in msg or "TPD" in msg


if __name__ == "__main__":
    from agent.agent import run_agent

    NUM_TRIALS = 3
    results = []

    trial_specs = [(TASKS[i % len(TASKS)][0], TASKS[i % len(TASKS)][1], i + 1)
                   for i in range(NUM_TRIALS)]

    for task, base_label, trial_num in trial_specs:
        label = f"{base_label}_trial{trial_num}"
        reset_fixtures()

        print(f"\n{'='*55}")
        print(f"Running: {label}")

        try:
            result = run_agent(task)
            rollback_history = result.get("rollback_history", [])
            anomalies_by_step = result.get("anomalies_by_step", {})

            print(f"  Status: {result['status']}")
            print(f"  Steps: {len(result.get('trajectory', []))}")
            print(f"  Rollbacks: {len(rollback_history)}")
            for rb in rollback_history:
                print(f"    attempt {rb['attempt']}: rolled back to step {rb['rollback_point']}, "
                      f"anomaly_types={rb['anomaly_types']}, severity={rb['severity']}")
            if anomalies_by_step:
                print(f"  Anomalies flagged at steps: {list(anomalies_by_step.keys())}")

            results.append({
                "label": label,
                "task": task,
                "status": result["status"],
                "steps": len(result.get("trajectory", [])),
                "rollback_history": rollback_history,
                "anomalies_by_step": {k: v for k, v in anomalies_by_step.items()},
                "cost_summary": result.get("cost_summary"),
            })

        except Exception as e:
            if is_daily_quota_error(e):
                print(f"\nHit a DAILY token quota limit — stopping early.\n  {e}")
                results.append({"label": label, "task": task, "error": str(e)})
                break
            print(f"  Error: {e}")
            results.append({"label": label, "task": task, "error": str(e)})

        if trial_num < NUM_TRIALS:
            print("\nWaiting 20s before next trial...")
            time.sleep(20)

    output_path = "tests/fixtures/rollback_behavior_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    for r in results:
        if "error" in r:
            print(f"  {r['label']}: ERROR — {r['error'][:100]}")
        else:
            print(f"  {r['label']}: status={r['status']}, steps={r['steps']}, "
                  f"rollbacks={len(r['rollback_history'])}")

    reset_fixtures()
    print(f"\nFull results saved to {output_path}")
    print(f"\nNext step: share these results back and I'll ground MAX_ROLLBACKS_PER_TASK "
          f"with whatever this actually showed.")
