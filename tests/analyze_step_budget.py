"""
Grounds MAX_TOTAL_STEPS (agent/agent.py) against every real trial's step
count collected so far across this project — no new live runs needed, since
step count is already recorded in every existing results file.

MAX_TOTAL_STEPS is a last-resort circuit breaker, not the primary control —
that's supposed to be the anomaly detectors + rollback system, which should
catch a genuinely stuck trajectory well before a step-count cap ever
matters. This asks a narrower question than "what's the optimal cap": given
every real task shape actually run so far, how much headroom does 25 give
over the most demanding LEGITIMATE trajectory observed, and is there any
real evidence it's too tight or too loose?

Data sources pulled together:
- tests/fixtures/token_baseline_results.json: clean bug-fix trials (3 steps
  each) and the deliberately larger "write 25 test functions" structural
  task (10-11 steps) — the widest legitimate task-difficulty spread
  available.
- tests/fixtures/first_step_baseline_results.json: the adversarial
  write-first trial (2 steps).
- runs/ (saved dashboard runs): included for completeness but flagged
  separately — the one saved run there hit MAX_TOTAL_STEPS=25 by looping on
  an already-completed task, which is exactly the bug ROLLBACK_SEVERITY_
  THRESHOLD's fix (0.8 -> 0.45) was built to catch. That run is NOT
  legitimate task difficulty; it's the pre-fix bug in action, so it's
  excluded from the "legitimate step count" analysis rather than treated as
  a real data point about how many steps a hard task needs.

Run with:
    python3 -m tests.analyze_step_budget
"""

import glob
import json

MAX_TOTAL_STEPS = 25  # current value, from agent/agent.py


def load_json_safe(path):
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return []


if __name__ == "__main__":
    legitimate_runs = []

    for r in load_json_safe("tests/fixtures/token_baseline_results.json"):
        if "steps" in r:
            legitimate_runs.append((r["label"], r["steps"], r["base_label"]))

    for r in load_json_safe("tests/fixtures/first_step_baseline_results.json"):
        if r.get("steps"):
            legitimate_runs.append((r["label"], r["steps"], "adversarial_first_step"))

    print(f"Legitimate real trial step counts collected so far ({len(legitimate_runs)} trials):\n")
    for label, steps, category in sorted(legitimate_runs, key=lambda x: x[1]):
        print(f"  {steps:3} steps — {label:35} ({category})")

    step_counts = [s for _, s, _ in legitimate_runs]
    if step_counts:
        max_steps = max(step_counts)
        print(f"\nMax legitimate step count observed: {max_steps}")
        print(f"Current MAX_TOTAL_STEPS: {MAX_TOTAL_STEPS}")
        print(f"Headroom: {round(MAX_TOTAL_STEPS / max_steps, 2)}x the hardest real task seen so far")

    print("\nExcluded from the analysis above (not legitimate task difficulty):")
    for path in sorted(glob.glob("runs/*.json")):
        run = load_json_safe(path)
        traj_len = len(run.get("trajectory", []))
        rollback_count = len(run.get("rollback_history", []))
        print(f"  {path}: {traj_len} steps, {rollback_count} rollbacks, status={run.get('status')} "
              f"— predates the ROLLBACK_SEVERITY_THRESHOLD fix (agent looped on an "
              f"already-completed task instead of stopping); not a real measure of task difficulty.")

    print("\nCAVEAT: the widest real task difficulty tested so far tops out at 11 steps (the "
          "deliberately larger structural task). No real trial has ever exercised a genuinely "
          "complex, multi-file, multi-verification task in the 12-24 step range, so this analysis "
          "can confirm 25 has real margin over everything actually observed, but can't rule out "
          "that a legitimately harder real task might need to sit closer to the cap. Unlike "
          "MIN_STEPS_FOR_TOKEN_CHECK or LOW_CONFIDENCE_PENALTY, there's no labeled scenario set "
          "or sweep possible here — this is real data bounding a plausible range, not a validated "
          "optimum.")
