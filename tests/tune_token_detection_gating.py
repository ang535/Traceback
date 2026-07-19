"""
Validates MIN_STEPS_FOR_TOKEN_CHECK and ABSOLUTE_TOKEN_CEILING (monitor/
detector.py) by replaying the REAL per-step token counts already collected
in tests/fixtures/token_baseline_results.json — the same real Groq trial
data TOKEN_MIN_RATIO was validated against. No new live agent runs needed.

Why these two constants matter together: check_token_explosion (the
rolling-average check) only activates once len(trajectory) >=
MIN_STEPS_FOR_TOKEN_CHECK prior steps exist. check_absolute_token_ceiling
has no such requirement and is active from step 1 — it exists specifically
to cover the blind spot while the rolling check is still warming up. If
MIN_STEPS_FOR_TOKEN_CHECK is set too high, real early spikes can fall
through BOTH checks: too early for the rolling average, not large enough in
absolute terms to break the ceiling.

This script reconstructs each real trial's trajectory and simulates
check_token_explosion with MIN_STEPS_FOR_TOKEN_CHECK swept across
[1, 2, 3, 4, 5], using the real, already-validated TOKEN_MIN_RATIO=2.2.
For each candidate: does it ever false-positive on a CLEAN trial (bad), and
what's the earliest step it catches a real spike in a VERBOSE trial (lower
is better, since earlier detection means less wasted spend before rollback).

Run with:
    python3 -m tests.tune_token_detection_gating
"""

import json

from monitor.scorer import TOKEN_MIN_RATIO  # the real, already-validated value

RESULTS_PATH = "tests/fixtures/token_baseline_results.json"


def check_token_explosion_param(trajectory: list, current_step: dict,
                                 min_steps: int, multiplier: float = TOKEN_MIN_RATIO):
    """Mirrors monitor.detector.check_token_explosion exactly, with
    MIN_STEPS_FOR_TOKEN_CHECK exposed as a parameter to sweep instead of a
    hardcoded module-level constant."""
    if len(trajectory) < min_steps:
        return None
    rolling_avg = sum(s["token_count"] for s in trajectory) / len(trajectory)
    current_tokens = current_step["token_count"]
    if current_tokens > rolling_avg * multiplier:
        return {"ratio": round(current_tokens / rolling_avg, 2)}
    return None


def load_trials():
    with open(RESULTS_PATH) as f:
        raw = json.load(f)
    # drop failed API calls (rate limits etc.) — no token_counts to replay
    return [r for r in raw if "token_counts" in r]


def simulate(trial: dict, min_steps: int) -> dict:
    """Replay one trial step-by-step, returns the first step (if any) where
    the rolling-average check fires."""
    token_counts = trial["token_counts"]
    trajectory = []
    for i, tokens in enumerate(token_counts):
        current_step = {"token_count": tokens}
        result = check_token_explosion_param(trajectory, current_step, min_steps)
        trajectory.append(current_step)
        if result:
            return {"fired_at_step": i + 1, "ratio": result["ratio"]}
    return {"fired_at_step": None, "ratio": None}


if __name__ == "__main__":
    trials = load_trials()
    clean_trials = [t for t in trials if t["base_label"].startswith("clean")]
    verbose_trials = [t for t in trials if t["base_label"].startswith("verbose")]

    print(f"Replaying {len(clean_trials)} clean trials and {len(verbose_trials)} verbose trials "
          f"from real Groq runs (tests/fixtures/token_baseline_results.json), "
          f"TOKEN_MIN_RATIO fixed at {TOKEN_MIN_RATIO} (the real, validated value).\n")

    print("=== MIN_STEPS_FOR_TOKEN_CHECK sweep ===\n")
    for min_steps in [1, 2, 3, 4, 5]:
        false_positives = []
        for t in clean_trials:
            r = simulate(t, min_steps)
            if r["fired_at_step"] is not None:
                false_positives.append((t["label"], r))

        earliest_detections = []
        for t in verbose_trials:
            r = simulate(t, min_steps)
            earliest_detections.append((t["label"], r["fired_at_step"], r["ratio"]))

        marker = " <-- current default" if min_steps == 3 else ""
        print(f"min_steps={min_steps}{marker}")
        if false_positives:
            print(f"  FALSE POSITIVES on clean trials: {false_positives}")
        else:
            print(f"  No false positives on any clean trial.")
        for label, step, ratio in earliest_detections:
            detected = f"step {step} (ratio={ratio})" if step else "NEVER detected by rolling check"
            print(f"  {label}: {detected}")
        print()

    print("=== ABSOLUTE_TOKEN_CEILING check ===\n")
    print("Step-1 token counts observed across ordinary real trials (the only step the "
          "rolling-average check can never cover, regardless of MIN_STEPS_FOR_TOKEN_CHECK, "
          "since there's no prior history at all):")
    step1_counts = [t["token_counts"][0] for t in trials]
    print(f"  {sorted(step1_counts)}")
    print(f"  max observed (ordinary tasks): {max(step1_counts)}, current ABSOLUTE_TOKEN_CEILING: 4000")

    try:
        with open("tests/fixtures/first_step_baseline_results.json") as f:
            first_step_trials = json.load(f)
        valid = [t for t in first_step_trials if t.get("first_step_tool") == "write_file"]
        if valid:
            adversarial_step1 = [t["step1_tokens"] for t in valid]
            print(f"\n  Adversarial real data (tests/measure_first_step_baseline.py — forced "
                  f"first action was a large write, not a read): {adversarial_step1}")
            print(f"  Largest real adversarial step 1: {max(adversarial_step1)} "
                  f"({round(100 * max(adversarial_step1) / 4000, 1)}% of the ceiling, "
                  f"margin {round(4000 / max(adversarial_step1), 2)}x)")
            print(f"  4000 stays above the most adversarial real case measured; margin is "
                  f"{round(4000 / max(step1_counts), 1)}x against ordinary-task data alone.")
        else:
            print("\n  first_step_baseline_results.json exists but has no valid trials "
                  "(model read a file first in every trial).")
    except FileNotFoundError:
        print("\n  CAVEAT: no real trial has had an explosive first step — every ordinary "
              "task's step 1 is a small read. Confirms 4000 is safe against everything "
              "observed so far but there's no data point for a genuinely explosive step 1. "
              "Run tests/measure_first_step_baseline.py to get one.")
