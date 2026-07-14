"""
Measures real token usage per step from actual agent runs.

This is the foundation for calibrating TOKEN_MIN_RATIO — the multiplier
above which a step's token count is considered a spike. Without knowing
what "normal" token usage looks like on your actual model and tasks, any
ratio threshold is a guess.

A single clean run and a single verbose run is NOT enough to trust a
threshold derived from them — one data point per bucket can't distinguish
"this is what normal variance looks like" from "this run happened to be
unusually calm/spiky." So this script runs multiple trials per task
category (see each entry's "trials" count in base_tasks below) to build a
real distribution:
1. Two DIFFERENT clean, solvable tasks (different bug, different file),
   run a couple of times each — establishes the normal range of
   step-vs-rolling-average ratios. Repeats of the SAME clean task are
   deterministic at temperature=0 with a reset fixture, so real coverage
   comes from task diversity, not from repeating one task many times.
2. A task that forces a genuinely larger output, run repeatedly —
   establishes how reliably a real spike clears any given threshold. This
   one DOES vary meaningfully run-to-run even at temperature=0, so more
   repeats of it are worth the quota.

The "spike" task asks the agent to write a companion file containing a
fixed number of enumerated test functions, rather than asking it to
"explain in extensive detail." That earlier phrasing relied on the model
choosing to ramble, which temperature=0 (needed for reliable tool-calling)
actively suppresses — measured spikes shrank to the point of overlapping
with normal variance. Asking for N discrete, explicitly-counted items is a
structural token cost the model can't shortcut regardless of temperature.

tasks/buggy_add.py is mutated by every trial's write_file calls, so without
a reset, later trials silently inherit whatever an earlier trial left
behind (they stop being independent, comparable runs). reset_fixture() is
called before every trial to guarantee every one starts from the same
known-buggy state.

Run with:
    python3 -m tests.measure_token_baseline

Requires GROQ_API_KEY (or GOOGLE_API_KEY, depending on agent.agent.PROVIDER)
in .env. Results saved to tests/fixtures/token_baseline_results.json.
"""

import json
import os
import time
from dotenv import load_dotenv
from monitor.logger import TrajectoryLogger

load_dotenv()

MAX_STEPS = 15

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

# A second, genuinely different clean fixture (different bug, different
# operator, different file) rather than just repeating buggy_add.py.
# At temperature=0 with a reset fixture, repeated trials of the SAME task
# produce byte-identical output — that confirms reproducibility, not the
# range of normal variance. Real diversity for the clean bucket has to come
# from different tasks, not repeats of one.
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

TEST_FILE_PATH = "tasks/buggy_add_tests.py"


def reset_fixture():
    """Restore both buggy task fixtures to their known-buggy pristine state.

    Every trial's write_file calls physically overwrite these files on
    disk. Without resetting before each trial, later trials silently
    inherit whatever an earlier trial left behind — sometimes already
    fixed, making the "find and fix the bug" task a no-op and producing a
    truncated, non-comparable trajectory. Resets both fixtures unconditionally
    on every trial (cheap, and avoids any cross-task contamination if a
    trial ordering ever interleaves the two clean tasks).
    """
    with open(BUGGY_ADD_PATH, "w") as f:
        f.write(BUGGY_ADD_FIXTURE)
    with open(BUGGY_MULTIPLY_PATH, "w") as f:
        f.write(BUGGY_MULTIPLY_FIXTURE)
    # the spike task's companion test file should also start fresh each
    # trial, not accumulate or get partially overwritten across runs
    if os.path.exists(TEST_FILE_PATH):
        os.remove(TEST_FILE_PATH)


def run_task_and_measure(task: str, label: str) -> dict:
    from agent.agent import build_agent

    reset_fixture()

    print(f"\n{'='*55}")
    print(f"Running: {label}")
    print(f"Task: {task[:70]}")

    logger = TrajectoryLogger()
    agent = build_agent()
    messages = [{"role": "user", "content": task}]
    pending_tool_call = None
    steps_taken = 0

    try:
        for chunk in agent.stream({"messages": messages}, stream_mode="values"):
            if steps_taken >= MAX_STEPS:
                print(f"  Hit MAX_STEPS cap ({MAX_STEPS}), stopping.")
                break

            latest = chunk["messages"][-1]
            msg_type = getattr(latest, "type", None)

            if msg_type == "ai" and getattr(latest, "tool_calls", None):
                call = latest.tool_calls[0]
                usage = getattr(latest, "usage_metadata", None)
                token_count = usage.get("total_tokens", 0) if usage else 0
                pending_tool_call = {
                    "tool_used": call["name"],
                    "input_summary": call["args"],
                    "token_count": token_count,
                }

            elif msg_type == "tool" and pending_tool_call:
                merged = {**pending_tool_call,
                          "output_summary": str(latest.content)[:500]}
                logger.log_step(
                    tool_used=merged["tool_used"],
                    input_summary=merged["input_summary"],
                    output_summary=merged["output_summary"],
                    token_count=merged["token_count"],
                )
                steps_taken += 1
                pending_tool_call = None

                traj = logger.get_active_trajectory()
                last = traj[-1]
                print(f"  Step {last['step_number']}: "
                      f"{last['tool_used']:<12} "
                      f"tokens={last['token_count']:<8} "
                      f"out: {str(last['output_summary'])[:50]}")

        trajectory = logger.get_active_trajectory()
        token_counts = [s["token_count"] for s in trajectory]

        if token_counts:
            avg = sum(token_counts) / len(token_counts)
            max_tok = max(token_counts)
            min_tok = min(token_counts)
            # compute rolling average at each step and the ratio vs that average
            ratios = []
            for i in range(1, len(token_counts)):
                rolling = sum(token_counts[:i]) / i
                if rolling > 0:
                    ratios.append(round(token_counts[i] / rolling, 2))
        else:
            avg = max_tok = min_tok = 0
            ratios = []

        print(f"\n  Steps: {len(trajectory)}")
        print(f"  Token counts per step: {token_counts}")
        print(f"  Min: {min_tok}  Max: {max_tok}  Avg: {round(avg, 1)}")
        print(f"  Step-vs-rolling-avg ratios: {ratios}")
        if ratios:
            print(f"  Max ratio observed: {max(ratios)}")

        return {
            "label": label,
            "task": task,
            "steps": len(trajectory),
            "token_counts": token_counts,
            "min_tokens": min_tok,
            "max_tokens": max_tok,
            "avg_tokens": round(avg, 1),
            "step_vs_rolling_ratios": ratios,
            "max_ratio_observed": max(ratios) if ratios else None,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {"label": label, "task": task, "error": str(e)}


if __name__ == "__main__":
    MAX_RETRIES = 3

    # Clean tasks get fewer repeats each because, at temperature=0 with a
    # reset fixture, repeating the SAME task is deterministic — extra trials
    # of one clean task don't add real coverage. Two DIFFERENT clean tasks
    # run twice each gives actual diversity for the same token budget that
    # five repeats of one task would have spent on redundant data.
    # The verbose/spike task showed genuine trial-to-trial variance even at
    # temperature=0 (different token counts each run), so more repeats of
    # it are worth the quota.
    base_tasks = [
        {
            "task": (
                "Read the file tasks/buggy_add.py, find the bug, "
                "fix it, and run it to confirm it works."
            ),
            "label": "clean_buggy_add",
            "trials": 2,
        },
        {
            "task": (
                "Read the file tasks/buggy_multiply.py, find the bug, "
                "fix it, and run it to confirm it works."
            ),
            "label": "clean_buggy_multiply",
            "trials": 2,
        },
        {
            "task": (
                "Read the file tasks/buggy_add.py, find the bug, and fix it. "
                "Then create a new file tasks/buggy_add_tests.py containing "
                "exactly 25 separate test functions named test_case_1 through "
                "test_case_25. Each function must call add() with a different "
                "pair of integers of your choosing, and assert the result "
                "equals the correct sum, with a one-line comment above each "
                "function stating the two inputs and the expected output. "
                "After writing the file, run it to confirm all 25 tests pass."
            ),
            "label": "verbose_buggy_add",
            "trials": 5,
        },
    ]

    # Interleave trials round-robin across task categories rather than
    # running all of one then all of the next — spreads any time-of-day/API
    # -load effects evenly instead of concentrating them in one bucket.
    trial_runs = []
    max_trials = max(t["trials"] for t in base_tasks)
    for trial_num in range(1, max_trials + 1):
        for t in base_tasks:
            if trial_num <= t["trials"]:
                trial_runs.append({
                    "task": t["task"],
                    "base_label": t["label"],
                    "label": f"{t['label']}_trial{trial_num}",
                })

    def is_daily_quota_error(r: dict) -> bool:
        # A daily token quota (TPD) rejection is unrecoverable by retrying —
        # unlike Groq's transient malformed-tool-call errors, more attempts
        # just waste whatever quota is left. Distinguishing this lets the
        # loop stop immediately instead of churning through every remaining
        # trial only to have each one fail the same way.
        err = r.get("error", "")
        return "tokens per day" in err or "TPD" in err

    results = []
    stopped_early = False
    for i, t in enumerate(trial_runs):
        result = run_task_and_measure(t["task"], t["label"])

        # Groq's tool-calling models occasionally malform their own
        # tool-call JSON on a fresh turn (400 "Failed to call a function"),
        # unrelated to anything about the task itself. Retrying the same
        # task a couple of times clears it without polluting the baseline
        # with a fabricated result — but not if it's a daily quota error,
        # which retrying cannot fix.
        attempt = 1
        while "error" in result and not is_daily_quota_error(result) and attempt < MAX_RETRIES:
            attempt += 1
            print(f"\n  Retry {attempt}/{MAX_RETRIES} for '{t['label']}' "
                  f"after error: {result['error'][:100]}")
            time.sleep(5)
            result = run_task_and_measure(t["task"], t["label"])

        result["base_label"] = t["base_label"]
        results.append(result)

        if is_daily_quota_error(result):
            print(f"\nHit a DAILY token quota limit — stopping the run early instead "
                  f"of burning remaining quota on doomed retries.\n  {result['error'][:200]}")
            stopped_early = True
            break

        if i < len(trial_runs) - 1:
            print(f"\nWaiting 20s before next run... ({i + 1}/{len(trial_runs)} done)")
            time.sleep(20)

    if stopped_early:
        print(f"\n{len(results)}/{len(trial_runs)} trials completed before the quota hit.")

    output_path = "tests/fixtures/token_baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print("FINAL SUMMARY")
    print(f"{'='*55}")

    for base_label in {t["label"] for t in base_tasks}:
        trials = [r for r in results if r.get("base_label") == base_label]
        ok = [r for r in trials if "error" not in r]
        failed = [r for r in trials if "error" in r]

        print(f"\n{base_label}: {len(ok)}/{len(trials)} trials succeeded")
        for r in ok:
            print(f"  {r['label']}: max_ratio={r['max_ratio_observed']}  "
                  f"tokens={r['token_counts']}")
        for r in failed:
            print(f"  {r['label']}: ERROR — {r['error'][:80]}")

        all_ratios = [ratio for r in ok for ratio in r["step_vs_rolling_ratios"]]
        if all_ratios:
            print(f"  Pooled across {len(ok)} trials: "
                  f"min={min(all_ratios)}  max={max(all_ratios)}")

    # leave the repo in a clean state rather than whatever the last trial's
    # write_file calls happened to produce
    reset_fixture()

    print(f"\nFull results saved to {output_path}")
    print(f"\nNext step: python3 -m tests.tune_token_threshold_real")