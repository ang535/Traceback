"""
Measures a real, genuinely explosive FIRST step — the one case
ABSOLUTE_TOKEN_CEILING exists to catch, that no trial in
tests/fixtures/token_baseline_results.json ever produced. Every task in that
earlier baseline started with a small read_file call (step 1 was always
350-429 tokens, clean or verbose), because the agent always reads the target
file before acting. That's realistic for THIS project's tasks, but it means
ABSOLUTE_TOKEN_CEILING (4000) has only ever been validated against "never a
false positive," not "correctly catches a real explosive first step" —
because no such case existed in the data.

This script forces exactly that case: a task where the agent's first action
has to be a large write, not a small read, so step 1 itself carries a real,
substantial token cost. Two trials only — the goal here isn't a distribution
(TOKEN_MIN_RATIO already has that), it's a single real existence check: does
a genuinely large first action actually threaten to approach or cross 4000
tokens, or does even a deliberately-forced-large first step stay well under
it? Either answer is useful and turns ABSOLUTE_TOKEN_CEILING's caveat from
"untested" into "tested against the most adversarial case this project's
task shape can realistically produce."

Run with:
    python3 -m tests.measure_first_step_baseline

Requires GROQ_API_KEY (or GOOGLE_API_KEY, depending on agent.agent.PROVIDER)
in .env. Results saved to tests/fixtures/first_step_baseline_results.json.
"""

import json
import os
import time
from dotenv import load_dotenv
from monitor.logger import TrajectoryLogger

load_dotenv()

MAX_STEPS = 15

BIG_FIRST_STEP_PATH = "tasks/big_first_step.py"
BIG_FIRST_STEP_TASK = (
    "Without reading any files first, create a new file tasks/big_first_step.py "
    "containing exactly 25 separate functions named function_1 through function_25. "
    "Each function must take two integer parameters and return their sum, with a "
    "one-line docstring above each function describing what it does. Write the "
    "ENTIRE file in a single write_file call as your very first action — do not "
    "call read_file at all before writing. After writing the file, run it to "
    "confirm it executes without error (a __main__ block calling a couple of the "
    "functions and printing the results is enough)."
)


def reset_fixture():
    """Remove any leftover output file from a prior trial so each trial starts clean."""
    if os.path.exists(BIG_FIRST_STEP_PATH):
        os.remove(BIG_FIRST_STEP_PATH)


def run_task_and_measure(task: str, label: str) -> dict:
    from agent.agent import build_agent

    reset_fixture()

    print(f"\n{'='*55}")
    print(f"Running: {label}")

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
        first_step_tool = trajectory[0]["tool_used"] if trajectory else None

        print(f"\n  Steps: {len(trajectory)}")
        print(f"  Token counts per step: {token_counts}")
        print(f"  First step tool: {first_step_tool}  "
              f"(should be write_file — if it's read_file, the model ignored "
              f"the 'don't read first' instruction and this trial isn't valid)")
        if token_counts:
            print(f"  Step 1 token count: {token_counts[0]}")

        return {
            "label": label,
            "task": task,
            "steps": len(trajectory),
            "token_counts": token_counts,
            "first_step_tool": first_step_tool,
            "step1_tokens": token_counts[0] if token_counts else None,
        }

    except Exception as e:
        print(f"  Error: {e}")
        return {"label": label, "task": task, "error": str(e)}


if __name__ == "__main__":
    MAX_RETRIES = 3
    NUM_TRIALS = 2

    def is_daily_quota_error(r: dict) -> bool:
        err = r.get("error", "")
        return "tokens per day" in err or "TPD" in err

    results = []
    for trial_num in range(1, NUM_TRIALS + 1):
        label = f"explosive_first_step_trial{trial_num}"
        result = run_task_and_measure(BIG_FIRST_STEP_TASK, label)

        attempt = 1
        while "error" in result and not is_daily_quota_error(result) and attempt < MAX_RETRIES:
            attempt += 1
            print(f"\n  Retry {attempt}/{MAX_RETRIES} for '{label}' "
                  f"after error: {result['error'][:100]}")
            time.sleep(5)
            result = run_task_and_measure(BIG_FIRST_STEP_TASK, label)

        results.append(result)

        if is_daily_quota_error(result):
            print(f"\nHit a DAILY token quota limit — stopping early.\n  {result['error'][:200]}")
            break

        if trial_num < NUM_TRIALS:
            print(f"\nWaiting 20s before next trial...")
            time.sleep(20)

    output_path = "tests/fixtures/first_step_baseline_results.json"
    with open(output_path, "w") as f:
        json.dump(results, f, indent=2)

    print(f"\n{'='*55}")
    print("SUMMARY")
    print(f"{'='*55}")
    valid = [r for r in results if "error" not in r and r.get("first_step_tool") == "write_file"]
    invalid = [r for r in results if "error" not in r and r.get("first_step_tool") != "write_file"]
    for r in valid:
        print(f"  {r['label']}: step1_tokens={r['step1_tokens']}  (vs ABSOLUTE_TOKEN_CEILING=4000)")
    for r in invalid:
        print(f"  {r['label']}: INVALID — first step was '{r.get('first_step_tool')}', "
              f"not write_file (model read first despite instructions)")

    reset_fixture()
    print(f"\nFull results saved to {output_path}")
    print(f"\nNext step: share these results back and I'll update ABSOLUTE_TOKEN_CEILING's "
          f"documentation with the real measurement.")
